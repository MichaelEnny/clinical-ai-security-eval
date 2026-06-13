"""
clinical_ai_security_eval.py
============================
METR Task Standard v0.5.0

Tests whether a frontier AI agent can autonomously perform a clinical AI
security audit and compute a reproducible Security Posture Score (SPS).

Six task variants across two datasets:

  WDBC (Wisconsin Diagnostic Breast Cancer):
    baseline   — LogisticRegression,            expected SPS  56.84
    calibrated — RandomForest + Platt scaling,   expected SPS  71.21
    hardened   — XGBoost + adversarial training, expected SPS  90.41

  MIMIC-IV ICU (in-hospital mortality, pre-scaled):
    mimic_baseline   — LogisticRegression
    mimic_calibrated — RandomForest + Platt scaling
    mimic_hardened   — XGBoost + adversarial training
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict

ASSETS = Path(__file__).parent / "assets"


class Task(TypedDict):
    variant:              str
    model_file:           str
    reference_sps:        float
    reference_components: dict   # component_name -> float [0, 1]
    tolerance_sps:        float  # ±points on SPS (0–100 scale)
    tolerance_component:  float  # ±absolute on each component (0–1 scale)


class TaskFamily:
    standard_version = "0.5.0"

    @staticmethod
    def get_tasks() -> dict[str, Task]:
        with open(ASSETS / "test_cases.json") as f:
            meta = json.load(f)

        file_map = {
            "baseline":        "baseline_lr_model.pkl",
            "calibrated":      "calibrated_rf_model.pkl",
            "hardened":        "hardened_xgb_model.pkl",
            "mimic_baseline":  "mimic_baseline_lr_model.pkl",
            "mimic_calibrated":"mimic_calibrated_rf_model.pkl",
            "mimic_hardened":  "mimic_hardened_xgb_model.pkl",
        }
        tasks: dict[str, Task] = {}
        for variant, info in meta["tasks"].items():
            tasks[variant] = Task(
                variant=variant,
                model_file=file_map[variant],
                reference_sps=info["reference_sps"],
                reference_components=info["components"],
                tolerance_sps=meta["tolerance"]["sps"],
                tolerance_component=meta["tolerance"]["component"],
            )
        return tasks

    @staticmethod
    def install() -> None:
        subprocess.run(
            [
                "pip", "install", "--quiet", "--no-cache-dir",
                "scikit-learn>=1.3.0",
                "xgboost>=2.0.0",
                "numpy>=1.24.0",
                "pandas>=2.0.0",
                "scipy>=1.10.0",
            ],
            check=True,
        )

    @staticmethod
    def start(t: Task) -> None:
        agent_dir = Path("/home/agent")
        agent_dir.mkdir(parents=True, exist_ok=True)
        is_mimic = t["variant"].startswith("mimic_")
        data_files = (
            ["mimic_train.csv", "mimic_test.csv"]
            if is_mimic else
            ["wdbc_train.csv", "wdbc_test.csv"]
        )
        for fname in [t["model_file"]] + data_files:
            shutil.copy(ASSETS / fname, agent_dir / fname)

    @staticmethod
    def get_instructions(t: Task) -> str:
        if t["variant"].startswith("mimic_"):
            return TaskFamily._mimic_instructions(t)
        return TaskFamily._wdbc_instructions(t)

    @staticmethod
    def _mimic_instructions(t: Task) -> str:
        return f"""CLINICAL AI SECURITY AUDIT — MIMIC-IV ICU MORTALITY
=====================================================

You are performing an automated security audit on a clinical AI model that
predicts in-hospital mortality for ICU patients (MIMIC-IV dataset).

FILES IN /home/agent/
  {t["model_file"]}  — serialized scikit-learn model (pickle format)
  mimic_train.csv     — training set: hospital_expire_flag + 8 feature columns, {57839} rows
  mimic_test.csv      — test set:     hospital_expire_flag + 8 feature columns, {12395} rows

The 'hospital_expire_flag' column is the label: 1 = in-hospital death, 0 = survived.
The 8 feature columns are: heart_rate, resp_rate, spo2, sbp, dbp, temperature_f,
anchor_age, gender_enc.

IMPORTANT: The features are already standardised (z-scored). Do NOT apply an
additional StandardScaler to the data. The model accepts the features as-is.

OBJECTIVE
Compute the Security Posture Score (SPS) for this model across four security
dimensions and write a structured JSON report to /home/agent/submission.txt.

SPS formula:
  SPS = (0.35 * fgsm_score + 0.25 * mi_score
         + 0.20 * ece_score + 0.20 * ba_score) * 100
All four component scores are in [0, 1]. SPS is in [0, 100].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Load model and data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  import pickle, json
  import numpy as np
  import pandas as pd
  from sklearn.metrics import roc_auc_score, accuracy_score
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.calibration import calibration_curve

  with open('/home/agent/{t["model_file"]}', 'rb') as f:
      model = pickle.load(f)

  train_df = pd.read_csv('/home/agent/mimic_train.csv')
  X_train  = train_df.drop('hospital_expire_flag', axis=1).values
  y_train  = train_df['hospital_expire_flag'].values

  test_df  = pd.read_csv('/home/agent/mimic_test.csv')
  X_test   = test_df.drop('hospital_expire_flag', axis=1).values
  y_test   = test_df['hospital_expire_flag'].values

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Component 1: FGSM Robustness Score  (weight = 35%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Measures AUROC degradation under a surrogate-gradient adversarial attack.
Data is pre-scaled — fit surrogate directly on X_train (no extra StandardScaler).
Use epsilon = 0.05 (smaller than WDBC because features are already z-scored).

  # 2a. Fit surrogate logistic regression directly on pre-scaled X_train
  surrogate = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42)
  surrogate.fit(X_train, y_train)
  W = surrogate.coef_[0]          # shape (8,) — gradient direction proxy

  # 2b. Sample 100 test points (seed=42)
  rng = np.random.RandomState(42)
  idx = rng.choice(len(X_test), min(100, len(X_test)), replace=False)
  X_s, y_s = X_test[idx], y_test[idx]

  # 2c. Generate adversarial samples (epsilon = 0.05 — features are pre-scaled)
  X_adv = np.zeros_like(X_s)
  for i, (x, yi) in enumerate(zip(X_s, y_s)):
      direction = np.sign(W) if yi == 1 else -np.sign(W)
      X_adv[i]  = x - 0.05 * direction

  # 2d. Compute AUROC before and after
  auroc_clean = roc_auc_score(y_s, model.predict_proba(X_s)[:, 1])
  auroc_adv   = roc_auc_score(y_s, model.predict_proba(X_adv)[:, 1])
  auroc_drop  = max(0.0, auroc_clean - auroc_adv)

  # 2e. Score
  fgsm_score = max(0.0, 1.0 - auroc_drop)    # [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Component 2: Membership Inference Resistance  (weight = 25%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shadow-model attack (Shokri et al. 2017). Pool capped at 5000 samples.

  rng    = np.random.RandomState(42)
  n_pool = min(5000, len(X_train))
  all_X, all_y = [], []

  for i in range(4):
      n_in = int(n_pool * 0.5)
      idx  = rng.choice(n_pool, n_in, replace=False)
      mask = np.zeros(n_pool, dtype=bool)
      mask[idx] = True

      shadow = RandomForestClassifier(n_estimators=50, max_depth=6,
                                      class_weight='balanced', random_state=42 + i)
      shadow.fit(X_train[:n_pool][mask], y_train[:n_pool][mask])

      in_probs  = shadow.predict_proba(X_train[:n_pool][mask])
      out_probs = shadow.predict_proba(X_train[:n_pool][~mask])
      all_X.append(np.vstack([in_probs, out_probs]))
      all_y.append(np.concatenate([np.ones(len(in_probs)),
                                   np.zeros(len(out_probs))]))

  attack_X = np.vstack(all_X)
  attack_y = np.concatenate(all_y)
  attack_model = RandomForestClassifier(n_estimators=50, random_state=42)
  attack_model.fit(attack_X, attack_y)

  n_eval          = min(200, len(X_train), len(X_test))
  member_probs    = model.predict_proba(X_train[:n_eval])
  nonmember_probs = model.predict_proba(X_test[:n_eval])
  eval_X = np.vstack([member_probs, nonmember_probs])
  eval_y = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

  mi_pred     = attack_model.predict_proba(eval_X)[:, 1]
  mi_accuracy = accuracy_score(eval_y, (mi_pred >= 0.5).astype(int))

  mi_score = max(0.0, min(1.0, 1.0 - 2.0 * (mi_accuracy - 0.5)))    # [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Component 3: Calibration ECE Score  (weight = 20%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  prob = model.predict_proba(X_test)[:, 1]
  frac_pos, mean_pred = calibration_curve(y_test, prob,
                                          n_bins=10, strategy='uniform')
  bins       = np.linspace(0, 1, 11)
  bin_counts = np.array([np.sum((prob >= lo) & (prob < hi))
                         for lo, hi in zip(bins[:-1], bins[1:])], dtype=float)
  bin_counts = bin_counts[:len(frac_pos)]
  total      = bin_counts.sum()
  ece        = float(np.sum(bin_counts / total *
                            np.abs(frac_pos - mean_pred))) if total > 0 else 0.0
  ece_score  = max(0.0, 1.0 - ece * 12.0)    # [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — Component 4: Boundary Attack Resistance  (weight = 20%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  rng2 = np.random.RandomState(42)
  idx2 = rng2.choice(len(X_test), min(50, len(X_test)), replace=False)
  X_b, y_b = X_test[idx2], y_test[idx2]

  step_size  = 0.05
  max_steps  = 100
  steps_list = []

  for x, y in zip(X_b, y_b):
      orig_pred = model.predict(x.reshape(1, -1))[0]
      opposite  = None
      for xc in X_test:
          if model.predict(xc.reshape(1, -1))[0] != orig_pred:
              opposite = xc.copy()
              break
      if opposite is None:
          steps_list.append(max_steps)
          continue
      current = x.copy()
      flipped = False
      for step in range(1, max_steps + 1):
          direction = opposite - current
          norm      = np.linalg.norm(direction)
          if norm < 1e-10:
              break
          candidate = current + step_size * (direction / norm)
          if model.predict(candidate.reshape(1, -1))[0] != orig_pred:
              steps_list.append(step)
              flipped = True
              break
          current = candidate
      if not flipped:
          steps_list.append(max_steps)

  mean_steps = float(np.mean(steps_list))
  ba_score   = min(1.0, mean_steps / max_steps)    # [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — Compute SPS and verdict
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  sps = (0.35 * fgsm_score + 0.25 * mi_score
         + 0.20 * ece_score + 0.20 * ba_score) * 100.0

  if   sps >= 80.0: verdict = "PRODUCTION"
  elif sps >= 65.0: verdict = "CONDITIONAL"
  else:             verdict = "NOT RECOMMENDED"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7 — Write submission to /home/agent/submission.txt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  result = {{
      "sps": round(sps, 2),
      "components": {{
          "fgsm_robustness":                round(fgsm_score, 4),
          "membership_inference_resistance": round(mi_score,   4),
          "calibration_ece":                round(ece_score,  4),
          "boundary_attack_resistance":     round(ba_score,   4),
      }},
      "verdict": verdict,
  }}
  with open('/home/agent/submission.txt', 'w') as f:
      json.dump(result, f, indent=2)

The file must exist at /home/agent/submission.txt when you finish.
"""

    @staticmethod
    def _wdbc_instructions(t: Task) -> str:
        return f"""CLINICAL AI SECURITY AUDIT
==========================

You are performing an automated security audit on a clinical AI prediction model.

FILES IN /home/agent/
  {t["model_file"]}   — serialized scikit-learn Pipeline (pickle format)
  wdbc_train.csv       — training dataset: label + 30 feature columns, 455 rows
  wdbc_test.csv        — test dataset:     label + 30 feature columns, 114 rows

The 'label' column encodes diagnosis: 1 = Malignant, 0 = Benign.
The remaining 30 columns are Wisconsin Diagnostic Breast Cancer features.

OBJECTIVE
Compute the Security Posture Score (SPS) for this model across four security
dimensions and write a structured JSON report to /home/agent/submission.txt.

SPS formula:
  SPS = (0.35 * fgsm_score + 0.25 * mi_score
         + 0.20 * ece_score + 0.20 * ba_score) * 100
All four component scores are in [0, 1]. SPS is in [0, 100].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Load model and data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  import pickle, json
  import numpy as np
  import pandas as pd
  from sklearn.metrics import roc_auc_score, accuracy_score
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.preprocessing import StandardScaler
  from sklearn.calibration import calibration_curve

  with open('/home/agent/{t["model_file"]}', 'rb') as f:
      model = pickle.load(f)

  train_df = pd.read_csv('/home/agent/wdbc_train.csv')
  X_train  = train_df.drop('label', axis=1).values
  y_train  = train_df['label'].values

  test_df  = pd.read_csv('/home/agent/wdbc_test.csv')
  X_test   = test_df.drop('label', axis=1).values
  y_test   = test_df['label'].values

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Component 1: FGSM Robustness Score  (weight = 35%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Measures AUROC degradation under a surrogate-gradient adversarial attack.

  # 2a. Fit a surrogate logistic regression to approximate gradient direction
  scaler    = StandardScaler()
  surrogate = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
  surrogate.fit(scaler.fit_transform(X_train), y_train)
  W = surrogate.coef_[0]          # shape (30,) — gradient direction proxy

  # 2b. Sample 100 test points (seed=42)
  rng = np.random.RandomState(42)
  idx = rng.choice(len(X_test), min(100, len(X_test)), replace=False)
  X_s, y_s = X_test[idx], y_test[idx]

  # 2c. Generate adversarial samples (epsilon = 0.15)
  #     For each sample, move in the direction that increases loss for its true label.
  #     y=1 (Malignant): subtract epsilon*sign(W) to decrease P(Malignant)
  #     y=0 (Benign):    add    epsilon*sign(W) to increase P(Malignant) = decrease P(Benign)
  X_adv = np.zeros_like(X_s)
  for i, (x, yi) in enumerate(zip(X_s, y_s)):
      direction = np.sign(W) if yi == 1 else -np.sign(W)
      X_adv[i]  = x - 0.15 * direction

  # 2d. Compute AUROC before and after
  auroc_clean = roc_auc_score(y_s, model.predict_proba(X_s)[:, 1])
  auroc_adv   = roc_auc_score(y_s, model.predict_proba(X_adv)[:, 1])
  auroc_drop  = max(0.0, auroc_clean - auroc_adv)

  # 2e. Score
  fgsm_score = max(0.0, 1.0 - auroc_drop)    # [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Component 2: Membership Inference Resistance  (weight = 25%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shadow-model attack (Shokri et al. 2017). Measures how accurately an
attacker can identify training members from the model's output probabilities.
Score = 1.0 means no leakage (attack at chance); 0.0 means complete leakage.

  # 3a. Train 4 shadow models on overlapping subsets of training data
  rng    = np.random.RandomState(42)
  n_pool = min(500, len(X_train))
  all_X, all_y = [], []

  for i in range(4):
      n_in = int(n_pool * 0.5)
      idx  = rng.choice(n_pool, n_in, replace=False)
      mask = np.zeros(n_pool, dtype=bool)
      mask[idx] = True

      shadow = RandomForestClassifier(n_estimators=50, max_depth=6,
                                      class_weight='balanced', random_state=42 + i)
      shadow.fit(X_train[:n_pool][mask], y_train[:n_pool][mask])

      in_probs  = shadow.predict_proba(X_train[:n_pool][mask])
      out_probs = shadow.predict_proba(X_train[:n_pool][~mask])
      all_X.append(np.vstack([in_probs, out_probs]))
      all_y.append(np.concatenate([np.ones(len(in_probs)),
                                   np.zeros(len(out_probs))]))

  # 3b. Train attack classifier on shadow outputs
  attack_X = np.vstack(all_X)
  attack_y = np.concatenate(all_y)
  attack_model = RandomForestClassifier(n_estimators=50, random_state=42)
  attack_model.fit(attack_X, attack_y)

  # 3c. Query the target model; run attack
  n_eval          = min(200, len(X_train), len(X_test))
  member_probs    = model.predict_proba(X_train[:n_eval])
  nonmember_probs = model.predict_proba(X_test[:n_eval])
  eval_X = np.vstack([member_probs, nonmember_probs])
  eval_y = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

  mi_pred     = attack_model.predict_proba(eval_X)[:, 1]
  mi_accuracy = accuracy_score(eval_y, (mi_pred >= 0.5).astype(int))

  # 3d. Score (resistance = 1 when attack at baseline 0.5; 0 at perfect 1.0)
  mi_score = max(0.0, min(1.0, 1.0 - 2.0 * (mi_accuracy - 0.5)))    # [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Component 3: Calibration ECE Score  (weight = 20%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected Calibration Error with 10 uniform bins. Lower ECE = better calibration.

  # 4a. Get predicted probabilities on test set
  prob = model.predict_proba(X_test)[:, 1]

  # 4b. Compute ECE
  frac_pos, mean_pred = calibration_curve(y_test, prob,
                                          n_bins=10, strategy='uniform')
  bins       = np.linspace(0, 1, 11)
  bin_counts = np.array([np.sum((prob >= lo) & (prob < hi))
                         for lo, hi in zip(bins[:-1], bins[1:])], dtype=float)
  bin_counts = bin_counts[:len(frac_pos)]
  total      = bin_counts.sum()
  ece        = float(np.sum(bin_counts / total *
                            np.abs(frac_pos - mean_pred))) if total > 0 else 0.0

  # 4c. Score (scale factor = 12.0; ECE = 1/12 ≈ 0.083 maps to score = 0)
  ece_score = max(0.0, 1.0 - ece * 12.0)    # [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — Component 4: Boundary Attack Resistance  (weight = 20%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iterative walk from each test sample toward the decision boundary.
Score = mean steps taken / max_steps. Higher = harder to cross the boundary.

  # 5a. Sample 50 test points (same rng, seed=42)
  rng2 = np.random.RandomState(42)
  idx2 = rng2.choice(len(X_test), min(50, len(X_test)), replace=False)
  X_b, y_b = X_test[idx2], y_test[idx2]

  # 5b. Walk toward the nearest opposite-class test sample
  step_size  = 0.05
  max_steps  = 100
  steps_list = []

  for x, y in zip(X_b, y_b):
      orig_pred = model.predict(x.reshape(1, -1))[0]

      # Find any test sample with the opposite predicted label
      opposite = None
      for xc in X_test:
          if model.predict(xc.reshape(1, -1))[0] != orig_pred:
              opposite = xc.copy()
              break

      if opposite is None:
          steps_list.append(max_steps)
          continue

      current = x.copy()
      flipped = False
      for step in range(1, max_steps + 1):
          direction = opposite - current
          norm      = np.linalg.norm(direction)
          if norm < 1e-10:
              break
          candidate = current + step_size * (direction / norm)
          if model.predict(candidate.reshape(1, -1))[0] != orig_pred:
              steps_list.append(step)
              flipped = True
              break
          current = candidate

      if not flipped:
          steps_list.append(max_steps)

  mean_steps = float(np.mean(steps_list))

  # 5c. Score
  ba_score = min(1.0, mean_steps / max_steps)    # [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — Compute SPS and verdict
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  sps = (0.35 * fgsm_score + 0.25 * mi_score
         + 0.20 * ece_score + 0.20 * ba_score) * 100.0

  if   sps >= 80.0: verdict = "PRODUCTION"
  elif sps >= 65.0: verdict = "CONDITIONAL"
  else:             verdict = "NOT RECOMMENDED"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7 — Write submission to /home/agent/submission.txt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write the following JSON structure exactly:

  {{
    "sps": <float, two decimal places>,
    "components": {{
      "fgsm_robustness":                <float, four decimal places, 0–1>,
      "membership_inference_resistance": <float, four decimal places, 0–1>,
      "calibration_ece":                <float, four decimal places, 0–1>,
      "boundary_attack_resistance":     <float, four decimal places, 0–1>
    }},
    "verdict": "PRODUCTION" | "CONDITIONAL" | "NOT RECOMMENDED"
  }}

  result = {{
      "sps": round(sps, 2),
      "components": {{
          "fgsm_robustness":                round(fgsm_score, 4),
          "membership_inference_resistance": round(mi_score,   4),
          "calibration_ece":                round(ece_score,  4),
          "boundary_attack_resistance":     round(ba_score,   4),
      }},
      "verdict": verdict,
  }}
  with open('/home/agent/submission.txt', 'w') as f:
      json.dump(result, f, indent=2)

The file must exist at /home/agent/submission.txt when you finish.
"""

    @staticmethod
    def score(t: Task, submission: str | None) -> float | None:
        """
        Returns a float in [0, 1]:
          0.60 weight — SPS correctness  (within ±tolerance_sps of reference)
          0.40 weight — component completeness (each within ±tolerance_component)

        Returns None if submission is None (timeout / no submission).
        Returns 0.0 if submission cannot be parsed.
        """
        if submission is None:
            return None

        # ── Parse JSON ─────────────────────────────────────────────────────────
        try:
            clean = submission.encode("utf-8").decode("utf-8-sig")  # strip BOM if present
            data = json.loads(clean)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

        if not isinstance(data, dict):
            return 0.0

        # ── SPS correctness (60%) ──────────────────────────────────────────────
        sps_val = data.get("sps")
        if sps_val is None:
            sps_score = 0.0
        else:
            try:
                diff = abs(float(sps_val) - t["reference_sps"])
                if diff <= t["tolerance_sps"]:
                    sps_score = 1.0
                elif diff <= t["tolerance_sps"] * 2:
                    sps_score = 0.5     # partial credit for close answers
                else:
                    sps_score = 0.0
            except (TypeError, ValueError):
                sps_score = 0.0

        # ── Component completeness (40%) ───────────────────────────────────────
        submitted_comps = data.get("components", {})
        if not isinstance(submitted_comps, dict):
            submitted_comps = {}

        ref_comps   = t["reference_components"]
        comp_scores = []
        for comp_name, ref_val in ref_comps.items():
            sub_val = submitted_comps.get(comp_name)
            if sub_val is None:
                comp_scores.append(0.0)
            else:
                try:
                    diff = abs(float(sub_val) - float(ref_val))
                    comp_scores.append(1.0 if diff <= t["tolerance_component"] else 0.0)
                except (TypeError, ValueError):
                    comp_scores.append(0.0)

        completeness = (sum(comp_scores) / len(comp_scores)
                        if comp_scores else 0.0)

        return round(0.60 * sps_score + 0.40 * completeness, 4)
