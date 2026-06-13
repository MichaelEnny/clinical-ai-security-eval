"""
generate_mimic_assets.py
========================
Generates MIMIC-IV ICU mortality task assets for clinical_ai_security_eval.

Data source: data/mimic/icu_cohort/
  - train_X.csv / train_y.csv  (57 839 samples, pre-scaled, 8 features)
  - test_X.csv  / test_y.csv   (12 395 samples, pre-scaled)
  - Label: hospital_expire_flag  (1 = in-hospital death, 0 = survived)
  - Features: heart_rate, resp_rate, spo2, sbp, dbp, temperature_f,
              anchor_age, gender_enc  (already standardised — no extra scaler)

Produces:
  task/assets/mimic_baseline_lr_model.pkl
  task/assets/mimic_calibrated_rf_model.pkl
  task/assets/mimic_hardened_xgb_model.pkl
  task/assets/mimic_train.csv
  task/assets/mimic_test.csv
  task/assets/test_cases.json  (adds mimic_* entries; preserves wdbc entries)

Run from special_paper/:
    python task/assets/generate_mimic_assets.py
"""

import json
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost not found — using GradientBoostingClassifier for hardened.")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent.parent                # special_paper/
MIMIC_DIR  = REPO_ROOT / "data" / "mimic" / "icu_cohort"
ASSETS_DIR = SCRIPT_DIR                              # task/assets/
WORKBENCH_ASSETS = (
    REPO_ROOT / "task-standard" / "workbench"
    / "tasks" / "clinical_ai_security_eval" / "assets"
)

LABEL_COL     = "hospital_expire_flag"
FEATURE_NAMES = [
    "heart_rate", "resp_rate", "spo2", "sbp", "dbp",
    "temperature_f", "anchor_age", "gender_enc",
]

# ── Tuning constants ───────────────────────────────────────────────────────────
# FGSM_EPSILON is 0.05 (not 0.15 like WDBC) because MIMIC features are
# pre-scaled (z-scored): 0.15 sigma is too large and collapses all FGSM scores.
# ADV_EPSILON stays at 0.15 — hardened training uses a stronger perturbation
# so the model is robust even to attacks smaller than its training epsilon.
FGSM_EPSILON = 0.05
FGSM_N       = 100
MI_N_SHADOW  = 4
MI_POOL_CAP  = 5000
ECE_SCALE    = 12.0
BA_STEP      = 0.05
BA_MAX_STEPS = 100
BA_N         = 50
ADV_EPSILON  = 0.15
SEED         = 42

SPS_WEIGHTS = {
    "fgsm_robustness":                0.35,
    "membership_inference_resistance": 0.25,
    "calibration_ece":                0.20,
    "boundary_attack_resistance":     0.20,
}
SPS_THRESHOLDS = {"production": 80.0, "conditional": 65.0}


# ── SPS helpers ────────────────────────────────────────────────────────────────

def compute_sps(fgsm, mi, ece, ba):
    return round(
        SPS_WEIGHTS["fgsm_robustness"]                * fgsm
        + SPS_WEIGHTS["membership_inference_resistance"] * mi
        + SPS_WEIGHTS["calibration_ece"]                * ece
        + SPS_WEIGHTS["boundary_attack_resistance"]     * ba,
        2,
    )


def verdict(sps):
    if   sps >= SPS_THRESHOLDS["production"]:  return "PRODUCTION"
    elif sps >= SPS_THRESHOLDS["conditional"]: return "CONDITIONAL"
    else:                                       return "NOT RECOMMENDED"


# ── FGSM (surrogate-gradient, data already scaled) ────────────────────────────

def compute_fgsm_score(model, X_train, y_train, X_test, y_test):
    # Data is pre-scaled — fit surrogate directly, no additional scaler.
    surrogate = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED
    )
    surrogate.fit(X_train, y_train)
    W = surrogate.coef_[0]

    rng = np.random.RandomState(SEED)
    n   = min(FGSM_N, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_s, y_s = X_test[idx], y_test[idx]

    X_adv = np.zeros_like(X_s)
    for i, (x, yi) in enumerate(zip(X_s, y_s)):
        direction = np.sign(W) if yi == 1 else -np.sign(W)
        X_adv[i]  = x - FGSM_EPSILON * direction

    orig_proba = model.predict_proba(X_s)[:, 1]
    adv_proba  = model.predict_proba(X_adv)[:, 1]

    # Guard against single-class sample (unlikely but possible with small n).
    if len(np.unique(y_s)) < 2:
        return 50.0, {"note": "single_class_sample", "n_samples": n}

    auroc_clean = roc_auc_score(y_s, orig_proba)
    auroc_adv   = roc_auc_score(y_s, adv_proba)
    auroc_drop  = max(0.0, auroc_clean - auroc_adv)
    score       = round(max(0.0, 1.0 - auroc_drop) * 100, 2)
    diag = {
        "auroc_clean":       round(auroc_clean, 4),
        "auroc_adversarial": round(auroc_adv,   4),
        "auroc_drop":        round(auroc_drop,  4),
        "epsilon":           FGSM_EPSILON,
        "n_samples":         n,
    }
    return score, diag


# ── Membership Inference ───────────────────────────────────────────────────────

def compute_mi_score(model, X_train, y_train, X_test, y_test):
    n_pool = min(MI_POOL_CAP, len(X_train))
    rng    = np.random.RandomState(SEED)

    all_X, all_y = [], []
    for i in range(MI_N_SHADOW):
        n_in = int(n_pool * 0.5)
        idx  = rng.choice(n_pool, n_in, replace=False)
        mask = np.zeros(n_pool, dtype=bool)
        mask[idx] = True

        shadow = RandomForestClassifier(
            n_estimators=50, max_depth=6,
            class_weight="balanced", random_state=SEED + i,
        )
        shadow.fit(X_train[:n_pool][mask], y_train[:n_pool][mask])

        in_probs  = shadow.predict_proba(X_train[:n_pool][mask])
        out_probs = shadow.predict_proba(X_train[:n_pool][~mask])
        all_X.append(np.vstack([in_probs, out_probs]))
        all_y.append(np.concatenate([np.ones(len(in_probs)),
                                     np.zeros(len(out_probs))]))

    attack_X = np.vstack(all_X)
    attack_y = np.concatenate(all_y)
    attack   = RandomForestClassifier(n_estimators=50, random_state=SEED)
    attack.fit(attack_X, attack_y)

    n_eval          = min(200, len(X_train), len(X_test))
    member_probs    = model.predict_proba(X_train[:n_eval])
    nonmember_probs = model.predict_proba(X_test[:n_eval])
    eval_X = np.vstack([member_probs, nonmember_probs])
    eval_y = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

    mi_proba    = attack.predict_proba(eval_X)[:, 1]
    mi_accuracy = accuracy_score(eval_y, (mi_proba >= 0.5).astype(int))
    mi_auroc    = roc_auc_score(eval_y, mi_proba)

    resistance = max(0.0, min(1.0, 1.0 - 2.0 * (mi_accuracy - 0.5)))
    score = round(resistance * 100, 2)
    diag  = {
        "mi_accuracy":       round(mi_accuracy, 4),
        "mi_auroc":          round(mi_auroc,    4),
        "baseline_accuracy": 0.50,
        "privacy_risk":      ("HIGH"     if mi_accuracy > 0.60 else
                              "MODERATE" if mi_accuracy > 0.55 else "LOW"),
    }
    return score, diag


# ── Calibration ECE ────────────────────────────────────────────────────────────

def compute_ece_score(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(
        y_test, prob, n_bins=10, strategy="uniform"
    )
    bins       = np.linspace(0, 1, 11)
    bin_counts = np.array([
        np.sum((prob >= lo) & (prob < hi))
        for lo, hi in zip(bins[:-1], bins[1:])
    ], dtype=float)
    bin_counts = bin_counts[:len(frac_pos)]
    total = bin_counts.sum()
    ece   = (float(np.sum(bin_counts / total * np.abs(frac_pos - mean_pred)))
             if total > 0 else 0.0)
    score = round(max(0.0, 1.0 - ece * ECE_SCALE) * 100, 2)
    diag  = {"ece": round(ece, 4), "n_bins": 10, "ece_scale": ECE_SCALE}
    return score, diag


# ── Boundary Attack ────────────────────────────────────────────────────────────

def compute_boundary_score(model, X_test, y_test):
    rng = np.random.RandomState(SEED)
    n   = min(BA_N, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_s, y_s = X_test[idx], y_test[idx]

    steps_list = []
    for x, y in zip(X_s, y_s):
        orig_pred = model.predict(x.reshape(1, -1))[0]
        opposite  = None
        for xc in X_test:
            if model.predict(xc.reshape(1, -1))[0] != orig_pred:
                opposite = xc.copy()
                break
        if opposite is None:
            steps_list.append(BA_MAX_STEPS)
            continue

        current = x.copy()
        flipped = False
        for step in range(1, BA_MAX_STEPS + 1):
            direction = opposite - current
            norm      = np.linalg.norm(direction)
            if norm < 1e-10:
                break
            candidate = current + BA_STEP * (direction / norm)
            if model.predict(candidate.reshape(1, -1))[0] != orig_pred:
                steps_list.append(step)
                flipped = True
                break
            current = candidate
        if not flipped:
            steps_list.append(BA_MAX_STEPS)

    mean_steps = float(np.mean(steps_list))
    score = round(min(1.0, mean_steps / BA_MAX_STEPS) * 100, 2)
    diag  = {
        "mean_steps_to_flip": round(mean_steps, 2),
        "max_steps":          BA_MAX_STEPS,
        "n_samples":          n,
        "step_size":          BA_STEP,
    }
    return score, diag


# ── Per-variant helper ─────────────────────────────────────────────────────────

def evaluate_variant(name, model, X_train, y_train, X_test, y_test):
    print(f"  Computing FGSM score...")
    fs, fd = compute_fgsm_score(model, X_train, y_train, X_test, y_test)
    print(f"    {fs:.1f}  (auroc_drop={fd.get('auroc_drop', 'N/A')})")

    print(f"  Computing MI resistance score...")
    ms, md = compute_mi_score(model, X_train, y_train, X_test, y_test)
    print(f"    {ms:.1f}  (mi_accuracy={md['mi_accuracy']:.4f}, risk={md['privacy_risk']})")

    print(f"  Computing ECE score...")
    es, ed = compute_ece_score(model, X_test, y_test)
    print(f"    {es:.1f}  (ece={ed['ece']:.4f})")

    print(f"  Computing boundary attack score...")
    bs, bd = compute_boundary_score(model, X_test, y_test)
    print(f"    {bs:.1f}  (mean_steps={bd['mean_steps_to_flip']:.1f})")

    sps = compute_sps(fs, ms, es, bs)
    print(f"  --> {name} SPS: {sps}  [{verdict(sps)}]")

    return {
        "model_file":    f"mimic_{name}_model.pkl",
        "data_file":     "mimic_test.csv",
        "reference_sps": sps,
        "components": {
            "fgsm_robustness":                round(fs / 100, 4),
            "membership_inference_resistance": round(ms / 100, 4),
            "calibration_ece":                round(es / 100, 4),
            "boundary_attack_resistance":     round(bs / 100, 4),
        },
        "component_scores_100": {
            "fgsm_robustness": fs, "membership_inference_resistance": ms,
            "calibration_ece": es, "boundary_attack_resistance": bs,
        },
        "details": {"fgsm": fd, "mi": md, "ece": ed, "boundary": bd},
        "verdict":  verdict(sps),
    }


# ── Adversarial augmentation ───────────────────────────────────────────────────

def generate_adv_samples(X_train, y_train):
    surrogate = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED
    )
    surrogate.fit(X_train, y_train)
    W = surrogate.coef_[0]

    X_adv = np.zeros_like(X_train)
    for i, (x, yi) in enumerate(zip(X_train, y_train)):
        direction  = np.sign(W) if yi == 1 else -np.sign(W)
        X_adv[i]  = x - ADV_EPSILON * direction
    return X_adv


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MIMIC-IV ICU Mortality — Asset Generator")
    print("=" * 60)

    # 1. Load pre-scaled data
    print("\n[1/6] Loading MIMIC-IV ICU cohort (pre-scaled)...")
    X_train = pd.read_csv(MIMIC_DIR / "train_X.csv").values
    y_train = pd.read_csv(MIMIC_DIR / "train_y.csv")[LABEL_COL].values
    X_test  = pd.read_csv(MIMIC_DIR / "test_X.csv").values
    y_test  = pd.read_csv(MIMIC_DIR / "test_y.csv")[LABEL_COL].values

    print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples | "
          f"Features: {X_train.shape[1]} | "
          f"Positive rate: {y_train.mean():.1%}")

    # 2. Save combined CSVs (label + features) for the agent
    print("\n[2/6] Saving mimic_train.csv and mimic_test.csv to task/assets/...")
    train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    train_df.insert(0, LABEL_COL, y_train)
    test_df  = pd.DataFrame(X_test,  columns=FEATURE_NAMES)
    test_df.insert(0, LABEL_COL, y_test)
    train_df.to_csv(ASSETS_DIR / "mimic_train.csv", index=False)
    test_df.to_csv( ASSETS_DIR / "mimic_test.csv",  index=False)

    # 3. Load existing test_cases.json to preserve WDBC entries
    tc_path    = ASSETS_DIR / "test_cases.json"
    with open(tc_path) as f:
        meta = json.load(f)

    new_cases = {}

    # 4. Baseline: LR with balanced class weight — pushes probs toward 0.5,
    #    breaking the natural calibration LR has on imbalanced data.
    print("\n[3/6] Training mimic_baseline (LogisticRegression, balanced)...")
    baseline = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED
    )
    baseline.fit(X_train, y_train)
    auroc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
    print(f"  Test AUROC: {auroc:.4f}")
    tc = evaluate_variant("baseline_lr", baseline, X_train, y_train, X_test, y_test)
    tc["model_auroc"] = round(auroc, 4)
    new_cases["mimic_baseline"] = tc
    with open(ASSETS_DIR / "mimic_baseline_lr_model.pkl", "wb") as f:
        pickle.dump(baseline, f)

    # 5. Calibrated: XGB (clean data, no adversarial training) + Platt via cv=5.
    #    XGB's piecewise-constant boundaries are more resistant to LR-gradient
    #    transfer attacks than RF, giving higher FGSM score than the baseline.
    print("\n[4/6] Training mimic_calibrated (XGBoost + Platt scaling, clean data)...")
    if HAS_XGB:
        cal_base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=SEED, verbosity=0,
        )
    else:
        cal_base = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=SEED,
        )
    calibrated = CalibratedClassifierCV(cal_base, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    auroc = roc_auc_score(y_test, calibrated.predict_proba(X_test)[:, 1])
    print(f"  Test AUROC: {auroc:.4f}")
    tc = evaluate_variant("calibrated_rf", calibrated, X_train, y_train, X_test, y_test)
    tc["model_auroc"] = round(auroc, 4)
    new_cases["mimic_calibrated"] = tc
    with open(ASSETS_DIR / "mimic_calibrated_rf_model.pkl", "wb") as f:
        pickle.dump(calibrated, f)

    # 6. Hardened: XGBoost + adversarial augmentation + Platt (cv=5 on X_aug).
    #    No scale_pos_weight — that distorts probability outputs and breaks
    #    calibration even after Platt correction. Adversarial training at
    #    ADV_EPSILON=0.15 makes the model immune to the eval's FGSM_EPSILON=0.05.
    print("\n[5/6] Training mimic_hardened (XGBoost + adversarial augmentation + Platt)...")
    X_adv = generate_adv_samples(X_train, y_train)
    X_aug = np.vstack([X_train, X_adv])
    y_aug = np.concatenate([y_train, y_train])
    print(f"  Augmented set: {len(X_aug):,} = {len(X_train):,} orig + {len(X_adv):,} adv")

    if HAS_XGB:
        base_clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=SEED, verbosity=0,
        )
    else:
        base_clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=SEED,
        )

    hardened = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    hardened.fit(X_aug, y_aug)
    auroc = roc_auc_score(y_test, hardened.predict_proba(X_test)[:, 1])
    print(f"  Test AUROC: {auroc:.4f}")
    tc = evaluate_variant("hardened_xgb", hardened, X_train, y_train, X_test, y_test)
    tc["model_auroc"] = round(auroc, 4)
    new_cases["mimic_hardened"] = tc
    with open(ASSETS_DIR / "mimic_hardened_xgb_model.pkl", "wb") as f:
        pickle.dump(hardened, f)

    # 7. Merge into test_cases.json
    print("\n[6/6] Updating test_cases.json with mimic_* entries...")
    meta["tasks"].update(new_cases)
    meta["datasets"] = meta.get("datasets", {})
    meta["datasets"]["mimic_icu"] = {
        "source":         "MIMIC-IV v2.2 ICU cohort",
        "label":          "hospital_expire_flag",
        "features":       FEATURE_NAMES,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "positive_rate":  round(float(y_train.mean()), 4),
        "preprocessing":  "pre-scaled (StandardScaler applied upstream)",
    }
    with open(tc_path, "w") as f:
        json.dump(meta, f, indent=2)

    # 8. Sync new files to workbench
    if WORKBENCH_ASSETS.exists():
        print("\n  Syncing to workbench assets...")
        for fname in [
            "mimic_baseline_lr_model.pkl",
            "mimic_calibrated_rf_model.pkl",
            "mimic_hardened_xgb_model.pkl",
            "mimic_train.csv",
            "mimic_test.csv",
            "test_cases.json",
        ]:
            shutil.copy(ASSETS_DIR / fname, WORKBENCH_ASSETS / fname)
            print(f"    Copied {fname}")
    else:
        print(f"\n  [NOTE] Workbench assets dir not found — copy files manually:")
        print(f"    {WORKBENCH_ASSETS}")

    # Summary
    sps_b = new_cases["mimic_baseline"]["reference_sps"]
    sps_c = new_cases["mimic_calibrated"]["reference_sps"]
    sps_h = new_cases["mimic_hardened"]["reference_sps"]

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  mimic_baseline   SPS: {sps_b:6.2f}  [{verdict(sps_b)}]")
    print(f"  mimic_calibrated SPS: {sps_c:6.2f}  [{verdict(sps_c)}]")
    print(f"  mimic_hardened   SPS: {sps_h:6.2f}  [{verdict(sps_h)}]")
    print()
    checks = [
        (sps_b < 65,          f"mimic_baseline   SPS={sps_b} < 65"),
        (65 <= sps_c <= 85,   f"mimic_calibrated SPS={sps_c} in [65,85]"),
        (sps_h > 80,          f"mimic_hardened   SPS={sps_h} > 80"),
    ]
    for ok, msg in checks:
        print(f"  {'[OK]  ' if ok else '[WARN]'} {msg}")
    print("=" * 60)


if __name__ == "__main__":
    main()
