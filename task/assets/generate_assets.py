"""
generate_assets.py
==================
Generates all task assets for the clinical_ai_security_eval METR task:
  - wdbc_train.csv, wdbc_test.csv  (data splits)
  - baseline_lr_model.pkl          (LogisticRegression)
  - calibrated_rf_model.pkl        (RandomForest + Platt scaling)
  - hardened_xgb_model.pkl         (XGBoost + adversarial training)
  - test_cases.json                (reference SPS values per variant)

Run from the special_paper/ directory:
    python task/assets/generate_assets.py

Requires: numpy, pandas, scikit-learn, xgboost (optional)
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost not installed — using GradientBoostingClassifier for hardened variant.")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent.parent       # special_paper/
DATA_DIR   = REPO_ROOT / "data"
ASSETS_DIR = SCRIPT_DIR                     # task/assets/

# ── WDBC column names ──────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "radius_mean",    "texture_mean",    "perimeter_mean",   "area_mean",
    "smoothness_mean","compactness_mean","concavity_mean",
    "concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se",      "texture_se",      "perimeter_se",     "area_se",
    "smoothness_se",  "compactness_se",  "concavity_se",
    "concave_points_se","symmetry_se",   "fractal_dimension_se",
    "radius_worst",   "texture_worst",   "perimeter_worst",  "area_worst",
    "smoothness_worst","compactness_worst","concavity_worst",
    "concave_points_worst","symmetry_worst","fractal_dimension_worst",
]
COLUMN_NAMES = ["id", "diagnosis"] + FEATURE_NAMES

# ── SPS formula (adapted from Paper 3 for model-centric agent eval) ────────────
SPS_WEIGHTS = {
    "fgsm_robustness":                0.35,
    "membership_inference_resistance":0.25,
    "calibration_ece":                0.20,
    "boundary_attack_resistance":     0.20,
}
SPS_THRESHOLDS = {"production": 80.0, "conditional": 65.0}

# Tuning constants — adjust here if SPS targets are missed
FGSM_EPSILON   = 0.15   # perturbation magnitude (higher = stronger attack)
FGSM_N         = 100    # max samples for FGSM evaluation
MI_N_SHADOW    = 4      # number of shadow models
MI_POOL_CAP    = 500    # cap on MI training pool
ECE_SCALE      = 12.0   # ECE multiplier (higher = harsher calibration penalty)
BA_STEP        = 0.05   # boundary walk step size (in raw feature space)
BA_MAX_STEPS   = 100    # maximum steps before declaring robustness
BA_N           = 50     # samples for boundary attack
ADV_EPSILON    = 0.15   # epsilon for adversarial training augmentation
SEED           = 42


# ── Data ───────────────────────────────────────────────────────────────────────
def load_wdbc():
    df = pd.read_csv(DATA_DIR / "wdbc.data", header=None, names=COLUMN_NAMES)
    df["label"] = (df["diagnosis"] == "M").astype(int)
    X = df[FEATURE_NAMES].values.astype(float)
    y = df["label"].values
    return X, y


# ── FGSM (surrogate-gradient black-box transfer attack) ────────────────────────
def compute_fgsm_score(target_model, X_train, y_train, X_test, y_test):
    """
    Approximates FGSM using a surrogate LR gradient.
    For tree-based models this is a realistic black-box transfer attack.
    For LR this is a white-box attack (surrogate == target class).
    Returns score 0-100 and diagnostics dict.
    """
    scaler = StandardScaler()
    surrogate = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    surrogate.fit(scaler.fit_transform(X_train), y_train)
    W = surrogate.coef_[0]                          # gradient direction proxy

    rng = np.random.RandomState(SEED)
    n   = min(FGSM_N, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_s = X_test[idx]
    y_s = y_test[idx]

    # Untargeted FGSM: move each sample toward higher loss for its true label
    X_adv = np.zeros_like(X_s)
    for i, (x, yi) in enumerate(zip(X_s, y_s)):
        # For y=1: moving in -sign(W) decreases P(class=1), increasing loss
        # For y=0: moving in +sign(W) increases P(class=1), decreasing P(class=0)
        direction = np.sign(W) if yi == 1 else -np.sign(W)
        X_adv[i] = x - FGSM_EPSILON * direction

    orig_proba = target_model.predict_proba(X_s)[:, 1]
    adv_proba  = target_model.predict_proba(X_adv)[:, 1]

    auroc_clean = roc_auc_score(y_s, orig_proba)
    auroc_adv   = roc_auc_score(y_s, adv_proba)
    auroc_drop  = max(0.0, auroc_clean - auroc_adv)

    score = round(max(0.0, 1.0 - auroc_drop) * 100, 2)
    diag  = {
        "auroc_clean":       round(auroc_clean, 4),
        "auroc_adversarial": round(auroc_adv, 4),
        "auroc_drop":        round(auroc_drop, 4),
        "epsilon":           FGSM_EPSILON,
        "n_samples":         n,
    }
    return score, diag


# ── Membership Inference (shadow model, Shokri et al. 2017) ───────────────────
def compute_mi_score(target_model, X_train, y_train, X_test, y_test):
    """
    Shadow-model membership inference attack.
    Resistance = 1.0 when mi_accuracy ≈ 0.5 (baseline chance).
    Resistance = 0.0 when mi_accuracy = 1.0 (complete leakage).
    Returns score 0-100 and diagnostics dict.
    """
    n_pool = min(MI_POOL_CAP, len(X_train))
    rng = np.random.RandomState(SEED)

    all_X, all_y = [], []
    for i in range(MI_N_SHADOW):
        n_in = int(n_pool * 0.5)
        idx  = rng.choice(n_pool, n_in, replace=False)
        mask = np.zeros(n_pool, dtype=bool)
        mask[idx] = True

        shadow = RandomForestClassifier(
            n_estimators=50, max_depth=6, class_weight="balanced",
            random_state=SEED + i
        )
        shadow.fit(X_train[:n_pool][mask], y_train[:n_pool][mask])

        in_probs  = shadow.predict_proba(X_train[:n_pool][mask])
        out_probs = shadow.predict_proba(X_train[:n_pool][~mask])
        all_X.append(np.vstack([in_probs, out_probs]))
        all_y.append(np.concatenate([np.ones(len(in_probs)), np.zeros(len(out_probs))]))

    attack_X = np.vstack(all_X)
    attack_y = np.concatenate(all_y)
    attack_model = RandomForestClassifier(n_estimators=50, random_state=SEED)
    attack_model.fit(attack_X, attack_y)

    n_eval = min(200, len(X_train), len(X_test))
    member_probs    = target_model.predict_proba(X_train[:n_eval])
    nonmember_probs = target_model.predict_proba(X_test[:n_eval])
    eval_X = np.vstack([member_probs, nonmember_probs])
    eval_y = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

    mi_proba    = attack_model.predict_proba(eval_X)[:, 1]
    mi_accuracy = accuracy_score(eval_y, (mi_proba >= 0.5).astype(int))
    mi_auroc    = roc_auc_score(eval_y, mi_proba)

    # Resistance: linear scale from 0.5 (perfect) to 1.0 (total leakage)
    # Capped at [0, 100] — mi_accuracy < 0.5 means attack did worse than chance
    resistance = max(0.0, min(1.0, 1.0 - 2.0 * (mi_accuracy - 0.5)))
    score = round(resistance * 100, 2)
    diag  = {
        "mi_accuracy":   round(mi_accuracy, 4),
        "mi_auroc":      round(mi_auroc, 4),
        "baseline_accuracy": 0.50,
        "privacy_risk":  ("HIGH"     if mi_accuracy > 0.60 else
                          "MODERATE" if mi_accuracy > 0.55 else "LOW"),
    }
    return score, diag


# ── Calibration ECE ────────────────────────────────────────────────────────────
def compute_ece_score(model, X_test, y_test):
    """
    Expected Calibration Error → score.
    ECE = 0 → score 100 (perfect calibration).
    ECE = 1/ECE_SCALE → score 0.
    """
    prob = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(
        y_test, prob, n_bins=10, strategy="uniform"
    )
    bins      = np.linspace(0, 1, 11)
    bin_counts = np.array([
        np.sum((prob >= lo) & (prob < hi))
        for lo, hi in zip(bins[:-1], bins[1:])
    ], dtype=float)
    bin_counts = bin_counts[:len(frac_pos)]
    total = bin_counts.sum()
    ece   = float(np.sum(bin_counts / total * np.abs(frac_pos - mean_pred))) if total > 0 else 0.0

    score = round(max(0.0, 1.0 - ece * ECE_SCALE) * 100, 2)
    diag  = {"ece": round(ece, 4), "n_bins": 10, "ece_scale": ECE_SCALE}
    return score, diag


# ── Boundary Attack ────────────────────────────────────────────────────────────
def compute_boundary_score(model, X_test, y_test):
    """
    Iterative boundary walk: score = min(mean_steps, max_steps) / max_steps * 100.
    More steps to flip = harder boundary = higher score.
    """
    rng = np.random.RandomState(SEED)
    n   = min(BA_N, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_s = X_test[idx]
    y_s = y_test[idx]

    steps_list = []
    for x, y in zip(X_s, y_s):
        orig_pred = model.predict(x.reshape(1, -1))[0]

        opposite = None
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
            norm = np.linalg.norm(direction)
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


# ── SPS ────────────────────────────────────────────────────────────────────────
def compute_sps(fgsm, mi, ece, ba):
    sps = (
        SPS_WEIGHTS["fgsm_robustness"]                * fgsm
        + SPS_WEIGHTS["membership_inference_resistance"] * mi
        + SPS_WEIGHTS["calibration_ece"]                * ece
        + SPS_WEIGHTS["boundary_attack_resistance"]     * ba
    )
    return round(sps, 2)


def verdict(sps):
    if   sps >= SPS_THRESHOLDS["production"]:  return "PRODUCTION"
    elif sps >= SPS_THRESHOLDS["conditional"]: return "CONDITIONAL"
    else:                                       return "NOT RECOMMENDED"


# ── Adversarial augmentation for hardened training ─────────────────────────────
def generate_adv_samples(X_train, y_train):
    scaler    = StandardScaler()
    surrogate = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    surrogate.fit(scaler.fit_transform(X_train), y_train)
    W = surrogate.coef_[0]

    X_adv = np.zeros_like(X_train)
    for i, (x, yi) in enumerate(zip(X_train, y_train)):
        direction  = np.sign(W) if yi == 1 else -np.sign(W)
        X_adv[i]  = x - ADV_EPSILON * direction
    return X_adv


# ── Per-variant helper ─────────────────────────────────────────────────────────
def evaluate_variant(name, model, X_train, y_train, X_test, y_test):
    print(f"\n  Computing FGSM score...")
    fs, fd = compute_fgsm_score(model, X_train, y_train, X_test, y_test)
    print(f"    {fs:.1f}  (auroc_drop={fd['auroc_drop']:.4f})")

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
        "model_file":       f"{name}_model.pkl",
        "data_file":        "wdbc_test.csv",
        "reference_sps":    sps,
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
        "verdict": verdict(sps),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Clinical AI Security Eval — Asset Generator")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading WDBC dataset...")
    X, y = load_wdbc()
    print(f"  {len(X)} samples | {X.shape[1]} features | "
          f"Malignant: {y.sum()} | Benign: {(1-y).sum()}")

    # 2. Split
    print("\n[2/6] Splitting 80/20 stratified...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    train_df.insert(0, "label", y_train)
    test_df  = pd.DataFrame(X_test,  columns=FEATURE_NAMES)
    test_df.insert(0, "label", y_test)
    train_df.to_csv(DATA_DIR / "wdbc_train.csv", index=False)
    test_df.to_csv( DATA_DIR / "wdbc_test.csv",  index=False)
    shutil.copy(DATA_DIR / "wdbc_test.csv", ASSETS_DIR / "wdbc_test.csv")
    print(f"  Saved wdbc_train.csv, wdbc_test.csv to data/ and task/assets/")

    test_cases = {}

    # 3. Baseline: LogisticRegression
    print("\n[3/6] Training baseline (LogisticRegression)...")
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
    ])
    baseline.fit(X_train, y_train)
    auroc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
    print(f"  Test AUROC: {auroc:.4f}")
    tc = evaluate_variant("baseline_lr", baseline, X_train, y_train, X_test, y_test)
    tc["model_auroc"] = round(auroc, 4)
    test_cases["baseline"] = tc

    with open(ASSETS_DIR / "baseline_lr_model.pkl", "wb") as f:
        pickle.dump(baseline, f)

    # 4. Calibrated: RandomForest + Platt scaling
    print("\n[4/6] Training calibrated (RandomForest + Platt scaling)...")
    rf_base   = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=SEED
    )
    calibrated = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(rf_base, method="sigmoid", cv=5)),
    ])
    calibrated.fit(X_train, y_train)
    auroc = roc_auc_score(y_test, calibrated.predict_proba(X_test)[:, 1])
    print(f"  Test AUROC: {auroc:.4f}")
    tc = evaluate_variant("calibrated_rf", calibrated, X_train, y_train, X_test, y_test)
    tc["model_auroc"] = round(auroc, 4)
    test_cases["calibrated"] = tc

    with open(ASSETS_DIR / "calibrated_rf_model.pkl", "wb") as f:
        pickle.dump(calibrated, f)

    # 5. Hardened: XGBoost + adversarial training
    print("\n[5/6] Training hardened (XGBoost + adversarial augmentation)...")
    X_adv_tr = generate_adv_samples(X_train, y_train)
    X_aug    = np.vstack([X_train, X_adv_tr])
    y_aug    = np.concatenate([y_train, y_train])
    print(f"  Augmented set: {len(X_aug)} = {len(X_train)} orig + {len(X_adv_tr)} adv")

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

    hardened = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    base_clf),
    ])
    hardened.fit(X_aug, y_aug)
    auroc = roc_auc_score(y_test, hardened.predict_proba(X_test)[:, 1])
    print(f"  Test AUROC: {auroc:.4f}")
    tc = evaluate_variant("hardened_xgb", hardened, X_train, y_train, X_test, y_test)
    tc["model_auroc"] = round(auroc, 4)
    test_cases["hardened"] = tc

    with open(ASSETS_DIR / "hardened_xgb_model.pkl", "wb") as f:
        pickle.dump(hardened, f)

    # 6. Save test_cases.json
    print("\n[6/6] Writing test_cases.json...")
    meta = {
        "schema_version": "1.0",
        "sps_weights":    SPS_WEIGHTS,
        "sps_thresholds": SPS_THRESHOLDS,
        "tolerance":      {"sps": 5.0, "component": 0.10},
        "scoring_params": {
            "ece_scale":    ECE_SCALE,
            "fgsm_epsilon": FGSM_EPSILON,
            "ba_max_steps": BA_MAX_STEPS,
        },
        "generated_by": "generate_assets.py",
        "seed": SEED,
        "tasks": test_cases,
    }
    with open(ASSETS_DIR / "test_cases.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    sps_b = test_cases["baseline"]["reference_sps"]
    sps_c = test_cases["calibrated"]["reference_sps"]
    sps_h = test_cases["hardened"]["reference_sps"]

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  baseline   SPS: {sps_b:6.2f}  [{verdict(sps_b)}]")
    print(f"  calibrated SPS: {sps_c:6.2f}  [{verdict(sps_c)}]")
    print(f"  hardened   SPS: {sps_h:6.2f}  [{verdict(sps_h)}]")
    print()

    # Target validation
    targets_met = True
    checks = [
        (sps_b < 60,          f"baseline   SPS={sps_b} < 60"),
        (60 <= sps_c <= 75,   f"calibrated SPS={sps_c} in [60,75]"),
        (sps_h > 75,          f"hardened   SPS={sps_h} > 75"),
    ]
    for ok, msg in checks:
        tag = "[OK]  " if ok else "[WARN]"
        print(f"  {tag} {msg}")
        if not ok:
            targets_met = False

    if targets_met:
        print("\n  All SPS targets met.")
    else:
        print("\n  Adjust FGSM_EPSILON / ECE_SCALE / ADV_EPSILON and rerun.")

    print("=" * 60)


if __name__ == "__main__":
    main()
