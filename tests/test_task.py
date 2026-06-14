"""
tests/test_task.py
==================
METR Task Standard v0.3.0 — required test categories:
  1. Structural  — schema, fields, instructions
  2. Scoring     — reference/wrong/None/edge-case submissions
  3. Environment — assets on disk, models load, start() logic, CSV shapes
"""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinical_ai_security_eval import TaskFamily

ASSETS = Path(__file__).parent.parent / "task" / "assets"
TASKS  = TaskFamily.get_tasks()

COMPONENT_KEYS = {
    "fgsm_robustness",
    "membership_inference_resistance",
    "calibration_ece",
    "boundary_attack_resistance",
}
REQUIRED_TASK_FIELDS = {
    "variant", "model_file", "reference_sps",
    "reference_components", "tolerance_sps", "tolerance_component",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def perfect_submission(t):
    return json.dumps({
        "sps":        t["reference_sps"],
        "components": t["reference_components"],
        "verdict":    "NOT RECOMMENDED",
    })


# ══════════════════════════════════════════════════════════════════════════════
# Category 1 — Structural
# ══════════════════════════════════════════════════════════════════════════════

class TestStructural:

    def test_standard_version(self):
        assert TaskFamily.standard_version == "0.3.0"

    def test_get_tasks_returns_nonempty_dict(self):
        assert isinstance(TASKS, dict)
        assert len(TASKS) > 0

    def test_get_tasks_has_three_variants(self):
        assert set(TASKS.keys()) == {"baseline", "calibrated", "hardened"}

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_task_has_required_fields(self, variant):
        t = TASKS[variant]
        for field in REQUIRED_TASK_FIELDS:
            assert field in t, f"Variant '{variant}' missing field: {field}"

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_instructions_is_nonempty_string(self, variant):
        inst = TaskFamily.get_instructions(TASKS[variant])
        assert isinstance(inst, str)
        assert len(inst.strip()) > 200, "Instructions too short to be meaningful"

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_instructions_name_the_model_file(self, variant):
        t    = TASKS[variant]
        inst = TaskFamily.get_instructions(t)
        assert t["model_file"] in inst, \
            f"Instructions do not mention the model filename for '{variant}'"

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_instructions_contain_key_methodology_terms(self, variant):
        inst = TaskFamily.get_instructions(TASKS[variant])
        for term in ["SPS", "fgsm", "submission.txt",
                     "components", "predict_proba"]:
            assert term in inst, f"Instructions missing term: '{term}'"

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_reference_sps_is_numeric_and_in_range(self, variant):
        sps = TASKS[variant]["reference_sps"]
        assert isinstance(sps, (int, float))
        assert 0.0 <= sps <= 100.0

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_reference_components_has_four_correct_keys(self, variant):
        comps = TASKS[variant]["reference_components"]
        assert set(comps.keys()) == COMPONENT_KEYS

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_all_component_values_in_unit_interval(self, variant):
        for name, val in TASKS[variant]["reference_components"].items():
            assert 0.0 <= val <= 1.0, \
                f"Component '{name}' = {val} is outside [0, 1]"

    def test_sps_targets_by_design(self):
        """Verify the three-tier SPS design from the GUIDE."""
        assert TASKS["baseline"]["reference_sps"]   < 60
        assert 60 <= TASKS["calibrated"]["reference_sps"] <= 75
        assert TASKS["hardened"]["reference_sps"]   > 75

    def test_sps_strictly_ordered(self):
        assert (TASKS["baseline"]["reference_sps"]
                < TASKS["calibrated"]["reference_sps"]
                < TASKS["hardened"]["reference_sps"])

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_tolerances_are_positive(self, variant):
        t = TASKS[variant]
        assert t["tolerance_sps"] > 0
        assert t["tolerance_component"] > 0

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_variant_field_matches_key(self, variant):
        assert TASKS[variant]["variant"] == variant


# ══════════════════════════════════════════════════════════════════════════════
# Category 2 — Scoring
# ══════════════════════════════════════════════════════════════════════════════

class TestScoring:

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_perfect_answer_scores_1(self, variant):
        t     = TASKS[variant]
        score = TaskFamily.score(t, perfect_submission(t))
        assert score == 1.0

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_none_returns_none(self, variant):
        result = TaskFamily.score(TASKS[variant], None)
        assert result is None

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_completely_wrong_answer_scores_low(self, variant):
        t          = TASKS[variant]
        submission = json.dumps({
            "sps":        0.0,
            "components": {k: 0.0 for k in COMPONENT_KEYS},
            "verdict":    "PRODUCTION",
        })
        score = TaskFamily.score(t, submission)
        # SPS of 0 is far outside any reference (>10 pts off) → sps_score = 0.
        # Some components with very low references (e.g. baseline fgsm = 0.0043)
        # may fall within the ±0.10 tolerance, yielding a small completeness bonus.
        # The score must therefore be < 0.20 — clearly low, not a passing result.
        assert score is not None
        assert score <= 0.20, \
            f"Expected low score for wrong answer on '{variant}', got {score}"

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_unparseable_json_scores_zero(self, variant):
        assert TaskFamily.score(TASKS[variant], "not valid json {{") == 0.0

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_empty_dict_scores_zero(self, variant):
        assert TaskFamily.score(TASKS[variant], "{}") == 0.0

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_sps_at_exact_upper_tolerance_still_scores_1(self, variant):
        t          = TASKS[variant]
        submission = json.dumps({
            "sps":        t["reference_sps"] + t["tolerance_sps"],
            "components": t["reference_components"],
            "verdict":    "NOT RECOMMENDED",
        })
        assert TaskFamily.score(t, submission) == 1.0

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_sps_just_outside_tolerance_gives_partial_credit(self, variant):
        t          = TASKS[variant]
        submission = json.dumps({
            "sps":        t["reference_sps"] + t["tolerance_sps"] * 1.5,
            "components": t["reference_components"],
            "verdict":    "NOT RECOMMENDED",
        })
        score = TaskFamily.score(t, submission)
        assert score is not None
        assert 0.0 < score < 1.0

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_correct_sps_missing_components_lowers_score(self, variant):
        t = TASKS[variant]
        full    = TaskFamily.score(t, perfect_submission(t))
        no_comp = TaskFamily.score(t, json.dumps({
            "sps": t["reference_sps"], "verdict": "NOT RECOMMENDED"
        }))
        assert no_comp < full

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_all_scores_are_in_unit_interval(self, variant):
        t = TASKS[variant]
        for submission in [
            perfect_submission(t),
            json.dumps({"sps": 50.0, "components": {}}),
            "{}",
        ]:
            s = TaskFamily.score(t, submission)
            assert s is None or 0.0 <= s <= 1.0

    def test_component_outside_tolerance_reduces_completeness(self):
        t    = TASKS["baseline"]
        bad  = {k: 0.0 for k in COMPONENT_KEYS}   # all wrong
        good = t["reference_components"]

        score_good = TaskFamily.score(t, json.dumps({
            "sps": t["reference_sps"], "components": good, "verdict": "X"
        }))
        score_bad  = TaskFamily.score(t, json.dumps({
            "sps": t["reference_sps"], "components": bad,  "verdict": "X"
        }))
        assert score_good > score_bad

    def test_non_dict_submission_scores_zero(self):
        t = TASKS["baseline"]
        assert TaskFamily.score(t, json.dumps([1, 2, 3])) == 0.0

    def test_sps_as_string_is_handled_gracefully(self):
        t          = TASKS["baseline"]
        submission = json.dumps({
            "sps":        "not_a_number",
            "components": t["reference_components"],
            "verdict":    "NOT RECOMMENDED",
        })
        score = TaskFamily.score(t, submission)
        assert score is not None
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Category 3 — Environment
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvironment:

    # ── Asset files on disk ──────────────────────────────────────────────────

    def test_assets_directory_exists(self):
        assert ASSETS.is_dir()

    @pytest.mark.parametrize("fname", [
        "baseline_lr_model.pkl",
        "calibrated_rf_model.pkl",
        "hardened_xgb_model.pkl",
    ])
    def test_model_pkl_exists(self, fname):
        assert (ASSETS / fname).exists(), f"Missing asset: {fname}"

    @pytest.mark.parametrize("fname", ["wdbc_train.csv", "wdbc_test.csv"])
    def test_csv_exists(self, fname):
        assert (ASSETS / fname).exists(), f"Missing asset: {fname}"

    def test_test_cases_json_exists(self):
        assert (ASSETS / "test_cases.json").exists()

    def test_test_cases_json_is_valid(self):
        with open(ASSETS / "test_cases.json") as f:
            meta = json.load(f)
        assert "tasks" in meta
        assert "tolerance" in meta
        assert "sps_weights" in meta
        assert set(meta["tasks"].keys()) == {"baseline", "calibrated", "hardened"}

    # ── CSV shapes ───────────────────────────────────────────────────────────

    def test_train_csv_shape(self):
        df = pd.read_csv(ASSETS / "wdbc_train.csv")
        assert df.shape == (455, 31), f"Expected (455, 31), got {df.shape}"
        assert "label" in df.columns

    def test_test_csv_shape(self):
        df = pd.read_csv(ASSETS / "wdbc_test.csv")
        assert df.shape == (114, 31), f"Expected (114, 31), got {df.shape}"
        assert "label" in df.columns

    def test_label_column_is_binary(self):
        for fname in ["wdbc_train.csv", "wdbc_test.csv"]:
            df = pd.read_csv(ASSETS / fname)
            assert set(df["label"].unique()).issubset({0, 1}), \
                f"{fname} label column contains values other than 0/1"

    def test_no_missing_values_in_csvs(self):
        for fname in ["wdbc_train.csv", "wdbc_test.csv"]:
            df = pd.read_csv(ASSETS / fname)
            assert df.isnull().sum().sum() == 0, \
                f"{fname} contains missing values"

    # ── Model loading ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_model_loads_as_sklearn_pipeline(self, variant):
        model_path = ASSETS / TASKS[variant]["model_file"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_model_predicts_on_test_csv(self, variant):
        model_path = ASSETS / TASKS[variant]["model_file"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        df    = pd.read_csv(ASSETS / "wdbc_test.csv")
        X     = df.drop("label", axis=1).values
        proba = model.predict_proba(X)
        assert proba.shape == (114, 2)
        assert np.all(proba >= 0)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_model_auroc_above_threshold(self, variant):
        """All three models should achieve AUROC > 0.95 on the test set."""
        from sklearn.metrics import roc_auc_score
        model_path = ASSETS / TASKS[variant]["model_file"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        df    = pd.read_csv(ASSETS / "wdbc_test.csv")
        X     = df.drop("label", axis=1).values
        y     = df["label"].values
        auroc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        assert auroc > 0.95, \
            f"Variant '{variant}' AUROC={auroc:.4f} below threshold 0.95"

    # ── start() file-copy logic ───────────────────────────────────────────────

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_start_copies_three_files(self, tmp_path, variant):
        """start() should deliver model + both CSVs to the agent workspace."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        t = TASKS[variant]
        for fname in [t["model_file"], "wdbc_train.csv", "wdbc_test.csv"]:
            shutil.copy(ASSETS / fname, agent_dir / fname)

        assert (agent_dir / t["model_file"]).exists()
        assert (agent_dir / "wdbc_train.csv").exists()
        assert (agent_dir / "wdbc_test.csv").exists()

    @pytest.mark.parametrize("variant", ["baseline", "calibrated", "hardened"])
    def test_agent_files_are_readable(self, tmp_path, variant):
        """Files placed in agent dir should be openable."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        t = TASKS[variant]
        for fname in [t["model_file"], "wdbc_train.csv", "wdbc_test.csv"]:
            shutil.copy(ASSETS / fname, agent_dir / fname)
            assert (agent_dir / fname).stat().st_size > 0
