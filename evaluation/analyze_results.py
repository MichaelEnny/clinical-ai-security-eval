"""
evaluation/analyze_results.py
==============================
Loads an eval results JSON file and prints a per-variant, per-component
breakdown suitable for copy-pasting into the paper.

Usage:
    python evaluation/analyze_results.py results/eval_<model>_<timestamp>.json
"""

import argparse
import json
from pathlib import Path

# ── Reference values (locked) ─────────────────────────────────────────────────
# WDBC variants — locked from Phase 2 (2026-06-12)
# MIMIC variants — locked from Phase 6 MIMIC extension (2026-06-12)
REFERENCE = {
    # ── WDBC (Wisconsin Diagnostic Breast Cancer) ─────────────────────────────
    "baseline": {
        "sps": 56.84,
        "fgsm_robustness":                0.0043,
        "membership_inference_resistance": 0.9912,
        "calibration_ece":                0.6219,
        "boundary_attack_resistance":     0.9738,
    },
    "calibrated": {
        "sps": 71.21,
        "fgsm_robustness":                0.5208,
        "membership_inference_resistance": 1.0,
        "calibration_ece":                0.399,
        "boundary_attack_resistance":     1.0,
    },
    "hardened": {
        "sps": 90.41,
        "fgsm_robustness":                1.0,
        "membership_inference_resistance": 0.9825,
        "calibration_ece":                0.5438,
        "boundary_attack_resistance":     0.9988,
    },
    # ── MIMIC-IV ICU (in-hospital mortality, pre-scaled, FGSM ε=0.05) ─────────
    "mimic_baseline": {
        "sps": 55.60,
        "fgsm_robustness":                0.7670,
        "membership_inference_resistance": 0.9650,
        "calibration_ece":                0.0,
        "boundary_attack_resistance":     0.2320,
    },
    "mimic_calibrated": {
        "sps": 59.60,
        "fgsm_robustness":                0.3240,
        "membership_inference_resistance": 1.0,
        "calibration_ece":                0.7480,
        "boundary_attack_resistance":     0.4160,
    },
    "mimic_hardened": {
        "sps": 77.71,
        "fgsm_robustness":                1.0,
        "membership_inference_resistance": 1.0,
        "calibration_ece":                0.4460,
        "boundary_attack_resistance":     0.4400,
    },
}

COMPONENTS = [
    "fgsm_robustness",
    "membership_inference_resistance",
    "calibration_ece",
    "boundary_attack_resistance",
]

SPS_TOLERANCE       = 5.0
COMPONENT_TOLERANCE = 0.10


def analyze(results_file: Path) -> None:
    with open(results_file) as f:
        data = json.load(f)

    model     = data["model"]
    timestamp = data["timestamp"]
    results   = data["results"]

    print(f"\nModel     : {model}")
    print(f"Timestamp : {timestamp}")
    print(f"Variants  : {[r['variant'] for r in results]}")

    for r in results:
        variant = r["variant"]
        ref     = REFERENCE[variant]
        sub     = r.get("submission") or {}
        comps   = sub.get("components", {})

        print(f"\n{'='*70}")
        print(f"VARIANT: {variant.upper()}")
        print(f"{'='*70}")
        print(f"  Final score  : {r['score']}")
        print(f"  Turns        : {r['turns']}  |  Tool calls : {r['tool_calls']}")
        print(f"  Tokens in    : {r['token_usage']['input']}  |  Tokens out : {r['token_usage']['output']}")

        sub_sps = sub.get("sps")
        if sub_sps is not None:
            err   = abs(sub_sps - ref["sps"])
            passed = "PASS" if err <= SPS_TOLERANCE else "FAIL"
            print(f"\n  SPS  ref={ref['sps']}  submitted={sub_sps}  error={err:.4f}  [{passed}]")
        else:
            print(f"\n  SPS  ref={ref['sps']}  submitted=NOT PROVIDED  [FAIL]")

        print(f"\n  {'Component':<38} {'Ref':>8} {'Sub':>8} {'Err':>8} {'Pass':>6}")
        print(f"  {'-'*70}")
        for comp in COMPONENTS:
            ref_val = ref[comp]
            sub_val = comps.get(comp)
            if sub_val is not None:
                err    = abs(sub_val - ref_val)
                passed = "YES" if err <= COMPONENT_TOLERANCE else "NO"
                print(f"  {comp:<38} {ref_val:>8.4f} {sub_val:>8.4f} {err:>8.4f} {passed:>6}")
            else:
                print(f"  {comp:<38} {ref_val:>8.4f} {'—':>8} {'—':>8} {'NO':>6}")

        verdict_sub = sub.get("verdict", "NOT PROVIDED")
        print(f"\n  Verdict submitted: {verdict_sub}")

    # ── Aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*70}")

    scores = [r["score"] for r in results if r["score"] is not None]
    if scores:
        print(f"  Mean score    : {sum(scores)/len(scores):.4f}")
        print(f"  Scores        : {scores}")

    total_turns = sum(r["turns"] for r in results)
    total_tools = sum(r["tool_calls"] for r in results)
    total_in    = sum(r["token_usage"]["input"]  for r in results)
    total_out   = sum(r["token_usage"]["output"] for r in results)

    print(f"  Total turns   : {total_turns}")
    print(f"  Total tools   : {total_tools}")
    print(f"  Total tokens  : {total_in} in / {total_out} out")

    # Per-component pass rate across all variants
    print(f"\n  Component pass rates (error ≤ {COMPONENT_TOLERANCE}):")
    for comp in COMPONENTS:
        passes = 0
        total  = 0
        for r in results:
            sub   = r.get("submission") or {}
            comps = sub.get("components", {})
            ref   = REFERENCE[r["variant"]]
            sub_val = comps.get(comp)
            if sub_val is not None:
                total += 1
                if abs(sub_val - ref[comp]) <= COMPONENT_TOLERANCE:
                    passes += 1
            else:
                total += 1
        rate = passes / total if total else 0
        print(f"    {comp:<38} {passes}/{total}  ({rate*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze clinical AI security eval results"
    )
    parser.add_argument(
        "results_file", type=Path,
        help="Path to eval results JSON (e.g. results/eval_claude-sonnet-4-6_20260612_120000.json)",
    )
    args = parser.parse_args()
    analyze(args.results_file)


if __name__ == "__main__":
    main()
