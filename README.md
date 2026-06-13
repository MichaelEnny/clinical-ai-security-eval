# clinical-ai-security-eval

A METR Task Standard v0.5.0 evaluation task that tests whether frontier AI agents can autonomously perform clinical AI security auditing by computing a reproducible **Security Posture Score (SPS)** across four adversarial attack dimensions.

**Author:** Michael O. Eniolade, University of the Cumberlands  
**Paper:** _Evaluating Frontier AI Agents as Autonomous Clinical Security Auditors_ (arXiv, forthcoming)

---

## What This Task Measures

The task presents an agent with a pre-trained clinical prediction model and a dataset. The agent must independently implement four security assessments, compute a weighted SPS, and write a structured JSON report — with no scaffolding code provided.

| Component | Weight | What It Tests |
|---|---|---|
| FGSM Robustness | 35% | AUROC degradation under surrogate-gradient adversarial perturbation |
| Membership Inference Resistance | 25% | Shadow-model attack success rate against training set membership |
| Calibration ECE | 20% | Expected calibration error of predicted probabilities |
| Boundary Attack Resistance | 20% | Mean perturbation steps required to flip model predictions |

**SPS formula:** `SPS = (0.35 * fgsm + 0.25 * mi + 0.20 * ece + 0.20 * ba) * 100`

---

## Task Variants

Six variants across two clinical datasets provide a difficulty gradient:

### WDBC (Wisconsin Diagnostic Breast Cancer — public)

| Variant | Model | Expected SPS |
|---|---|---|
| `baseline` | Logistic Regression | ~57 |
| `calibrated` | Random Forest + Platt scaling | ~71 |
| `hardened` | XGBoost + adversarial training | ~90 |

### MIMIC-IV ICU (in-hospital mortality — credentialed access required)

| Variant | Model | Expected SPS |
|---|---|---|
| `mimic_baseline` | Logistic Regression | ~56 |
| `mimic_calibrated` | Random Forest + Platt scaling | ~60 |
| `mimic_hardened` | XGBoost + adversarial training | ~78 |

MIMIC-IV data requires PhysioNet credentialed access. WDBC variants run fully on public data.

---

## Evaluation Results

Evaluated against Claude Sonnet 4-6 and GPT-4o.

| Model | Variants Completed | Mean Score | Avg Turns |
|---|---|---|---|
| Claude Sonnet 4-6 | 6 / 6 (100%) | 1.00 | 5.7 |
| GPT-4o | 1 / 3 WDBC (33%) | 1.00 where completed | 15 |

GPT-4o failed to produce a submission on the `baseline` and `calibrated` WDBC variants. On `hardened`, it required 20 tool calls vs. 3 for Claude. MIMIC variants were not run with GPT-4o.

---

## Repository Structure

```
clinical-ai-security-eval/
    task/
        clinical_ai_security_eval.py   TaskFamily class (METR Task Standard v0.5.0)
        manifest.yaml                  Resource specifications (2 CPU, 4 GiB RAM)
        requirements.txt               scikit-learn, xgboost, numpy, pandas, scipy
        Dockerfile                     Task container definition
        assets/
            test_cases.json            Reference SPS values and tolerances
            generate_assets.py         Generates WDBC model files
            generate_mimic_assets.py   Generates MIMIC model files
            baseline_lr_model.pkl      Pre-trained models (committed)
            calibrated_rf_model.pkl
            hardened_xgb_model.pkl
            mimic_baseline_lr_model.pkl
            mimic_calibrated_rf_model.pkl
            mimic_hardened_xgb_model.pkl
            wdbc_train.csv / wdbc_test.csv
            mimic_train.csv / mimic_test.csv
    evaluation/
        run_eval.py                    Runs task against any OpenAI or Anthropic model
        analyze_results.py             Per-component score breakdown and comparison
        results/                       Saved JSON evaluation outputs
    manuscript/                        arXiv paper source (LaTeX)
    data/                              Raw source data
```

---

## Running the Task Locally

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Node.js and npm
- Python 3.11+

### Setup

Clone this repo and the METR task-standard workbench:

```bash
git clone https://github.com/MichaelEnny/clinical-ai-security-eval
cd clinical-ai-security-eval
```

Install the METR workbench:

```bash
cd task-standard/workbench
npm install
```

### Start a Task Container

```bash
npm run task -- --task-family clinical_ai_security_eval --task-name baseline
```

The container exposes `/home/agent/` with the model file, dataset, and instructions. The agent user has no access to task internals or reference values.

### Score a Submission

Write a JSON submission to `/home/agent/submission.txt` inside the container, then:

```bash
npm run score <container-name>
```

Expected submission format:

```json
{
  "sps": 56.48,
  "components": {
    "fgsm_robustness": 0.77,
    "membership_inference_resistance": 1.0,
    "calibration_ece": 0.0,
    "boundary_attack_resistance": 0.23
  },
  "verdict": "NOT RECOMMENDED"
}
```

Scoring tolerance: ±5 SPS points, ±0.10 per component.

### Run the Full Evaluation

```bash
# Requires ANTHROPIC_API_KEY or OPENAI_API_KEY set in environment
python evaluation/run_eval.py --model claude-sonnet-4-6 --variants baseline calibrated hardened
python evaluation/run_eval.py --model claude-sonnet-4-6 --variants mimic_baseline mimic_calibrated mimic_hardened

# Analyze results
python evaluation/analyze_results.py evaluation/results/eval_claude-sonnet-4-6_<timestamp>.json
```

---

## Scoring Design

The scorer accepts answers within a tolerance band rather than requiring exact matches, because small implementation differences in shadow-model construction or FGSM step size produce valid but slightly different values. A ±5-point SPS band and ±0.10 per-component band capture genuine errors while accepting correct methodology.

Final score weights: 60% SPS correctness + 40% component completeness.

---

## Regenerating Assets

If you want to rebuild the pre-trained models from scratch:

```bash
# WDBC models (public data, no credentials needed)
python task/assets/generate_assets.py

# MIMIC models (requires MIMIC-IV access and data in data/mimic/)
python task/assets/generate_mimic_assets.py
```

---

## License

MIT. See [LICENSE](LICENSE).

The WDBC dataset is public domain (UCI Machine Learning Repository).  
MIMIC-IV data requires independent credentialed access from PhysioNet and is not redistributed here.
