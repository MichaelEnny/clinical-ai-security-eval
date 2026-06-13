# METR Eval Task: Development and arXiv Publication Guide

Author: Michael O. Eniolade
Institution: University of the Cumberlands
Date: June 2026

---

## What You Are Building

Two deliverables come out of this project:

1. An open-source evaluation task following the METR Task Standard v0.5.0
2. A research paper for arXiv describing the task design, methodology, and evaluation results

The eval task tests AI agent capability to perform clinical AI security assessment. Specifically, the agent receives a clinical prediction model, a dataset, and a set of evaluation instructions. The agent must compute the Security Posture Score across multiple adversarial attack vectors and produce a structured security report.

This aligns directly with the third-paper SPS framework and the broader thesis on clinical AI security evaluation. The task measures a real autonomous capability: whether frontier AI agents can replace or assist human evaluators in clinical AI security auditing.

---

## Part 1: METR Overview

### Who METR Is

METR is a research nonprofit focused on measuring whether AI systems pose risks of catastrophic harm. The organization builds and maintains standardized benchmarks for autonomous AI capabilities.

METR's evaluations focus on:
- Task completion across long-horizon software and research assignments
- AI research acceleration capabilities
- Evaluation integrity threats such as reward hacking and sandbagging
- Monitoring evasion and dangerous autonomous behaviors

METR partners with Anthropic, OpenAI, Google DeepMind, and Meta to run evaluations on frontier models before deployment.

### The Three Core Repositories

**task-standard** (https://github.com/METR/task-standard)
The specification and Python package for defining eval tasks. Every task follows the TaskFamily class pattern defined here. The standard is at version 0.5.0.

**vivaria** (https://github.com/METR/vivaria)
The open-source platform for running evaluations. Vivaria spins up Docker containers, runs agents, and records results. METR is transitioning new projects to UK AISI's Inspect framework, but Vivaria remains fully functional.

**public-tasks** (https://github.com/METR/public-tasks)
A collection of 31 task families with roughly 186 individual tasks. These serve as reference implementations and examples.

---

## Part 2: The METR Task Standard

### How a Task Works

A task defines three things: an environment for an agent to operate in, a string of instructions for the agent, and a scoring function to evaluate the result.

The environment runs inside a Docker container. The agent operates as a non-root user named `agent`. The agent receives only the instruction string at the start, with no access to the internal task parameters.

### TaskFamily Class Structure

Every task is a Python class named `TaskFamily` in a `.py` file. The file name matches the task family name.

```
clinical_ai_security_eval/
    clinical_ai_security_eval.py     # Required: TaskFamily class
    manifest.yaml                     # Optional: resource specs
    requirements.txt                  # Optional: Python dependencies
    Dockerfile                        # Optional: custom environment
    assets/
        test_cases.json
        sample_model.pkl
        setup.sh
```

### Required Methods

Every TaskFamily must implement three methods:

**get_tasks()** returns a dictionary mapping task names to Task objects. Each key is a string task identifier. Each value is a TypedDict containing all parameters for one task variant.

**get_instructions(t)** takes a Task object and returns the instruction string the agent sees. The agent receives nothing else at start.

**score(t, submission)** takes a Task object and the agent's final string submission, then returns a float between 0 and 1. Return None if the task is unscoreable due to timeout or error.

### Optional Methods

**install()** runs once during environment setup. Use subprocess calls to install packages or download dependencies.

**start()** runs after environment creation but before the agent starts. Use it to copy files, write configuration, or seed the workspace.

**get_permissions(t)** returns a list of permission strings. The only current permission is `"full_internet"`. Without it, the agent gets network access to LLM APIs only.

**required_environment_variables()** returns variable names the task needs from the environment.

### Scoring Mechanics

The standard supports two scoring approaches.

End-of-task scoring uses `score(t, submission)`. The agent writes its final answer to a submission file or returns a string. This is the most common approach.

Intermediate scoring uses `intermediate_score(env)` and `aggregate_scores(score_log)`. The scorer runs periodically during execution and the final score aggregates the log. Use this for tasks where the agent's process matters as much as the final output.

### Type Definitions

```python
from typing import TypedDict, NotRequired

class GPUSpec(TypedDict):
    count: int
    model: str  # "v100", "a10", "a100", or "h100"

class VMSpec(TypedDict):
    cpus: int
    cpu_architecture: NotRequired[str]  # "x64" or "arm64"
    gpu: NotRequired[GPUSpec]
    ram_gib: int
    base_image: NotRequired[str]  # "debian-12" or "ubuntu-20.04-cuda"
```

---

## Part 3: Designing the Clinical AI Security Eval Task

### Task Concept

The task asks an AI agent to perform a full security audit on a clinical prediction model. The agent receives:
- A pre-trained scikit-learn model (pickle file)
- A test dataset (CSV)
- Evaluation instructions specifying which attack vectors to assess

The agent must:
1. Load and inspect the model
2. Run adversarial robustness tests (FGSM-equivalent)
3. Run membership inference probing
4. Compute calibration error metrics
5. Generate a Security Posture Score
6. Write a structured security report

The final submission is a JSON string containing the SPS value, per-component scores, and a pass/fail determination.

### Why This Qualifies as a METR-Style Task

METR tasks measure real autonomous capability. This task measures whether an AI agent can independently perform clinical AI security evaluation, a multi-step technical task requiring code execution, domain knowledge, and structured output generation. The scoring is objective and automated. The task has clear difficulty variation across task variants.

### Task Variants

Three variants provide difficulty gradient:

**baseline**: The agent evaluates a logistic regression model on the Wisconsin Breast Cancer dataset. No adversarial hardening. The expected SPS falls below 60.

**calibrated**: The agent evaluates a random forest model with Platt scaling applied. Moderate security posture. Expected SPS between 60 and 75.

**hardened**: The agent evaluates an XGBoost model with adversarial training and subgroup calibration. High security posture. Expected SPS above 75.

Each variant has a known SPS value computed ahead of time. The scoring function checks whether the agent's computed SPS falls within a tolerance band of the reference value.

---

## Part 4: Environment Setup

### Prerequisites

- Python 3.11 or higher
- Docker (OrbStack recommended on macOS; Docker Desktop on Windows)
- Node.js and npm (for the Vivaria workbench)
- Git

### Install the METR Task Standard Package

```bash
pip install metr-task-standard pytest
```

The package provides:
- Type definitions in `metr_task_standard.types`
- Pytest fixtures in `metr_task_standard.pytest_plugin`

### Clone Vivaria for Local Testing

```bash
git clone https://github.com/METR/vivaria.git
cd vivaria
```

On macOS or Linux, run the setup script:

```bash
./scripts/setup-docker-compose.sh
```

On Windows PowerShell:

```powershell
.\scripts\setup-docker-compose.ps1
```

Add your API key to `.env.server`:

```
ANTHROPIC_API_KEY=your_key_here
```

Launch the platform:

```bash
docker compose up --pull always --detach --wait
```

Access the web interface at `https://localhost:4000`.

### Clone the Workbench for Development

The workbench provides npm scripts for local task development without the full Vivaria stack.

```bash
cd vivaria/task-standard/workbench
npm install
```

---

## Part 5: Task Implementation

See `task/clinical_ai_security_eval.py` for the full implementation. The key design decisions are documented below.

### Scoring the SPS Submission

The agent submits a JSON string with this structure:

```json
{
  "sps": 72.4,
  "components": {
    "fgsm_robustness": 0.61,
    "membership_inference": 0.78,
    "calibration_ece": 0.82,
    "boundary_attack": 0.55
  },
  "verdict": "MODERATE"
}
```

The scorer parses the JSON and checks whether the SPS value falls within 5 points of the reference. Per-component scores within 0.10 of reference earn partial credit. The final score weights correctness at 60% and completeness at 40%.

### Tolerance Design

A strict exact-match scorer would penalize correct methodology with slight implementation differences. A tolerance band of plus or minus 5 SPS points and plus or minus 0.10 per component captures genuine errors while accepting correct approaches.

### Asset Preparation

The `assets/` directory contains pre-computed reference values and the models the agent evaluates. These are generated once using `evaluation/generate_assets.py` and committed alongside the task code.

---

## Part 6: Local Development Workflow

### Step 1: Build the Task Environment

From the workbench directory:

```bash
npm run task -- --task-family clinical_ai_security_eval --task-name baseline
```

This creates a Docker container with the task environment. The first run takes roughly five minutes while Docker pulls the base image.

### Step 2: Inspect the Environment

```bash
docker exec -it <container-id> bash
# Now inside the container as root
ls /home/agent/
# Switch to agent user
su agent
```

Verify the agent sees the right files and instructions.

### Step 3: Test Scoring Manually

Place a test submission in the container:

```bash
docker exec <container-id> bash -c "echo '{\"sps\": 72.4, \"components\": {...}, \"verdict\": \"MODERATE\"}' > /home/agent/submission.txt"
```

Run the scorer:

```bash
npm run score -- --task-family clinical_ai_security_eval --task-name baseline
```

### Step 4: Run Pytest Tests

```bash
npm run test -- --task-family clinical_ai_security_eval
```

Or run directly:

```bash
cd task/
pytest tests/ -v
```

### Step 5: Clean Up

```bash
npm run destroy
```

---

## Part 7: Testing Requirements

All tasks submitted to METR must pass three test categories.

**Structural tests**: `get_tasks()` returns a non-empty dictionary. All task variants contain required fields. `get_instructions()` returns a non-empty string for every variant.

**Scoring tests**: Submitting the reference answer scores 1.0. Submitting a wrong answer scores 0.0. Submitting None returns None without raising an exception.

**Environment tests**: The `start()` method copies all required files to `/home/agent/`. The agent user has read access to all assets. The `install()` method completes without errors.

See `tests/test_task.py` for the full test suite.

---

## Part 8: Contributing to METR

### Task Bounty Program

METR runs a Task Bounty Program at https://taskdev.metr.org/introduction/. External contributors submit tasks and receive payment upon acceptance.

The submission process:
1. Visit taskdev.metr.org and read the submission guidelines
2. Develop the task following Task Standard v0.5.0
3. Run all pytest tests locally
4. Submit via the bounty platform
5. Iterate based on METR's feedback
6. Receive payment upon acceptance

### Key Restrictions

Do not train frontier models on submitted tasks. Do not publish unprotected solutions. Report exposed solutions to tasks@metr.org. METR specifically restricts publishing solutions for several task families; check the current list before publishing.

### Submission Checklist

- [ ] TaskFamily class with `standard_version = "0.5.0"`
- [ ] `get_tasks()` returns at least two variants
- [ ] `get_instructions()` returns complete, unambiguous instructions
- [ ] `score()` returns float in [0, 1] or None
- [ ] All pytest tests pass
- [ ] `manifest.yaml` present with resource specifications
- [ ] `assets/` contains all dependencies committed to the repository
- [ ] No hard-coded absolute paths
- [ ] Dockerfile tested and working

---

## Part 9: Writing the arXiv Paper

### Target Categories

Submit to `cs.AI` with cross-listing to `cs.CR` (cryptography and security). The primary contribution is a new evaluation methodology for clinical AI security, which fits cs.AI. The security angle warrants the cs.CR cross-list.

### Paper Structure

**Abstract** (150 words): State the problem (no standardized eval for clinical AI security autonomous assessment), the contribution (a new METR-standard eval task), and the key result (SPS values across task variants and model types).

**Introduction**: Frame the gap. Security evaluation of deployed clinical AI is labor-intensive and non-standardized. Autonomous AI agents offer a path to scalable, reproducible auditing. This work tests whether frontier agents perform that auditing reliably.

**Related Work**: Cover METR's evaluation framework, existing clinical AI safety work, adversarial robustness benchmarks, and membership inference attack literature.

**Task Design**: Describe the three task variants, the scoring methodology, tolerance design decisions, and the SPS components. Include the TaskFamily class structure.

**Evaluation**: Run the task against two or three frontier models. Report per-variant scores, per-component accuracy, and SPS deviation from reference values. Identify which security assessment components agents perform well and which they miss.

**Discussion**: Interpret failures. Agents likely underestimate membership inference risk and overestimate FGSM robustness. Discuss what this means for autonomous clinical AI auditing.

**Conclusion**: State the open-source availability of the task and invite community extensions.

### arXiv Submission Process

1. Write the paper in LaTeX. See `manuscript/main.tex` for the template.
2. Compile to PDF and verify all figures and tables render correctly.
3. Create an arXiv account at arxiv.org if not already registered.
4. Upload the source files (`.tex`, `.bib`, figures) as a `.tar.gz` archive.
5. Select primary category `cs.AI`, secondary category `cs.CR`.
6. Enter the abstract, title, and author list.
7. Submit. arXiv processes submissions within one business day.

### Recommended arXiv Template Style

Use the standard arXiv LaTeX template (`\documentclass[11pt]{article}` with `\usepackage{arxiv}` or plain article class). See `manuscript/main.tex` for the full setup.

### Connecting the Paper to the Task

The paper must reference the public task repository so readers can reproduce experiments. Use this format in the Data Availability section:

"The eval task, scoring code, and pre-computed reference values are available at https://github.com/MichaelEnny/clinical-ai-security-eval under the MIT License. The task follows the METR Task Standard v0.5.0 and runs in the Vivaria platform."

---

## Part 10: Project Structure Reference

```
special_paper/
    GUIDE.md                            This file - full development and paper guide
    code/                               All Python source code for the eval tool and SPS computation
    task/                               METR task definition and environment files
        clinical_ai_security_eval.py    TaskFamily class implementation
        manifest.yaml                   Task metadata and resource specifications
        requirements.txt                Python dependencies
        Dockerfile                      Custom Docker environment definition
        assets/
            test_cases.json             Task variants with reference SPS values
            generate_assets.py          Script to generate pre-trained model files
            baseline_lr_model.pkl       Generated: logistic regression model
            calibrated_rf_model.pkl     Generated: random forest with Platt scaling
            hardened_xgb_model.pkl      Generated: XGBoost with adversarial hardening
            wdbc_test.csv               Generated: test dataset
    tests/
        test_task.py                    Pytest test suite (structural, scoring, environment)
    evaluation/
        run_eval.py                     Script to run eval against frontier models
        analyze_results.py              Score analysis and per-component breakdown
        results/                        Evaluation output files (git-ignored)
    manuscript/
        main.tex                        arXiv paper main file
        sections/
            abstract.tex
            introduction.tex
            related_work.tex
            task_design.tex
            evaluation.tex
            discussion.tex
            conclusion.tex
        references.bib
    data/                               Raw data and intermediate outputs
```

---

## Part 11: Key Resources

METR Task Standard repository: https://github.com/METR/task-standard

Vivaria platform: https://github.com/METR/vivaria

Public task examples: https://github.com/METR/public-tasks

Task Bounty Program: https://taskdev.metr.org/introduction/

METR evaluation guide: https://evaluations.metr.org/

UK AISI Inspect framework (METR's recommended path for new projects): https://inspect.aisi.org.uk/

arXiv submission guide: https://arxiv.org/help/submit

Contact for task submissions: tasks@metr.org

Contact for Vivaria issues: vivaria@metr.org

---

## Part 12: Full Development Checklist

### Task Development

- [ ] Define three task variants with reference SPS values
- [ ] Implement TaskFamily class following standard v0.5.0
- [ ] Write `install()` to set up scikit-learn, XGBoost, scipy
- [ ] Write `start()` to copy model files and test data to /home/agent/
- [ ] Write `score()` with tolerance-based JSON parsing
- [ ] Commit all assets to repository
- [ ] Verify Dockerfile builds without errors
- [ ] Run all pytest tests locally
- [ ] Test each variant manually in the workbench container

### Evaluation Run

- [ ] Run task against Claude Sonnet or GPT-4
- [ ] Run task against at least one other frontier model for comparison
- [ ] Record per-variant scores and per-component accuracy
- [ ] Analyze which SPS components agents compute correctly
- [ ] Document failure modes

### arXiv Paper

- [ ] Write all manuscript sections
- [ ] Add figures: task architecture diagram, per-component score heatmap, SPS deviation bar chart
- [ ] Include TaskFamily class listing as a code figure
- [ ] Compile PDF and check all references resolve
- [ ] Run through AI detection tool before submission
- [ ] Submit to arXiv cs.AI with cs.CR cross-list
- [ ] Post GitHub repository link in paper

### METR Contribution (Optional)

- [ ] Review current Task Bounty Program guidelines
- [ ] Ensure no hard-coded paths or local-only dependencies
- [ ] Add README to task directory
- [ ] Submit via taskdev.metr.org
