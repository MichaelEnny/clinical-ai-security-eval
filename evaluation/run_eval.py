"""
evaluation/run_eval.py
======================
Runs the clinical_ai_security_eval METR task against a frontier AI model
and records scores for all three variants.

Supports both Anthropic (claude-*) and OpenAI (gpt-*, o1, o3, o4-*) models.
Provider is auto-detected from the model name.

Usage:
    python evaluation/run_eval.py                          # default: gpt-4o
    python evaluation/run_eval.py --model gpt-4o
    python evaluation/run_eval.py --model claude-sonnet-4-6
    python evaluation/run_eval.py --model gpt-4o --variants baseline

Requirements:
    pip install openai anthropic
    OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable set
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
WORKBENCH   = ROOT / "task-standard" / "workbench"
TASK_DIR    = "tasks/clinical_ai_security_eval"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Ensure Docker is on PATH (Windows) ───────────────────────────────────────
if os.name == "nt":
    docker_bin = r"C:\Program Files\Docker\Docker\resources\bin"
    if docker_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + ";" + docker_bin

# ── Constants ─────────────────────────────────────────────────────────────────
NPM           = "npm.cmd" if os.name == "nt" else "npm"
VARIANTS      = ["baseline", "calibrated", "hardened",
                 "mimic_baseline", "mimic_calibrated", "mimic_hardened"]
DEFAULT_MODEL = "gpt-4o"
MAX_TURNS     = 40


# ── Provider detection ────────────────────────────────────────────────────────

def get_provider(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(
        f"Cannot detect provider for model '{model}'. "
        "Model must start with 'claude', 'gpt-', 'o1', 'o3', or 'o4'."
    )


# ── Docker helpers ────────────────────────────────────────────────────────────

ENC = {"encoding": "utf-8", "errors": "replace"}


def start_container(variant: str) -> str:
    print(f"\n{'='*60}")
    print(f"[{variant.upper()}] Starting task container...")
    print("="*60)
    result = subprocess.run(
        [NPM, "run", "task", "--", TASK_DIR, variant],
        cwd=WORKBENCH,
        capture_output=True,
        **ENC,
    )
    # Show output so progress is visible
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)

    result2 = subprocess.run(
        ["docker", "ps",
         "--filter", f"name=clinical_ai_security_eval-{variant}",
         "--format", "{{.Names}}"],
        capture_output=True, **ENC, check=True,
    )
    container = result2.stdout.strip().split("\n")[0]
    if not container:
        raise RuntimeError(f"No running container found for variant '{variant}'")
    print(f"[{variant}] Container: {container}")
    return container


def docker_exec(container: str, command: str) -> str:
    result = subprocess.run(
        ["docker", "exec", "-u", "agent", "-w", "/home/agent",
         container, "bash", "-c", command],
        capture_output=True, **ENC,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    return (stdout + stderr).strip()


def get_instructions(container: str) -> str:
    return docker_exec(container, "cat /home/agent/instructions.txt")


def score_container(container: str) -> float | None:
    result = subprocess.run(
        [NPM, "run", "score", "--", container],
        cwd=WORKBENCH, capture_output=True, **ENC,
    )
    output = (result.stdout or "") + (result.stderr or "")
    for line in output.splitlines():
        if "Score:" in line:
            try:
                return float(line.split("Score:")[-1].strip())
            except ValueError:
                pass
    return None


def destroy_container(container: str) -> None:
    subprocess.run(
        [NPM, "run", "destroy", "--", container],
        cwd=WORKBENCH, capture_output=True, **ENC,
    )
    print(f"[container] Destroyed: {container}")


# ── Tool definition (shared schema) ──────────────────────────────────────────

TOOL_DESCRIPTION = (
    "Run a bash command in the task container as the agent user. "
    "Use this to read files, execute Python scripts, inspect outputs, "
    "and write your final JSON answer to /home/agent/submission.txt."
)

TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Bash command to run inside /home/agent/",
        }
    },
    "required": ["command"],
}

# Anthropic format
ANTHROPIC_TOOL = {
    "name": "bash",
    "description": TOOL_DESCRIPTION,
    "input_schema": TOOL_PARAMETERS,
}

# OpenAI format
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": TOOL_DESCRIPTION,
        "parameters": TOOL_PARAMETERS,
    },
}


# ── Anthropic agent loop ──────────────────────────────────────────────────────

def run_agent_anthropic(
    variant: str, container: str, instructions: str, model: str
) -> dict:
    import anthropic
    client = anthropic.Anthropic()

    print(f"[{variant}] Provider: Anthropic | Model: {model}")

    messages = [{
        "role": "user",
        "content": (
            instructions
            + "\n\nComplete the task step by step. When you have computed all "
              "four component scores and the SPS, write your JSON answer to "
              "/home/agent/submission.txt."
        ),
    }]

    turns = tool_calls = token_in = token_out = 0

    while turns < MAX_TURNS:
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=8096,
                    tools=[ANTHROPIC_TOOL],
                    messages=messages,
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  [retry {attempt+1}/3] API error: {e}. Retrying in 10s...")
                time.sleep(10)
        turns    += 1
        token_in  += response.usage.input_tokens
        token_out += response.usage.output_tokens

        for block in response.content:
            if hasattr(block, "text") and block.text.strip():
                print(f"  [agent] {block.text.strip()[:300]}")

        if response.stop_reason == "end_turn":
            print(f"[{variant}] Agent finished in {turns} turns.")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls += 1
                    cmd = block.input.get("command", "")
                    print(f"  [tool {tool_calls}] $ {cmd[:100]}")
                    output = docker_exec(container, cmd)
                    print(f"  [out]  {output[:200]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})
        else:
            print(f"[{variant}] Unexpected stop reason: {response.stop_reason}")
            break

    return {"turns": turns, "tool_calls": tool_calls,
            "token_usage": {"input": token_in, "output": token_out}}


# ── OpenAI agent loop ─────────────────────────────────────────────────────────

def run_agent_openai(
    variant: str, container: str, instructions: str, model: str
) -> dict:
    from openai import OpenAI
    client = OpenAI()

    print(f"[{variant}] Provider: OpenAI | Model: {model}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI security auditor. Complete the task "
                "methodically, step by step. Always verify your calculations "
                "before writing the final submission."
            ),
        },
        {
            "role": "user",
            "content": (
                instructions
                + "\n\nComplete the task step by step. When you have computed all "
                  "four component scores and the SPS, write your JSON answer to "
                  "/home/agent/submission.txt."
            ),
        },
    ]

    turns = tool_calls = token_in = token_out = 0

    while turns < MAX_TURNS:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[OPENAI_TOOL],
                    tool_choice="auto",
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  [retry {attempt+1}/3] API error: {e}. Retrying in 10s...")
                time.sleep(10)
        turns     += 1
        token_in  += response.usage.prompt_tokens
        token_out += response.usage.completion_tokens

        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if msg.content:
            print(f"  [agent] {str(msg.content).strip()[:300]}")

        if finish_reason == "stop":
            print(f"[{variant}] Agent finished in {turns} turns.")
            break

        if finish_reason == "tool_calls":
            # Append assistant message with tool_calls
            messages.append({
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            # Execute each tool call
            for tc in msg.tool_calls:
                tool_calls += 1
                args = json.loads(tc.function.arguments)
                cmd  = args.get("command", "")
                print(f"  [tool {tool_calls}] $ {cmd[:100]}")
                output = docker_exec(container, cmd)
                print(f"  [out]  {output[:200]}")
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      output,
                })
        else:
            print(f"[{variant}] Unexpected finish reason: {finish_reason}")
            break

    return {"turns": turns, "tool_calls": tool_calls,
            "token_usage": {"input": token_in, "output": token_out}}


# ── Dispatcher ────────────────────────────────────────────────────────────────

def run_agent(
    variant: str, container: str, instructions: str, model: str
) -> dict:
    provider = get_provider(model)
    if provider == "anthropic":
        return run_agent_anthropic(variant, container, instructions, model)
    return run_agent_openai(variant, container, instructions, model)


# ── Per-variant evaluation ────────────────────────────────────────────────────

def evaluate_variant(variant: str, model: str) -> dict:
    container    = start_container(variant)
    instructions = get_instructions(container)

    agent_stats  = run_agent(variant, container, instructions, model)

    submission_raw = docker_exec(
        container,
        "cat /home/agent/submission.txt 2>/dev/null || echo '__NO_SUBMISSION__'",
    )
    submission_parsed = None
    if submission_raw != "__NO_SUBMISSION__":
        try:
            submission_parsed = json.loads(submission_raw.lstrip("﻿"))
        except json.JSONDecodeError:
            pass

    score = score_container(container)
    print(f"\n[{variant}] *** Score: {score} ***")

    destroy_container(container)

    return {
        "variant":        variant,
        "model":          model,
        "provider":       get_provider(model),
        "score":          score,
        "submission":     submission_parsed,
        "submission_raw": submission_raw,
        **agent_stats,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clinical_ai_security_eval against a frontier model"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model ID — claude-* uses Anthropic, gpt-*/o* uses OpenAI (default: gpt-4o)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=VARIANTS, choices=VARIANTS,
        help="Which variants to evaluate (default: all six)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of independent runs per variant (default: 1)",
    )
    args = parser.parse_args()

    provider = get_provider(args.model)
    print(f"Provider : {provider}")
    print(f"Model    : {args.model}")
    print(f"Variants : {args.variants}")
    print(f"Runs     : {args.runs}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file  = RESULTS_DIR / f"eval_{args.model}_{timestamp}.json"

    all_results = []
    for run_idx in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n{'#'*60}")
            print(f"  RUN {run_idx} of {args.runs}")
            print(f"{'#'*60}")
        for variant in args.variants:
            result = evaluate_variant(variant, args.model)
            result["run"] = run_idx
            all_results.append(result)
            with open(out_file, "w") as f:
                json.dump(
                    {"model": args.model, "provider": provider,
                     "timestamp": timestamp, "runs": args.runs,
                     "results": all_results},
                    f, indent=2,
                )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    for r in all_results:
        sps = r["submission"].get("sps", "N/A") if r["submission"] else "N/A"
        run_label = f" run={r.get('run', 1)}" if args.runs > 1 else ""
        print(
            f"  {r['variant']:20}{run_label} | Score: {str(r['score']):>4} "
            f"| SPS submitted: {sps} "
            f"| Turns: {r['turns']} | Tool calls: {r['tool_calls']}"
        )
    print(f"\nResults saved to: {out_file}")
    print(f"\nTo analyze:\n  python evaluation/analyze_results.py {out_file}")


if __name__ == "__main__":
    main()
