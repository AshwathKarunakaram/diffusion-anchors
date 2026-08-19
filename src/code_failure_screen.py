"""Premise check for a single-canvas DiffusionGemma code-failure monitor.

This script does *not* train a probe.  It first establishes whether the model
produces enough passing and failing samples of the same short coding prompt to
make a within-prompt probe study meaningful.

For every task and random seed it:
  1. runs exactly one 256-token canvas through the instrumented denoising loop;
  2. executes the resulting function against fixed tests;
  3. saves every denoising-step canvas;
  4. summarizes the pass rate per prompt.

Tasks with a pass rate between 20% and 80% are candidates for the next stage:
reading decoder activations at fixed denoising steps and testing whether they
predict test failure better than the visible unfinished canvas.

Run on the A100/Colab runtime:
    python src/code_failure_screen.py --n-seeds 6 --overwrite
"""

import argparse
import json
import os
from dataclasses import dataclass

from code_repair_pilot import CodeTask, TASKS, extract_python, run_tests, serializable_steps
from config import CANVAS_LENGTH
from custom_denoise import run_denoising
from model_utils import build_chat_inputs, load_model


DATA_DIR = "data/code_failure_screen"
RESULT_PATH = "results/code_failure_screen.jsonl"
SUMMARY_PATH = "results/code_failure_screen_summary.json"


EXTRA_TASKS = (
    CodeTask(
        name="two_sum_indices",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "two_sum_indices(nums, target), returning the indices of two "
            "different elements of nums whose sum is target. Return the "
            "smaller index first. Assume exactly one answer exists."
        ),
        expected_name="two_sum_indices",
        tests=(
            (([2, 7, 11, 15], 9), [0, 1]),
            (([3, 2, 4], 6), [1, 2]),
            (([3, 3], 6), [0, 1]),
        ),
    ),
    CodeTask(
        name="product_except_self",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "product_except_self(nums), returning a list where item i is the "
            "product of every nums item except nums[i]. Do not use division."
        ),
        expected_name="product_except_self",
        tests=(
            (([1, 2, 3, 4],), [24, 12, 8, 6]),
            (([0, 1],), [1, 0]),
            (([-1, 1, 0, -3, 3],), [0, 0, 9, 0, 0]),
        ),
    ),
    CodeTask(
        name="run_length_encode",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "run_length_encode(s), returning a list of (character, count) "
            "tuples for consecutive runs in s."
        ),
        expected_name="run_length_encode",
        tests=(
            (("",), []),
            (("aaabbc",), [("a", 3), ("b", 2), ("c", 1)]),
            (("abcd",), [("a", 1), ("b", 1), ("c", 1), ("d", 1)]),
        ),
    ),
    CodeTask(
        name="spiral_order",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "spiral_order(matrix), returning all values of a rectangular "
            "matrix in clockwise spiral order. Return an empty list for an "
            "empty matrix."
        ),
        expected_name="spiral_order",
        tests=(
            (([],), []),
            (([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), [1, 2, 3, 6, 9, 8, 7, 4, 5]),
            (([[1, 2, 3, 4]],), [1, 2, 3, 4]),
        ),
    ),
)

SCREEN_TASKS = TASKS + EXTRA_TASKS


def run_one(model, tokenizer, inputs, seed: int):
    final, steps = run_denoising(
        model,
        inputs["input_ids"],
        inputs.get("attention_mask"),
        seed=seed,
        disable_compile=True,
    )
    return tokenizer.decode(final[0].tolist(), skip_special_tokens=True), steps


def task_summary(rows):
    """Return only the facts needed to select a probe dataset."""
    by_task = {}
    for row in rows:
        bucket = by_task.setdefault(
            row["task"],
            {"attempts": 0, "passes": 0, "step_counts": [], "failure_reasons": {}},
        )
        bucket["attempts"] += 1
        bucket["passes"] += int(row["tests"]["passed"])
        bucket["step_counts"].append(row["n_steps"])
        if not row["tests"]["passed"]:
            reason = row["tests"]["reason"]
            bucket["failure_reasons"][reason] = bucket["failure_reasons"].get(reason, 0) + 1

    summary = {}
    for task, bucket in by_task.items():
        pass_rate = bucket["passes"] / bucket["attempts"]
        summary[task] = {
            **bucket,
            "pass_rate": pass_rate,
            "mean_steps": sum(bucket["step_counts"]) / len(bucket["step_counts"]),
            "probe_candidate": 0.2 <= pass_rate <= 0.8,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=6)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--task", choices=[task.name for task in SCREEN_TASKS], default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace earlier screen outputs instead of appending to them",
    )
    args = parser.parse_args()

    if CANVAS_LENGTH != 256:
        raise RuntimeError("The code-failure screen is restricted to one 256-token canvas.")
    if args.n_seeds < 2:
        raise ValueError("Use at least two seeds; this screen needs within-prompt variation.")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    if args.overwrite:
        for path in (RESULT_PATH, SUMMARY_PATH):
            if os.path.exists(path):
                os.remove(path)

    tasks = [task for task in SCREEN_TASKS if args.task in (None, task.name)]
    print("Loading DiffusionGemma for single-canvas code-failure screen...")
    model, processor = load_model()
    tokenizer = processor.tokenizer
    rows = []

    for task in tasks:
        inputs = build_chat_inputs(processor, task.prompt, model.device)
        for seed in range(args.start_seed, args.start_seed + args.n_seeds):
            text, steps = run_one(model, tokenizer, inputs, seed)
            test = run_tests(text, task)
            row = {
                "task": task.name,
                "seed": seed,
                "prompt": task.prompt,
                "final_text": text,
                "final_code": extract_python(text),
                "tests": test,
                "n_steps": len(steps),
                "single_canvas": True,
            }
            rows.append(row)
            cache_path = os.path.join(DATA_DIR, f"{task.name}_seed_{seed:04d}.json")
            with open(cache_path, "w") as handle:
                json.dump({**row, "steps": serializable_steps(steps)}, handle)
            outcome = "PASS" if test["passed"] else f"FAIL:{test['reason']}"
            print(f"{task.name} seed={seed}: {outcome} steps={len(steps)}")

    summary = task_summary(rows)
    with open(RESULT_PATH, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with open(SUMMARY_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTask summary:")
    for task, stats in summary.items():
        tag = "PROBE CANDIDATE" if stats["probe_candidate"] else "not mixed"
        print(
            f"{task}: {stats['passes']}/{stats['attempts']} pass "
            f"({stats['pass_rate']:.0%}), mean steps={stats['mean_steps']:.1f} — {tag}"
        )
    print(f"\nWrote {len(rows)} rows to {RESULT_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
