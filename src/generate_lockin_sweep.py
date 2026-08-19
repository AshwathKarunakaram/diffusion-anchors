"""Replicate answer correction and wrong-answer lock-in across prompt families.

This is the premise check for the lock-in project.  It runs only a single
DiffusionGemma canvas, saves every intermediate draft, and records whether the
final integer answer is correct.  The automatic labels are deliberately
conservative aids for triage; qualitative inspection of cached trajectories is
required before making a claim about correction or lock-in.

Run on the A100/Colab runtime:
    python src/generate_lockin_sweep.py --n-seeds 10 --overwrite
"""

import argparse
import json
import os
import re

from config import CANVAS_LENGTH
from custom_denoise import run_denoising
from lockin_prompts import PROMPTS
from model_utils import build_chat_inputs, load_model


DATA_DIR = "data/lockin_sweep"
RESULT_PATH = "results/lockin_sweep.jsonl"
SUMMARY_PATH = "results/lockin_sweep_summary.json"

ANSWER_PATTERNS = (
    re.compile(r"(?:the\s+)?answer\s+is\s*[:=]?\s*\**\s*(-?\d[\d,]*)", re.I),
    re.compile(r"^.*?(?:there\s+are|there\s+is)\s+\**\s*(-?\d[\d,]*)", re.I | re.S),
)
FIRST_INT = re.compile(r"-?\d[\d,]*")


def extract_answer(text: str):
    """Read the answer-first integer, returning None when no integer is visible."""
    text = text.replace("thought\n", "", 1).strip()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1).replace(",", ""))
    match = FIRST_INT.search(text)
    return int(match.group().replace(",", "")) if match else None


def trajectory_label(per_step_answers, final_answer, gold):
    """A triage label, not a proof of the model's reasoning algorithm."""
    visible = [answer for answer in per_step_answers if answer is not None]
    if final_answer == gold:
        if any(answer != gold for answer in visible[:-1]):
            return "possible_corrector"
        return "always_or_early_correct"
    if final_answer is None:
        return "no_final_integer"
    # A wrong answer observed at least twice is more likely to be a genuine
    # stable draft than a single noisy early canvas.
    if visible.count(final_answer) >= 2:
        return "possible_locked_wrong"
    return "wrong_unstable_or_other"


def serializable_steps(tokenizer, steps):
    result = []
    for index, step in enumerate(steps):
        text = tokenizer.decode(step["argmax_canvas"], skip_special_tokens=True)
        result.append(
            {
                "step_index": index,
                "cur_step": step["cur_step"],
                "argmax_canvas": step["argmax_canvas"],
                "accepted_mask": step["accepted_mask"],
                "text": text,
                "first_answer": extract_answer(text),
            }
        )
    return result


def summarize(rows):
    summary = {}
    for row in rows:
        bucket = summary.setdefault(
            row["prompt_name"],
            {
                "gold_answer": row["gold_answer"],
                "attempts": 0,
                "correct_final": 0,
                "possible_correctors": 0,
                "possible_locked_wrong": 0,
                "labels": {},
                "step_counts": [],
            },
        )
        bucket["attempts"] += 1
        bucket["correct_final"] += int(row["final_answer"] == row["gold_answer"])
        bucket["possible_correctors"] += int(row["trajectory_label"] == "possible_corrector")
        bucket["possible_locked_wrong"] += int(row["trajectory_label"] == "possible_locked_wrong")
        label = row["trajectory_label"]
        bucket["labels"][label] = bucket["labels"].get(label, 0) + 1
        bucket["step_counts"].append(row["n_steps"])
    for bucket in summary.values():
        bucket["correct_final_rate"] = bucket["correct_final"] / bucket["attempts"]
        bucket["mean_steps"] = sum(bucket["step_counts"]) / bucket["attempts"]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--prompt", choices=[prompt.name for prompt in PROMPTS], default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if CANVAS_LENGTH != 256:
        raise RuntimeError("This experiment is restricted to one 256-token canvas.")
    if args.n_seeds < 2:
        raise ValueError("Use at least two seeds to distinguish trajectory types.")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    if args.overwrite:
        for path in (RESULT_PATH, SUMMARY_PATH):
            if os.path.exists(path):
                os.remove(path)

    prompts = [prompt for prompt in PROMPTS if args.prompt in (None, prompt.name)]
    print("Loading DiffusionGemma for answer-first lock-in sweep...")
    model, processor = load_model()
    tokenizer = processor.tokenizer
    rows = []

    for prompt in prompts:
        inputs = build_chat_inputs(processor, prompt.prompt, model.device)
        for seed in range(args.start_seed, args.start_seed + args.n_seeds):
            final, steps = run_denoising(
                model,
                inputs["input_ids"],
                inputs.get("attention_mask"),
                seed=seed,
                disable_compile=True,
            )
            final_text = tokenizer.decode(final[0].tolist(), skip_special_tokens=True)
            saved_steps = serializable_steps(tokenizer, steps)
            per_step_answers = [step["first_answer"] for step in saved_steps]
            final_answer = extract_answer(final_text)
            label = trajectory_label(per_step_answers, final_answer, prompt.answer)
            row = {
                "prompt_name": prompt.name,
                "prompt": prompt.prompt,
                "gold_answer": prompt.answer,
                "seed": seed,
                "final_text": final_text,
                "final_answer": final_answer,
                "n_steps": len(steps),
                "trajectory_label": label,
                "single_canvas": True,
            }
            rows.append(row)
            path = os.path.join(DATA_DIR, f"{prompt.name}_seed_{seed:04d}.json")
            with open(path, "w") as handle:
                json.dump({**row, "steps": saved_steps}, handle)
            print(
                f"{prompt.name} seed={seed}: final={final_answer} gold={prompt.answer} "
                f"steps={len(steps)} {label}"
            )

    summary = summarize(rows)
    with open(RESULT_PATH, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with open(SUMMARY_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nPrompt summary:")
    for name, stats in summary.items():
        print(
            f"{name}: {stats['correct_final']}/{stats['attempts']} correct; "
            f"possible correctors={stats['possible_correctors']}; "
            f"possible locked wrong={stats['possible_locked_wrong']}; "
            f"mean steps={stats['mean_steps']:.1f}"
        )
    print(f"\nWrote {len(rows)} rows to {RESULT_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
