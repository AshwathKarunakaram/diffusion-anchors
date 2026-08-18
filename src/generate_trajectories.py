"""Step 1 (replication): run DiffusionGemma on short GSM8K problems and cache
every intermediate canvas to disk.

Run inside tmux:  python src/generate_trajectories.py

Output: data/trajectories/problem_{i:04d}.json with
  question, gold_answer, prompt_token_ids, seed,
  steps: [{step, token_ids, text}], final_text
All downstream analysis runs on these files with NO GPU.

`seed` is recorded (and set right before the `generate()` call) so
`intervene_swap.py` can reseed identically and replay this exact trajectory
up to an injection step before editing it live -- see custom_denoise.py and
intervene_swap.py for why that matters. `disable_compile=True` matches what
the replay uses too, keeping both runs on the same eager decoder path.
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset

from config import N_PROBLEMS, MAX_SOLUTION_CHARS, TRAJ_DIR, CANVAS_LENGTH, MAX_DENOISING_STEPS
from model_utils import load_model, CanvasRecorder, build_inputs


def gold_answer(sol: str) -> str:
    m = re.search(r"####\s*([\-\d,\.]+)", sol)
    return m.group(1).replace(",", "") if m else ""


def select_problems(n_problems: int):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    picked = []
    for ex in ds:
        if len(ex["answer"]) <= MAX_SOLUTION_CHARS:
            picked.append({"question": ex["question"], "gold": gold_answer(ex["answer"])})
        if len(picked) >= n_problems:
            break
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-problems",
        type=int,
        default=N_PROBLEMS,
        help=f"how many GSM8K problems to generate (default: {N_PROBLEMS})",
    )
    args = parser.parse_args()

    os.makedirs(TRAJ_DIR, exist_ok=True)
    print("Loading model...")
    t_load = time.time()
    model, processor = load_model()
    print(f"Model loaded ({time.time() - t_load:.0f}s).")
    problems = select_problems(args.n_problems)
    print(f"Selected {len(problems)} short GSM8K problems. Starting generation.\n")

    t_run_start = time.time()
    for i, prob in enumerate(problems):
        out_path = os.path.join(TRAJ_DIR, f"problem_{i:04d}.json")
        if os.path.exists(out_path):
            print(f"[{i+1}/{len(problems)}] idx={i:04d} already cached, skipping")
            continue

        t0 = time.time()
        seed = i
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = build_inputs(processor, prob["question"], model.device)
        rec = CanvasRecorder(tokenizer=processor.tokenizer)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=CANVAS_LENGTH,
                streamer=rec,
                disable_compile=True,  # matches the eager path intervene_swap.py's replay uses
            )

        steps = [
            {"step": s, "token_ids": ids, "text": processor.tokenizer.decode(ids)}
            for s, ids in enumerate(rec.draft_history)
        ]
        final_text = processor.tokenizer.decode(
            out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        with open(out_path, "w") as f:
            json.dump({
                "idx": i,
                "question": prob["question"],
                "gold_answer": prob["gold"],
                "prompt_len": int(inputs["input_ids"].shape[1]),
                "seed": seed,
                "n_steps": len(steps),
                "steps": steps,
                "final_text": final_text,
            }, f)
        elapsed = time.time() - t0
        avg = (time.time() - t_run_start) / (i + 1)
        eta_min = avg * (len(problems) - i - 1) / 60
        print(f"[{i+1}/{len(problems)}] idx={i:04d} steps={len(steps)} ({elapsed:.1f}s, "
              f"ETA {eta_min:.0f}m) final={final_text[-60:]!r}")


if __name__ == "__main__":
    main()
