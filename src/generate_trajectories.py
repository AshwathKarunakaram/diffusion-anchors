"""Step 1 (replication): run DiffusionGemma on short GSM8K problems and cache
every intermediate canvas to disk.

Run inside tmux:  python src/generate_trajectories.py

Output: data/trajectories/problem_{i:04d}.json with
  question, gold_answer, prompt_token_ids,
  steps: [{step, token_ids, text}], final_text
All downstream analysis runs on these files with NO GPU.
"""

import json
import os
import re

import torch
from datasets import load_dataset

from config import N_PROBLEMS, MAX_SOLUTION_CHARS, TRAJ_DIR, CANVAS_LENGTH, MAX_DENOISING_STEPS
from model_utils import load_model, CanvasRecorder, build_inputs


def gold_answer(sol: str) -> str:
    m = re.search(r"####\s*([\-\d,\.]+)", sol)
    return m.group(1).replace(",", "") if m else ""


def select_problems():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    picked = []
    for ex in ds:
        if len(ex["answer"]) <= MAX_SOLUTION_CHARS:
            picked.append({"question": ex["question"], "gold": gold_answer(ex["answer"])})
        if len(picked) >= N_PROBLEMS:
            break
    return picked


def main():
    os.makedirs(TRAJ_DIR, exist_ok=True)
    model, processor = load_model()
    problems = select_problems()
    print(f"Selected {len(problems)} short GSM8K problems")

    for i, prob in enumerate(problems):
        out_path = os.path.join(TRAJ_DIR, f"problem_{i:04d}.json")
        if os.path.exists(out_path):
            continue

        inputs = build_inputs(processor, prob["question"], model.device)
        rec = CanvasRecorder(tokenizer=processor.tokenizer)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=CANVAS_LENGTH,
                streamer=rec,
                # T=0-style determinism if supported; VERIFY sampler config
                # options in DiffusionGemmaGenerationConfig on the pod.
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
                "n_steps": len(steps),
                "steps": steps,
                "final_text": final_text,
            }, f)
        print(f"[{i:04d}] steps={len(steps)} final={final_text[-60:]!r}")


if __name__ == "__main__":
    main()
