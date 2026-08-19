"""Generate Neel's 5.1.2 squares trajectories over many seeds.

Prompt is answer-first ON PURPOSE -- we want the 9→8 self-correction
(or 9→9 lock-in) they documented, not GSM8K last-line anchoring.

  python src/generate_squares.py --n-seeds 20
"""

import argparse
import json
import os
import time

import torch

from config import CANVAS_LENGTH, SQUARES_DIR, SQUARES_PROMPT
from model_utils import CanvasRecorder, build_chat_inputs, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(SQUARES_DIR, exist_ok=True)
    print("Loading model...")
    t_load = time.time()
    model, processor = load_model()
    print(f"Model loaded ({time.time() - t_load:.0f}s).")
    print(f"Prompt: {SQUARES_PROMPT}\n")

    inputs = build_chat_inputs(processor, SQUARES_PROMPT, model.device)
    t_run = time.time()
    n = args.n_seeds
    for k, seed in enumerate(range(args.start_seed, args.start_seed + n)):
        out_path = os.path.join(SQUARES_DIR, f"seed_{seed:04d}.json")
        if os.path.exists(out_path):
            print(f"[{k+1}/{n}] seed={seed:04d} already cached, skipping")
            continue

        t0 = time.time()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rec = CanvasRecorder(tokenizer=processor.tokenizer)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=CANVAS_LENGTH,
                streamer=rec,
                disable_compile=True,
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
                "seed": seed,
                "prompt": SQUARES_PROMPT,
                "n_steps": len(steps),
                "steps": steps,
                "final_text": final_text,
            }, f)
        elapsed = time.time() - t0
        eta = (time.time() - t_run) / (k + 1) * (n - k - 1) / 60
        print(f"[{k+1}/{n}] seed={seed:04d} steps={len(steps)} ({elapsed:.1f}s, "
              f"ETA {eta:.0f}m) final={final_text[:80]!r}")


if __name__ == "__main__":
    main()
