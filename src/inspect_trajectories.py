"""CPU-only viewer for saved generate_trajectories JSON.

The live generate() log is a wall of special tokens and pads. This reads
data/trajectories/problem_*.json and prints the useful bits.

  python src/inspect_trajectories.py           # one-line table
  python src/inspect_trajectories.py --idx 2   # full write-up for one problem
  python src/inspect_trajectories.py --idx 1 --steps   # first / mid / last canvas
"""

import argparse
import glob
import json
import os
import re
import sys

from config import TRAJ_DIR
from parse_commitment import extract_answer

PAD_RE = re.compile(r"(?:<pad>)+")


def clean(text: str) -> str:
    text = PAD_RE.sub("", text)
    for tag in ("<bos>", "<eos>", "<turn|>", "<|turn>", "<channel|>", "<|channel>"):
        text = text.replace(tag, "")
    return text.strip()


def content_len(token_ids):
    """Tokens before the first obvious pad/eos-ish run at the end is hard
    without the tokenizer; use decoded text length as a rough 'did it fill
    the page' check instead, plus whether the last line exists."""
    return len(token_ids)


def load_all():
    paths = sorted(glob.glob(os.path.join(TRAJ_DIR, "problem_*.json")))
    if not paths:
        sys.exit(f"no trajectories in {TRAJ_DIR}/ -- run generate_trajectories.py first")
    return [json.load(open(p)) for p in paths]


def finished(text: str) -> bool:
    return extract_answer(text) is not None and "The answer is" in text


def print_table(trajs):
    print(f"{'idx':>4} {'steps':>5} {'done':>4} {'ans':>8} {'gold':>8}  question")
    print("-" * 88)
    n_done = 0
    for t in trajs:
        ans = extract_answer(t["final_text"])
        done = finished(t["final_text"])
        n_done += int(done)
        q = t["question"].replace("\n", " ")
        if len(q) > 48:
            q = q[:45] + "..."
        print(f"{t['idx']:4d} {t['n_steps']:5d} {'Y' if done else 'N':>4} "
              f"{(ans or '?'):>8} {t['gold_answer']:>8}  {q}")
    print(f"\n{n_done}/{len(trajs)} actually wrote a last-line answer on this one page.")
    print("N = we cut them off at 256 tokens (max_new_tokens), not 'the model got it wrong'.")


def print_one(traj, show_steps: bool):
    ans = extract_answer(traj["final_text"])
    print("=" * 72)
    print(f"problem {traj['idx']:04d}  |  {traj['n_steps']} denoising steps  |  "
          f"seed={traj.get('seed')}")
    print(f"gold={traj['gold_answer']}  extracted={ans}  "
          f"finished={finished(traj['final_text'])}")
    print("-" * 72)
    print("QUESTION")
    print(traj["question"])
    print("-" * 72)
    print("FINAL WRITE-UP (pads stripped)")
    print(clean(traj["final_text"]))
    if show_steps and traj["steps"]:
        picks = [0, len(traj["steps"]) // 2, len(traj["steps"]) - 1]
        labels = ["first step (mostly noise + early guesses)",
                  "midpoint",
                  "last step"]
        for lab, s in zip(labels, picks):
            step = traj["steps"][s]
            print("-" * 72)
            print(f"CANVAS at {lab} (step {step['step']})")
            print(clean(step["text"])[:1500])
            if len(clean(step["text"])) > 1500:
                print("... [truncated]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--steps", action="store_true",
                        help="also print first/mid/last canvas (needs --idx)")
    args = parser.parse_args()

    trajs = load_all()
    if args.idx is None:
        print_table(trajs)
        print("\nFor the actual write-up:  python src/inspect_trajectories.py --idx 0")
        return
    match = [t for t in trajs if t["idx"] == args.idx]
    if not match:
        sys.exit(f"no problem_{args.idx:04d}.json")
    print_one(match[0], show_steps=args.steps)


if __name__ == "__main__":
    main()
