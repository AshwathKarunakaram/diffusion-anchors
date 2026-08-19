"""Classify squares trajectories: 9→8 corrector vs 9→9 locked vs other.

CPU only. Looks at the FIRST integer in each step's decoded canvas
(answer-first prompt). Paper: by step 4 the top guess is often 9, then
later reasoning can correct to 8 (21^2..28^2). 9 usually includes 20^2=400.

  python src/parse_squares.py
"""

import glob
import json
import os
import re

from config import SQUARES_DIR, SQUARES_GOLD

FIRST_INT = re.compile(r"(-?\d+)")


def first_int(text: str):
    m = FIRST_INT.search(text.replace(",", ""))
    return m.group(1) if m else None


def classify(per_step, final):
    """corrector: some early-ish step is 9 and final is 8.
    locked_9: saw 9 and finished 9.
    always_8: never showed 9, finished 8.
    other: everything else."""
    early = per_step[: max(1, min(5, len(per_step)))]
    saw_9 = "9" in per_step
    saw_9_early = "9" in early
    if saw_9_early and final == SQUARES_GOLD:
        return "corrector_9to8"
    if saw_9 and final == "9":
        return "locked_9"
    if final == SQUARES_GOLD and not saw_9:
        return "always_8"
    if final == "9":
        return "final_9_no_early"
    return "other"


def main():
    paths = sorted(glob.glob(os.path.join(SQUARES_DIR, "seed_*.json")))
    if not paths:
        raise SystemExit(f"no files in {SQUARES_DIR}/ -- run generate_squares.py")

    print(f"{'seed':>5} {'n':>4} {'class':<16} per-step first-int  |  final")
    print("-" * 88)
    counts = {}
    for path in paths:
        traj = json.load(open(path))
        per = [first_int(s["text"]) or "?" for s in traj["steps"]]
        final = first_int(traj["final_text"])
        cls = classify(per, final)
        counts[cls] = counts.get(cls, 0) + 1
        seq = ",".join(per)
        if len(seq) > 40:
            seq = seq[:37] + "..."
        snippet = traj["final_text"].replace("\n", " ")[:40]
        print(f"{traj['seed']:5d} {traj['n_steps']:4d} {cls:<16} {seq:<22} | {final}  {snippet!r}")

    print("\ncounts:", dict(sorted(counts.items())))
    n_corr = counts.get("corrector_9to8", 0)
    n_lock = counts.get("locked_9", 0)
    print(f"corrector_9to8={n_corr}  locked_9={n_lock}")
    if n_corr and n_lock:
        print("BOTH natural classes exist -> do resample patching (#2).")
    elif n_corr and not n_lock:
        print("Almost all correct (or no lock-in) -> do entropy/sharpen lock-in (#1).")
    elif n_lock and not n_corr:
        print("Lock-in without 9→8. Try more seeds, or flatten-S to manufacture correction.")
    else:
        print("No clean 9/8 story. Read a few --idx files before intervening.")


if __name__ == "__main__":
    main()
