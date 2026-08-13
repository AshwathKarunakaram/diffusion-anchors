"""Step 2 (replication analysis, CPU-only): compute answer-commitment and
reasoning-convergence steps from cached trajectories.

Produces results/commitment.csv with one row per problem:
  idx, n_steps, final_answer, gold_answer, correct,
  answer_commit_step, reasoning_converge_step, commit_lead
The headline replication check: answer_commit_step << reasoning_converge_step
on a substantial fraction of problems (answer-first ordering, cf.
arXiv 2608.05687 which found this for LLaDA/Dream).
"""

import csv
import glob
import json
import os
import re

from config import TRAJ_DIR, REASONING_MATCH_FRAC

ANSWER_RE = re.compile(r"[Tt]he answer is\s*\$?(-?[\d,]+(?:\.\d+)?)")


def extract_answer(text: str):
    """Regex cascade (mirrors 2608.05687 App. B style)."""
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    m = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)\s*$", text.strip())
    return m[-1].replace(",", "") if m else None


def commit_step(traj):
    """First step s such that extract_answer(step_text) == final answer for
    ALL steps >= s (stability, not first appearance)."""
    final = extract_answer(traj["final_text"])
    if final is None:
        return None, None
    per_step = [extract_answer(s["text"]) for s in traj["steps"]]
    commit = None
    for s in range(len(per_step)):
        if all(a == final for a in per_step[s:]):
            commit = s
            break
    return commit, final


def reasoning_converge_step(traj):
    """First step from which >= REASONING_MATCH_FRAC of final-canvas tokens
    (excluding the answer line) already match, at every later step."""
    final_ids = traj["steps"][-1]["token_ids"]
    n = len(final_ids)
    if n == 0:
        return None
    for s in range(len(traj["steps"])):
        ok = True
        for later in traj["steps"][s:]:
            ids = later["token_ids"]
            match = sum(1 for a, b in zip(ids, final_ids) if a == b) / n
            if match < REASONING_MATCH_FRAC:
                ok = False
                break
        if ok:
            return s
    return None


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(TRAJ_DIR, "problem_*.json"))):
        traj = json.load(open(path))
        c, final = commit_step(traj)
        r = reasoning_converge_step(traj)
        rows.append({
            "idx": traj["idx"],
            "n_steps": traj["n_steps"],
            "final_answer": final,
            "gold_answer": traj["gold_answer"],
            "correct": final == traj["gold_answer"],
            "answer_commit_step": c,
            "reasoning_converge_step": r,
            "commit_lead": (r - c) if (c is not None and r is not None) else None,
        })

    os.makedirs("results", exist_ok=True)
    with open("results/commitment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    leads = [r["commit_lead"] for r in rows if r["commit_lead"] is not None]
    early = sum(1 for l in leads if l > 0)
    print(f"{len(rows)} problems | answer commits before reasoning converges on "
          f"{early}/{len(leads)} ({100*early/max(len(leads),1):.0f}%)")
    print("If this fraction is small, the premise fails on DiffusionGemma -> pivot.")


if __name__ == "__main__":
    main()
