"""Step 2 (replication analysis, CPU-only): compute answer-commitment and
reasoning-convergence steps from cached trajectories.

Produces results/commitment.csv with one row per problem:
  idx, n_steps, final_answer, gold_answer, correct,
  answer_commit_step, reasoning_converge_step, commit_lead
The headline replication check: answer_commit_step << reasoning_converge_step
on a substantial fraction of problems (answer-first ordering, cf.
arXiv 2608.05687 which found this for LLaDA/Dream).

Needs the tokenizer (not the 26B weights) to exclude the answer span from
the convergence metric -- see `reasoning_converge_step` docstring for why
that exclusion matters. No GPU needed, but `transformers` must be
importable now (it wasn't before this fix).
"""

import csv
import glob
import json
import os
import re

from transformers import AutoTokenizer, GenerationConfig

from config import MODEL_ID, TRAJ_DIR, REASONING_MATCH_FRAC
from model_utils import find_answer_token_span

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


def reasoning_converge_step(traj, tokenizer, eos_ids, final_answer):
    """First step from which >= REASONING_MATCH_FRAC of REASONING-region
    tokens (final canvas, excluding the answer span and anything at/after
    the first EOS token) already match, at every later step.

    Two exclusions, both necessary -- the previous version compared all 256
    canvas positions with neither:

    1. The answer span. It's stable from `answer_commit_step` onward by
       definition, so leaving it in inflates the match fraction and can make
       "convergence" look like it happens at or before commitment -- which
       defeats the point of measuring the commit-to-converge lag in the
       first place.
    2. Everything at/after the first EOS token. Those canvas positions carry
       no content, so the model has little signal for what belongs there --
       they can stay high-entropy (and keep getting renoised) far longer
       than the actual reasoning text. Left in, they can drag the match
       fraction below threshold until the very last recorded step
       regardless of when the real reasoning stabilized, biasing
       `reasoning_converge_step` toward `n_steps - 1` on every problem.
    """
    final_ids = traj["steps"][-1]["token_ids"]
    content_end = next((i for i, t in enumerate(final_ids) if t in eos_ids), len(final_ids))
    if content_end == 0:
        return None

    span = find_answer_token_span(tokenizer, final_ids[:content_end], final_answer) if final_answer else None
    excluded = set(range(span[0], span[1])) if span else set()
    positions = [j for j in range(content_end) if j not in excluded]
    if not positions:
        return None

    for s in range(len(traj["steps"])):
        ok = True
        for later in traj["steps"][s:]:
            ids = later["token_ids"]
            match = sum(1 for j in positions if ids[j] == final_ids[j]) / len(positions)
            if match < REASONING_MATCH_FRAC:
                ok = False
                break
        if ok:
            return s
    return None


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eos_ids = set(GenerationConfig.from_pretrained(MODEL_ID).eos_token_id or [])

    rows = []
    for path in sorted(glob.glob(os.path.join(TRAJ_DIR, "problem_*.json"))):
        traj = json.load(open(path))
        c, final = commit_step(traj)
        r = reasoning_converge_step(traj, tokenizer, eos_ids, final)
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
