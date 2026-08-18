"""Step 3 (core experiment): soft answer-swap intervention.

For each problem where the answer committed at step c but reasoning converged
later at step r: reseed identically to the cached trajectory and REPLAY
denoising from scratch via `custom_denoise.run_denoising`. At injection step
s = c + frac*(r-c), edit the LIVE denoising canvas in place (not a cached
argmax snapshot -- see note below) through `intervention_fn`, then let
denoising continue naturally. No restart, no temperature reset -- see
`custom_denoise.py` for why that's sound (it calls the same private
`_denoising_step` HF's `generate()` uses, so there's no reimplementation to
drift out of sync).

Conditions per problem/injection-step:
  swap      -- answer -> matched wrong value, SAME token length (treatment)
  noop      -- re-inject the SAME answer (edit-disruption control)
  random    -- perturb a random already-stable non-answer position (recovery control)

WHY current_canvas, NOT the cached argmax snapshot
`traj["steps"][s]["token_ids"]` is `argmax_canvas` (the streamer's "best
guess" -- see model_utils.py / custom_denoise.py docstrings), not the actual
noisy resumption canvas `current_canvas` the model would see next. Splicing
into the argmax snapshot and resuming FROM it (as the old, broken version of
this file did) quietly replaces every not-yet-accepted position's real noise
with the model's own confident guess -- injecting information that wasn't
there. This version finds the answer span using the cached argmax ids (they
match `current_canvas` at accepted/stable positions almost always, since
accepted positions are by construction the lowest-entropy ones -- sample and
argmax essentially always agree there), but performs the actual edit on the
LIVE `current_canvas` from the replay.

REPLAY SAFETY
Reseeding before a fresh run reproduces the same RNG draws in the same order
as the cached trajectory (see custom_denoise.py's RNG note) ONLY if the run
is otherwise identical, and this model's MoE routing has no deterministic
CUDA kernel (torch.histc -- see custom_denoise.py), so an exact replay is
likely but not guaranteed. Before applying any edit, `make_intervention`
verifies the live `argmax_canvas` at the injection step matches the cached
trajectory's canvas at that step; on a mismatch it raises `ReplayMismatch`
and the problem/step is skipped and logged, rather than silently
intervening on a trajectory different from the one commit/converge steps
were measured on.
"""

import csv
import glob
import json
import os
import random as pyrandom
import time

import torch

from config import ANSWER_DELTAS, INJECTION_FRACTIONS, INTERV_DIR, TRAJ_DIR, USER_SUFFIX
from custom_denoise import run_denoising
from model_utils import build_inputs, find_answer_token_span, load_model
from parse_commitment import extract_answer


class ReplayMismatch(Exception):
    pass


def matched_wrong_answer(tokenizer, ans: str, span_len: int) -> str | None:
    """Same TOKEN length as the original answer's span in the canvas (not
    just same digit-string length -- BPE doesn't guarantee those coincide,
    and a length mismatch would shift every later canvas position)."""
    try:
        v = float(ans) if "." in ans else int(ans)
    except ValueError:
        return None
    for d in ANSWER_DELTAS:
        cand = v + d
        if cand <= 0:
            continue
        cand_str = str(cand)
        if len(tokenizer.encode(cand_str, add_special_tokens=False)) == span_len:
            return cand_str
    return None


def splice_answer(tokenizer, token_ids, span, new_answer: str):
    """Returns None (caller must handle) instead of silently producing a
    canvas of the wrong length if `new_answer` doesn't tokenize to exactly
    `span`'s length in this context."""
    new_ids = tokenizer.encode(new_answer, add_special_tokens=False)
    if len(new_ids) != span[1] - span[0]:
        return None
    return token_ids[:span[0]] + new_ids + token_ids[span[1]:]


def make_intervention(target_idx, edit_fn, expected_argmax_ids):
    """intervention_fn for run_denoising: no-op until the `target_idx`-th
    step (0-indexed, matching cached trajectory step indices), then verifies
    the replay matches the cached trajectory before calling `edit_fn`."""
    counter = {"i": -1}

    def intervention_fn(state):
        counter["i"] += 1
        if counter["i"] != target_idx:
            return None
        live_ids = state["argmax_canvas"][0].tolist()
        if live_ids != expected_argmax_ids:
            n_diff = sum(a != b for a, b in zip(live_ids, expected_argmax_ids))
            raise ReplayMismatch(
                f"replay diverged from cached trajectory at step {target_idx}: "
                f"{n_diff}/{len(live_ids)} positions differ"
            )
        return edit_fn(state)

    return intervention_fn


def make_splice_edit(tokenizer, span, new_answer):
    def edit_fn(state):
        canvas_ids = state["current_canvas"][0].tolist()
        new_ids = splice_answer(tokenizer, canvas_ids, span, new_answer)
        if new_ids is None:
            return None  # length mismatch in this context -- leave canvas untouched
        return {"current_canvas": torch.tensor([new_ids], device=state["current_canvas"].device)}
    return edit_fn


def make_random_edit(span):
    def edit_fn(state):
        canvas_ids = state["current_canvas"][0].tolist()
        candidates = [k for k in range(len(canvas_ids)) if not (span[0] <= k < span[1])]
        j = pyrandom.choice(candidates)
        new_ids = list(canvas_ids)
        new_ids[j] = pyrandom.choice(canvas_ids)
        return {"current_canvas": torch.tensor([new_ids], device=state["current_canvas"].device)}
    return edit_fn


def run_condition(model, tokenizer, inputs, seed, target_idx, edit_fn, expected_argmax_ids, label, meta):
    intervention_fn = make_intervention(target_idx, edit_fn, expected_argmax_ids)
    try:
        final_canvas, _steps = run_denoising(
            model, inputs["input_ids"], inputs.get("attention_mask"),
            intervention_fn=intervention_fn, seed=seed, disable_compile=True,
        )
    except ReplayMismatch as e:
        return {**meta, "condition": label, "final_text": None, "final_answer": None, "error": str(e)}
    final_text = tokenizer.decode(final_canvas[0].tolist(), skip_special_tokens=True)
    final_ans = extract_answer(final_text)
    return {**meta, "condition": label, "final_text": final_text, "final_answer": final_ans}


def main():
    os.makedirs(INTERV_DIR, exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tok = processor.tokenizer
    commit = {int(r["idx"]): r for r in csv.DictReader(open("results/commitment.csv"))}

    out_path = os.path.join(INTERV_DIR, "interventions.jsonl")
    paths = sorted(glob.glob(os.path.join(TRAJ_DIR, "problem_*.json")))
    print(f"{len(paths)} cached trajectories found. Starting interventions.\n")

    n_eligible = n_skipped_window = n_no_seed = n_no_span = n_no_match = n_rows = n_errors = 0
    t_start = time.time()
    for i, path in enumerate(paths):
        traj = json.load(open(path))
        row = commit.get(traj["idx"])
        lead = (row or {}).get("commit_lead")
        # Old 0-9 JSONs have no user_suffix (verbose prompt). Replaying them
        # with the current brief prompt would be a different generation.
        if traj.get("user_suffix") != USER_SUFFIX:
            n_skipped_window += 1
            continue
        if not row or lead in ("", None) or int(lead) < 2:
            n_skipped_window += 1
            continue  # need a window between commitment and convergence
        c, r = int(row["answer_commit_step"]), int(row["reasoning_converge_step"])
        orig = row["final_answer"]
        seed = traj.get("seed")
        if seed is None:
            n_no_seed += 1
            print(f"[{i+1}/{len(paths)}] idx={traj['idx']:04d} no recorded seed -- re-run "
                  f"generate_trajectories.py to get one, skipping")
            continue

        n_eligible += 1
        inputs = build_inputs(processor, traj["question"], model.device)

        for frac in INJECTION_FRACTIONS:
            s = c + int(frac * (r - c))
            argmax_ids = traj["steps"][s]["token_ids"]
            span = find_answer_token_span(tok, argmax_ids, orig)
            if span is None:
                n_no_span += 1
                print(f"[{i+1}/{len(paths)}] idx={traj['idx']:04d} frac={frac}: "
                      f"answer '{orig}' not found in canvas at step {s}, skipping")
                continue
            span_len = span[1] - span[0]
            wrong = matched_wrong_answer(tok, orig, span_len)
            if wrong is None:
                n_no_match += 1
                print(f"[{i+1}/{len(paths)}] idx={traj['idx']:04d} frac={frac}: "
                      f"no token-length-matched wrong answer for '{orig}' "
                      f"(span_len={span_len}), skipping")
                continue

            meta = {"idx": traj["idx"], "inject_step": s, "frac": frac,
                    "orig_answer": orig, "commit_step": c, "converge_step": r}

            conditions = [
                ("swap", make_splice_edit(tok, span, wrong), {"injected_answer": wrong}),
                ("noop", make_splice_edit(tok, span, orig), {"injected_answer": orig}),
                ("random", make_random_edit(span), {"injected_answer": None}),
            ]

            results = [
                run_condition(model, tok, inputs, seed, s, edit_fn, argmax_ids, label, {**meta, **extra})
                for label, edit_fn, extra in conditions
            ]
            n_rows += len(results)
            n_errors += sum(1 for res in results if res.get("error"))

            with open(out_path, "a") as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")
            elapsed_min = (time.time() - t_start) / 60
            print(f"[{i+1}/{len(paths)}] idx={traj['idx']:04d} frac={frac} step={s} "
                  f"({elapsed_min:.1f}m elapsed) swap->{results[0]['final_answer']} "
                  f"(orig {orig}, inj {wrong})"
                  + (f"  ** {sum(1 for res in results if res.get('error'))} REPLAY MISMATCH(ES) **"
                     if any(res.get("error") for res in results) else ""))

    print(f"\nDone. {n_eligible} problems had a usable commit/converge window "
          f"({n_skipped_window} skipped, {n_no_seed} missing seed), "
          f"{n_rows} intervention rows written to {out_path} "
          f"({n_no_span} spans not found, {n_no_match} no length-matched wrong answer, "
          f"{n_errors} replay mismatches).")


if __name__ == "__main__":
    main()
