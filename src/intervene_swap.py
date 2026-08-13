"""Step 3 (core experiment): soft answer-swap intervention.

For each problem where the answer committed at step c but reasoning converged
later at step r: take the cached canvas at injection step
s = c + frac*(r-c), swap the answer tokens to a matched wrong answer, and
CONTINUE denoising from the edited canvas via decoder_input_ids.

Conditions per problem/step:
  swap      -- answer -> matched wrong answer (the treatment)
  noop      -- re-inject the SAME answer (edit-disruption control)
  random    -- perturb a random non-answer, already-stable token (recovery control)

Outcomes recorded: final answer in {injected, original, other}; full final text
(judged later by judge.py for whether post-injection reasoning supports the
injected answer).

!! HONESTY BLOCK -- verify before trusting results !!
(a) decoder_input_ids restarts generate()'s denoising loop on the given canvas.
    The temperature schedule (t_max -> t_min) will restart unless overridden.
    Mitigation attempted below: pass a generation config with t_max set to the
    schedule value at the injection step. VERIFY the config plumbing on-pod by
    reading generation_diffusion_gemma.py source.
(b) Un-denoised positions in the cached canvas hold noise/mask token ids.
    VERIFY generate() accepts a partially-noised canvas as decoder_input_ids
    without re-noising everything (read the sampler's initialize_canvas /
    renoise_canvas to see how a provided canvas is treated).
(c) If (a)/(b) prove wrong, fallback: subclass EntropyBoundSampler /
    copy the generation loop into a custom function -- ~1-2h of work, and you
    get exact mid-trajectory continuation plus a place to patch
    self-conditioning vectors later. This fallback is the mech-interp-grade
    version anyway.
"""

import json
import os
import random

import torch

from config import (ANSWER_DELTAS, INJECTION_FRACTIONS, INTERV_DIR, TRAJ_DIR,
                    CANVAS_LENGTH)
from model_utils import load_model, build_inputs
from parse_commitment import extract_answer


def matched_wrong_answer(ans: str) -> str | None:
    """Same digit count, small perturbation."""
    try:
        v = float(ans) if "." in ans else int(ans)
    except ValueError:
        return None
    for d in ANSWER_DELTAS:
        cand = v + d
        if cand > 0 and len(str(cand)) == len(str(ans)):
            return str(cand)
    return None


def find_answer_token_span(tokenizer, token_ids, answer: str):
    """Locate the token span of the final-answer NUMBER in the canvas.
    Returns (start, end) or None. Strategy: decode with offsets is not
    available for all tokenizers -> decode cumulative prefixes (slow but
    correct enough for 256 tokens)."""
    text = ""
    positions = []
    for j, tid in enumerate(token_ids):
        piece = tokenizer.decode([tid])
        positions.append((len(text), len(text) + len(piece)))
        text += piece
    k = text.rfind(answer)
    if k < 0:
        return None
    span = [j for j, (a, b) in enumerate(positions) if a < k + len(answer) and b > k]
    return (min(span), max(span) + 1) if span else None


def splice_answer(tokenizer, token_ids, span, new_answer: str):
    new_ids = tokenizer.encode(new_answer, add_special_tokens=False)
    return token_ids[:span[0]] + new_ids + token_ids[span[1]:]


def run_condition(model, processor, traj, canvas_ids, label, meta):
    inputs = build_inputs(processor, traj["question"], model.device)
    canvas = torch.tensor([canvas_ids[:CANVAS_LENGTH]], device=model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            decoder_input_ids=canvas,       # continue from edited canvas
            max_new_tokens=CANVAS_LENGTH,
            # TODO(VERIFY a): generation_config with t_max lowered to the
            # schedule temperature at the injection step.
        )
    final_text = processor.tokenizer.decode(
        out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    final_ans = extract_answer(final_text)
    return {**meta, "condition": label, "final_text": final_text,
            "final_answer": final_ans}


def main():
    os.makedirs(INTERV_DIR, exist_ok=True)
    model, processor = load_model()
    tok = processor.tokenizer
    import csv, glob
    commit = {int(r["idx"]): r for r in csv.DictReader(open("results/commitment.csv"))}

    results = []
    for path in sorted(glob.glob(os.path.join(TRAJ_DIR, "problem_*.json"))):
        traj = json.load(open(path))
        row = commit.get(traj["idx"])
        if not row or not row["commit_lead"] or int(row["commit_lead"]) <= 2:
            continue  # need a window between commitment and convergence
        c, r = int(row["answer_commit_step"]), int(row["reasoning_converge_step"])
        orig = row["final_answer"]
        wrong = matched_wrong_answer(orig)
        if wrong is None:
            continue

        for frac in INJECTION_FRACTIONS:
            s = c + int(frac * (r - c))
            ids = traj["steps"][s]["token_ids"]
            span = find_answer_token_span(tok, ids, orig)
            if span is None:
                continue
            meta = {"idx": traj["idx"], "inject_step": s, "frac": frac,
                    "orig_answer": orig, "injected_answer": wrong,
                    "commit_step": c, "converge_step": r}

            # treatment: swap to wrong answer
            results.append(run_condition(
                model, processor, traj, splice_answer(tok, ids, span, wrong),
                "swap", meta))
            # control 1: re-inject same answer (edit disruption)
            results.append(run_condition(
                model, processor, traj, splice_answer(tok, ids, span, orig),
                "noop", meta))
            # control 2: perturb a random stable non-answer token
            j = random.choice([k for k in range(len(ids))
                               if not (span[0] <= k < span[1])])
            rand_ids = list(ids); rand_ids[j] = random.choice(ids)
            results.append(run_condition(
                model, processor, traj, rand_ids, "random", meta))

            with open(os.path.join(INTERV_DIR, "interventions.jsonl"), "a") as f:
                for res in results[-3:]:
                    f.write(json.dumps(res) + "\n")
            print(f"[{traj['idx']:04d}] frac={frac} "
                  f"swap->{results[-3]['final_answer']} (orig {orig}, inj {wrong})")


if __name__ == "__main__":
    main()
