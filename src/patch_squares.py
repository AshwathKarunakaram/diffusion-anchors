"""#2: resample-patch S between a 9→8 corrector and 9→9 locked squares runs.

Donor is seed 11 (last visible 9 at step 6, then 8). Recipients are locked
seeds. At that step we copy the donor's self-conditioning into the locked
run (reasoning-only / answer-only / all) and let denoising continue.

  python src/patch_squares.py
"""

import argparse
import json
import os
import re

import torch

from config import SQUARES_DIR, SQUARES_PROMPT, INTERV_DIR
from custom_denoise import run_denoising
from model_utils import build_chat_inputs, find_first_token_span, load_model

FIRST_INT = re.compile(r"(-?\d+)")


def first_int(text: str):
    m = FIRST_INT.search(text.replace(",", ""))
    return m.group(1) if m else None


def load_traj(seed: int):
    path = os.path.join(SQUARES_DIR, f"seed_{seed:04d}.json")
    return json.load(open(path))


def answer_span(tokenizer, token_ids, digit="9"):
    for needle in (f"**{digit}**", f" {digit} ", digit):
        span = find_first_token_span(tokenizer, token_ids, needle)
        if span is not None:
            return span
    return None


def capture_at(k: int, store: dict):
    n = {"i": -1}

    def fn(state):
        n["i"] += 1
        if n["i"] == k:
            store["S"] = state["self_conditioning_logits"].detach().clone()
            store["canvas"] = state["current_canvas"].detach().clone()
            store["argmax"] = state["argmax_canvas"].detach().clone()
        return None
    return fn


def make_s_patch(k, donor_S, region, span, expected_argmax):
    n = {"i": -1}

    def fn(state):
        n["i"] += 1
        if n["i"] != k:
            return None
        live = state["argmax_canvas"][0].tolist()
        if expected_argmax is not None and live != expected_argmax:
            ndiff = sum(a != b for a, b in zip(live, expected_argmax))
            print(f"  replay diff {ndiff}/{len(live)} at step {k} (continuing anyway)")
        S = state["self_conditioning_logits"].clone()
        C = S.shape[1]
        if region == "all":
            S = donor_S
        elif region == "answer":
            if span is None:
                return None
            a, b = span
            S[:, a:b, :] = donor_S[:, a:b, :]
        elif region == "reasoning":
            if span is None:
                return None
            b = span[1]
            if b < C:
                S[:, b:, :] = donor_S[:, b:, :]
        return {"self_conditioning_logits": S}
    return fn


def run_one(model, inputs, tok, seed, intervention_fn=None):
    final, steps = run_denoising(
        model, inputs["input_ids"], inputs.get("attention_mask"),
        intervention_fn=intervention_fn, seed=seed, disable_compile=True,
    )
    text = tok.decode(final[0].tolist(), skip_special_tokens=True)
    return first_int(text), text, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=int, default=11)
    parser.add_argument("--k", type=int, default=6,
                        help="last step donor still shows 9 (0-indexed)")
    parser.add_argument("--locked", default="0,1,5,7,12")
    args = parser.parse_args()
    locked_seeds = [int(x) for x in args.locked.split(",") if x.strip() != ""]

    os.makedirs(INTERV_DIR, exist_ok=True)
    print("Loading model...")
    model, processor = load_model()
    tok = processor.tokenizer
    inputs = build_chat_inputs(processor, SQUARES_PROMPT, model.device)
    k = args.k

    donor_traj = load_traj(args.donor)
    per = [first_int(s["text"]) for s in donor_traj["steps"]]
    print(f"donor seed {args.donor} cached per-step={per} final={first_int(donor_traj['final_text'])}")
    print(f"patch at k={k} (want this to still be 9 on the donor)")

    captured = {}
    d_ans, d_text, _ = run_one(model, inputs, tok, args.donor, capture_at(k, captured))
    print(f"donor REPLAY final={d_ans}  (want 8)\n  {d_text[:180]!r}")
    if "S" not in captured:
        raise SystemExit(f"did not capture S at step {k} — donor had too few steps")
    donor_S = captured["S"]
    donor_span = answer_span(tok, captured["argmax"][0].tolist(), "9")
    print(f"donor answer span at k={k}: {donor_span}")

    rows = []
    out_path = os.path.join(INTERV_DIR, "squares_patch.jsonl")

    def log(row):
        rows.append(row)
        print(f"  {row['label']:40s} seed={row['seed']} -> {row['final']}")
        with open(out_path, "a") as f:
            f.write(json.dumps({**row, "text": row["text"][:500]}) + "\n")

    # sanity: locked replay with no patch should stay 9
    for seed in locked_seeds[:2]:
        ans, text, _ = run_one(model, inputs, tok, seed, None)
        log({"label": "locked_replay", "seed": seed, "final": ans, "text": text,
             "region": None})

    for seed in locked_seeds:
        traj = load_traj(seed)
        if traj["n_steps"] <= k:
            print(f"skip seed {seed}: only {traj['n_steps']} steps")
            continue
        expected = traj["steps"][k]["token_ids"]
        live_span = answer_span(tok, expected, "9") or donor_span
        for region in ("reasoning", "answer", "all"):
            ans, text, _ = run_one(
                model, inputs, tok, seed,
                make_s_patch(k, donor_S, region, live_span, expected),
            )
            log({"label": f"locked+donor_S_{region}", "seed": seed,
                 "final": ans, "text": text, "region": region})

    # reverse: corrector + locked seed 0's S at k
    rev = {}
    run_one(model, inputs, tok, locked_seeds[0], capture_at(k, rev))
    if "S" in rev:
        locked_span = answer_span(tok, rev["argmax"][0].tolist(), "9") or donor_span
        expected_d = donor_traj["steps"][k]["token_ids"]
        for region in ("reasoning", "answer", "all"):
            ans, text, _ = run_one(
                model, inputs, tok, args.donor,
                make_s_patch(k, rev["S"], region, locked_span, expected_d),
            )
            log({"label": f"donor+locked{locked_seeds[0]}_S_{region}",
                 "seed": args.donor, "final": ans, "text": text, "region": region})

    print(f"\nwrote {out_path}")
    print("want: locked+donor_S_reasoning -> 8  (reasoning S causes correction)")
    print("      locked+donor_S_answer    -> 9  (answer S is not the trigger)")
    print("      donor+locked_S_reasoning -> 9  (reverse: steal lock-in)")


if __name__ == "__main__":
    main()
