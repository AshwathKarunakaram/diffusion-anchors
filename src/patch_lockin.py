"""Causal test: patch a correcting run's self-conditioning into a locked run.

Generalizes patch_squares.py from one hand-picked squares pair to any
lock-in sweep family. Donors are possible_corrector runs, recipients are
possible_locked_wrong runs (both from results/lockin_sweep.jsonl). At a
chosen denoising step k, the recipient's self-conditioning logits S are
replaced by the donor's (answer window only, or the whole canvas) and
denoising continues. Success = the recipient's final answer flips to gold.

Conditions per (recipient, step, region):
  * noop    -- plain replay (must reproduce the locked answer, or the pair
               is discarded as nondeterministic);
  * donor   -- the real patch;
  * shuffle -- donor S with the patched positions randomly permuted
               (controls for "any perturbation frees the lock").

Run:
    python src/patch_lockin.py --prompt two_aces_hand --steps 1,2,4,6
"""

import argparse
import json
import os
import random

import torch

from capture_lens import answer_span_and_targets, select_runs, LOCKED_LABELS, CORRECT_LABELS
from custom_denoise import run_denoising
from lockin_answers import extract_answer
from lockin_prompts import PROMPTS
from model_utils import build_chat_inputs, load_model

OUT_PATH = "results/patch_lockin.jsonl"


def replay(model, tokenizer, inputs, seed, intervention_fn=None):
    final, steps = run_denoising(
        model,
        inputs["input_ids"],
        inputs.get("attention_mask"),
        intervention_fn=intervention_fn,
        seed=seed,
        disable_compile=True,
    )
    final_ids = final[0].tolist()
    text = tokenizer.decode(final_ids, skip_special_tokens=True)
    return final_ids, extract_answer(text), len(steps)


def capture_s_at(step_k: int, store: dict):
    call = {"i": -1}

    def fn(state):
        call["i"] += 1
        if call["i"] == step_k:
            store["S"] = state["self_conditioning_logits"].detach().clone()
        return None

    return fn


def make_patch(step_k: int, donor_S, window, shuffle: bool, seed: int):
    """Replace S at step k inside [window[0], window[1]) (None = full canvas)."""
    call = {"i": -1, "fired": False}

    def fn(state):
        call["i"] += 1
        if call["i"] != step_k:
            return None
        call["fired"] = True
        S = state["self_conditioning_logits"].clone()
        donor = donor_S.to(S.device, S.dtype)
        start, end = (0, S.shape[1]) if window is None else window
        patch = donor[:, start:end, :]
        if shuffle:
            generator = random.Random(seed)
            order = list(range(end - start))
            generator.shuffle(order)
            patch = patch[:, order, :]
        S[:, start:end, :] = patch
        return {"self_conditioning_logits": S}

    fn.state = call
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", action="append",
                        choices=[prompt.name for prompt in PROMPTS], default=None)
    parser.add_argument("--steps", type=str, default="1,2,4,6",
                        help="comma-separated step indices at which to patch")
    parser.add_argument("--regions", type=str, default="answer,all")
    parser.add_argument("--max-per-label", type=int, default=12)
    parser.add_argument("--skip-shuffle", action="store_true")
    args = parser.parse_args()
    prompt_names = args.prompt or ["two_aces_hand"]
    step_ks = [int(v) for v in args.steps.split(",")]
    regions = args.regions.split(",")
    prompt_by_name = {prompt.name: prompt for prompt in PROMPTS}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print("Loading DiffusionGemma for lock-in patching...")
    model, processor = load_model()
    tokenizer = processor.tokenizer
    results = []

    for prompt_name in prompt_names:
        prompt = prompt_by_name[prompt_name]
        rows = select_runs(prompt_name, args.max_per_label)
        donors = [row for row in rows if row["trajectory_label"] in CORRECT_LABELS]
        recipients = [row for row in rows if row["trajectory_label"] in LOCKED_LABELS]
        if not donors or not recipients:
            print(f"{prompt_name}: need both donors and recipients, skipping")
            continue
        donor = donors[0]
        inputs = build_chat_inputs(processor, prompt.prompt, model.device)
        print(f"{prompt_name}: donor seed={donor['seed']}, "
              f"{len(recipients)} recipients, steps={step_ks}, regions={regions}")

        # Capture donor S at every requested step in one replay per step.
        donor_S = {}
        for step_k in step_ks:
            store = {}
            _, donor_final, donor_steps = replay(
                model, tokenizer, inputs, donor["seed"], capture_s_at(step_k, store))
            if "S" not in store:
                print(f"  donor run has only {donor_steps} steps; step {step_k} skipped")
                continue
            donor_S[step_k] = store["S"]
        if not donor_S:
            continue

        for recipient in recipients:
            seed = recipient["seed"]
            final_ids, noop_answer, n_steps = replay(model, tokenizer, inputs, seed)
            if noop_answer != recipient["final_answer"]:
                print(f"  seed={seed}: noop replay {noop_answer} != sweep "
                      f"{recipient['final_answer']}; skipping (nondeterministic)")
                continue
            targets = answer_span_and_targets(tokenizer, final_ids, recipient["gold_answer"])
            window = None
            if targets is not None:
                start, end = targets["window"]
                window = (start, end)

            for step_k, S in donor_S.items():
                if step_k >= n_steps:
                    continue
                for region in regions:
                    region_window = window if region == "answer" else None
                    if region == "answer" and window is None:
                        continue
                    conditions = ["donor"] if args.skip_shuffle else ["donor", "shuffle"]
                    for condition in conditions:
                        patch = make_patch(step_k, S, region_window,
                                           shuffle=(condition == "shuffle"), seed=seed)
                        _, patched_answer, patched_steps = replay(
                            model, tokenizer, inputs, seed, patch)
                        row = {
                            "prompt_name": prompt_name,
                            "donor_seed": donor["seed"],
                            "recipient_seed": seed,
                            "step_k": step_k,
                            "region": region,
                            "condition": condition,
                            "fired": patch.state["fired"],
                            "noop_answer": noop_answer,
                            "patched_answer": patched_answer,
                            "gold_answer": recipient["gold_answer"],
                            "flipped_to_gold": patched_answer == recipient["gold_answer"],
                            "n_steps": patched_steps,
                        }
                        results.append(row)
                        print(f"  seed={seed} k={step_k} {region}/{condition}: "
                              f"{noop_answer} -> {patched_answer} "
                              f"{'FLIPPED' if row['flipped_to_gold'] else ''}")

    with open(OUT_PATH, "a") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")

    flips = sum(row["flipped_to_gold"] and row["condition"] == "donor" for row in results)
    donor_rows = sum(row["condition"] == "donor" for row in results)
    shuffle_flips = sum(row["flipped_to_gold"] and row["condition"] == "shuffle" for row in results)
    shuffle_rows = sum(row["condition"] == "shuffle" for row in results)
    print(f"\ndonor patches: {flips}/{donor_rows} flipped to gold")
    if shuffle_rows:
        print(f"shuffle controls: {shuffle_flips}/{shuffle_rows} flipped to gold")
    print(f"Wrote {len(results)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
