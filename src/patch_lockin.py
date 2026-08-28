"""Causal test: patch a correcting run's self-conditioning into a locked run.

Donors are possible_corrector runs, recipients are possible_locked_wrong runs
(both read from results/lockin_sweep.jsonl). At a chosen denoising step k the
recipient's self-conditioning logits S are replaced by the donor's -- over the
answer window, everything outside it, or the whole canvas -- and denoising
continues. Success = the recipient's final answer flips to gold.

Conditions per (recipient, donor, step, region):
  * noop         -- plain replay, run before every patch. It must reproduce
                    the recipient's locked answer or the run is discarded as
                    nondeterministic rather than silently counted.
  * donor        -- the real patch, from a corrector.
  * shuffle      -- donor S with the patched positions randomly permuted.
                    Controls for "any perturbation frees the lock", but note
                    it only discriminates when the prompt's natural correct
                    rate is well below 1: on an easy family a shuffle is just
                    a reroll and lands at the natural rate.
  * locked_donor -- S from a DIFFERENT locked run. This is the control the
                    shuffle cannot provide: coherent state that is not
                    correcting state. If it rescues as well as a corrector,
                    the result reduces to "any coherent S resets the run".

Read every arm against the family's natural correct rate (analyze_patch.py
prints it), never against zero.

Run:
    python src/patch_lockin.py --prompt two_aces_hand --steps 1,2 \
        --n-donors 3 --locked-donor
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


def region_slices(window, canvas_len, region: str):
    """Position ranges the patch may write.

    answer     -- the answer window only
    not_answer -- everything EXCEPT the answer window (does the fate live
                  outside the answer positions?)
    all        -- the whole canvas
    """
    if region == "all":
        return [(0, canvas_len)]
    if window is None:
        return []
    start, end = window
    if region == "answer":
        return [(start, end)]
    if region == "not_answer":
        return [span for span in ((0, start), (end, canvas_len)) if span[1] > span[0]]
    raise ValueError(f"unknown region: {region}")


def make_patch(step_k: int, donor_S, window, region: str, shuffle: bool, seed: int):
    """Replace S at step k inside the region's position ranges."""
    call = {"i": -1, "fired": False}

    def fn(state):
        call["i"] += 1
        if call["i"] != step_k:
            return None
        S = state["self_conditioning_logits"].clone()
        donor = donor_S.to(S.device, S.dtype)
        spans = region_slices(window, S.shape[1], region)
        if not spans:
            return None
        generator = random.Random(seed)
        delta_sq, base_sq = 0.0, 0.0
        for start, end in spans:
            patch = donor[:, start:end, :]
            if shuffle:
                order = list(range(end - start))
                generator.shuffle(order)
                patch = patch[:, order, :]
            original = S[:, start:end, :]
            # How far this patch actually moves the state. A locked donor's S
            # resembles a locked recipient's more than a corrector's does, so
            # "locked_donor rescues less" could otherwise be explained by it
            # simply being a smaller edit. Logging the magnitude lets the
            # analysis check that rather than assume it.
            delta_sq += float((patch.float() - original.float()).pow(2).sum())
            base_sq += float(original.float().pow(2).sum())
            S[:, start:end, :] = patch
        call["fired"] = True
        call["delta_norm"] = delta_sq ** 0.5
        call["relative_delta"] = (delta_sq / base_sq) ** 0.5 if base_sq > 0 else None
        return {"self_conditioning_logits": S}

    fn.state = call
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", action="append",
                        choices=[prompt.name for prompt in PROMPTS], default=None)
    parser.add_argument("--steps", type=str, default="1,2,4,6",
                        help="comma-separated step indices at which to patch")
    parser.add_argument("--regions", type=str, default="answer,all",
                        help="comma-separated: answer, not_answer, all")
    parser.add_argument("--max-per-label", type=int, default=12)
    parser.add_argument("--skip-shuffle", action="store_true")
    parser.add_argument("--n-donors", type=int, default=1,
                        help="how many corrector donors to use (donor-idiosyncrasy check)")
    parser.add_argument("--locked-donor", action="store_true",
                        help="also transplant S from OTHER locked runs: the "
                             "coherent-but-wrong control the shuffle cannot provide")
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
        correctors = [row for row in rows if row["trajectory_label"] in CORRECT_LABELS]
        recipients = [row for row in rows if row["trajectory_label"] in LOCKED_LABELS]
        if not correctors or not recipients:
            print(f"{prompt_name}: need both donors and recipients, skipping")
            continue

        # Donor pool: N correctors, plus (for the coherent-but-wrong control)
        # two locked runs so every recipient can be given a locked donor that
        # is not itself.
        donor_specs = [(row["seed"], "donor") for row in correctors[: args.n_donors]]
        locked_pool = [row["seed"] for row in recipients[:2]]
        if args.locked_donor:
            donor_specs += [(seed, "locked_donor") for seed in locked_pool]

        inputs = build_chat_inputs(processor, prompt.prompt, model.device)
        print(f"{prompt_name}: donors={donor_specs}, {len(recipients)} recipients, "
              f"steps={step_ks}, regions={regions}")

        # Capture every donor's S at every requested step (one replay each).
        donor_S = {}
        for donor_seed, kind in donor_specs:
            for step_k in step_ks:
                store = {}
                _, _, donor_steps = replay(
                    model, tokenizer, inputs, donor_seed, capture_s_at(step_k, store))
                if "S" not in store:
                    print(f"  donor seed={donor_seed} has only {donor_steps} steps; "
                          f"step {step_k} skipped")
                    continue
                donor_S[(donor_seed, step_k)] = (store["S"], kind)
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

            for (donor_seed, step_k), (S, kind) in donor_S.items():
                if step_k >= n_steps or donor_seed == seed:
                    continue  # a run cannot donate to itself
                for region in regions:
                    if region != "all" and window is None:
                        continue
                    conditions = [kind]
                    if kind == "donor" and not args.skip_shuffle:
                        conditions.append("shuffle")
                    for condition in conditions:
                        patch = make_patch(step_k, S, window, region,
                                           shuffle=(condition == "shuffle"), seed=seed)
                        _, patched_answer, patched_steps = replay(
                            model, tokenizer, inputs, seed, patch)
                        row = {
                            "prompt_name": prompt_name,
                            "donor_seed": donor_seed,
                            "recipient_seed": seed,
                            "step_k": step_k,
                            "region": region,
                            "condition": condition,
                            "fired": patch.state["fired"],
                            "delta_norm": patch.state.get("delta_norm"),
                            "relative_delta": patch.state.get("relative_delta"),
                            "noop_answer": noop_answer,
                            "patched_answer": patched_answer,
                            "gold_answer": recipient["gold_answer"],
                            "flipped_to_gold": patched_answer == recipient["gold_answer"],
                            "n_steps": patched_steps,
                        }
                        results.append(row)
                        if not row["fired"]:
                            print(f"  WARNING seed={seed} k={step_k} {region}: patch "
                                  f"never fired (step past end, or empty region); "
                                  f"this row is excluded from rates")
                        print(f"  seed={seed} <- d{donor_seed} k={step_k} "
                              f"{region}/{condition}: {noop_answer} -> {patched_answer} "
                              f"{'FLIPPED' if row['flipped_to_gold'] else ''}")

    with open(OUT_PATH, "a") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")

    print()
    for condition in ("donor", "locked_donor", "shuffle"):
        subset = [row for row in results if row["condition"] == condition]
        if not subset:
            continue
        flips = sum(row["flipped_to_gold"] for row in subset)
        print(f"{condition}: {flips}/{len(subset)} flipped to gold "
              f"({flips / len(subset):.0%})")
        for region in sorted({row["region"] for row in subset}):
            rows_r = [row for row in subset if row["region"] == region]
            flips_r = sum(row["flipped_to_gold"] for row in rows_r)
            print(f"    {region}: {flips_r}/{len(rows_r)}")
    print(f"Wrote {len(results)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
