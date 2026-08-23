"""Feature-level test: is there a single direction that controls self-correction?

The transplant experiment moves a whole donor state. This asks the sharper
question: take the self-conditioning pathway's HIDDEN output (d=2816, not the
262k-vocab logits), average it over the answer window at an early denoising
step, and compute the difference of group means

    v = mean(hidden | correcting runs) - mean(hidden | locked runs)

then add alpha * v back into a locked run's self-conditioning output at that
step. If a single direction rescues locked runs with a dose-response in
alpha -- and the norm-matched random direction does not -- then "can this run
still correct itself" is carried by a linear feature, not merely by donor
state as a whole.

Three tests, in increasing strength:
  1. rescue      -- +alpha*v into locked runs; expect flips to gold.
  2. random      -- norm-matched random direction; expect no flips.
  3. induce      -- -alpha*v into CORRECTING runs; if they lock in, control is
                    bidirectional, which is much harder to explain by "any
                    perturbation helps".
  4. transfer    -- with --direction-from, build v on one prompt family and
                    apply it to another: does a general correctability
                    direction exist, or is it prompt-specific?

Hooks the module output rather than the state dict because run_denoising's
intervention callback exposes the vocab-space logits, while the direction is
only tractable (and only interpretable as a feature) in hidden space.

Run:
    python src/steer_lockin.py --smoke                    # alignment check
    python src/steer_lockin.py --prompt two_aces_hand
    python src/steer_lockin.py --prompt increasing_three_digits \
        --direction-from two_aces_hand                    # cross-prompt transfer
"""

import argparse
import json
import os

import numpy as np
import torch

from capture_lens import answer_span_and_targets, select_runs, CORRECT_LABELS, LOCKED_LABELS
from custom_denoise import run_denoising
from lockin_answers import extract_answer
from config import SELF_CONDITIONING_PATH
from lockin_prompts import PROMPTS
from model_utils import build_chat_inputs, get_module, load_model

OUT_PATH = "results/steer_lockin.jsonl"
DIR_DIR = "data/steer_directions"


def sc_module(model):
    return get_module(model, SELF_CONDITIONING_PATH)


def attach_capture(model, store):
    """Record the self-conditioning module's hidden output, one entry per call."""

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        store.append(hidden.detach()[0].to("cpu", torch.float32))

    return sc_module(model).register_forward_hook(hook)


def attach_steer(model, step_k, delta, window):
    """Add `delta` to the self-conditioning hidden output at one step."""
    state = {"i": -1, "fired": False}

    def hook(module, args, output):
        state["i"] += 1
        if state["i"] != step_k:
            return None
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        start, end = window if window is not None else (0, hidden.shape[1])
        hidden[0, start:end, :] += delta.to(hidden.device, hidden.dtype)
        state["fired"] = True
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    handle = sc_module(model).register_forward_hook(hook)
    return handle, state


def replay(model, tokenizer, inputs, seed, capture_store=None, steer=None):
    handles = []
    state = None
    if capture_store is not None:
        handles.append(attach_capture(model, capture_store))
    if steer is not None:
        handle, state = attach_steer(model, *steer)
        handles.append(handle)
    try:
        final, steps = run_denoising(
            model, inputs["input_ids"], inputs.get("attention_mask"),
            seed=seed, disable_compile=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    text = tokenizer.decode(final[0].tolist(), skip_special_tokens=True)
    return final[0].tolist(), extract_answer(text), len(steps), state


def window_for(tokenizer, final_ids, gold_answer):
    targets = answer_span_and_targets(tokenizer, final_ids, gold_answer)
    return tuple(targets["window"]) if targets else None


def build_direction(model, tokenizer, inputs, rows, step_k, gold_answer, mode="pooled"):
    """Difference of group means in the self-conditioning hidden space.

    mode="pooled"  -> one (d,) vector: the window is averaged away first, so
                      this asks whether a SINGLE direction carries the fate.
    mode="perpos"  -> an (L, d) tensor keeping per-position structure, where L
                      is the shortest answer window across runs and positions
                      are aligned to the window start. Strictly more
                      expressive; closer to what the full transplant moves.
    """
    slices = {"donor": [], "locked": []}
    for row in rows:
        store = []
        final_ids, answer, n_steps, _ = replay(
            model, tokenizer, inputs, row["seed"], capture_store=store)
        if answer != row["final_answer"]:
            print(f"    seed={row['seed']}: replay mismatch, excluded from direction")
            continue
        if len(store) <= step_k:
            print(f"    seed={row['seed']}: only {len(store)} sc calls, excluded")
            continue
        window = window_for(tokenizer, final_ids, gold_answer)
        if window is None:
            print(f"    seed={row['seed']}: no answer window, excluded")
            continue
        start, end = window
        group = "donor" if row["trajectory_label"] in CORRECT_LABELS else "locked"
        slices[group].append(store[step_k][start:end])  # (win_len, d)
    if not slices["donor"] or not slices["locked"]:
        return None

    if mode == "pooled":
        stacks = {g: torch.stack([s.mean(dim=0) for s in v]) for g, v in slices.items()}
        length = None
    else:
        length = min(s.shape[0] for v in slices.values() for s in v)
        stacks = {g: torch.stack([s[:length] for s in v]) for g, v in slices.items()}
    mean_correct = stacks["donor"].mean(dim=0)
    mean_locked = stacks["locked"].mean(dim=0)
    return {
        "v": mean_correct - mean_locked,
        "length": length,
        "mean_correct_norm": float(mean_correct.norm()),
        "mean_locked_norm": float(mean_locked.norm()),
        "n_correct": len(slices["donor"]),
        "n_locked": len(slices["locked"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="two_aces_hand",
                        choices=[prompt.name for prompt in PROMPTS])
    parser.add_argument("--direction-from", default=None,
                        choices=[prompt.name for prompt in PROMPTS],
                        help="build the direction on this prompt, steer the other one")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--alphas", default="1,2,4,8,16")
    parser.add_argument("--mode", choices=("pooled", "perpos"), default="perpos",
                        help="pooled = one direction; perpos keeps per-position structure")
    parser.add_argument("--max-per-label", type=int, default=8)
    parser.add_argument("--skip-induce", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="check sc-call alignment on one run, then stop")
    args = parser.parse_args()
    alphas = [float(v) for v in args.alphas.split(",")]
    prompt_by_name = {prompt.name: prompt for prompt in PROMPTS}
    prompt = prompt_by_name[args.prompt]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(DIR_DIR, exist_ok=True)
    print("Loading DiffusionGemma for steering...")
    model, processor = load_model()
    tokenizer = processor.tokenizer
    inputs = build_chat_inputs(processor, prompt.prompt, model.device)

    if args.smoke:
        rows = select_runs(args.prompt, 1)
        store = []
        final_ids, answer, n_steps, _ = replay(
            model, tokenizer, inputs, rows[0]["seed"], capture_store=store)
        print(f"\n--- smoke ---\nseed={rows[0]['seed']} answer={answer}")
        print(f"denoising steps={n_steps}  self_conditioning calls={len(store)}")
        print(f"hidden shape per call: {tuple(store[0].shape)}")
        print(f"window={window_for(tokenizer, final_ids, rows[0]['gold_answer'])}")
        print("calls should equal steps (or steps+1); hidden should be (256, 2816)")
        return

    # --- build the direction -------------------------------------------------
    source_name = args.direction_from or args.prompt
    source_prompt = prompt_by_name[source_name]
    source_rows = select_runs(source_name, args.max_per_label)
    source_inputs = (inputs if source_name == args.prompt
                     else build_chat_inputs(processor, source_prompt.prompt, model.device))
    print(f"building {args.mode} direction on {source_name} at step {args.step} "
          f"from {len(source_rows)} runs...")
    direction = build_direction(model, tokenizer, source_inputs, source_rows,
                                args.step, source_rows[0]["gold_answer"], mode=args.mode)
    if direction is None:
        raise SystemExit("could not build a direction: need both groups represented")
    v = direction["v"]
    print(f"  direction from {direction['n_correct']} correcting / "
          f"{direction['n_locked']} locked runs; ||v||={v.norm():.3f}, "
          f"||mean_correct||={direction['mean_correct_norm']:.3f}")
    np.save(os.path.join(DIR_DIR, f"v_{source_name}_step{args.step}_{args.mode}.npy"),
            v.numpy())

    generator = torch.Generator().manual_seed(0)
    v_random = torch.randn(v.shape, generator=generator)
    v_random = v_random / v_random.norm() * v.norm()

    # --- steer ---------------------------------------------------------------
    rows = select_runs(args.prompt, args.max_per_label)
    locked = [row for row in rows if row["trajectory_label"] in LOCKED_LABELS]
    correcting = [row for row in rows if row["trajectory_label"] in CORRECT_LABELS]
    results = []

    def run_condition(row, condition, delta_base, sign):
        final_ids, noop_answer, n_steps, _ = replay(model, tokenizer, inputs, row["seed"])
        if noop_answer != row["final_answer"]:
            print(f"  seed={row['seed']}: noop {noop_answer} != sweep "
                  f"{row['final_answer']}; skipping")
            return
        window = window_for(tokenizer, final_ids, row["gold_answer"])
        if window is None or args.step >= n_steps:
            return
        if args.mode == "perpos":
            # delta is (L, d); patch exactly L positions from the window start.
            length = delta_base.shape[0]
            if window[1] - window[0] < length:
                print(f"  seed={row['seed']}: window shorter than direction, skipping")
                return
            window = (window[0], window[0] + length)
        for alpha in alphas:
            delta = sign * alpha * delta_base
            _, answer, steps_run, state = replay(
                model, tokenizer, inputs, row["seed"],
                steer=(args.step, delta, window))
            flipped = answer == row["gold_answer"]
            locked_in = (answer is not None and answer != row["gold_answer"])
            record = {
                "prompt_name": args.prompt,
                "direction_from": source_name,
                "mode": args.mode,
                "seed": row["seed"],
                "label": row["trajectory_label"],
                "condition": condition,
                "alpha": alpha,
                "step_k": args.step,
                "fired": state["fired"],
                "noop_answer": noop_answer,
                "steered_answer": answer,
                "gold_answer": row["gold_answer"],
                "flipped_to_gold": flipped,
                "left_gold": locked_in if condition == "induce" else None,
                "n_steps": steps_run,
            }
            results.append(record)
            tag = ""
            if condition == "induce":
                tag = "INDUCED LOCK-IN" if locked_in else ""
            elif flipped:
                tag = "RESCUED"
            print(f"  seed={row['seed']} {condition} a={alpha}: "
                  f"{noop_answer} -> {answer} {tag}")

    print(f"\nrescue: +alpha*v into {len(locked)} locked runs")
    for row in locked:
        run_condition(row, "rescue", v, +1.0)

    print(f"\nrandom control: norm-matched random direction, {len(locked)} locked runs")
    for row in locked:
        run_condition(row, "random", v_random, +1.0)

    if not args.skip_induce:
        print(f"\ninduce: -alpha*v into {len(correcting)} correcting runs")
        for row in correcting:
            run_condition(row, "induce", v, -1.0)

    with open(OUT_PATH, "a") as handle:
        for record in results:
            handle.write(json.dumps(record) + "\n")

    print("\n=== summary ===")
    for condition in ("rescue", "random", "induce"):
        subset = [r for r in results if r["condition"] == condition]
        if not subset:
            continue
        print(f"{condition}:")
        for alpha in alphas:
            at_alpha = [r for r in subset if r["alpha"] == alpha]
            if not at_alpha:
                continue
            if condition == "induce":
                hits = sum(bool(r["left_gold"]) for r in at_alpha)
                print(f"  alpha={alpha}: {hits}/{len(at_alpha)} left the gold answer")
            else:
                hits = sum(r["flipped_to_gold"] for r in at_alpha)
                print(f"  alpha={alpha}: {hits}/{len(at_alpha)} rescued to gold")
    print(f"\nWrote {len(results)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
