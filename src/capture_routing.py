"""Exploratory: do locked and correcting runs route to different experts?

DiffusionGemma's decoder is mixture-of-experts -- every layer has a router
that sends each position to a subset of experts. Nobody has looked at expert
routing in a text diffusion model, so this is the highest-variance experiment
in the repo: it may show a clean divergence at the first denoising step, or
nothing at all. It is deliberately kept separate from the main pipeline and
none of the headline claims depend on it.

Router outputs differ between transformers versions, so --smoke prints the
raw shapes first and the parser handles the common shapes rather than
assuming one. If --smoke shows something unexpected, stop and read it; do
not trust a divergence number computed off a misparsed tensor.

Measures, at one early step, per layer:
  * mean Jensen-Shannon divergence between the two groups' mean routing
    distributions over the answer window (0 = identical routing);
  * a within-group baseline: the same divergence computed between two random
    halves of the SAME group. The between-group number only means something
    if it clearly exceeds this.

Run:
    python src/capture_routing.py --smoke
    python src/capture_routing.py --prompt two_aces_hand --step 1
"""

import argparse
import json
import os

import numpy as np
import torch

from capture_lens import select_runs, LOCKED_LABELS
from config import DECODER_LAYERS_PATH
from custom_denoise import run_denoising
from lockin_answers import extract_answer
from lockin_prompts import PROMPTS
from model_utils import build_chat_inputs, get_module, load_model

OUT_PATH = "results/routing_divergence.json"
WINDOW = (0, 24)


def parse_router_output(output):
    """Return per-position expert weights (n_pos, n_experts), or None.

    Handles the shapes routers commonly return: a bare logits tensor, or a
    tuple whose first 3-D/2-D member is the logits.
    """
    candidates = output if isinstance(output, tuple) else (output,)
    for item in candidates:
        if not isinstance(item, torch.Tensor) or item.dtype in (torch.int32, torch.int64):
            continue
        tensor = item.detach().float()
        if tensor.dim() == 3:
            tensor = tensor[0]
        if tensor.dim() == 2 and tensor.shape[0] >= WINDOW[1]:
            return torch.softmax(tensor, dim=-1).cpu().numpy()
    return None


def attach_router_hooks(model, store):
    """store[layer_index] gets one entry per router call."""
    layers = get_module(model, DECODER_LAYERS_PATH)
    handles = []

    def make_hook(index):
        def hook(module, args, output):
            store.setdefault(index, []).append(output)
        return hook

    for index, layer in enumerate(layers):
        if not hasattr(layer, "router"):
            raise AttributeError(
                f"decoder layer {index} has no .router; this model build is "
                f"not the MoE layout this script assumes")
        handles.append(layer.router.register_forward_hook(make_hook(index)))
    return handles, len(layers)


def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions, base 2."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * (np.log2(a) - np.log2(b)), axis=-1)
    return float(np.mean(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def collect(model, tokenizer, inputs, rows, step_k, n_layers, smoke=False):
    groups = {"locked": {}, "correcting": {}}
    for row in rows:
        store = {}
        handles, _ = attach_router_hooks(model, store)
        try:
            final, steps = run_denoising(
                model, inputs["input_ids"], inputs.get("attention_mask"),
                seed=row["seed"], disable_compile=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        answer = extract_answer(tokenizer.decode(final[0].tolist(),
                                                 skip_special_tokens=True))
        if answer != row["final_answer"]:
            print(f"  seed={row['seed']}: replay mismatch, excluded")
            continue

        if smoke:
            print(f"\n--- smoke: seed={row['seed']} ---")
            print(f"denoising steps: {len(steps)}")
            print(f"layers hooked: {n_layers}; router calls on layer 0: "
                  f"{len(store.get(0, []))}")
            raw = store[0][0]
            kinds = ([f"{type(t).__name__}{tuple(t.shape)}" for t in raw]
                     if isinstance(raw, tuple)
                     else [f"{type(raw).__name__}{tuple(getattr(raw, 'shape', ()))}"])
            print(f"layer-0 router output: {kinds}")
            parsed = parse_router_output(raw)
            print(f"parsed weights: "
                  f"{None if parsed is None else parsed.shape}")
            if parsed is not None:
                print(f"row sums (should be ~1): {parsed[:3].sum(axis=-1)}")
            return None

        group = "locked" if row["trajectory_label"] in LOCKED_LABELS else "correcting"
        for layer_index in range(n_layers):
            calls = store.get(layer_index, [])
            if len(calls) <= step_k:
                continue
            weights = parse_router_output(calls[step_k])
            if weights is None:
                continue
            pooled = weights[WINDOW[0]:WINDOW[1]].mean(axis=0)
            groups[group].setdefault(layer_index, []).append(pooled)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="two_aces_hand",
                        choices=[prompt.name for prompt in PROMPTS])
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-per-label", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prompt = {p.name: p for p in PROMPTS}[args.prompt]

    print("Loading DiffusionGemma for routing capture...")
    model, processor = load_model()
    tokenizer = processor.tokenizer
    inputs = build_chat_inputs(processor, prompt.prompt, model.device)
    rows = select_runs(args.prompt, 1 if args.smoke else args.max_per_label)
    n_layers = len(get_module(model, DECODER_LAYERS_PATH))

    groups = collect(model, tokenizer, inputs, rows, args.step, n_layers, args.smoke)
    if args.smoke or groups is None:
        print("\nIf the parsed shape is not (positions, experts) with rows summing "
              "to ~1, fix parse_router_output before trusting any divergence.")
        return

    rng = np.random.default_rng(0)
    report = {"prompt": args.prompt, "step": args.step, "layers": {}}
    print(f"\nlayer  between-group JS   within-group baseline   n_locked/n_correct")
    for layer_index in range(n_layers):
        locked = groups["locked"].get(layer_index, [])
        correct = groups["correcting"].get(layer_index, [])
        if len(locked) < 2 or len(correct) < 2:
            continue
        locked_arr, correct_arr = np.stack(locked), np.stack(correct)
        between = js_divergence(locked_arr.mean(0), correct_arr.mean(0))

        # Baseline: split the larger group in half and compare halves.
        pool = correct_arr if len(correct_arr) >= len(locked_arr) else locked_arr
        index = rng.permutation(len(pool))
        half = len(pool) // 2
        within = js_divergence(pool[index[:half]].mean(0), pool[index[half:2 * half]].mean(0))

        report["layers"][layer_index] = {
            "between_group_js": between, "within_group_js": within,
            "n_locked": len(locked), "n_correct": len(correct),
        }
        flag = "  <-- exceeds baseline" if between > within * 2 else ""
        print(f"{layer_index:5d}  {between:16.5f}   {within:20.5f}   "
              f"{len(locked)}/{len(correct)}{flag}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {OUT_PATH}")
    print("A between-group value that does not clearly exceed the within-group "
          "baseline is a null result: routing does not distinguish the groups.")


if __name__ == "__main__":
    main()
