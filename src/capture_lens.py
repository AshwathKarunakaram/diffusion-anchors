"""Logit-lens capture across the (layer x denoising-step) grid.

Replays lock-in sweep runs (same seeds, same single-canvas loop) with forward
hooks on every decoder layer. After each replay it projects each layer's
hidden states through the model's own final norm + lm_head and stores small
reductions per (layer, step):

  * agreement of the lens top-1 with the model's own step output and with the
    final canvas (the layer x step "does the lens work at all" heatmap);
  * top-1 ids + logprobs for an answer window (span +/- WINDOW_PAD tokens);
  * logprob of the gold-answer tokens and of the final-canvas tokens at the
    answer span (when tokenizations align);
  * the self-conditioning channel's top-5 per position per step, captured via
    the intervention callback (no model change, returns None).

Notes:
  * lm_head is applied without any final logit softcapping the generation
    path may add; softcap is monotone, so top-k and rank comparisons are
    unaffected, and all logprobs here share the same transform.
  * Replays are seeded like the sweep; hooks and reductions consume no RNG,
    so trajectories should reproduce (spot-checked via final answers).

Run (after generate_lockin_sweep + relabel produced results/lockin_sweep.jsonl):
    python src/capture_lens.py --smoke                      # one run, sanity prints
    python src/capture_lens.py --prompt two_aces_hand       # full contrast set
"""

import argparse
import json
import os

import numpy as np
import torch

from config import CANVAS_LENGTH, DECODER_LAYERS_PATH, DECODER_NORM_PATH
from custom_denoise import run_denoising
from lockin_answers import extract_answer
from model_utils import build_chat_inputs, find_first_token_span, get_module, load_model
from lockin_prompts import PROMPTS

RESULT_PATH = "results/lockin_sweep.jsonl"
OUT_DIR = "data/lens_capture"
WINDOW_PAD = 4
TOPK_S = 5

LOCKED_LABELS = ("possible_locked_wrong",)
CORRECT_LABELS = ("possible_corrector",)


def select_runs(prompt_name: str, max_per_label: int):
    """Locked runs plus an equal number of correctors, from the sweep JSONL."""
    locked, correct = [], []
    with open(RESULT_PATH) as handle:
        for line in handle:
            row = json.loads(line)
            if row["prompt_name"] != prompt_name:
                continue
            if row["trajectory_label"] in LOCKED_LABELS:
                locked.append(row)
            elif row["trajectory_label"] in CORRECT_LABELS:
                correct.append(row)
    locked.sort(key=lambda row: row["seed"])
    correct.sort(key=lambda row: row["seed"])
    locked = locked[:max_per_label]
    correct = correct[: max(len(locked), 1) if max_per_label <= 0 else max_per_label]
    return locked + correct


def attach_layer_hooks(model):
    """Store each decoder layer's output hidden states, per forward call."""
    captures = []  # captures[call_index][layer_index] = (256, d) fp16 cpu
    layers = get_module(model, DECODER_LAYERS_PATH)
    current = {}

    def make_hook(layer_index):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach()
            # Keep only canvas positions if the sequence carries anything else.
            if hidden.shape[1] > CANVAS_LENGTH:
                hidden = hidden[:, -CANVAS_LENGTH:, :]
            current[layer_index] = hidden[0].to("cpu", torch.float16)
            if layer_index == len(layers) - 1:
                captures.append([current[i] for i in range(len(layers))])
                current.clear()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    return captures, handles


def make_s_recorder(store: list):
    """Intervention callback: record self-conditioning top-5, change nothing."""

    def fn(state):
        logits = state["self_conditioning_logits"][0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        top = torch.topk(logprobs, TOPK_S, dim=-1)
        store.append(
            (top.indices.to("cpu", torch.int32), top.values.to("cpu", torch.float16))
        )
        return None

    return fn


def answer_span_and_targets(tokenizer, final_ids, gold_answer):
    """Answer-window positions plus gold/final token id sequences."""
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
    final_value = extract_answer(final_text)
    span = None
    if final_value is not None:
        for needle in (f"{final_value:,}", str(final_value)):
            span = find_first_token_span(tokenizer, final_ids, needle)
            if span is not None:
                break
    if span is None:
        return None
    start, end = span
    window = (max(0, start - WINDOW_PAD), min(len(final_ids), end + WINDOW_PAD))
    gold_ids = None
    for rendering in (f"{gold_answer:,}", str(gold_answer)):
        ids = tokenizer.encode(rendering, add_special_tokens=False)
        if len(ids) == end - start:
            gold_ids = ids
            break
    return {
        "span": (start, end),
        "window": window,
        "final_value": final_value,
        "final_span_ids": [int(i) for i in final_ids[start:end]],
        "gold_span_ids": gold_ids,
    }


@torch.no_grad()
def lens_reduce(model, captures, steps, final_ids, targets, device):
    """Project every (layer, step) hidden through norm + lm_head; reduce."""
    norm = get_module(model, DECODER_NORM_PATH)
    lm_head = model.lm_head
    n_layers = len(get_module(model, DECODER_LAYERS_PATH))
    n_steps = len(steps)

    # Align hook calls to denoising steps from the END (any warm-up decoder
    # forward before the loop shows up as extra leading captures).
    if len(captures) < n_steps:
        raise RuntimeError(f"hook calls ({len(captures)}) < denoising steps ({n_steps})")
    captures = captures[-n_steps:]

    w_start, w_end = targets["window"]
    s_start, s_end = targets["span"]
    span_len = s_end - s_start
    window_len = w_end - w_start
    final_canvas = torch.tensor(final_ids, device=device)
    gold_ids = targets["gold_span_ids"]
    gold_t = torch.tensor(gold_ids, device=device) if gold_ids else None
    final_span_t = torch.tensor(targets["final_span_ids"], device=device)

    agree_step = np.zeros((n_layers, n_steps), dtype=np.float16)
    agree_final = np.zeros((n_layers, n_steps), dtype=np.float16)
    top1_win = np.zeros((n_layers, n_steps, window_len), dtype=np.int32)
    top1_lp_win = np.zeros((n_layers, n_steps, window_len), dtype=np.float16)
    final_lp_span = np.zeros((n_layers, n_steps, span_len), dtype=np.float16)
    gold_lp_span = (
        np.zeros((n_layers, n_steps, span_len), dtype=np.float16) if gold_t is not None else None
    )

    for step_index, per_layer in enumerate(captures):
        step_argmax = torch.tensor(steps[step_index]["argmax_canvas"], device=device)
        for layer_index in range(n_layers):
            hidden = per_layer[layer_index].to(device, torch.bfloat16)
            logits = lm_head(norm(hidden)).float()
            logprobs = torch.log_softmax(logits, dim=-1)
            top1 = logprobs.argmax(dim=-1)
            agree_step[layer_index, step_index] = (top1 == step_argmax).float().mean().item()
            agree_final[layer_index, step_index] = (top1 == final_canvas).float().mean().item()
            win_top1 = top1[w_start:w_end]
            top1_win[layer_index, step_index] = win_top1.cpu().numpy()
            top1_lp_win[layer_index, step_index] = (
                logprobs[torch.arange(w_start, w_end, device=device), win_top1]
                .to(torch.float16).cpu().numpy()
            )
            span_rows = torch.arange(s_start, s_end, device=device)
            final_lp_span[layer_index, step_index] = (
                logprobs[span_rows, final_span_t].to(torch.float16).cpu().numpy()
            )
            if gold_t is not None:
                gold_lp_span[layer_index, step_index] = (
                    logprobs[span_rows, gold_t].to(torch.float16).cpu().numpy()
                )
    arrays = {
        "agree_step": agree_step,
        "agree_final": agree_final,
        "top1_win": top1_win,
        "top1_lp_win": top1_lp_win,
        "final_lp_span": final_lp_span,
    }
    if gold_lp_span is not None:
        arrays["gold_lp_span"] = gold_lp_span
    return arrays


def reduce_s_records(s_records, n_steps):
    s_records = s_records[-n_steps:]
    ids = np.stack([pair[0].numpy() for pair in s_records])      # (S, 256, K)
    lps = np.stack([pair[1].numpy() for pair in s_records])      # (S, 256, K)
    return {"s_top5_ids": ids, "s_top5_lp": lps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", action="append",
                        choices=[prompt.name for prompt in PROMPTS], default=None)
    parser.add_argument("--max-per-label", type=int, default=12)
    parser.add_argument("--smoke", action="store_true",
                        help="capture a single locked run and print sanity info")
    args = parser.parse_args()
    prompt_names = args.prompt or ["two_aces_hand"]
    prompt_by_name = {prompt.name: prompt for prompt in PROMPTS}

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading DiffusionGemma for lens capture...")
    model, processor = load_model()
    tokenizer = processor.tokenizer
    device = model.device

    for prompt_name in prompt_names:
        prompt = prompt_by_name[prompt_name]
        rows = select_runs(prompt_name, args.max_per_label)
        if args.smoke:
            rows = rows[:1]
        print(f"{prompt_name}: capturing {len(rows)} runs "
              f"({sum(r['trajectory_label'] in LOCKED_LABELS for r in rows)} locked)")
        inputs = build_chat_inputs(processor, prompt.prompt, device)

        for row in rows:
            seed = row["seed"]
            captures, handles = attach_layer_hooks(model)
            s_records = []
            try:
                final, steps = run_denoising(
                    model,
                    inputs["input_ids"],
                    inputs.get("attention_mask"),
                    intervention_fn=make_s_recorder(s_records),
                    seed=seed,
                    disable_compile=True,
                )
            finally:
                for handle in handles:
                    handle.remove()

            final_ids = final[0].tolist()
            final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
            replay_answer = extract_answer(final_text)
            if replay_answer != row["final_answer"]:
                print(f"  WARNING {prompt_name} seed={seed}: replay final "
                      f"{replay_answer} != sweep final {row['final_answer']}; skipping")
                continue
            targets = answer_span_and_targets(tokenizer, final_ids, row["gold_answer"])
            if targets is None:
                print(f"  WARNING {prompt_name} seed={seed}: no answer span; skipping")
                continue

            arrays = lens_reduce(model, captures, steps, final_ids, targets, device)
            arrays.update(reduce_s_records(s_records, len(steps)))
            meta = {
                "prompt_name": prompt_name,
                "seed": seed,
                "label": row["trajectory_label"],
                "gold_answer": row["gold_answer"],
                "final_answer": row["final_answer"],
                "n_steps": len(steps),
                "n_hook_calls": len(captures),
                "span": targets["span"],
                "window": targets["window"],
                "gold_span_ids": targets["gold_span_ids"],
                "final_span_ids": targets["final_span_ids"],
                "final_text": final_text,
            }
            stem = os.path.join(OUT_DIR, f"{prompt_name}_seed_{seed:04d}")
            np.savez_compressed(stem + ".npz", **arrays)
            with open(stem + ".json", "w") as handle:
                json.dump(meta, handle)
            gold_aligned = "aligned" if targets["gold_span_ids"] else "UNALIGNED"
            print(f"  {prompt_name} seed={seed}: {row['trajectory_label']} "
                  f"steps={len(steps)} span={targets['span']} gold_tokens={gold_aligned}")

            if args.smoke:
                agree = arrays["agree_step"]
                print("\n--- smoke summary ---")
                print(f"hook calls={meta['n_hook_calls']} steps={meta['n_steps']}")
                print(f"agree_step layer 29 (should be ~1.0 at every step): "
                      f"{np.round(agree[-1].astype(float), 3).tolist()}")
                print(f"agree_step layer 15 mid-run: "
                      f"{np.round(agree[15].astype(float), 3).tolist()}")
                print(f"agree_step layer 0: {np.round(agree[0].astype(float), 3).tolist()}")
                return

    print("done")


if __name__ == "__main__":
    main()
