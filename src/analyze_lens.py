"""Analyze lens captures: where the logit lens works, and what it says about
lock-in.

Reads data/lens_capture/*.npz (+ .json sidecars) produced by capture_lens.py.
Loads only the tokenizer (AutoProcessor), never the model — runs on CPU.

Outputs (results/plots/lens/):
  * heatmap_agree_step.png  -- mean lens/top-1 agreement with the model's own
    step output, layer x step-fraction (the paper's open question, mapped);
  * heatmap_agree_final.png -- same against the final canvas;
  * gold_readability.png    -- mean gold-token logprob at the answer span,
    best layer per step, correcting vs locked runs;
  * gold_vs_final_margin.png-- (gold - final-token) logprob margin at the
    span for locked runs only: was the right answer ever winning inside?
  * lens_events.json        -- per run: earliest (step fraction, layer) where
    the decoded answer window contains the gold string.

Run:  python src/analyze_lens.py
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor

from config import MODEL_ID

CAPTURE_DIR = "data/lens_capture"
PLOT_DIR = "results/plots/lens"
EVENTS_PATH = os.path.join(PLOT_DIR, "lens_events.json")
N_STEP_BINS = 10

LOCKED = "possible_locked_wrong"
CORRECT = "possible_corrector"


def load_runs():
    runs = []
    for npz_path in sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.npz"))):
        meta_path = npz_path[:-4] + ".json"
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as handle:
            meta = json.load(handle)
        runs.append({"meta": meta, "arrays": dict(np.load(npz_path))})
    if not runs:
        raise SystemExit(f"No captures found in {CAPTURE_DIR}/")
    return runs


def step_fractions(n_steps):
    return (np.arange(n_steps) + 0.5) / n_steps


def binned_heatmap(runs, key):
    """Mean over runs of arrays[key], binned to a common step-fraction axis."""
    n_layers = runs[0]["arrays"][key].shape[0]
    total = np.zeros((n_layers, N_STEP_BINS))
    count = np.zeros((n_layers, N_STEP_BINS))
    for run in runs:
        grid = run["arrays"][key].astype(np.float64)  # (L, S)
        bins = np.minimum((step_fractions(grid.shape[1]) * N_STEP_BINS).astype(int),
                          N_STEP_BINS - 1)
        for step_index, bin_index in enumerate(bins):
            total[:, bin_index] += grid[:, step_index]
            count[:, bin_index] += 1
    return total / np.maximum(count, 1)


def plot_heatmap(grid, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=1,
                      extent=(0, 1, -0.5, grid.shape[0] - 0.5), cmap="viridis")
    ax.set_xlabel("denoising progress (step fraction)")
    ax.set_ylabel("decoder layer")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="top-1 agreement")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gold_curves(runs):
    """Best-layer mean gold logprob at the span, binned, per label group."""
    groups = {}
    for run in runs:
        arrays, meta = run["arrays"], run["meta"]
        if "gold_lp_span" not in arrays:
            continue
        per_step = arrays["gold_lp_span"].astype(np.float64).mean(axis=2).max(axis=0)  # (S,)
        bins = np.minimum((step_fractions(len(per_step)) * N_STEP_BINS).astype(int),
                          N_STEP_BINS - 1)
        bucket = groups.setdefault(meta["label"], [np.zeros(N_STEP_BINS), np.zeros(N_STEP_BINS)])
        for step_index, bin_index in enumerate(bins):
            bucket[0][bin_index] += per_step[step_index]
            bucket[1][bin_index] += 1
    return {
        label: total / np.maximum(count, 1)
        for label, (total, count) in groups.items()
    }


def locked_margin(runs):
    """Gold-minus-final logprob margin (best layer) for locked runs."""
    curves = []
    for run in runs:
        arrays, meta = run["arrays"], run["meta"]
        if meta["label"] != LOCKED or "gold_lp_span" not in arrays:
            continue
        margin = (arrays["gold_lp_span"].astype(np.float64)
                  - arrays["final_lp_span"].astype(np.float64)).mean(axis=2).max(axis=0)
        curves.append((meta, margin))
    return curves


def gold_visible_events(runs, tokenizer):
    """Earliest (step_fraction, layer) whose decoded window contains gold."""
    events = []
    for run in runs:
        arrays, meta = run["arrays"], run["meta"]
        gold = meta["gold_answer"]
        needles = (f"{gold:,}", str(gold))
        top1 = arrays["top1_win"]  # (L, S, W)
        found = None
        for step_index in range(top1.shape[1]):
            for layer_index in range(top1.shape[0]):
                text = tokenizer.decode(top1[layer_index, step_index].tolist())
                if any(needle in text for needle in needles):
                    found = {
                        "step_index": step_index,
                        "step_fraction": float((step_index + 0.5) / top1.shape[1]),
                        "layer": int(layer_index),
                        "decoded": text,
                    }
                    break
            if found:
                break
        events.append({
            "prompt_name": meta["prompt_name"],
            "seed": meta["seed"],
            "label": meta["label"],
            "final_answer": meta["final_answer"],
            "gold_answer": gold,
            "n_steps": meta["n_steps"],
            "gold_first_visible": found,
        })
    return events


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    runs = load_runs()
    print(f"loaded {len(runs)} captures "
          f"({sum(r['meta']['label'] == LOCKED for r in runs)} locked, "
          f"{sum(r['meta']['label'] == CORRECT for r in runs)} correcting)")

    plot_heatmap(binned_heatmap(runs, "agree_step"),
                 "Logit lens vs model's own step output",
                 os.path.join(PLOT_DIR, "heatmap_agree_step.png"))
    plot_heatmap(binned_heatmap(runs, "agree_final"),
                 "Logit lens vs final canvas",
                 os.path.join(PLOT_DIR, "heatmap_agree_final.png"))

    curves = gold_curves(runs)
    if curves:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = (np.arange(N_STEP_BINS) + 0.5) / N_STEP_BINS
        for label, curve in sorted(curves.items()):
            ax.plot(x, curve, marker="o", label=label)
        ax.set_xlabel("denoising progress (step fraction)")
        ax.set_ylabel("gold-token logprob at answer span (best layer)")
        ax.set_title("Internal readability of the correct answer")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "gold_readability.png"), dpi=160)
        plt.close(fig)

    margins = locked_margin(runs)
    if margins:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for meta, margin in margins:
            ax.plot(step_fractions(len(margin)), margin, alpha=0.7,
                    label=f"{meta['prompt_name']} s{meta['seed']}")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("denoising progress (step fraction)")
        ax.set_ylabel("logprob margin: gold - locked token (best layer)")
        ax.set_title("Locked runs: was the right answer winning internally?")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "gold_vs_final_margin.png"), dpi=160)
        plt.close(fig)

    tokenizer = AutoProcessor.from_pretrained(MODEL_ID).tokenizer
    events = gold_visible_events(runs, tokenizer)
    with open(EVENTS_PATH, "w") as handle:
        json.dump(events, handle, indent=2)

    print("\ngold-first-visible (internal), by run:")
    for event in events:
        visible = event["gold_first_visible"]
        where = (f"step {visible['step_index']} ({visible['step_fraction']:.2f}) "
                 f"layer {visible['layer']}") if visible else "never"
        print(f"  {event['prompt_name']} seed={event['seed']} [{event['label']}]: {where}")
    print(f"\nwrote plots to {PLOT_DIR}/ and events to {EVENTS_PATH}")


if __name__ == "__main__":
    main()
