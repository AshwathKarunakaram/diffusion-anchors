"""Can lock-in be predicted from internal state before the answer exists?

If the fate is already carried at denoising step 1, a linear probe on the
self-conditioning hidden state at that step should separate runs that will
correct themselves from runs that will lock in. That turns the mechanistic
finding into something usable: detect a doomed run early and resample it
instead of spending the remaining steps on it.

Two feature sets, both taken at one early step:
  * "canvas"  -- mean over ALL 256 positions. Uses no knowledge of where the
                 answer will end up, so it is the honest deployable version.
  * "window"  -- mean over a fixed early position window (see FIXED_WINDOW).
                 Still no peeking at the outcome; just a prior about where
                 answer-first prompts put their answer.

Evaluation avoids the two ways this could look better than it is:
  * leave-one-run-out cross-validation within a prompt (no run scores itself);
  * cross-prompt transfer -- fit on one family, test on another, which is the
    only test of whether "doomed-ness" is represented in a shared way rather
    than memorised per prompt.

Capture needs the GPU; training and evaluation are CPU and reuse the cache,
so re-running with different settings after the first pass is free.

Run:  python src/doom_detector.py --prompt two_aces_hand \\
                                  --prompt increasing_three_digits
"""

import argparse
import json
import os

import numpy as np

CACHE_DIR = "data/doom_detector"
OUT_PATH = "results/doom_detector.json"
FIXED_WINDOW = (0, 24)  # answer-first prompts place the answer early


def capture(prompt_names, step_k, max_per_label):
    """Replay runs with a self-conditioning hook; cache pooled features."""
    import torch  # imported here so --train-only needs no GPU stack

    from capture_lens import select_runs, LOCKED_LABELS
    from custom_denoise import run_denoising
    from lockin_answers import extract_answer
    from lockin_prompts import PROMPTS
    from model_utils import build_chat_inputs, load_model
    from steer_lockin import attach_capture

    prompt_by_name = {prompt.name: prompt for prompt in PROMPTS}
    print("Loading DiffusionGemma for doom-detector capture...")
    model, processor = load_model()
    tokenizer = processor.tokenizer

    for prompt_name in prompt_names:
        path = os.path.join(CACHE_DIR, f"{prompt_name}_step{step_k}.npz")
        if os.path.exists(path):
            print(f"{prompt_name}: cache exists, skipping capture")
            continue
        prompt = prompt_by_name[prompt_name]
        rows = select_runs(prompt_name, max_per_label)
        inputs = build_chat_inputs(processor, prompt.prompt, model.device)
        canvas_features, window_features, labels, seeds = [], [], [], []
        print(f"{prompt_name}: capturing {len(rows)} runs at step {step_k}")

        for row in rows:
            store = []
            handle = attach_capture(model, store)
            try:
                final, steps = run_denoising(
                    model, inputs["input_ids"], inputs.get("attention_mask"),
                    seed=row["seed"], disable_compile=True,
                )
            finally:
                handle.remove()
            answer = extract_answer(tokenizer.decode(final[0].tolist(),
                                                     skip_special_tokens=True))
            if answer != row["final_answer"]:
                print(f"  seed={row['seed']}: replay mismatch, excluded")
                continue
            if len(store) <= step_k:
                print(f"  seed={row['seed']}: only {len(store)} sc calls, excluded")
                continue
            hidden = store[step_k].numpy()  # (256, d)
            canvas_features.append(hidden.mean(axis=0))
            window_features.append(hidden[FIXED_WINDOW[0]:FIXED_WINDOW[1]].mean(axis=0))
            labels.append(int(row["trajectory_label"] in LOCKED_LABELS))
            seeds.append(row["seed"])

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(
            path,
            canvas=np.stack(canvas_features), window=np.stack(window_features),
            labels=np.array(labels), seeds=np.array(seeds),
        )
        print(f"  cached {len(labels)} runs ({sum(labels)} locked) -> {path}")


def load_cache(prompt_name, step_k):
    path = os.path.join(CACHE_DIR, f"{prompt_name}_step{step_k}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run without --train-only first")
    return dict(np.load(path))


def auc(scores, labels):
    """Rank-based AUC, ties averaged. 0.5 = chance."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tied score groups
    for value in np.unique(scores):
        mask = scores == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    positives, negatives = labels == 1, labels == 0
    n_pos, n_neg = positives.sum(), negatives.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_logistic(features, labels, l2=1.0, steps=400, lr=0.5):
    """Small L2 logistic regression by gradient descent on standardised inputs.

    Written out rather than pulled from sklearn so the exact regularisation
    and standardisation are visible: with ~40 runs and 2816 dimensions, those
    two choices are what stand between this and memorisation.
    """
    mean, std = features.mean(axis=0), features.std(axis=0) + 1e-6
    x = (features - mean) / std
    y = labels.astype(float)
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(steps):
        logits = x @ weights + bias
        preds = 1 / (1 + np.exp(-logits))
        error = preds - y
        grad_w = x.T @ error / len(y) + l2 * weights / len(y)
        grad_b = error.mean()
        weights -= lr * grad_w
        bias -= lr * grad_b
    return {"weights": weights, "bias": bias, "mean": mean, "std": std}


def score(model, features):
    x = (features - model["mean"]) / model["std"]
    return x @ model["weights"] + model["bias"]


def leave_one_out(features, labels, l2):
    scores = np.zeros(len(labels))
    for index in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        mask[index] = False
        if len(np.unique(labels[mask])) < 2:
            scores[index] = 0.0
            continue
        model = fit_logistic(features[mask], labels[mask], l2=l2)
        scores[index] = score(model, features[index:index + 1])[0]
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-per-label", type=int, default=20)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--train-only", action="store_true",
                        help="skip GPU capture and reuse the cache")
    args = parser.parse_args()
    prompt_names = args.prompt or ["two_aces_hand", "increasing_three_digits"]

    if not args.train_only:
        capture(prompt_names, args.step, args.max_per_label)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    data = {name: load_cache(name, args.step) for name in prompt_names}
    report = {"step": args.step, "l2": args.l2, "within_prompt": {}, "cross_prompt": {}}

    print(f"\n=== within-prompt, leave-one-run-out (step {args.step}) ===")
    for name, cache in data.items():
        labels = cache["labels"]
        print(f"\n{name}: {len(labels)} runs, {labels.sum()} locked")
        if len(np.unique(labels)) < 2:
            print("  only one class present; skipping")
            continue
        report["within_prompt"][name] = {}
        for feature_set in ("canvas", "window"):
            scores = leave_one_out(cache[feature_set], labels, args.l2)
            value = auc(scores, labels)
            print(f"  {feature_set:7s} AUC = {value:.3f}"
                  f"{'  (chance = 0.500)' if value is not None else ''}")
            report["within_prompt"][name][feature_set] = value

    if len(prompt_names) >= 2:
        print("\n=== cross-prompt transfer (fit on one family, test on another) ===")
        for train_name in prompt_names:
            for test_name in prompt_names:
                if train_name == test_name:
                    continue
                train, test = data[train_name], data[test_name]
                if len(np.unique(train["labels"])) < 2 or len(np.unique(test["labels"])) < 2:
                    continue
                key = f"{train_name}->{test_name}"
                report["cross_prompt"][key] = {}
                for feature_set in ("canvas", "window"):
                    model = fit_logistic(train[feature_set], train["labels"], l2=args.l2)
                    value = auc(score(model, test[feature_set]), test["labels"])
                    print(f"  {key} [{feature_set}] AUC = {value:.3f}")
                    report["cross_prompt"][key][feature_set] = value

    with open(OUT_PATH, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {OUT_PATH}")
    print("AUC near 0.5 = fate not linearly readable at this step; near 1.0 = "
          "readable before the answer exists. Cross-prompt AUC is the one that "
          "shows a shared representation rather than per-prompt memorisation.")


if __name__ == "__main__":
    main()
