"""Analyze the patching results: point-of-no-return curve and control stats.

Reads results/patch_lockin.jsonl (all runs, appended across invocations) and
produces, per prompt family:

  * pnr_curve_<prompt>.png -- rescue rate vs the denoising step the patch was
    applied at, one line per region. This is the point-of-no-return figure:
    a cliff means the fate becomes uneditable at a specific step, a gradual
    decay means it consolidates progressively, a flat line means the run can
    be rescued at any time and simply never rescues itself.
  * conditions_<prompt>.png -- flip rate by condition, with the family's own
    natural correct rate drawn as a reference line.
  * printed exact tests for every condition pair that has data.

The natural correct rate comes from results/lockin_sweep.jsonl for the same
prompt and is the number every arm must be read against: a "reroll the dice"
explanation of any intervention predicts exactly that rate, so an arm sitting
at it carries no evidence, and only arms clearly above or below it do.

CPU only.  Run:  python src/analyze_patch.py
"""

import json
import os
from collections import defaultdict
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATCH_PATH = "results/patch_lockin.jsonl"
SWEEP_PATH = "results/lockin_sweep.jsonl"
PLOT_DIR = "results/plots/patch"
SUMMARY_PATH = os.path.join(PLOT_DIR, "patch_summary.json")

CONDITION_ORDER = ("donor", "locked_donor", "shuffle")
REGION_ORDER = ("all", "answer", "not_answer")


def fisher_two_sided(a, b, c, d):
    """Exact test on a 2x2 table by summing tables no likelier than observed."""
    n, row1, col1 = a + b + c + d, a + b, a + c
    def prob(x):
        return comb(col1, x) * comb(n - col1, row1 - x) / comb(n, row1)
    observed = prob(a)
    low = max(0, row1 - (n - col1))
    return sum(prob(x) for x in range(low, min(col1, row1) + 1)
               if prob(x) <= observed * (1 + 1e-9))


def load_rows(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run the pipeline steps in order")
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def natural_rates(sweep_rows):
    stats = defaultdict(lambda: [0, 0])
    for row in sweep_rows:
        stats[row["prompt_name"]][0] += int(row["final_answer"] == row["gold_answer"])
        stats[row["prompt_name"]][1] += 1
    return {name: hits / total for name, (hits, total) in stats.items() if total}


def fired_only(rows):
    """Drop rows whose patch never actually fired.

    make_patch is a no-op when the requested step index is past the end of a
    run or the region resolves to no positions. Those rows carry an unpatched
    outcome, so counting them as failed rescues would silently deflate every
    rate. They are reported separately instead.
    """
    return [row for row in rows if row.get("fired", True)]


def rate(rows):
    rows = fired_only(rows)
    if not rows:
        return None
    return sum(row["flipped_to_gold"] for row in rows) / len(rows)


def plot_pnr(prompt, rows, natural, path):
    steps = sorted({row["step_k"] for row in rows})
    regions = [r for r in REGION_ORDER if any(row["region"] == r for row in rows)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for region in regions:
        ys, xs = [], []
        for step in steps:
            subset = fired_only([row for row in rows
                                 if row["condition"] == "donor" and row["region"] == region
                                 and row["step_k"] == step])
            if subset:
                xs.append(step)
                ys.append(rate(subset))
        if xs:
            ax.plot(xs, ys, marker="o", label=f"donor / {region}")
    if natural is not None:
        ax.axhline(natural, linestyle="--", linewidth=1, color="grey",
                   label=f"natural correct rate ({natural:.0%})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("denoising step the patch was applied at")
    ax.set_ylabel("fraction of locked runs rescued")
    ax.set_title(f"Point of no return: {prompt}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_conditions(prompt, rows, natural, path):
    labels, values, counts = [], [], []
    for condition in CONDITION_ORDER:
        for region in REGION_ORDER:
            subset = fired_only([row for row in rows
                                 if row["condition"] == condition and row["region"] == region])
            if subset:
                labels.append(f"{condition}\n{region}")
                values.append(rate(subset))
                counts.append(len(subset))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.5))
    bars = ax.bar(range(len(labels)), values,
                  color=["#2a6f6a" if label.startswith("donor") else "#9a9a9a"
                         for label in labels])
    for index, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(index, bar.get_height() + 0.02, f"n={count}",
                ha="center", fontsize=7)
    if natural is not None:
        ax.axhline(natural, linestyle="--", linewidth=1, color="black",
                   label=f"natural correct rate ({natural:.0%})")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of locked runs rescued")
    ax.set_title(f"Rescue rate by condition: {prompt}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    patch_rows = load_rows(PATCH_PATH)
    natural = natural_rates(load_rows(SWEEP_PATH))
    summary = {}

    by_prompt = defaultdict(list)
    for row in patch_rows:
        by_prompt[row["prompt_name"]].append(row)

    for prompt, rows in sorted(by_prompt.items()):
        base = natural.get(prompt)
        print(f"\n=== {prompt} ===")
        print(f"natural correct rate (fresh seed): "
              f"{base:.1%}" if base is not None else "natural rate unavailable")
        print("unpatched locked recipients are 0% by construction "
              "(noop replay is verified before every patch)")

        skipped = len(rows) - len(fired_only(rows))
        if skipped:
            print(f"NOTE: {skipped} rows whose patch never fired are excluded "
                  f"(unpatched outcome; counting them would deflate every rate)")
        rows = fired_only(rows)

        entry = {"natural_rate": base, "conditions": {}, "n_never_fired": skipped,
                 "by_step": {}}
        for condition in CONDITION_ORDER:
            subset = [row for row in rows if row["condition"] == condition]
            if not subset:
                continue
            hits = sum(row["flipped_to_gold"] for row in subset)
            print(f"\n{condition}: {hits}/{len(subset)} ({hits / len(subset):.0%})")
            entry["conditions"][condition] = {"hits": hits, "n": len(subset)}
            deltas = [row["relative_delta"] for row in subset
                      if row.get("relative_delta") is not None]
            if deltas:
                mean_delta = sum(deltas) / len(deltas)
                print(f"    mean relative change to S: {mean_delta:.3f}")
                entry["conditions"][condition]["mean_relative_delta"] = mean_delta
            for region in REGION_ORDER:
                region_rows = [row for row in subset if row["region"] == region]
                if region_rows:
                    region_hits = sum(row["flipped_to_gold"] for row in region_rows)
                    print(f"    {region}: {region_hits}/{len(region_rows)}")
                    entry["conditions"][condition][region] = {
                        "hits": region_hits, "n": len(region_rows)}

        # Exact tests between every pair of conditions that has data.
        print()
        for i, first in enumerate(CONDITION_ORDER):
            for second in CONDITION_ORDER[i + 1:]:
                one = [row for row in rows if row["condition"] == first]
                two = [row for row in rows if row["condition"] == second]
                if not one or not two:
                    continue
                a, b = sum(r["flipped_to_gold"] for r in one), 0
                b = len(one) - a
                c = sum(r["flipped_to_gold"] for r in two)
                d = len(two) - c
                p = fisher_two_sided(a, b, c, d)
                print(f"{first} {a}/{a + b} vs {second} {c}/{c + d}: p = {p:.3e}")
                entry.setdefault("tests", {})[f"{first}_vs_{second}"] = p
                if base is not None:
                    for name, hits, total in ((first, a, a + b), (second, c, c + d)):
                        if abs(hits / total - base) < 0.05:
                            print(f"    note: {name} sits at the natural rate "
                                  f"-- this arm has no discriminating power here")

        for step in sorted({row["step_k"] for row in rows}):
            donor_rows = [row for row in rows
                          if row["condition"] == "donor" and row["step_k"] == step]
            if donor_rows:
                entry["by_step"][step] = {
                    "hits": sum(r["flipped_to_gold"] for r in donor_rows),
                    "n": len(donor_rows),
                }

        plot_pnr(prompt, rows, base, os.path.join(PLOT_DIR, f"pnr_curve_{prompt}.png"))
        plot_conditions(prompt, rows, base,
                        os.path.join(PLOT_DIR, f"conditions_{prompt}.png"))
        summary[prompt] = entry

    with open(SUMMARY_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote plots to {PLOT_DIR}/ and summary to {SUMMARY_PATH}")
    print("\nCheck the mean relative change per condition before reading the "
          "rates: if locked_donor edits S much less than donor does, part of "
          "its lower rescue rate is edit size rather than edit content.")


if __name__ == "__main__":
    main()
