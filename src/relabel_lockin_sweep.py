"""Rebuild lock-in sweep labels from cached trajectories after extractor fixes.

Torch-free and GPU-free: every draft canvas is already decoded and cached in
``data/lockin_sweep/``, so answers and labels can be recomputed offline. The
cache files are updated in place (final_answer, per-step first_answer,
trajectory_label) and the results JSONL and summary are regenerated from
scratch, replacing the mislabeled originals.

Run from the repo root:
    python src/relabel_lockin_sweep.py
"""

import glob
import json
import os

from lockin_answers import extract_answer, summarize, trajectory_label


DATA_DIR = "data/lockin_sweep"
RESULT_PATH = "results/lockin_sweep.jsonl"
SUMMARY_PATH = "results/lockin_sweep_summary.json"

ROW_FIELDS = (
    "prompt_name",
    "prompt",
    "gold_answer",
    "seed",
    "final_text",
    "final_answer",
    "n_steps",
    "trajectory_label",
    "single_canvas",
)


def relabel_record(record: dict) -> tuple[dict, bool]:
    """Recompute answers/labels in a cached record; report whether they moved."""
    old = (record.get("final_answer"), record.get("trajectory_label"))
    for step in record["steps"]:
        step["first_answer"] = extract_answer(step["text"])
    record["final_answer"] = extract_answer(record["final_text"])
    per_step = [step["first_answer"] for step in record["steps"]]
    record["trajectory_label"] = trajectory_label(
        per_step, record["final_answer"], record["gold_answer"]
    )
    return record, (record["final_answer"], record["trajectory_label"]) != old


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not paths:
        raise SystemExit(f"No cached trajectories found in {DATA_DIR}/")

    rows, changed = [], 0
    for path in paths:
        with open(path) as handle:
            record = json.load(handle)
        record, moved = relabel_record(record)
        if moved:
            changed += 1
            print(
                f"relabeled {record['prompt_name']} seed={record['seed']}: "
                f"final={record['final_answer']} gold={record['gold_answer']} "
                f"{record['trajectory_label']}"
            )
        with open(path, "w") as handle:
            json.dump(record, handle)
        rows.append({key: record[key] for key in ROW_FIELDS})

    with open(RESULT_PATH, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = summarize(rows)
    with open(SUMMARY_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n{changed}/{len(rows)} rows changed")
    print("\nPrompt summary:")
    for name, stats in summary.items():
        print(
            f"{name}: {stats['correct_final']}/{stats['attempts']} correct; "
            f"possible correctors={stats['possible_correctors']}; "
            f"possible locked wrong={stats['possible_locked_wrong']}; "
            f"possible lost correct={stats['possible_lost_correct']}; "
            f"mean steps={stats['mean_steps']:.1f}"
        )
    print(f"\nRewrote {RESULT_PATH} and {SUMMARY_PATH} from {len(rows)} cached trajectories")


if __name__ == "__main__":
    main()
