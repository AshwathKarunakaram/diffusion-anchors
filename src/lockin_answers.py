"""Answer extraction and trajectory labels for the lock-in sweep.

Torch-free so relabeling cached trajectories needs no GPU runtime.

The sweep's prompts demand the answer before the reasoning, so only the
region before the reasoning marker counts as the answer. Round 1 read
integers from anywhere in the canvas, which mislabeled runs two ways:
"There are 4 aces in a standard deck" matched a there-are pattern (final=4),
and restated questions fed the first-integer fallback (final=1 for
"from 1 through 999" while the bolded answer was the correct 499,500).
"""

import re

REASONING_MARKER = re.compile(r"#+\s*Reasoning|\*\*\s*Reasoning|\bReasoning\s*:", re.I)
ANSWER_IS = re.compile(r"answer\s+is\s*[:=]?\s*\**\s*(-?\d[\d,]*)", re.I)
BOLD_SPAN = re.compile(r"\*\*(.+?)\*\*", re.S)
INT = re.compile(r"-?\d[\d,]*")


def _to_int(text: str) -> int:
    return int(text.replace(",", ""))


def extract_answer(text: str):
    """Read the answer-first integer, or None when no integer is visible.

    Preference order inside the pre-reasoning segment:
      1. the last integer inside a **bold** span (the model bolds its answer,
         and in "**4 x 194,580 = 778,320**" the answer is the last integer);
      2. an "answer is N" phrase (covers unbalanced-asterisk drafts);
      3. the last plain integer (answer-first sentences end with the value).
    """
    text = text.replace("thought\n", "", 1).strip()
    marker = REASONING_MARKER.search(text)
    segment = text[: marker.start()] if marker else text

    bold_ints = [
        value
        for span in BOLD_SPAN.finditer(segment)
        for value in INT.findall(span.group(1))
    ]
    if bold_ints:
        return _to_int(bold_ints[-1])
    match = ANSWER_IS.search(segment)
    if match:
        return _to_int(match.group(1))
    ints = INT.findall(segment)
    return _to_int(ints[-1]) if ints else None


def trajectory_label(per_step_answers, final_answer, gold):
    """A triage label, not a proof of the model's reasoning algorithm."""
    visible = [answer for answer in per_step_answers if answer is not None]
    if final_answer == gold:
        if any(answer != gold for answer in visible[:-1]):
            return "possible_corrector"
        return "always_or_early_correct"
    if final_answer is None:
        return "no_final_integer"
    # A wrong answer observed at least twice is more likely to be a genuine
    # stable draft than a single noisy early canvas.
    if visible.count(final_answer) >= 2:
        return "possible_locked_wrong"
    return "wrong_unstable_or_other"


def summarize(rows):
    summary = {}
    for row in rows:
        bucket = summary.setdefault(
            row["prompt_name"],
            {
                "gold_answer": row["gold_answer"],
                "attempts": 0,
                "correct_final": 0,
                "possible_correctors": 0,
                "possible_locked_wrong": 0,
                "labels": {},
                "step_counts": [],
            },
        )
        bucket["attempts"] += 1
        bucket["correct_final"] += int(row["final_answer"] == row["gold_answer"])
        bucket["possible_correctors"] += int(row["trajectory_label"] == "possible_corrector")
        bucket["possible_locked_wrong"] += int(row["trajectory_label"] == "possible_locked_wrong")
        label = row["trajectory_label"]
        bucket["labels"][label] = bucket["labels"].get(label, 0) + 1
        bucket["step_counts"].append(row["n_steps"])
    for bucket in summary.values():
        bucket["correct_final_rate"] = bucket["correct_final"] / bucket["attempts"]
        bucket["mean_steps"] = sum(bucket["step_counts"]) / bucket["attempts"]
    return summary
