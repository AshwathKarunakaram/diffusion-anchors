"""Single-canvas feasibility pilot for endogenous code-consistency repair.

The model receives ordinary coding prompts, chooses its own local identifiers,
and is then replayed with only the *definition* of one repeated identifier
renamed.  The prompt does not require that identifier, so either restoring the
old name or consistently propagating the new name can produce correct code.

This pilot is deliberately small.  It asks whether a semantic inconsistency
ever yields a nontrivial repair trajectory before investing in internal
activation patching.

Run on an A100 after the normal DiffusionGemma setup:
    python src/code_repair_pilot.py --n-seeds 5

All outputs are single-canvas: ``run_denoising`` raises before a second canvas
could be created.
"""

import argparse
import ast
import json
import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from typing import Callable

import torch

from config import CANVAS_LENGTH
from custom_denoise import run_denoising
from model_utils import build_chat_inputs, load_model


DATA_DIR = "data/code_repair_pilot"
RESULT_PATH = "results/code_repair_pilot.jsonl"


@dataclass(frozen=True)
class CodeTask:
    name: str
    prompt: str
    expected_name: str
    tests: tuple[tuple[tuple, object], ...]


TASKS = (
    CodeTask(
        name="positive_sum",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "positive_sum(xs), returning the sum of the positive integers in xs."
        ),
        expected_name="positive_sum",
        tests=((([],), 0), (([1, -2, 3],), 4), (([-1, -4],), 0)),
    ),
    CodeTask(
        name="count_evens",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "count_evens(xs), returning the number of even integers in xs."
        ),
        expected_name="count_evens",
        tests=((([],), 0), (([1, 2, 4, 5],), 2), (([-2, 0, 3],), 2)),
    ),
    CodeTask(
        name="clamp_all",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "clamp_all(xs, low, high), returning a new list where every value "
            "in xs is clamped to the inclusive interval [low, high]."
        ),
        expected_name="clamp_all",
        tests=((([], 0, 5), []), (([-2, 3, 10], 0, 5), [0, 3, 5]), (([4], 0, 5), [4])),
    ),
    CodeTask(
        name="is_balanced",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "is_balanced(s), returning True exactly when the parentheses, "
            "square brackets, and curly braces in s are correctly balanced "
            "and nested. Ignore all other characters."
        ),
        expected_name="is_balanced",
        tests=((("",), True), (("([]){}",), True), (("([)]",), False), (("(a[b]{c})",), True)),
    ),
    CodeTask(
        name="longest_increasing_run",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "longest_increasing_run(xs), returning the length of the longest "
            "contiguous strictly increasing run in xs. Return 0 for an empty list."
        ),
        expected_name="longest_increasing_run",
        tests=((([],), 0), (([1, 2, 1, 2, 3],), 3), (([3, 2, 1],), 1), (([1, 2, 3],), 3)),
    ),
    CodeTask(
        name="eval_rpn",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "eval_rpn(tokens), which evaluates an expression in reverse Polish "
            "notation. Tokens are integer strings or one of +, -, *, /. Division "
            "truncates toward zero."
        ),
        expected_name="eval_rpn",
        tests=(
            ((["2", "1", "+", "3", "*"],), 9),
            ((["4", "13", "5", "/", "+"],), 6),
            ((["3", "-4", "+"],), -1),
        ),
    ),
    CodeTask(
        name="merge_intervals",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "merge_intervals(intervals), returning merged inclusive integer "
            "intervals sorted by start. Each input interval is [start, end]; "
            "intervals that overlap or touch must be merged."
        ),
        expected_name="merge_intervals",
        tests=(
            (([],), []),
            (([[1, 3], [2, 4], [6, 7], [7, 8]],), [[1, 4], [6, 8]]),
            (([[5, 6], [1, 2]],), [[1, 2], [5, 6]]),
        ),
    ),
    CodeTask(
        name="longest_unique_substring",
        prompt=(
            "Write only Python code, with no Markdown fences. Define "
            "longest_unique_substring(s), returning the length of the longest "
            "substring of s containing no repeated characters."
        ),
        expected_name="longest_unique_substring",
        tests=((("",), 0), (("abcabcbb",), 3), (("bbbbb",), 1), (("pwwkew",), 3)),
    ),
)

# Names are intentionally semantically neutral.  A valid rename should not
# change program behavior, only identifier consistency.
RENAME_CANDIDATES = ("total", "count", "result", "value", "acc", "out")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def extract_python(text: str) -> str:
    """Remove generation wrappers without repairing model-authored code.

    DiffusionGemma's chat decoding can expose a bare ``thought`` channel marker
    immediately before otherwise ordinary code.  It is not part of the
    assistant's Python response, and executing it causes a spurious
    ``NameError``.  Our prompts require a top-level function, so discard only
    material before the first top-level ``def`` after handling Markdown fences.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    function_start = re.search(r"(?m)^def\s+", text)
    if function_start is not None:
        text = text[function_start.start():]
    return text.strip()


def parse_python(text: str):
    try:
        return ast.parse(extract_python(text))
    except SyntaxError:
        return None


def repeated_local_bindings(text: str) -> list[str]:
    """Names assigned in executable code and used at least twice.

    This deliberately avoids inferring a full scope graph in the pilot.  The
    prompts have one short function, and the result is only used to select a
    candidate trajectory for the later, stricter analyses.
    """
    tree = parse_python(text)
    if tree is None:
        return []
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    counts = {name: len(re.findall(rf"\b{re.escape(name)}\b", text)) for name in bound}
    return sorted(name for name, count in counts.items() if count >= 3)


def nth_token_span(tokenizer, token_ids: list[int], needle: str, occurrence: int):
    """Return the token span for a numbered textual occurrence in a canvas."""
    prefixes = [""]
    for j in range(1, len(token_ids) + 1):
        prefixes.append(tokenizer.decode(token_ids[:j]))
    matches = list(re.finditer(rf"\b{re.escape(needle)}\b", prefixes[-1]))
    if occurrence >= len(matches):
        return None
    match = matches[occurrence]
    span = [
        j
        for j in range(len(token_ids))
        if len(prefixes[j]) < match.end() and len(prefixes[j + 1]) > match.start()
    ]
    return (min(span), max(span) + 1) if span else None


def token_matched_rename(tokenizer, old: str):
    old_ids = tokenizer.encode(old, add_special_tokens=False)
    for new in RENAME_CANDIDATES:
        if new == old:
            continue
        new_ids = tokenizer.encode(new, add_special_tokens=False)
        if len(new_ids) == len(old_ids):
            return new, old_ids, new_ids
    return None


def choose_injection(tokenizer, steps):
    """Find an early executable draft with a repeated, token-length-matched name."""
    for k, step in enumerate(steps[:-2]):
        text = tokenizer.decode(step["argmax_canvas"], skip_special_tokens=True)
        for name in repeated_local_bindings(text):
            rename = token_matched_rename(tokenizer, name)
            span = nth_token_span(tokenizer, step["argmax_canvas"], name, 0)
            if rename is not None and span is not None:
                new, old_ids, new_ids = rename
                if span[1] - span[0] == len(old_ids):
                    return {
                        "step_index": k,
                        "old_name": name,
                        "new_name": new,
                        "span": span,
                        "old_ids": old_ids,
                        "new_ids": new_ids,
                        "clean_text": text,
                    }
    return None


def make_rename_intervention(injection, mode: str):
    """Modify only the definition-site state after its recorded denoising step."""
    seen = {"step": -1}

    def intervention(state):
        seen["step"] += 1
        if seen["step"] != injection["step_index"]:
            return None
        start, end = injection["span"]
        old_ids, new_ids = injection["old_ids"], injection["new_ids"]
        overrides = {}

        if mode in {"canvas", "both"}:
            canvas = state["current_canvas"].clone()
            canvas[0, start:end] = torch.tensor(new_ids, device=canvas.device, dtype=canvas.dtype)
            overrides["current_canvas"] = canvas

        if mode in {"self_conditioning", "both"}:
            logits = state["self_conditioning_logits"].clone()
            # Exchange old/new logits: this preserves the row's logit multiset
            # rather than injecting an arbitrary high-norm logit vector.
            for pos, old_id, new_id in zip(range(start, end), old_ids, new_ids):
                old_logit = logits[0, pos, old_id].clone()
                logits[0, pos, old_id] = logits[0, pos, new_id]
                logits[0, pos, new_id] = old_logit
            overrides["self_conditioning_logits"] = logits
        return overrides

    return intervention


def _safe_exec_child(code: str, function_name: str, tests, queue):
    """Execute narrow generated code in an isolated child with restricted builtins."""
    try:
        tree = parse_python(code)
        if tree is None:
            queue.put({"passed": False, "reason": "syntax_error"})
            return
        forbidden = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.ClassDef)
        if any(isinstance(node, forbidden) for node in ast.walk(tree)):
            queue.put({"passed": False, "reason": "forbidden_syntax"})
            return
        if any(
            isinstance(node, ast.Name) and node.id.startswith("__")
            or isinstance(node, ast.Attribute) and node.attr.startswith("__")
            for node in ast.walk(tree)
        ):
            queue.put({"passed": False, "reason": "dunder"})
            return
        safe_builtins = {
            "abs": abs, "bool": bool, "enumerate": enumerate, "float": float,
            "dict": dict, "int": int, "len": len, "list": list, "max": max,
            "min": min, "range": range, "set": set, "sorted": sorted, "sum": sum,
        }
        namespace = {"__builtins__": safe_builtins}
        exec(compile(tree, "<generated>", "exec"), namespace, namespace)
        fn = namespace.get(function_name)
        if not callable(fn):
            queue.put({"passed": False, "reason": "missing_function"})
            return
        for args, expected in tests:
            if fn(*args) != expected:
                queue.put({"passed": False, "reason": "wrong_output"})
                return
        queue.put({"passed": True, "reason": "passed"})
    except BaseException as exc:
        queue.put({"passed": False, "reason": type(exc).__name__})


def run_tests(text: str, task: CodeTask) -> dict:
    queue = mp.Queue()
    proc = mp.Process(
        target=_safe_exec_child,
        args=(extract_python(text), task.expected_name, task.tests, queue),
    )
    proc.start()
    proc.join(timeout=2)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"passed": False, "reason": "timeout"}
    return queue.get() if not queue.empty() else {"passed": False, "reason": "no_result"}


def classify(text: str, injection: dict, task: CodeTask) -> dict:
    code = extract_python(text)
    old_count = len(re.findall(rf"\b{re.escape(injection['old_name'])}\b", code))
    new_count = len(re.findall(rf"\b{re.escape(injection['new_name'])}\b", code))
    syntax_valid = parse_python(code) is not None
    test = run_tests(code, task)
    if old_count and not new_count:
        label = "local_restoration"
    elif new_count >= 2 and not old_count and syntax_valid:
        label = "coherent_propagation"
    elif syntax_valid and test["passed"]:
        label = "global_rewrite"
    elif old_count and new_count:
        label = "mixed_inconsistent"
    else:
        label = "derailed"
    return {
        "outcome": label,
        "old_name_count": old_count,
        "new_name_count": new_count,
        "syntax_valid": syntax_valid,
        "tests": test,
    }


def run_one(model, tokenizer, inputs, seed: int, intervention_fn=None):
    final, steps = run_denoising(
        model,
        inputs["input_ids"],
        inputs.get("attention_mask"),
        intervention_fn=intervention_fn,
        seed=seed,
        disable_compile=True,
    )
    text = tokenizer.decode(final[0].tolist(), skip_special_tokens=True)
    return text, steps


def serializable_steps(steps):
    return [
        {
            "step_index": i,
            "cur_step": step["cur_step"],
            "argmax_canvas": step["argmax_canvas"],
            "accepted_mask": step["accepted_mask"],
        }
        for i, step in enumerate(steps)
    ]


def trace_identifier(tokenizer, steps, old_name: str, new_name: str):
    """Save readable per-step evidence for whether a rename spreads over time."""
    trace = []
    for index, step in enumerate(steps):
        text = tokenizer.decode(step["argmax_canvas"], skip_special_tokens=True)
        code = extract_python(text)
        trace.append(
            {
                "step_index": index,
                "cur_step": step["cur_step"],
                "old_name_count": len(re.findall(rf"\b{re.escape(old_name)}\b", code)),
                "new_name_count": len(re.findall(rf"\b{re.escape(new_name)}\b", code)),
                "code": code,
            }
        )
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--task", choices=[task.name for task in TASKS], default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the JSONL summary instead of appending a second pilot run",
    )
    args = parser.parse_args()
    if CANVAS_LENGTH != 256:
        raise RuntimeError("This pilot is validated only for the 256-token single-canvas setup.")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    if args.overwrite and os.path.exists(RESULT_PATH):
        os.remove(RESULT_PATH)
    tasks = [task for task in TASKS if args.task in (None, task.name)]

    print("Loading DiffusionGemma for single-canvas code-repair pilot...")
    model, processor = load_model()
    tokenizer = processor.tokenizer
    rows = []

    for task in tasks:
        inputs = build_chat_inputs(processor, task.prompt, model.device)
        for seed in range(args.start_seed, args.start_seed + args.n_seeds):
            clean_text, clean_steps = run_one(model, tokenizer, inputs, seed)
            injection = choose_injection(tokenizer, clean_steps)
            record = {
                "task": task.name,
                "seed": seed,
                "prompt": task.prompt,
                "clean_text": clean_text,
                "clean_tests": run_tests(clean_text, task),
                "n_steps": len(clean_steps),
                "single_canvas": True,
            }
            cache_path = os.path.join(DATA_DIR, f"{task.name}_seed_{seed:04d}.json")
            cache = {**record, "clean_steps": serializable_steps(clean_steps)}

            if injection is None:
                record["status"] = "no_eligible_identifier"
                rows.append(record)
                cache["status"] = record["status"]
                with open(cache_path, "w") as handle:
                    json.dump(cache, handle)
                print(f"{task.name} seed={seed}: no eligible repeated identifier")
                continue

            record["injection"] = {
                key: value for key, value in injection.items() if key not in {"old_ids", "new_ids", "clean_text"}
            }
            clean_replay, replay_steps = run_one(model, tokenizer, inputs, seed)
            record["replay_matches_clean"] = clean_replay == clean_text
            record["replay_tests"] = run_tests(clean_replay, task)

            for mode in ("canvas", "self_conditioning", "both"):
                final_text, intervention_steps = run_one(
                    model,
                    tokenizer,
                    inputs,
                    seed,
                    make_rename_intervention(injection, mode),
                )
                record[mode] = {
                    "final_text": final_text,
                    **classify(final_text, injection, task),
                    "trajectory_steps": len(intervention_steps),
                }
                cache.setdefault("interventions", {})[mode] = {
                    "steps": serializable_steps(intervention_steps),
                    "identifier_trace": trace_identifier(
                        tokenizer,
                        intervention_steps,
                        injection["old_name"],
                        injection["new_name"],
                    ),
                }
            record["status"] = "intervened"
            rows.append(record)
            cache.update(
                {
                    "status": record["status"],
                    "injection": record["injection"],
                    "replay_matches_clean": record["replay_matches_clean"],
                    "interventions": cache.get("interventions", {}),
                }
            )
            with open(cache_path, "w") as handle:
                json.dump(cache, handle)
            outcomes = ", ".join(f"{mode}={record[mode]['outcome']}" for mode in ("canvas", "self_conditioning", "both"))
            print(f"{task.name} seed={seed}: {outcomes}")

    with open(RESULT_PATH, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {RESULT_PATH}")


if __name__ == "__main__":
    main()
