#!/usr/bin/env python3
"""Download LIMO and evaluation benchmarks directly from Hugging Face.

The generated JSONL files use the schema expected by the original paper code.
All datasets can be overridden with CLI flags if a mirror is preferred.
"""

import argparse
import json
import os
import re
import tempfile


DEFAULT_DATASETS = {
    "limo": ("GAIR/LIMO", "train"),
    "aime24": ("math-ai/aime24", "test"),
    "aime25": ("math-ai/aime25", "test"),
    "amc12": ("AI-MO/aimo-validation-amc", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
}

EXPECTED_ROWS = {
    "limo": 817,
    "aime24": 30,
    "aime25": 30,
    "amc12": 83,
    "math500": 500,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
        help="Datasets to download and convert",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--revision", default="", help="Optional HF revision for every dataset")
    parser.add_argument("--overwrite", action="store_true")
    for task, (dataset_id, _) in DEFAULT_DATASETS.items():
        parser.add_argument(f"--{task}-dataset", default=dataset_id)
    return parser.parse_args()


def last_boxed(text):
    text = str(text or "")
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    opening = text.find("{", start)
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index].strip()
    return None


def normalize_answer(value):
    answer = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", answer):
        answer = answer.split(".", 1)[0]
    return answer


def first_present(row, keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def convert_row(task, row, index):
    if task == "limo":
        question = first_present(row, ("question", "problem"))
        solution = first_present(row, ("solution", "response"))
        answer = first_present(row, ("answer",)) or last_boxed(solution)
        if question is None or solution is None or answer is None:
            raise ValueError(f"LIMO row {index} is missing question/solution/answer")
        return {
            "question": str(question).strip(),
            "solution": str(solution).strip(),
            "answer": normalize_answer(answer),
        }

    question = first_present(row, ("problem", "question", "input"))
    solution = first_present(row, ("solution", "response"))
    answer = first_present(row, ("answer", "ground_truth")) or last_boxed(solution)
    if question is None or answer is None:
        raise ValueError(f"{task} row {index} is missing question/answer")
    record = {
        "idx": index,
        "question": str(question).strip(),
        "answer": normalize_answer(answer),
    }
    if solution is not None:
        record["solution"] = str(solution).strip()
    if task == "math500" and "solution" not in record:
        record["solution"] = f"\\boxed{{{record['answer']}}}"
    return record


def atomic_write_jsonl(path, rows):
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=output_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    args = parse_args()
    from datasets import load_dataset

    for task in args.tasks:
        output = os.path.join(args.data_root, task, "test.jsonl")
        if task == "limo":
            output = os.path.join(args.data_root, "limo", "train.jsonl")
        if os.path.exists(output) and not args.overwrite:
            print(f"{task:8s}: keeping existing {output} (use --overwrite to refresh)")
            continue

        dataset_id = getattr(args, f"{task}_dataset")
        split = DEFAULT_DATASETS[task][1]
        kwargs = {"split": split}
        if args.cache_dir:
            kwargs["cache_dir"] = args.cache_dir
        if args.revision:
            kwargs["revision"] = args.revision
        dataset = load_dataset(dataset_id, **kwargs)
        rows = [convert_row(task, dict(row), index) for index, row in enumerate(dataset)]
        if not rows:
            raise SystemExit(f"{dataset_id} returned no rows for split={split}")
        atomic_write_jsonl(output, rows)
        expected = EXPECTED_ROWS.get(task)
        note = "" if expected == len(rows) else f" (expected {expected}; upstream revision may have changed)"
        print(f"{task:8s}: {len(rows)} rows from {dataset_id}[{split}] -> {output}{note}")


if __name__ == "__main__":
    main()

