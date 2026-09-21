#!/usr/bin/env python3
"""Collect benchmark metric files into one machine-readable summary."""

import argparse
import glob
import json
import os


EXPECTED_TASKS = ("aime24", "aime25", "amc12", "math500")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Evaluation output root")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    candidates = {}
    for path in sorted(glob.glob(os.path.join(args.root, "**", "*_metrics.json"), recursive=True)):
        with open(path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        relative = os.path.relpath(path, args.root)
        task = relative.split(os.sep, 1)[0]
        if task not in EXPECTED_TASKS:
            continue
        candidates.setdefault(task, []).append(
            {
                "task": task,
                "acc": metrics.get("acc"),
                "acc_first": metrics.get("acc_first"),
                "pass_at_k": metrics.get("pass_at_k", {}),
                "num_questions": metrics.get("num_samples"),
                "num_scores": metrics.get("num_scores"),
                "metrics_file": relative,
                "modified_at": os.path.getmtime(path),
            }
        )

    if not candidates:
        raise SystemExit(f"No *_metrics.json files found under {args.root}")
    rows = []
    order = {task: index for index, task in enumerate(EXPECTED_TASKS)}
    for task, task_rows in sorted(candidates.items(), key=lambda item: order[item[0]]):
        selected = max(task_rows, key=lambda row: row["modified_at"])
        selected.pop("modified_at")
        rows.append(selected)
        if len(task_rows) > 1:
            print(
                f"Warning: found {len(task_rows)} metric files for {task}; "
                f"using newest: {selected['metrics_file']}"
            )

    expected = set(EXPECTED_TASKS)
    present = {row["task"] for row in rows}
    missing = sorted(expected - present)
    if missing:
        raise SystemExit("Missing benchmark results: " + ", ".join(missing))

    accs = [row["acc"] for row in rows if isinstance(row["acc"], (int, float))]
    pass_keys = sorted(
        {key for row in rows for key in row["pass_at_k"]},
        key=lambda key: int(key.split("@", 1)[1]),
    )
    macro_pass_at_k = {}
    for key in pass_keys:
        values = [
            row["pass_at_k"].get(key)
            for row in rows
            if isinstance(row["pass_at_k"].get(key), (int, float))
        ]
        if len(values) == len(rows):
            macro_pass_at_k[key] = round(sum(values) / len(values), 2)
    summary = {
        "metric": "pass@1 averaged over all generated completions",
        "macro_average": round(sum(accs) / len(accs), 2) if accs else None,
        "macro_pass_at_k": macro_pass_at_k,
        "results": rows,
    }
    output = args.output or os.path.join(args.root, "summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        f"{'benchmark':12s} {'pass@1':>8s} {'pass@6':>8s} "
        f"{'first':>8s} {'#q':>6s} {'#gen':>8s}"
    )
    for row in rows:
        pass6 = row["pass_at_k"].get("pass@6")
        pass6_text = f"{pass6:8.2f}" if isinstance(pass6, (int, float)) else f"{'-':>8s}"
        print(
            f"{row['task']:12s} {row['acc']:8.2f} {pass6_text} "
            f"{row['acc_first']:8.2f} "
            f"{row['num_questions']:6d} {row['num_scores']:8d}"
        )
    macro6 = macro_pass_at_k.get("pass@6")
    macro6_text = f"{macro6:8.2f}" if isinstance(macro6, (int, float)) else f"{'-':>8s}"
    print(f"{'MACRO AVG':12s} {summary['macro_average']:8.2f} {macro6_text}")
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
