#!/usr/bin/env python3
import argparse
import json
import os
import sys

from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from segment_utils import SEGMENT_MODES, split_segments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", default=os.path.join(ROOT, "data/limo/train.jsonl"))
    parser.add_argument("--output-data", default=os.path.join(ROOT, "data/limo/solution_segments.jsonl"))
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--segment-mode", choices=SEGMENT_MODES, default="paragraph")
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = []
    with open(args.input_data, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            solution = str(row.get("solution") or "")
            if not solution:
                raise ValueError(f"{args.input_data}:{line_number}: empty solution")
            row["segments"] = split_segments(solution, args.segment_mode)
            rows.append(row)

    if not rows:
        raise SystemExit(f"No samples found in {args.input_data}")
    token_lengths = [
        len(tokenizer(row["solution"], add_special_tokens=False)["input_ids"])
        for row in tqdm(rows, desc="Token statistics")
    ]
    segment_counts = [len(row["segments"]) for row in rows]
    print(
        "samples=%d mean_segments=%.2f max_tokens=%d mode=%s"
        % (len(rows), sum(segment_counts) / len(segment_counts), max(token_lengths), args.segment_mode)
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_data)), exist_ok=True)
    temporary = args.output_data + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output_data)
    print(f"Wrote {args.output_data}")


if __name__ == "__main__":
    main()
