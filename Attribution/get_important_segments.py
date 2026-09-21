import json
import numpy as np
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_data_file", type=str, required=True, help="Path to input jsonl")
    p.add_argument("--IG_score_data_file", type=str, required=True, help="Path to input IG scores")
    p.add_argument("--output_data_file", type=str, required=True, help="Path to output jsonl")
    p.add_argument("--cumulative_ratio", type=float, default=0.7)
    p.add_argument(
        "--coherence_max",
        "--consistency_max",
        dest="consistency_max",
        type=float,
        default=0.8,
    )
    return p.parse_args()

args = parse_args()
if not 0 < args.cumulative_ratio <= 1:
    raise ValueError("--cumulative_ratio must be in (0, 1]")
if not 0 <= args.consistency_max <= 1:
    raise ValueError("--consistency_max must be in [0, 1]")

input_data = []
with open(args.input_data_file, "r") as f: 
    for line in f:
        json_obj = json.loads(line.strip())  
        input_data.append(json_obj)

def to_compact(row):
    if isinstance(row, dict):
        return [tuple(segment) for segment in row["segments"]]
    compact = []
    for segment in row:
        token_count = len(segment)
        compact.append(
            (
                token_count,
                float(np.sum(np.abs(segment))) if token_count else 0.0,
                float(np.sum(segment)) if token_count else 0.0,
            )
        )
    return compact


all_IG_list = []
with open(args.IG_score_data_file, "r") as f: 
    for line in f:
        if line.strip():
            all_IG_list.append(to_compact(json.loads(line)))
print("sample number", len(input_data), len(all_IG_list))
if len(input_data) != len(all_IG_list):
    raise ValueError(
        "Input/IG row count mismatch: %d != %d" % (len(input_data), len(all_IG_list))
    )


ratio_list = []
for i in range(len(input_data)):
    assert len(all_IG_list[i]) == len(input_data[i]["segments"]), f"{i} {len(all_IG_list[i])}. {len(input_data[i]['segments'])}"
    cur_IGs = all_IG_list[i]
    if not cur_IGs:
        raise ValueError(f"Sample {i} has no segments")
        
    all_segs_IG_stres = []
    for token_count, sum_abs, _sum_signed in cur_IGs:
        all_segs_IG_stres.append(
            sum_abs / (token_count ** 0.5) if token_count else 0.0
        )
    indexed_sorted = sorted(enumerate(all_segs_IG_stres), key=lambda x: -x[1])
    sorted_indices = [idx for idx, val in indexed_sorted]
    sorted_inst_IG_stre = np.array([val for idx, val in indexed_sorted])

    total_strength = sorted_inst_IG_stre.sum()
    if total_strength <= 0 or not np.isfinite(total_strength):
        important_index = []
    else:
        normed = sorted_inst_IG_stre / total_strength
        cumsum = np.cumsum(normed)
        cutoff = int(np.searchsorted(cumsum, args.cumulative_ratio, side="left"))
        important_index = sorted(sorted_indices[: cutoff + 1])
        
    IG_dire_list = []
    for _token_count, denominator, signed_sum in cur_IGs:
        # Match the comparison folder: a zero-attribution segment is treated
        # as maximally coherent and therefore excluded by the 0.8 threshold.
        IG_dire_list.append(abs(signed_sum) / denominator if denominator else 1.0)
    
    select_span_ids = [
        index for index in important_index if IG_dire_list[index] <= args.consistency_max
    ]
    assert select_span_ids == sorted(select_span_ids)
    
    input_data[i]['selected_spans_ids'] = select_span_ids
    ratio_list.append(len(select_span_ids)/len(cur_IGs))
        
os.makedirs(os.path.dirname(os.path.abspath(args.output_data_file)), exist_ok=True)
with open(args.output_data_file, 'w') as f:
    for n in range(len(input_data)):
        f.write(json.dumps(input_data[n], ensure_ascii=False) + '\n')
print(np.mean(ratio_list))
