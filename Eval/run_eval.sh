#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${GPU_EVAL:-0}"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-${1:-Qwen/Qwen3-8B}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_qwen3_8b_selective}"
RUN_TAG="${RUN_TAG:-qwen3_8b_selective}"
TASK_SPECS="${TASK_SPECS:-aime24:32 aime25:32 amc12:32 math500:6}"
SEED="${SEED:-0}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MIN_P="${MIN_P:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
OVERWRITE="${OVERWRITE:-0}"

PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"

for spec in $TASK_SPECS; do
  task="${spec%%:*}"
  samples="${spec##*:}"
  echo "============================================================"
  echo "task=${task} samples_per_question=${samples} model=${MODEL_PATH}"
  echo "============================================================"

  eval_command=("$PYTHON_BIN" -u math_eval.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_name "$task" \
    --data_dir "${SCRIPT_DIR}/../data" \
    --output_dir "${OUTPUT_ROOT}/${task}/${RUN_TAG}" \
    --split test \
    --prompt_type deepseek-longcot \
    --num_test_sample -1 \
    --max_tokens_per_call "$MAX_TOKENS" \
    --seed "$SEED" \
    --temperature "$TEMPERATURE" \
    --n_sampling "$samples" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --min_p "$MIN_P" \
    --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --use_vllm \
    --save_outputs \
    --apply_chat_template \
    --enable-thinking \
    --prefill-think \
    --enable_prefix_caching)
  if [[ "$MAX_MODEL_LEN" != "0" ]]; then
    eval_command+=(--max_model_len "$MAX_MODEL_LEN")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    eval_command+=(--overwrite)
  fi
  "${eval_command[@]}"
done
