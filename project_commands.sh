#!/usr/bin/env bash
# Online end-to-end selective-SFT runner matching
# new_nothingnew_2/SegmentSelectiveSFT. Qwen3-8B is trained directly with the
# selective mask; unlike the Qwen2.5 experiment, there is no full-CoT warm-up.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
EVAL_VENV="${EVAL_VENV:-${ROOT_DIR}/.venv_eval}"
TRAIN_VENV="${TRAIN_VENV:-${ROOT_DIR}/.venv_train}"
EVAL_PY="${EVAL_VENV}/bin/python"
TRAIN_PY="${TRAIN_VENV}/bin/python"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false

BACKBONE_MODEL="${BACKBONE_MODEL:-Qwen/Qwen3-8B}"
ATTR_MODEL="${ATTR_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_EVAL="${GPU_EVAL:-0}"
GPU_ATTR="${GPU_ATTR:-0}"

# Fair-comparison recipe from new_nothingnew_2/SegmentSelectiveSFT/commands.sh.
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
OPTIM="${OPTIM:-adamw_torch}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPSILON="${ADAM_EPSILON:-1e-8}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Attribution and selection defaults from the comparison repository.
SEGMENT_MODE="${SEGMENT_MODE:-paragraph}"
IG_STEPS="${IG_STEPS:-20}"
IG_BATCH_SIZE="${IG_BATCH_SIZE:-4}"
ATTR_RESUME="${ATTR_RESUME:-0}"

EVAL_N="${EVAL_N:-3}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
EVAL_TAG="${EVAL_TAG:-selective_r16_ep3}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-0}"
DATA_OVERWRITE="${DATA_OVERWRITE:-0}"

ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
TRAIN_DATA="${ROOT_DIR}/data/s1k/train.jsonl"
SEGMENT_FILE="${ARTIFACT_DIR}/attribution/s1k_solution_segments.jsonl"
ATTRIBUTED_FILE="${ARTIFACT_DIR}/attribution/s1k_attributed_J${IG_STEPS}.jsonl"
IG_FILE="${ARTIFACT_DIR}/attribution/s1k_IG_J${IG_STEPS}.jsonl"
IG_COMPACT_FILE="${ARTIFACT_DIR}/attribution/s1k_IG_J${IG_STEPS}_compact.jsonl"
SELECTED_DATA="${ARTIFACT_DIR}/data/s1k_solutions_selected.jsonl"
TRAIN_OUTPUT="${ARTIFACT_DIR}/qwen3_8b_selective_lora_r${LORA_R}"
ADAPTER_MODEL="${TRAIN_OUTPUT}/final"
MERGED_MODEL="${TRAIN_OUTPUT}/final-merged"
EVAL_MODEL="${EVAL_MODEL:-${MERGED_MODEL}}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${ROOT_DIR}/Eval/outputs_${EVAL_TAG}}"

DRY_RUN="${DRY_RUN:-0}"
FORCE_SETUP="${FORCE_SETUP:-0}"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_command "$@"
  [[ "$DRY_RUN" == "1" ]] || "$@"
}

run_logged() {
  local name="$1"
  shift
  print_command "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

require_file() {
  [[ "$DRY_RUN" == "1" || -e "$1" ]] || die "Missing required path: $1"
}

require_model() {
  case "$1" in
    /*|./*|../*) require_file "$1" ;;
    *) : ;;
  esac
}

stage_setup() {
  log "Creating isolated train and eval environments"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die \
    "Cannot find ${PYTHON_BIN}; install Python 3.11 or set PYTHON_BIN"

  [[ -x "$EVAL_PY" ]] || run "$PYTHON_BIN" -m venv "$EVAL_VENV"
  [[ -x "$TRAIN_PY" ]] || run "$PYTHON_BIN" -m venv "$TRAIN_VENV"

  if [[ "$DRY_RUN" == "1" || "$FORCE_SETUP" == "1" || ! -f "${EVAL_VENV}/.ssft_ready" || "${ROOT_DIR}/requirements.txt" -nt "${EVAL_VENV}/.ssft_ready" ]]; then
    run "${EVAL_VENV}/bin/pip" install --upgrade pip
    run "${EVAL_VENV}/bin/pip" install -r "${ROOT_DIR}/requirements.txt"
    run "${EVAL_VENV}/bin/pip" install -e "${ROOT_DIR}/Eval/latex2sympy"
    run "$EVAL_PY" -c "import torch, transformers, vllm, datasets; print(torch.__version__, transformers.__version__, vllm.__version__)"
    [[ "$DRY_RUN" == "1" ]] || touch "${EVAL_VENV}/.ssft_ready"
  else
    log "Eval environment already prepared"
  fi

  if [[ "$DRY_RUN" == "1" || "$FORCE_SETUP" == "1" || ! -f "${TRAIN_VENV}/.ssft_ready" || "${ROOT_DIR}/SelectiveSFT/requirements.txt" -nt "${TRAIN_VENV}/.ssft_ready" ]]; then
    run "${TRAIN_VENV}/bin/pip" install --upgrade pip
    run "${TRAIN_VENV}/bin/pip" install -r "${ROOT_DIR}/SelectiveSFT/requirements.txt"
    run "$TRAIN_PY" -c "import torch, transformers, trl, unsloth, torchao, bitsandbytes; print(torch.__version__, transformers.__version__, trl.__version__)"
    [[ "$DRY_RUN" == "1" ]] || touch "${TRAIN_VENV}/.ssft_ready"
  else
    log "Training environment already prepared"
  fi
}

stage_download() {
  require_file "$EVAL_PY"
  log "Downloading the Qwen3-8B backbone and attribution model from Hugging Face"
  run_logged download_models "$EVAL_PY" "${ROOT_DIR}/download_hf_models.py" \
    "$BACKBONE_MODEL" "$ATTR_MODEL" --cache-dir "$HF_HUB_CACHE"
}

stage_data() {
  require_file "$EVAL_PY"
  log "Downloading s1K plus AIME24/AIME25/AMC12/MATH500"
  local data_command=("$EVAL_PY" "${ROOT_DIR}/prepare_hf_data.py"
    --tasks s1k aime24 aime25 amc12 math500
    --data-root "${ROOT_DIR}/data"
    --cache-dir "$HF_DATASETS_CACHE")
  [[ "$DATA_OVERWRITE" == "1" ]] && data_command+=(--overwrite)
  run_logged prepare_data "${data_command[@]}"
}

stage_attribution() {
  require_file "$EVAL_PY"
  require_file "$TRAIN_DATA"
  log "Paragraph segmentation + J=${IG_STEPS} IG and segment selection on s1K"
  run_logged segment env CUDA_VISIBLE_DEVICES="$GPU_ATTR" "$EVAL_PY" \
    "${ROOT_DIR}/Attribution/segment_split.py" \
    --input-data "$TRAIN_DATA" \
    --output-data "$SEGMENT_FILE" \
    --tokenizer "$ATTR_MODEL" \
    --segment-mode "$SEGMENT_MODE"

  local attribution_command=(env CUDA_VISIBLE_DEVICES="$GPU_ATTR" "$EVAL_PY"
    "${ROOT_DIR}/Attribution/grad_analyze.py"
    --model_name "$ATTR_MODEL"
    --input_data "$SEGMENT_FILE"
    --output_data_file "$ATTRIBUTED_FILE"
    --output_ig_file "$IG_FILE"
    --output_compact_file "$IG_COMPACT_FILE"
    --ig_steps "$IG_STEPS"
    --ig_batch_size "$IG_BATCH_SIZE")
  [[ "$ATTR_RESUME" == "1" ]] && attribution_command+=(--resume)
  run_logged attribution "${attribution_command[@]}"

  run_logged select_segments "$EVAL_PY" \
    "${ROOT_DIR}/Attribution/get_important_segments.py" \
    --input_data_file "$SEGMENT_FILE" \
    --IG_score_data_file "$IG_COMPACT_FILE" \
    --output_data_file "$SELECTED_DATA" \
    --cumulative_ratio 0.7 \
    --coherence_max 0.8
}

stage_train() {
  require_file "$TRAIN_PY"
  require_file "$SELECTED_DATA"
  [[ "$GPU_TRAIN" != *,* ]] || die \
    "The reference recipe trains one process on one GPU; set one GPU in GPU_TRAIN"
  log "Selective LoRA SFT (no full-CoT warm-up): r=${LORA_R}, epochs=${EPOCHS}, effective batch=$((TRAIN_BATCH_SIZE * GRAD_ACCUM))"
  local train_command=(env CUDA_VISIBLE_DEVICES="$GPU_TRAIN" "$TRAIN_PY"
    "${ROOT_DIR}/SelectiveSFT/train_mask.py"
    --model_name_or_path "$BACKBONE_MODEL"
    --data_names "$SELECTED_DATA"
    --output_dir "$TRAIN_OUTPUT"
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --optim "$OPTIM"
    --weight_decay "$WEIGHT_DECAY"
    --adam_beta1 "$ADAM_BETA1"
    --adam_beta2 "$ADAM_BETA2"
    --adam_epsilon "$ADAM_EPSILON"
    --lr_scheduler_type "$LR_SCHEDULER"
    --warmup_ratio "$WARMUP_RATIO"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --target_modules "$TARGET_MODULES"
    --dataset_num_proc 2
    --segment_mode "$SEGMENT_MODE"
    --think_prefix none
    --no-enable-thinking
    --mask)
  run_logged train_selective "${train_command[@]}"
}

stage_merge() {
  require_file "$TRAIN_PY"
  require_file "${ADAPTER_MODEL}/adapter_config.json"
  log "Merging LoRA adapter for vLLM"
  run_logged merge_lora "$TRAIN_PY" "${ROOT_DIR}/SelectiveSFT/merge_lora.py" \
    --adapter "$ADAPTER_MODEL" \
    --base_model "$BACKBONE_MODEL" \
    --output_dir "$MERGED_MODEL"
}

stage_eval() {
  require_file "$EVAL_PY"
  require_model "$EVAL_MODEL"
  for task in aime24 aime25 amc12 math500; do
    require_file "${ROOT_DIR}/data/${task}/test.jsonl"
  done
  log "Evaluating ${EVAL_MODEL} with the fair-comparison decoding recipe"
  run_logged eval env \
    GPU_EVAL="$GPU_EVAL" \
    PYTHON_BIN="$EVAL_PY" \
    MODEL_PATH="$EVAL_MODEL" \
    OUTPUT_ROOT="$EVAL_OUTPUT" \
    RUN_TAG="$EVAL_TAG" \
    TASK_SPECS="aime24:${EVAL_N} aime25:${EVAL_N} amc12:${EVAL_N} math500:${EVAL_N}" \
    MAX_TOKENS="$MAX_TOKENS" \
    TEMPERATURE="$TEMPERATURE" \
    TOP_P="$TOP_P" \
    REPETITION_PENALTY="$REPETITION_PENALTY" \
    OVERWRITE="$EVAL_OVERWRITE" \
    bash "${ROOT_DIR}/Eval/run_eval.sh"

  run_logged summarize "$EVAL_PY" "${ROOT_DIR}/Eval/summarize_results.py" "$EVAL_OUTPUT"
}

show_config() {
  cat <<EOF
backbone       : $BACKBONE_MODEL
training data  : baesad/s1K-1.1-deepseek-cot -> $TRAIN_DATA
training       : selective-only LoRA r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT
warm-up        : none
Qwen3 thinking : disabled (matches think_prefix=none behavior)
epochs / lr    : $EPOCHS / $LEARNING_RATE
sequence       : $MAX_SEQ_LENGTH
batch          : $TRAIN_BATCH_SIZE x $GRAD_ACCUM accumulation
optimizer      : $OPTIM betas=($ADAM_BETA1,$ADAM_BETA2) eps=$ADAM_EPSILON wd=$WEIGHT_DECAY
scheduler      : $LR_SCHEDULER warmup_ratio=$WARMUP_RATIO
eval           : n=$EVAL_N temp=$TEMPERATURE top_p=$TOP_P repetition_penalty=$REPETITION_PENALTY
adapter        : $ADAPTER_MODEL
merged model   : $MERGED_MODEL
eval output    : $EVAL_OUTPUT
attribution    : model=$ATTR_MODEL mode=$SEGMENT_MODE J=$IG_STEPS batch=$IG_BATCH_SIZE
EOF
}

usage() {
  cat <<'EOF'
Usage: bash project_commands.sh <stage> [stage ...]

Stages:
  setup        Create the train/eval environments and install dependencies
  download     Download Qwen3-8B and the attribution model from Hugging Face
  data         Download s1K and the four evaluation datasets
  train        Selective LoRA SFT directly from Qwen3-8B (no full-CoT warm-up)
  merge        Merge the final LoRA adapter for vLLM
  eval         Evaluate AIME24, AIME25, AMC12 and MATH500
  attribution  Paragraph split, IG attribution and segment selection
  all          setup, download, data, attribution, train, merge, eval
  config       Print the resolved configuration
EOF
}

run_stage() {
  case "$1" in
    setup) stage_setup ;;
    download) stage_download ;;
    data) stage_data ;;
    train) stage_train ;;
    merge) stage_merge ;;
    eval) stage_eval ;;
    attribution) stage_attribution ;;
    config) show_config ;;
    all)
      stage_setup
      stage_download
      stage_data
      stage_attribution
      stage_train
      stage_merge
      stage_eval
      ;;
    help|-h|--help) usage ;;
    *) die "Unknown stage '$1' (use: bash project_commands.sh help)" ;;
  esac
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 0
fi

for stage in "$@"; do
  run_stage "$stage"
done
