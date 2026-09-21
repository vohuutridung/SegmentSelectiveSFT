#!/usr/bin/env bash
# Online end-to-end runner for Segment-Selective SFT with Qwen3-8B.
#
# Usage:
#   bash project_commands.sh setup
#   bash project_commands.sh data download attribution train eval
#   bash project_commands.sh all
#
# Common overrides:
#   GPU_ATTR=0,1 GPU_TRAIN=0 GPU_EVAL=0,1 bash project_commands.sh all
#   HF_HOME=/mnt/cache/huggingface bash project_commands.sh download data
#   AIME_N=8 AMC_N=8 MATH_N=4 bash project_commands.sh eval
#   EVAL_MODEL=/path/to/another/model bash project_commands.sh eval

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

ATTR_MODEL="${ATTR_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
BACKBONE_MODEL="${BACKBONE_MODEL:-Qwen/Qwen3-8B}"
GPU_ATTR="${GPU_ATTR:-0,1}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_EVAL="${GPU_EVAL:-0}"

IG_STEPS="${IG_STEPS:-50}"
IG_BATCH_SIZE="${IG_BATCH_SIZE:-1}"
ATTR_RESUME="${ATTR_RESUME:-1}"
SEGMENT_MODE="${SEGMENT_MODE:-cue}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"

# Qwen schedule from the paper, applied to the requested Qwen3-8B backbone.
FULL_EPOCHS="${FULL_EPOCHS:-7}"
FULL_LR="${FULL_LR:-1.5e-5}"
SELECTIVE_EPOCHS="${SELECTIVE_EPOCHS:-4}"
SELECTIVE_LR="${SELECTIVE_LR:-8e-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

AIME_N="${AIME_N:-32}"
AMC_N="${AMC_N:-32}"
MATH_N="${MATH_N:-6}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
EVAL_TAG="${EVAL_TAG:-qwen3_8b_selective}"

ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
SEGMENT_FILE="${ARTIFACT_DIR}/attribution/solution_segments.jsonl"
ATTRIBUTED_FILE="${ARTIFACT_DIR}/attribution/solution_segments_attributed_7B_J${IG_STEPS}.jsonl"
IG_FILE="${ARTIFACT_DIR}/attribution/IG_7B_J${IG_STEPS}.jsonl"
SELECTED_DATA="${ARTIFACT_DIR}/data/solutions_top70_consistency80_7B_J${IG_STEPS}.jsonl"
FULL_OUTPUT="${ARTIFACT_DIR}/qwen3_8b_fullcot"
SELECTIVE_OUTPUT="${ARTIFACT_DIR}/qwen3_8b_selective"
FULL_MODEL="${FULL_OUTPUT}/final"
SELECTIVE_MODEL="${SELECTIVE_OUTPUT}/final"
EVAL_MODEL="${EVAL_MODEL:-${SELECTIVE_MODEL}}"
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
    *) : ;; # A Hugging Face repository ID is resolved online by Transformers/vLLM.
  esac
}

stage_setup() {
  log "Creating two isolated Python environments"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Cannot find ${PYTHON_BIN}; install Python 3.11 or set PYTHON_BIN"

  if [[ ! -x "$EVAL_PY" ]]; then
    run "$PYTHON_BIN" -m venv "$EVAL_VENV"
  fi
  if [[ ! -x "$TRAIN_PY" ]]; then
    run "$PYTHON_BIN" -m venv "$TRAIN_VENV"
  fi

  if [[ "$DRY_RUN" == "1" || "$FORCE_SETUP" == "1" || ! -f "${EVAL_VENV}/.ssft_ready" ]]; then
    run "${EVAL_VENV}/bin/pip" install --upgrade pip
    run "${EVAL_VENV}/bin/pip" install -r "${ROOT_DIR}/requirements.txt"
    run "${EVAL_VENV}/bin/pip" install -e "${ROOT_DIR}/Eval/latex2sympy"
    run "$EVAL_PY" -c "import torch, transformers, vllm, datasets; print(torch.__version__, transformers.__version__, vllm.__version__)"
    [[ "$DRY_RUN" == "1" ]] || touch "${EVAL_VENV}/.ssft_ready"
  else
    log "Eval/attribution environment already prepared"
  fi

  if [[ "$DRY_RUN" == "1" || "$FORCE_SETUP" == "1" || ! -f "${TRAIN_VENV}/.ssft_ready" ]]; then
    run "${TRAIN_VENV}/bin/pip" install --upgrade pip
    run "${TRAIN_VENV}/bin/pip" install -r "${ROOT_DIR}/SelectiveSFT/requirements.txt"
    run "$TRAIN_PY" -c "import torch, transformers, trl, unsloth, datasets; print(torch.__version__, transformers.__version__, trl.__version__)"
    [[ "$DRY_RUN" == "1" ]] || touch "${TRAIN_VENV}/.ssft_ready"
  else
    log "Training environment already prepared"
  fi
}

stage_download() {
  require_file "$EVAL_PY"
  log "Downloading Hugging Face model snapshots"
  run_logged download_models "$EVAL_PY" "${ROOT_DIR}/download_hf_models.py" \
    "$ATTR_MODEL" "$BACKBONE_MODEL" --cache-dir "$HF_HUB_CACHE"
}

stage_data() {
  require_file "$EVAL_PY"
  log "Downloading and converting LIMO + AIME24/AIME25/AMC12/MATH500"
  run_logged prepare_data "$EVAL_PY" "${ROOT_DIR}/prepare_hf_data.py" \
    --data-root "${ROOT_DIR}/data" \
    --cache-dir "$HF_DATASETS_CACHE" \
    --overwrite
}

stage_attribution() {
  require_file "$EVAL_PY"
  require_file "${ROOT_DIR}/data/limo/train.jsonl"
  log "Cue segmentation + J=${IG_STEPS} Integrated Gradients"
  run_logged segment env CUDA_VISIBLE_DEVICES="$GPU_ATTR" "$EVAL_PY" \
    "${ROOT_DIR}/Attribution/segment_split.py" \
    --input-data "${ROOT_DIR}/data/limo/train.jsonl" \
    --output-data "$SEGMENT_FILE" \
    --tokenizer "$ATTR_MODEL" \
    --segment-mode "$SEGMENT_MODE"

  local attribution_command=(env CUDA_VISIBLE_DEVICES="$GPU_ATTR" "$EVAL_PY"
    "${ROOT_DIR}/Attribution/grad_analyze.py"
    --model_name "$ATTR_MODEL"
    --input_data "$SEGMENT_FILE"
    --output_data_file "$ATTRIBUTED_FILE"
    --output_ig_file "$IG_FILE"
    --ig_steps "$IG_STEPS"
    --ig_batch_size "$IG_BATCH_SIZE")
  if [[ "$ATTR_RESUME" == "1" ]]; then
    attribution_command+=(--resume)
  fi
  run_logged attribution "${attribution_command[@]}"

  run_logged select_segments "$EVAL_PY" \
    "${ROOT_DIR}/Attribution/get_important_segments.py" \
    --input_data_file "$SEGMENT_FILE" \
    --IG_score_data_file "$IG_FILE" \
    --output_data_file "$SELECTED_DATA" \
    --cumulative_ratio 0.7 \
    --consistency_max 0.8
}

stage_full_sft() {
  require_file "$TRAIN_PY"
  require_file "${ROOT_DIR}/data/limo/train.jsonl"
  [[ "$GPU_TRAIN" != *,* ]] || die \
    "Training is single-process full-parameter SFT; set GPU_TRAIN to one sufficiently large GPU"
  log "Stage 1/2: full-CoT SFT of Qwen3-8B"
  run_logged train_full env CUDA_VISIBLE_DEVICES="$GPU_TRAIN" "$TRAIN_PY" \
    "${ROOT_DIR}/SelectiveSFT/train_mask.py" \
    --model_name_or_path "$BACKBONE_MODEL" \
    --data_names "${ROOT_DIR}/data/limo/train.jsonl" \
    --output_dir "$FULL_OUTPUT" \
    --epochs "$FULL_EPOCHS" \
    --learning_rate "$FULL_LR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --enable-thinking \
    --prefill-think
}

stage_selective_sft() {
  require_file "$TRAIN_PY"
  require_file "$FULL_MODEL"
  require_file "$SELECTED_DATA"
  [[ "$GPU_TRAIN" != *,* ]] || die \
    "Training is single-process full-parameter SFT; set GPU_TRAIN to one sufficiently large GPU"
  log "Stage 2/2: segment-selective SFT from the full-CoT checkpoint"
  run_logged train_selective env CUDA_VISIBLE_DEVICES="$GPU_TRAIN" "$TRAIN_PY" \
    "${ROOT_DIR}/SelectiveSFT/train_mask.py" \
    --model_name_or_path "$FULL_MODEL" \
    --data_names "$SELECTED_DATA" \
    --output_dir "$SELECTIVE_OUTPUT" \
    --epochs "$SELECTIVE_EPOCHS" \
    --learning_rate "$SELECTIVE_LR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --segment_mode "$SEGMENT_MODE" \
    --mask \
    --enable-thinking \
    --prefill-think
}

stage_train() {
  stage_full_sft
  stage_selective_sft
}

stage_eval() {
  require_file "$EVAL_PY"
  require_model "$EVAL_MODEL"
  for task in aime24 aime25 amc12 math500; do
    require_file "${ROOT_DIR}/data/${task}/test.jsonl"
  done
  log "Evaluating ${EVAL_MODEL} on AIME24, AIME25, AMC12 and MATH500"
  run_logged eval env \
    GPU_EVAL="$GPU_EVAL" \
    PYTHON_BIN="$EVAL_PY" \
    MODEL_PATH="$EVAL_MODEL" \
    OUTPUT_ROOT="$EVAL_OUTPUT" \
    RUN_TAG="$EVAL_TAG" \
    TASK_SPECS="aime24:${AIME_N} aime25:${AIME_N} amc12:${AMC_N} math500:${MATH_N}" \
    MAX_TOKENS="$MAX_TOKENS" \
    OVERWRITE=1 \
    bash "${ROOT_DIR}/Eval/run_eval.sh"

  run_logged summarize "$EVAL_PY" "${ROOT_DIR}/Eval/summarize_results.py" "$EVAL_OUTPUT"
}

show_config() {
  cat <<EOF
backbone model : $BACKBONE_MODEL
attribution    : $ATTR_MODEL
HF cache       : $HF_HOME
GPUs attr/train/eval: $GPU_ATTR / $GPU_TRAIN / $GPU_EVAL
IG             : mode=$SEGMENT_MODE steps=$IG_STEPS batch=$IG_BATCH_SIZE resume=$ATTR_RESUME
train stage 1  : epochs=$FULL_EPOCHS lr=$FULL_LR
train stage 2  : epochs=$SELECTIVE_EPOCHS lr=$SELECTIVE_LR
sequence length: $MAX_SEQ_LENGTH
eval samples   : AIME=$AIME_N AMC12=$AMC_N MATH500=$MATH_N
final model    : $EVAL_MODEL
eval output    : $EVAL_OUTPUT
EOF
}

usage() {
  sed -n '1,15p' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Stages: setup download data attribution full-sft selective-sft train eval all config
Run stages separately when scheduling attribution and training on different machines.
EOF
}

run_stage() {
  case "$1" in
    setup) stage_setup ;;
    download) stage_download ;;
    data) stage_data ;;
    attribution) stage_attribution ;;
    full-sft) stage_full_sft ;;
    selective-sft) stage_selective_sft ;;
    train) stage_train ;;
    eval) stage_eval ;;
    config) show_config ;;
    all)
      stage_setup
      stage_download
      stage_data
      stage_attribution
      stage_train
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
