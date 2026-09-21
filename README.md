# SegmentSelectiveSFT — online Qwen3-8B baseline

This branch ports the experiment in
`new_nothingnew_2/SegmentSelectiveSFT` from an offline, machine-specific server
to an ordinary Internet-connected NVIDIA server. Models and datasets are loaded
from Hugging Face, and there are no `/mnt/local` or `aiskylimit` paths.

The comparison recipe—not the paper appendix—is authoritative here. The only
intentional experiment change is the requested backbone:
`Qwen/Qwen2.5-7B-Instruct` becomes `Qwen/Qwen3-8B`.

## Run end to end

Requirements:

- Linux and Python 3.11.
- One sufficiently large NVIDIA GPU for training. The reference experiment is
  single-process and uses sequence length 32,768.
- Enough CPU RAM and disk to merge and store an 8B model.
- Hugging Face access; set `HF_TOKEN` if required.

```bash
bash project_commands.sh all
```

The stages can also be scheduled separately:

```bash
bash project_commands.sh setup
bash project_commands.sh download data
bash project_commands.sh train
bash project_commands.sh merge
bash project_commands.sh eval
```

Use `DRY_RUN=1 bash project_commands.sh all` to print the complete workflow,
or `bash project_commands.sh config` to print resolved settings.

## Fair-comparison configuration

These defaults mirror `new_nothingnew_2/SegmentSelectiveSFT/commands.sh`:

| Component | Default |
|---|---|
| Training data | `baesad/s1K-1.1-deepseek-cot`, 934 rows |
| Backbone | `Qwen/Qwen3-8B` |
| Training strategy | one-stage full-CoT SFT; no segment mask |
| Parameter-efficient tuning | LoRA |
| LoRA | r=16, alpha=16, dropout=0.05 |
| Target modules | q/k/v/o and gate/up/down projections |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Sequence length | 32,768 |
| Batch | 1 per device × 32 gradient accumulation |
| Optimizer | `adamw_torch`, betas=(0.9, 0.999), eps=1e-8 |
| Weight decay | 0.0 |
| Scheduler | cosine, warmup ratio 0.1 |
| Gradient checkpointing | Unsloth checkpointing enabled |
| Seed | 3407 |
| Eval samples | 3 per question for every benchmark |
| Eval decoding | temperature 0.6, top-p 0.9, repetition penalty 1.05 |
| Max generated tokens | 32,768 |
| Reported metrics | pass@1 and pass@3, plus first-sample accuracy |

The training prompt and raw s1K response are unchanged. As in the comparison
repository's `think_prefix=none` setting, no `<think>` token is manually
prefilled. The Qwen3 chat template is still used with its native thinking mode.

Environment variables override any setting without editing the script. For
example:

```bash
HF_HOME=/mnt/cache/huggingface GPU_TRAIN=1 GPU_EVAL=0 \
  bash project_commands.sh all

EVAL_N=1 MAX_TOKENS=4096 EVAL_TAG=smoke \
  EVAL_MODEL=Qwen/Qwen3-8B bash project_commands.sh eval
```

## Data and model flow

`prepare_hf_data.py` downloads and converts:

- `baesad/s1K-1.1-deepseek-cot`
- `math-ai/aime24`
- `math-ai/aime25`
- `AI-MO/aimo-validation-amc` as `amc12`
- `HuggingFaceH4/MATH-500`

Training writes the LoRA adapter to:

```text
artifacts/qwen3_8b_fullsft_lora_r16/final/
```

Because vLLM evaluates a complete model, the next stage merges that adapter
with Qwen3-8B:

```text
artifacts/qwen3_8b_fullsft_lora_r16/final-merged/
```

Evaluation writes per-question generations, per-task metrics, and the final
summary to:

```text
Eval/outputs_fullsft_r16_ep3/summary.json
```

Existing task metrics are reused by default, matching the resumable behavior of
the comparison repository. Set `EVAL_OVERWRITE=1` to regenerate them.

## Optional selective pipeline

The default `all` workflow is deliberately the full-CoT baseline above. The
original attribution/selective code remains available for separate experiments:

```bash
bash project_commands.sh attribution
```

Its defaults also follow the comparison folder: s1K data, paragraph
segmentation, 20 IG integration points, IG batch size 4, and no automatic
resume. It is not executed by the fair-comparison baseline.

## Paper

The underlying code implements *Segment-Level Attribution for Selective
Learning of Long Reasoning Traces* ([arXiv:2602.00425](https://arxiv.org/abs/2602.00425)).
This branch intentionally uses the separate experimental recipe documented
above rather than the paper's default training schedule.
