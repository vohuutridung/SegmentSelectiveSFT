# SegmentSelectiveSFT — online Qwen3-8B selective SFT

This branch ports the experiment in
`new_nothingnew_2/SegmentSelectiveSFT` from an offline, machine-specific server
to an ordinary Internet-connected NVIDIA server. Models and datasets are loaded
from Hugging Face, and there are no `/mnt/local` or `aiskylimit` paths.

The comparison recipe—not the paper appendix—is authoritative here. This run
uses `Qwen/Qwen3-8B` and trains it directly with selective SFT. The full-CoT
warm-up used for the Qwen2.5-7B experiment is deliberately omitted.

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
bash project_commands.sh attribution
bash project_commands.sh train
bash project_commands.sh merge
bash project_commands.sh eval
```

Use `DRY_RUN=1 bash project_commands.sh all` to print the complete workflow,
or `bash project_commands.sh config` to print resolved settings.

## Fair-comparison configuration

The hyperparameters and selective strategy follow
`new_nothingnew_2/SegmentSelectiveSFT`:

| Component | Default |
|---|---|
| Training data | `baesad/s1K-1.1-deepseek-cot`, 934 rows |
| Backbone | `Qwen/Qwen3-8B` |
| Training strategy | selective SFT directly from Qwen3-8B |
| Full-CoT warm-up | none |
| Attribution model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| Segmentation | paragraph (`\n\n`) |
| Integrated Gradients | 20 steps, batch size 4 |
| Selection | cumulative strength 0.7, consistency ≤ 0.8 |
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

The full reasoning trace remains in the input, but labels are `-100` outside
the selected segments. Following the comparison folder, the first,
penultimate, and final segments are always supervised in addition to the
attribution-selected segments.

The comparison folder uses Qwen2.5 with `think_prefix=none` during training and
does not add `<think>` during evaluation. To preserve that effective prompt
behavior after changing only the backbone, this branch explicitly sets
`enable_thinking=False` for Qwen3-8B in both training and evaluation. It also
keeps `think_prefix=none`, so no manual `<think>` prefix is appended.

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

The attribution stages create the selected training data at:

```text
artifacts/data/s1k_solutions_selected.jsonl
```

Selective training writes the LoRA adapter to:

```text
artifacts/qwen3_8b_selective_lora_r16/final/
```

Because vLLM evaluates a complete model, the next stage merges that adapter
with Qwen3-8B:

```text
artifacts/qwen3_8b_selective_lora_r16/final-merged/
```

Evaluation writes per-question generations, per-task metrics, and the final
summary to:

```text
Eval/outputs_selective_r16_ep3/summary.json
```

Existing task metrics are reused by default, matching the resumable behavior of
the comparison repository. Set `EVAL_OVERWRITE=1` to regenerate them.

The `all` workflow runs attribution and segment selection before training. It
never trains an intermediate full-CoT checkpoint.

## Paper

The underlying code implements *Segment-Level Attribution for Selective
Learning of Long Reasoning Traces* ([arXiv:2602.00425](https://arxiv.org/abs/2602.00425)).
This branch intentionally uses the separate experimental recipe documented
above rather than the paper's default training schedule.
