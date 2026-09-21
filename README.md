# Segment-Level Attribution for Selective Learning of Long Reasoning Traces

Implementation of the ICLR 2026 paper by Siyuan Wang, Yanchen Liu, and Xiang
Ren, adapted for an ordinary Internet-connected NVIDIA server.

This branch keeps the paper pipeline—LIMO, cue segmentation, 50-step
Integrated Gradients, strength/consistency selection, full-CoT SFT followed by
selective SFT—but uses the requested `Qwen/Qwen3-8B` train/eval backbone.
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` remains the attribution model, as in
the paper. Qwen3-8B is a new backbone experiment, so its scores are not expected
to reproduce the paper's Qwen2.5-7B table.

## One-command workflow

Requirements:

- Linux, Python 3.11, NVIDIA driver/CUDA-compatible GPUs.
- Enough disk for two model snapshots, two full Qwen3-8B training stages, and
  vLLM caches.
- A GPU large enough for full-parameter 8B training at sequence length 16,384.
  The training launcher is single-process and intentionally follows the paper's
  full-parameter setup; set `GPU_TRAIN` to exactly one sufficiently large GPU.
- Internet access to Hugging Face. Set `HF_TOKEN` when your environment or
  mirror requires authentication.

Run everything:

```bash
bash project_commands.sh all
```

This creates two isolated virtual environments, downloads models and datasets,
runs attribution, trains Qwen3-8B in two stages, then evaluates on AIME24,
AIME25, AMC12, and MATH500.

Long jobs can be scheduled independently:

```bash
bash project_commands.sh setup
bash project_commands.sh download data
bash project_commands.sh attribution
bash project_commands.sh full-sft
bash project_commands.sh selective-sft
bash project_commands.sh eval
```

`bash project_commands.sh config` prints the resolved configuration without
starting a job. `DRY_RUN=1 bash project_commands.sh all` prints every command.

## Defaults and overrides

The defaults are:

| Component | Default |
|---|---|
| Training data | `GAIR/LIMO`, 817 examples |
| Attribution model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| Backbone | `Qwen/Qwen3-8B` |
| Segmentation | transition cues |
| IG integration | 50 points, alpha = 1/50 ... 1 |
| Selection | top 70% attribution strength, consistency <= 0.8 |
| Stage 1 | full-CoT, 7 epochs, LR 1.5e-5 |
| Stage 2 | selective SFT, 4 epochs, LR 8e-6 |
| Context | 16,384 tokens |
| Eval decoding | paper sampling: Qwen3 thinking, temp 0.6, top-p 1.0 |

Useful overrides:

```bash
GPU_ATTR=0,1 GPU_TRAIN=2 GPU_EVAL=0,1 bash project_commands.sh all

HF_HOME=/mnt/cache/huggingface \
ARTIFACT_DIR=/mnt/checkpoints/segment-sft \
bash project_commands.sh all

# Cheaper validation run before the full evaluation (use a distinct tag).
EVAL_TAG=smoke AIME_N=2 AMC_N=2 MATH_N=2 MAX_TOKENS=4096 \
bash project_commands.sh eval

# Evaluate another HF model or local checkpoint with the same harness.
EVAL_MODEL=Qwen/Qwen3-8B EVAL_TAG=qwen3_base bash project_commands.sh eval
```

All options are environment variables so the committed script stays portable;
there are no `/mnt/local`, `aiskylimit`, or machine-specific model paths.
Attribution checkpoints each completed example and resumes by default; use
`ATTR_RESUME=0` to recompute it from the beginning.

## Data and outputs

`prepare_hf_data.py` downloads and converts:

- `GAIR/LIMO`
- `math-ai/aime24`
- `math-ai/aime25`
- `AI-MO/aimo-validation-amc` (83 modified AMC12 2022/2023 problems)
- `HuggingFaceH4/MATH-500`

Generated artifacts are ignored by Git:

```text
artifacts/
├── attribution/
├── data/solutions_top70_consistency80_7B_J50.jsonl
├── qwen3_8b_fullcot/final/
└── qwen3_8b_selective/final/

Eval/outputs_qwen3_8b_selective/
└── summary.json
```

The final evaluation prints a table and writes
`Eval/outputs_qwen3_8b_selective/summary.json`. `acc` is pass@1 averaged over
all generated completions; the summary also reports the unbiased pass@6
estimator used by the paper. `acc_first` is retained for comparison with the
original evaluator, which only reported completion zero.

## Method notes

For every LIMO reasoning trace, the pipeline:

1. Splits the trace at the paper's transition cues without deleting or changing
   any characters.
2. Uses the padding embedding as the IG baseline and estimates the path integral
   at `j / 50`, for `j = 1 ... 50`, against the log-probability of the correct
   final-answer tokens.
3. Computes segment strength as `sum(abs(IG)) / sqrt(token_count)` and direction
   consistency as `abs(sum(IG)) / sum(abs(IG))`.
4. Takes the smallest set of highest-strength segments whose cumulative strength
   reaches 70%, removes candidates with consistency above 0.8, then adds the
   first and last segments as required by the appendix.
5. Runs full-CoT SFT and continues from that checkpoint with loss enabled only
   on the selected segments.

The full reasoning trace remains in the input. Selective SFT masks labels with
`-100` outside selected segments; it is selective supervision, not trace
pruning. Following the paper appendix, the first and last segments are always
supervised in addition to the IG-selected segments.

Qwen3 already contains `<think>` and `</think>` tokens. Training and evaluation
therefore use the native thinking chat template and the same `<think>` prefill;
the vocabulary is not resized. The unchanged LIMO trace is placed inside that
thinking block, and training also supervises the closing `</think>` and
`<|im_end|>` format tokens so generation can terminate normally.

## Citation

Paper: [arXiv:2602.00425](https://arxiv.org/abs/2602.00425)

```bibtex
@inproceedings{wang2026segment,
  title     = {Segment-Level Attribution for Selective Learning of Long Reasoning Traces},
  author    = {Wang, Siyuan and Liu, Yanchen and Ren, Xiang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
