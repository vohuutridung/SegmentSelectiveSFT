#!/usr/bin/env python3
"""Selective SFT (or optional full-CoT SFT) for the experiment recipe."""

import argparse
import gc
import os
import sys

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")

# Unsloth must be imported before TRL so its compatibility patches are active.
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from segment_utils import SEGMENT_MODES, split_segments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default=os.path.join(ROOT, "data/s1k/train.jsonl"))
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset_num_proc", type=int, default=2)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--mask", action="store_true", help="Use segment-selective labels")
    parser.add_argument("--segment_mode", choices=SEGMENT_MODES, default="paragraph")
    parser.add_argument(
        "--think_prefix",
        choices=("none", "plain"),
        default="none",
        help="Match the comparison folder: none adds no manual <think> prefix",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Qwen3 thinking mode; disabled to preserve the Qwen2.5 prompt behavior",
    )
    args = parser.parse_args()
    args.target_modules = [
        module.strip() for module in args.target_modules.split(",") if module.strip()
    ]
    return args


def segment_char_bounds(segments, offset):
    bounds = []
    cursor = offset
    for segment in segments:
        bounds.append((cursor, cursor + len(segment)))
        cursor += len(segment)
    return bounds


def load_training_dataset(path_or_id, split):
    if os.path.isfile(path_or_id) or path_or_id.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=path_or_id, split="train")
    return load_dataset(path_or_id, split=split)


def main():
    args = parse_args()
    if args.max_seq_length <= 0:
        raise SystemExit("--max_seq_length must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("--learning_rate must be positive")
    if args.per_device_train_batch_size <= 0:
        raise SystemExit("--per_device_train_batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient_accumulation_steps must be positive")
    if args.load_in_4bit and args.full_finetune:
        raise SystemExit(
            "--load_in_4bit is supported only by the LoRA/QLoRA path"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=args.full_finetune,
    )
    if not tokenizer.is_fast:
        raise SystemExit("A fast tokenizer is required for exact character-offset masking")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_qwen3 = getattr(model.config, "model_type", "").lower().startswith("qwen3")
    if not is_qwen3:
        print(
            "Warning: the default workflow is designed for Qwen3; loaded model_type=%r"
            % getattr(model.config, "model_type", None)
        )

    use_gradient_checkpointing = not args.no_gradient_checkpointing
    if args.full_finetune:
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.gradient_checkpointing_disable()
        trainer_gradient_checkpointing = use_gradient_checkpointing
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=args.target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=(
                "unsloth" if use_gradient_checkpointing else False
            ),
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )
        # Unsloth owns checkpointing for the adapter path.
        trainer_gradient_checkpointing = False
    model.config.use_cache = False

    dataset = load_training_dataset(args.data_names, args.split)
    if args.mask and "selected_spans_ids" not in dataset.column_names:
        raise SystemExit(
            "--mask requires selected_spans_ids from Attribution/get_important_segments.py"
        )

    skipped = {"no_supervised_tokens": 0}

    def formatting_prompts_func(examples):
        questions = examples["question"]
        outputs = examples["solution"]
        selected_batch = examples.get("selected_spans_ids") or [[] for _ in questions]
        stored_segments_batch = examples.get("segments") or [None for _ in questions]

        input_ids_list = []
        labels_list = []
        for question, output, selected_ids, stored_segments in zip(
            questions, outputs, selected_batch, stored_segments_batch
        ):
            messages = [
                {
                    "role": "user",
                    "content": question
                    + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
                }
            ]
            chat_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            # The comparison folder uses Qwen2.5 with think_prefix=none, so its
            # effective prompt has no thinking block. Disable Qwen3's otherwise
            # enabled-by-default thinking mode to preserve that behavior.
            if is_qwen3:
                chat_kwargs["enable_thinking"] = args.enable_thinking
            prompt_text = tokenizer.apply_chat_template(messages, **chat_kwargs)
            think_prefix = "<think>\n" if args.think_prefix == "plain" else ""
            # Append the raw s1K trace after the native assistant prefix,
            # without an added EOS or thinking suffix.
            full_text = prompt_text + think_prefix + output
            response_char = len(prompt_text) + len(think_prefix)

            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=args.max_seq_length,
                return_offsets_mapping=True,
            )
            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            labels = [-100] * len(input_ids)

            bounds = None
            keep = None
            if args.mask:
                segments = split_segments(output, args.segment_mode)
                if stored_segments is not None and list(stored_segments) != segments:
                    raise ValueError(
                        "Stored segments do not match --segment_mode=%s" % args.segment_mode
                    )
                # Match the comparison repo: always keep first, penultimate,
                # and final segments in the optional selective-SFT path.
                keep = {
                    0,
                    len(segments) - 2,
                    len(segments) - 1,
                    *[int(index) for index in selected_ids],
                }
                keep = {index for index in keep if 0 <= index < len(segments)}
                bounds = segment_char_bounds(segments, response_char)

            supervised = 0
            for token_index, (char_start, char_end) in enumerate(offsets):
                if char_end <= char_start or char_start < response_char:
                    continue
                if bounds is None:
                    labels[token_index] = input_ids[token_index]
                    supervised += 1
                    continue
                for segment_index, (segment_start, segment_end) in enumerate(bounds):
                    if segment_start <= char_start < segment_end:
                        if segment_index in keep:
                            labels[token_index] = input_ids[token_index]
                            supervised += 1
                        break

            if supervised == 0:
                skipped["no_supervised_tokens"] += 1
                continue
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    original_columns = dataset.column_names
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=original_columns,
        load_from_cache_file=False,
        desc="Tokenizing and building selective labels",
    )
    if not len(dataset):
        raise SystemExit("No training samples remain after tokenization/truncation")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    effective_batch = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * int(os.environ.get("WORLD_SIZE", "1"))
    )
    print("=" * 72)
    print("model             :", args.model_name_or_path)
    print("mode              :", "selective" if args.mask else "full-CoT")
    print("finetuning        :", "full parameters" if args.full_finetune else "LoRA")
    if not args.full_finetune:
        print(
            "LoRA              : r=%d alpha=%d dropout=%s modules=%s"
            % (
                args.lora_r,
                args.lora_alpha,
                args.lora_dropout,
                ",".join(args.target_modules),
            )
        )
    print("samples           :", len(dataset), "skipped:", skipped)
    print("max sequence      :", args.max_seq_length)
    print("effective batch   :", effective_batch)
    print("epochs / lr       :", args.epochs, "/", args.learning_rate)
    print(
        "optimizer         : %s betas=(%s,%s) eps=%s wd=%s"
        % (
            args.optim,
            args.adam_beta1,
            args.adam_beta2,
            args.adam_epsilon,
            args.weight_decay,
        )
    )
    print(
        "scheduler         : %s warmup=%s"
        % (args.lr_scheduler_type, args.warmup_ratio)
    )
    print("output            :", args.output_dir)
    print("=" * 72)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            max_length=args.max_seq_length,
            dataset_num_proc=args.dataset_num_proc,
            packing=False,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            lr_scheduler_type=args.lr_scheduler_type,
            output_dir=args.output_dir,
            optim=args.optim,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            seed=args.seed,
            report_to=args.report_to,
            run_name=os.path.basename(os.path.abspath(args.output_dir)),
            save_strategy="epoch",
            overwrite_output_dir=True,
            save_total_limit=3,
            save_only_model=True,
            gradient_checkpointing=trainer_gradient_checkpointing,
            group_by_length=args.group_by_length,
            max_grad_norm=args.max_grad_norm,
        ),
    )
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    model.config.use_cache = True
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
