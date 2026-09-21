import os
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class IntegratedGradientsAttribution:
    """
    Compute Integrated Gradients (IG) attributions from intermediate tokens
    to the model's final answer tokens.

    Notes:
      - We attribute the summed log-probability of answer tokens to input token embeddings.
      - The baseline is a sequence filled with `baseline_token_id` (often the pad token).
    """
    def __init__(self, model_name, gradient_checkpointing=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not self.tokenizer.is_fast:
            raise SystemExit(
                "A fast tokenizer is required for exact segment token offsets"
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("pad_token_id", self.tokenizer.pad_token_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if gradient_checkpointing:
            attention_dropout = getattr(self.model.config, "attention_dropout", 0.0) or 0.0
            dropout_layers = [
                module.p
                for module in self.model.modules()
                if isinstance(module, torch.nn.Dropout) and module.p > 0
            ]
            if attention_dropout > 0 or dropout_layers:
                raise SystemExit(
                    "Gradient checkpointing requires train mode, but this model has "
                    "non-zero dropout. Re-run with --no-gradient-checkpointing."
                )
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Hugging Face checkpointing is active only in train mode. Qwen's
            # dropout is zero, so this does not change the IG function.
            self.model.train()
            print("gradient checkpointing: enabled")
        else:
            print("gradient checkpointing: disabled")
        self.input_device = self.model.get_input_embeddings().weight.device

    @torch.no_grad()
    def _embed(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def compute_step_to_answer_attribution_integrated(self, input_ids, step_indices, answer_indices, baseline_token_id=0, steps=50):
        return self.batch_compute_step_to_answer_attribution_integrated(
            input_ids=input_ids,
            step_indices=step_indices,
            answer_indices=answer_indices,
            baseline_token_id=baseline_token_id,
            steps=steps,
            batch_size=1,
        )
        

    def batch_compute_step_to_answer_attribution_integrated(
        self,
        input_ids,
        step_indices,
        answer_indices,
        baseline_token_id=0,
        steps=50,
        batch_size=1,   
    ):
        # [1, L]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.input_device).unsqueeze(0)

        with torch.no_grad():
            input_embeddings = self._embed(input_ids)  # [1, L, D]
            baseline_embeddings = self._embed(torch.full_like(input_ids, baseline_token_id))      # [1, L, D]

        # Match new_nothingnew_2/SegmentSelectiveSFT exactly: J evenly spaced
        # interpolation points including both 0 and 1.
        alphas = torch.linspace(
            0,
            1,
            steps,
            device=self.input_device,
            dtype=input_embeddings.dtype,
        ).view(steps, 1, 1)
        total_gradients = torch.zeros_like(input_embeddings)  # [1, L, D]

        ans_start, ans_end = answer_indices
        answer_token_ids = input_ids[0, ans_start:ans_end]          # [A]

        step_pos = 0
        while step_pos < steps:
            chunk_end = min(step_pos + batch_size, steps)
            chunk_size = chunk_end - step_pos
            alphas_chunk = alphas[step_pos:chunk_end]  # [chunk_size, 1, 1]

            # Expand embeddings to [chunk_size, L, D]
            input_expand = input_embeddings.expand(chunk_size, -1, -1)
            baseline_expand = baseline_embeddings.expand(chunk_size, -1, -1)

            # Interpolation path: baseline -> input
            interpolated_embeddings = baseline_expand + alphas_chunk * (input_expand - baseline_expand).detach()
            interpolated_embeddings = interpolated_embeddings.to(dtype=input_embeddings.dtype)
            interpolated_embeddings.requires_grad_(True)

            # forward：batch = chunk_size
            self.model.zero_grad(set_to_none=True)
            # Avoid materializing [batch, sequence, vocabulary] logits. Only
            # answer positions contribute to the paper's scalar objective.
            hidden = self.model.model(
                inputs_embeds=interpolated_embeddings,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            target_hidden = hidden[:, ans_start - 1 : ans_end - 1, :]
            lm_head_device = self.model.lm_head.weight.device
            target_logits = self.model.lm_head(target_hidden.to(lm_head_device))
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)  # [chunk_size, A, V]

            # answer_token_ids: [A] -> [chunk_size, A]
            answer_ids_expand = (
                answer_token_ids.to(target_logits.device)
                .unsqueeze(0)
                .expand(chunk_size, -1)
            )

            target_log_probs = log_probs.gather(
                dim=-1,
                index=answer_ids_expand.unsqueeze(-1)
            ).squeeze(-1) # [chunk_size, A]

            loss = target_log_probs.sum()
            grads_chunk = torch.autograd.grad(
                loss,
                interpolated_embeddings,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]   # [chunk_size, L, D]

            # Sum gradients over the chunk's interpolation points
            total_gradients += grads_chunk.sum(dim=0, keepdim=True)  # -> [1, L, D]
            del hidden, target_hidden, target_logits, log_probs, target_log_probs, grads_chunk
            step_pos = chunk_end

        avg_gradients = total_gradients / steps  # [1, L, D]
        input_diff = input_embeddings - baseline_embeddings  # [1, L, D]
        attributions = (input_diff * avg_gradients).sum(dim=-1).squeeze(0)  # [L]
        norm = attributions.norm()
        if torch.isfinite(norm) and norm > 0:
            attributions = attributions / norm

        step_scores = []
        for step_idx in step_indices:
            step_attr = attributions[step_idx[0]: step_idx[1]]
            step_scores.append(step_attr.detach().cpu().float().numpy().tolist())

        del input_embeddings, baseline_embeddings, total_gradients
        torch.cuda.empty_cache()

        return step_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or local path")
    p.add_argument("--input_data", type=str, required=True, help="Path to input jsonl")
    p.add_argument("--output_data_file", type=str, required=True, help="Path to attributed JSONL")
    p.add_argument("--output_ig_file", type=str, required=True, help="Path to output IG jsonl")
    p.add_argument(
        "--output_compact_file",
        type=str,
        default="",
        help="Compact per-segment IG summary used by the comparison pipeline",
    )
    p.add_argument("--ig_steps", type=int, default=20, help="Number of IG steps")
    p.add_argument("--ig_batch_size", type=int, default=1, help="Interpolation points per forward pass")
    p.add_argument(
        "--no_gradient_checkpointing",
        "--no-gradient-checkpointing",
        dest="no_gradient_checkpointing",
        action="store_true",
        help="Use more memory but avoid activation recomputation",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume a validated partial output_data_file instead of replacing it",
    )
    # p.add_argument("--baseline_token", type=str, default="pad", choices=["pad", "zero"], help="Baseline token choice")
    return p.parse_args()


if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    args = parse_args()
    if args.ig_steps <= 0:
        raise ValueError("--ig_steps must be positive")
    if args.ig_batch_size <= 0:
        raise ValueError("--ig_batch_size must be positive")
    attribution_calculator = IntegratedGradientsAttribution(
        args.model_name,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    input_data = []
    with open(args.input_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                input_data.append(json.loads(line))
    if not input_data:
        raise SystemExit(f"No rows found in {args.input_data}")

    attribution_config = {
        "model": args.model_name,
        "ig_steps": args.ig_steps,
        "baseline_token_id": attribution_calculator.tokenizer.pad_token_id,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output_data_file))
    os.makedirs(output_dir, exist_ok=True)

    done = 0
    write_mode = "w"
    if args.resume and os.path.exists(args.output_data_file):
        with open(args.output_data_file, encoding="utf-8") as f:
            raw_lines = f.readlines()
        existing = []
        truncated_tail = False
        for line_index, line in enumerate(raw_lines):
            if not line.strip():
                continue
            try:
                existing.append(json.loads(line))
            except json.JSONDecodeError as error:
                if any(tail.strip() for tail in raw_lines[line_index + 1 :]):
                    raise ValueError(
                        f"Corrupt non-final row in {args.output_data_file}"
                    ) from error
                truncated_tail = True
                break

        if len(existing) > len(input_data):
            raise ValueError(
                "Resume output contains more rows than the current input dataset"
            )
        for index, record in enumerate(existing):
            source = input_data[index]
            if (
                record.get("question") != source.get("question")
                or record.get("segments") != source.get("segments")
                or record.get("_attribution_config") != attribution_config
                or "attribution" not in record
            ):
                raise ValueError(
                    f"Resume output does not match input/config at row {index}; "
                    "remove it or run without --resume"
                )
        missing_final_newline = bool(
            raw_lines and raw_lines[-1].strip() and not raw_lines[-1].endswith("\n")
        )
        if truncated_tail or missing_final_newline:
            with open(args.output_data_file, "w", encoding="utf-8") as f:
                for record in existing:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Repaired the final row boundary in {args.output_data_file}")
        done = len(existing)
        write_mode = "a"
        print(f"Resuming attribution at row {done}/{len(input_data)}")

    with open(args.output_data_file, write_mode, encoding="utf-8") as f:
        input_template = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for n in tqdm(range(done, len(input_data)), initial=done, total=len(input_data)):
            each_data = input_data[n]
            user_msg = input_template.format(input=each_data["question"])
            user_tokens = attribution_calculator.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=True,
                add_generation_prompt=True,
            )
            
            pred_thoughts = each_data["segments"]
            if not pred_thoughts:
                raise ValueError(f"Sample {n} has no reasoning segments")

            # Tokenize the joined trace once. BPE tokenization is not prefix
            # stable, so summing separately tokenized prefixes can move a
            # segment boundary and assign the wrong IG values.
            joined_thoughts = "".join(pred_thoughts)
            encoded_thoughts = attribution_calculator.tokenizer(
                joined_thoughts,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            assistant_tokens = encoded_thoughts["input_ids"]
            token_offsets = encoded_thoughts["offset_mapping"]

            segment_starts = []
            cursor = 0
            for segment in pred_thoughts:
                segment_starts.append(cursor)
                cursor += len(segment)
            if cursor != len(joined_thoughts):
                raise RuntimeError(f"Segment reconstruction failed for sample {n}")

            first_token = {}
            last_token = {}
            segment_index = 0
            for token_index, (char_start, char_end) in enumerate(token_offsets):
                if char_end <= char_start:
                    continue
                while (
                    segment_index + 1 < len(segment_starts)
                    and char_start >= segment_starts[segment_index + 1]
                ):
                    segment_index += 1
                first_token.setdefault(segment_index, token_index)
                last_token[segment_index] = token_index
            assistant_token_spans = [
                (first_token[index], last_token[index] + 1)
                if index in first_token
                else (0, 0)
                for index in range(len(pred_thoughts))
            ]

            # Shift segment spans by user prompt length
            offset = len(user_tokens)
            adjusted_spans = [
                (start + offset, end + offset) if end > start else (0, 0)
                for start, end in assistant_token_spans
            ]

            answer_string = "</think> So, the final answer is \\boxed{" + each_data['answer'] + "}"
            
            # Preserve the comparison folder's token-level boxed-answer span
            # heuristic so attribution selects the same target tokens.
            answer_tokens = attribution_calculator.tokenizer(
                answer_string, add_special_tokens=False
            )["input_ids"]
            answer_tokens_split = attribution_calculator.tokenizer.convert_ids_to_tokens(
                answer_tokens
            )
            ans_start = None
            for token_index, token in enumerate(answer_tokens_split):
                if "boxed" in token:
                    ans_start = token_index + 1
            if ans_start is None or "{" not in answer_tokens_split[ans_start]:
                raise ValueError(f"Could not locate boxed answer for sample {n}")
            ans_end = len(answer_tokens_split)
            is_end = False
            stack = 0
            for token_index in range(ans_start, len(answer_tokens_split)):
                token = answer_tokens_split[token_index]
                if "{" in token or "}" in token:
                    for character in token:
                        if character == "{":
                            stack += 1
                        elif character == "}":
                            stack -= 1
                            if stack == 0:
                                ans_end = token_index
                                is_end = True
                                break
                    if is_end:
                        break
            
            full_tokens = user_tokens + assistant_tokens + answer_tokens
            answer_offset = len(user_tokens) + len(assistant_tokens)
            answer_indices = (
                ans_start + 1 + answer_offset,
                ans_end + answer_offset,
            )

            importance_scores = attribution_calculator.batch_compute_step_to_answer_attribution_integrated(
                full_tokens,
                adjusted_spans,
                answer_indices,
                baseline_token_id=attribution_calculator.tokenizer.pad_token_id,
                steps=args.ig_steps,
                batch_size=args.ig_batch_size,
            )
            input_data[n]["attribution"] = importance_scores
            input_data[n]["_attribution_config"] = attribution_config

            f.write(json.dumps(input_data[n], ensure_ascii=False) + '\n')
            f.flush()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_ig_file)), exist_ok=True)
    compact_path = args.output_compact_file
    if not compact_path:
        base = args.output_ig_file
        compact_path = (base[:-6] if base.endswith(".jsonl") else base) + "_compact.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(compact_path)), exist_ok=True)
    row_count = 0
    with open(args.output_data_file, encoding="utf-8") as source, open(
        args.output_ig_file, "w", encoding="utf-8"
    ) as destination, open(compact_path, "w", encoding="utf-8") as compact_output:
        for line in source:
            if not line.strip():
                continue
            attribution = json.loads(line)["attribution"]
            destination.write(json.dumps(attribution, ensure_ascii=False) + "\n")
            compact = []
            for segment in attribution:
                token_count = len(segment)
                sum_abs = float(np.sum(np.abs(segment))) if token_count else 0.0
                sum_signed = float(np.sum(segment)) if token_count else 0.0
                compact.append(
                    [token_count, round(sum_abs, 8), round(sum_signed, 8)]
                )
            compact_output.write(json.dumps({"segments": compact}) + "\n")
            row_count += 1
    if row_count != len(input_data):
        raise RuntimeError(
            f"Attributed row count mismatch: wrote {row_count}, expected {len(input_data)}"
        )
    print(f"Wrote {row_count} attribution rows to {args.output_ig_file}")
    print(f"Wrote compact attribution rows to {compact_path}")
