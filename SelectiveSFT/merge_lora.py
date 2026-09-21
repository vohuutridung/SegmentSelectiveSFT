#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model for vLLM evaluation."""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--base_model", default="")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--device_map", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = os.path.join(args.adapter, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise SystemExit(f"Not a LoRA adapter directory: {args.adapter}")
    with open(adapter_config_path, encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    base_model = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise SystemExit("Cannot determine the base model; pass --base_model")
    output_dir = args.output_dir or args.adapter.rstrip("/") + "-merged"

    print("base model :", base_model)
    print("adapter    :", args.adapter)
    print("output     :", output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    tokenizer_source = (
        args.adapter
        if os.path.isfile(os.path.join(args.adapter, "tokenizer_config.json"))
        else base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
