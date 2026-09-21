#!/usr/bin/env python3
"""Pre-download model snapshots into the configured Hugging Face cache."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--revision", default="")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    for model in args.models:
        kwargs = {"repo_id": model}
        if args.cache_dir:
            kwargs["cache_dir"] = args.cache_dir
        if args.revision:
            kwargs["revision"] = args.revision
        path = snapshot_download(**kwargs)
        print(f"{model} -> {path}")


if __name__ == "__main__":
    main()

