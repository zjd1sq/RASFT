#!/usr/bin/env python3
"""Download ASFT dataset from Hugging Face.

Usage:
  python download_data.py --output_dir data

This script downloads the dataset files to a local directory.
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ASFT dataset from Hugging Face")
    parser.add_argument("--output_dir", default="data", help="Directory to save downloaded files")
    args = parser.parse_args()

    repo_id = "chichi56/ASFT"
    files = [
        "numina_cot_10k.jsonl",
      
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in files:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=fname,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded {fname} -> {path}")


if __name__ == "__main__":
    main()
