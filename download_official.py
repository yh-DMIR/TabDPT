#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official TabDPT weight to a local directory.")
    parser.add_argument("--repo-id", default="Layer6/TabDPT")
    parser.add_argument("--filename", default="tabdpt1_1.safetensors")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out-dir", default="ckpt/TabDPT")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.filename} from {args.repo_id}@{args.revision} ...")
    cache_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
    )

    final_path = out_dir / args.filename
    shutil.copy2(cache_path, final_path)
    print(f"Saved to: {final_path.resolve()}")


if __name__ == "__main__":
    main()
