#!/usr/bin/env python3
"""Create and validate the fixed conversation-level dataset split."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.empathy_dataset import (  # noqa: E402
    load_or_create_split_manifest,
    validate_split_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        default="/mnt/HDD1/bk_dataset/generated_text/train_final_with_reference_images.json",
    )
    parser.add_argument(
        "--output", default="./artifacts/splits/conv_id_seed42.json"
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.json_path, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    manifest = load_or_create_split_manifest(
        raw_data, args.output, args.val_ratio, args.test_ratio, args.seed
    )
    validate_split_manifest(manifest, raw_data)
    sets = {name: set(ids) for name, ids in manifest["splits"].items()}
    report = {
        "manifest": os.path.abspath(args.output),
        "source_conv_count": manifest["source_conv_count"],
        "split_conv_counts": {name: len(ids) for name, ids in sets.items()},
        "overlap": {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        },
        "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
