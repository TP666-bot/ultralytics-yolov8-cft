#!/usr/bin/env python3
"""Verify VEDAI RGB/IR list pairing (same length, consistent scene IDs)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.data.dual_stream import load_img_list


def scene_id(path: str) -> str:
    name = Path(path).name
    m = re.match(r"(\d+)_", name)
    return m.group(1) if m else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ultralytics/cfg/datasets/vedai_dual.yaml")
    args = parser.parse_args()

    from ultralytics.data.dual_utils import check_dual_det_dataset

    data = check_dual_det_dataset(args.data)
    for split, rgb_k, ir_k in (
        ("train", "train_rgb", "train_ir"),
        ("val", "val_rgb", "val_ir"),
    ):
        rgb = load_img_list(data[rgb_k])
        ir = load_img_list(data[ir_k])
        print(f"{split}: rgb={len(rgb)} ir={len(ir)}")
        if len(rgb) != len(ir):
            print(f"  FAIL length mismatch")
            continue
        bad = [(r, i) for r, i in zip(rgb, ir) if scene_id(r) != scene_id(i)]
        if bad:
            print(f"  FAIL {len(bad)} scene-id mismatches (first 3):")
            for r, i in bad[:3]:
                print(f"    {Path(r).name} vs {Path(i).name}")
        else:
            print(f"  OK paired by sorted index")


if __name__ == "__main__":
    main()
