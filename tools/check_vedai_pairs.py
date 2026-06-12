#!/usr/bin/env python3
"""Verify VEDAI RGB/IR list pairing (same length, consistent scene IDs)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scene_id(path: str) -> str:
    name = Path(path).name
    m = re.match(r"(\d+)_", name)
    return m.group(1) if m else name


def yaml_paths_ok(data_yaml: Path) -> tuple[bool, str]:
    from ultralytics.utils import YAML

    data = YAML.load(data_yaml, append_filename=True)
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = root.resolve()  # same as check_dual_det_dataset (relative to cwd)
    for key in ("train_rgb", "val_rgb", "train_ir", "val_ir"):
        rel = data.get(key, "")
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        if not p.exists():
            return False, str(p)
    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ultralytics/cfg/datasets/vedai_dual.yaml")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml

    ok, missing = yaml_paths_ok(data_yaml)
    if not ok:
        print("VEDAI data not ready.")
        print(f"  Missing: {missing}")
        print()
        print("Steps:")
        print("  1. Download https://downloads.greyc.fr/vedai/")
        print("  2. python tools/prepare_vedai.py --src /path/to/vedai_raw")
        print("  3. bash tools/setup_vedai.sh")
        print("  4. python tools/check_vedai_pairs.py")
        sys.exit(1)

    from ultralytics.data.dual_stream import load_img_list
    from ultralytics.data.dual_utils import check_dual_det_dataset

    data = check_dual_det_dataset(str(data_yaml))
    for split, rgb_k, ir_k in (
        ("train", "train_rgb", "train_ir"),
        ("val", "val_rgb", "val_ir"),
    ):
        rgb = load_img_list(data[rgb_k])
        ir = load_img_list(data[ir_k])
        print(f"{split}: rgb={len(rgb)} ir={len(ir)}")
        if len(rgb) != len(ir):
            print("  FAIL length mismatch")
            sys.exit(1)
        bad = [(r, i) for r, i in zip(rgb, ir) if scene_id(r) != scene_id(i)]
        if bad:
            print(f"  FAIL {len(bad)} scene-id mismatches (first 3):")
            for r, i in bad[:3]:
                print(f"    {Path(r).name} vs {Path(i).name}")
            sys.exit(1)
        print("  OK paired by sorted index")
    print("VEDAI pairing check passed.")


if __name__ == "__main__":
    main()
