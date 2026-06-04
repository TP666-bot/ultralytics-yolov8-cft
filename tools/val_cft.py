#!/usr/bin/env python3
"""Validate YOLOv8 + CFT dual-stream checkpoint on LLVIP test split.

Example:
  python tools/val_cft.py \\
    model=runs/llvip/y8_cft/weights/best.pt \\
    data=ultralytics/cfg/datasets/llvip_dual.yaml \\
    imgsz=1024 batch=4 device=0 conf=0.001 iou=0.5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.detect.cft_train import CFTDetectionTrainer
from ultralytics.utils import DEFAULT_CFG


def parse_overrides(argv: list[str]) -> dict:
    overrides = {"mode": "val"}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if v.lower() in {"true", "false"}:
                v = v.lower() == "true"
            elif v.replace(".", "", 1).isdigit():
                v = float(v) if "." in v else int(v)
            overrides[k] = v
    return overrides


def main():
    overrides = parse_overrides(sys.argv[1:])
    overrides.setdefault("data", "ultralytics/cfg/datasets/llvip_dual.yaml")
    overrides.setdefault("task", "detect")
    overrides.setdefault("conf", 0.001)
    overrides.setdefault("iou", 0.5)
    trainer = CFTDetectionTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer.setup_val()
    trainer.validate()


if __name__ == "__main__":
    main()
