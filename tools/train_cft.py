#!/usr/bin/env python3
"""Train YOLOv8 + CFT (dual RGB/IR) on LLVIP.

Example:
  python tools/train_cft.py \\
    model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \\
    data=ultralytics/cfg/datasets/llvip_dual.yaml \\
    pretrained=yolov8l.pt epochs=200 imgsz=1024 batch=4 device=0 \\
    project=runs/llvip name=y8_cft
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
    overrides = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if v.lower() in {"true", "false"}:
                v = v.lower() == "true"
            elif v.isdigit():
                v = int(v)
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass
            overrides[k] = v
    return overrides


def main():
    overrides = parse_overrides(sys.argv[1:])
    if "model" not in overrides:
        overrides["model"] = "ultralytics/cfg/models/v8/yolov8l_fusion_add_llvip.yaml"
    if "data" not in overrides:
        overrides["data"] = "ultralytics/cfg/datasets/llvip_dual.yaml"
    if "pretrained" not in overrides and not str(overrides.get("model", "")).endswith(".pt"):
        overrides["pretrained"] = "yolov8l.pt"
    overrides.setdefault("task", "detect")
    trainer = CFTDetectionTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
