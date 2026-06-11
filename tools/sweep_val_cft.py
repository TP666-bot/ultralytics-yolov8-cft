#!/usr/bin/env python3
"""Sweep conf/iou for val_cft on a fixed checkpoint (reuse model + dataloader)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.detect.cft_train import CFTDetectionTrainer
from ultralytics.utils import DEFAULT_CFG


def parse_overrides(argv: list[str]) -> dict:
    overrides = {"mode": "val", "plots": False, "verbose": False}
    sweeps: list[tuple[float, float]] = []
    model = None
    for arg in argv:
        if arg.startswith("sweep="):
            for pair in arg.split("=", 1)[1].split(","):
                conf_s, iou_s = pair.split(":")
                sweeps.append((float(conf_s), float(iou_s)))
        elif "=" in arg:
            k, v = arg.split("=", 1)
            if k == "model":
                model = v
            elif v.lower() in {"true", "false"}:
                overrides[k] = v.lower() == "true"
            elif v.replace(".", "", 1).isdigit():
                overrides[k] = float(v) if "." in v else int(v)
            else:
                overrides[k] = v
    if model:
        overrides["model"] = model
    if not sweeps:
        sweeps = [(0.001, 0.5), (0.001, 0.6), (0.0005, 0.5), (0.001, 0.55)]
    return overrides, sweeps


def main():
    overrides, sweeps = parse_overrides(sys.argv[1:])
    overrides.setdefault("data", "ultralytics/cfg/datasets/llvip_dual.yaml")
    overrides.setdefault("task", "detect")
    overrides.setdefault(
        "model",
        "runs/detect/runs/llvip/y8_cft_v2/weights/best.pt",
    )
    trainer = CFTDetectionTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer.setup_val()

    print(f"Model: {overrides['model']}")
    print(f"{'conf':>8} {'iou':>6} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 54)
    for conf, iou in sweeps:
        trainer.args.conf = conf
        trainer.args.iou = iou
        trainer.validator.args.conf = conf
        trainer.validator.args.iou = iou
        metrics = trainer.validator(trainer=trainer)
        p = metrics.get("metrics/precision(B)", metrics.get("precision", 0))
        r = metrics.get("metrics/recall(B)", metrics.get("recall", 0))
        m50 = metrics.get("metrics/mAP50(B)", metrics.get("mAP50", 0))
        m5095 = metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95", 0))
        print(f"{conf:8.4f} {iou:6.2f} {p:8.4f} {r:8.4f} {m50:8.4f} {m5095:10.4f}")


if __name__ == "__main__":
    main()
