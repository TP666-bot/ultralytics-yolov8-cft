#!/usr/bin/env python3
"""Partially load GPT weights from YOLOv5 CFT checkpoint into YOLOv8 CFT model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks_dual import DualDetectionModel


def _ensure_yolov5_import_path() -> None:
    """YOLOv5 CFT .pt pickles reference ``models.*``; add sibling repo if present."""
    for candidate in (
        ROOT.parent / "multispectral-object-detection",
        ROOT / "multispectral-object-detection",
    ):
        if (candidate / "models" / "yolo.py").exists() or (candidate / "models" / "common.py").exists():
            p = str(candidate.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y8-cfg", required=True)
    parser.add_argument("--y8-weights", default="yolov8l.pt")
    parser.add_argument("--y5-cft", required=True)
    parser.add_argument("--out", default="runs/llvip/y8_cft_init.pt")
    args = parser.parse_args()

    model = DualDetectionModel(args.y8_cfg, ch=3, verbose=True)
    model.load(args.y8_weights)

    _ensure_yolov5_import_path()
    ckpt = torch.load(args.y5_cft, map_location="cpu", weights_only=False)
    y5_sd = ckpt["model"].float().state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = model.state_dict()
    n = 0
    keys = ("GPT", "Add2", "trans_blocks", "pos_emb", "Add.")
    for k, v in y5_sd.items():
        if not any(x in k for x in keys):
            continue
        suffix = k.split("model.", 1)[-1]
        for tk in state:
            if tk.endswith(suffix) or suffix in tk:
                if state[tk].shape == v.shape:
                    state[tk] = v.clone()
                    n += 1
                    break
    model.load_state_dict(state, strict=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.float(), "epoch": -1}, out)
    print(f"Saved {out} with {n} GPT-related tensors from Y5 CFT")


if __name__ == "__main__":
    main()
