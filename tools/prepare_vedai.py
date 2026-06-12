#!/usr/bin/env python3
"""Prepare VEDAI for YOLOv8+CFT (DocF-compatible layout).

Expected raw download (https://downloads.greyc.fr/vedai/) in --src:
  Vehicules1024/          # *_co.png, *_ir.png
  fold01.txt              # train image IDs (one per line)
  fold01test.txt          # val image IDs
  annotation1024_cleaned.txt

Output (default: <WORKSPACE>/VEDAI/Vehicules1024/Images/):
  color/00000000_co.png + 00000000_co.txt
  ir/00000000_ir.png
  color/fold01.txt, fold01test.txt
  ir/fold01.txt, fold01test.txt

Then: bash tools/setup_vedai.sh && python tools/check_vedai_pairs.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# VEDAI original class id -> YOLO 0-based (9 classes, aligned with vedai_dual.yaml)
CLASS_MAP = {1: 0, 2: 1, 4: 3, 5: 4, 7: 7, 8: 7, 9: 8, 10: 7, 11: 5, 23: 2, 31: 6}

ANNOT_COLUMNS = [
    "Image_ID",
    "x_centre",
    "y_centre",
    "orientation",
    "corner1_x",
    "corner2_x",
    "corner3_x",
    "corner4_x",
    "corner1_y",
    "corner2_y",
    "corner3_y",
    "corner4_y",
    "class",
    "is_contained",
    "is_occluded",
]


def parse_image_id(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    name = Path(line.replace("./", "")).name
    m = re.search(r"(\d{8})", name)
    return m.group(1) if m else ""


def read_id_list(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        i = parse_image_id(line)
        if i:
            ids.append(i)
    return ids


def load_annotations(annot_path: Path, img_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    from PIL import Image

    labels: dict[str, list[tuple[int, float, float, float, float]]] = {}
    size_cache: dict[str, tuple[int, int]] = {}

    for line in annot_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < len(ANNOT_COLUMNS):
            continue
        rec = dict(zip(ANNOT_COLUMNS, parts[: len(ANNOT_COLUMNS)]))
        if int(float(rec["is_contained"])) != 1:
            continue
        cls_raw = int(float(rec["class"]))
        if cls_raw not in CLASS_MAP:
            continue
        cls = CLASS_MAP[cls_raw]
        img_id = str(int(float(rec["Image_ID"]))).zfill(8)
        co = img_dir / f"{img_id}_co.png"
        if img_id not in size_cache:
            if not co.exists():
                continue
            with Image.open(co) as im:
                size_cache[img_id] = im.size
        w, h = size_cache[img_id]
        xs = [min(max(float(rec[k]), 0), w) for k in ("corner1_x", "corner2_x", "corner3_x", "corner4_x")]
        ys = [min(max(float(rec[k]), 0), h) for k in ("corner1_y", "corner2_y", "corner3_y", "corner4_y")]
        bw = (max(xs) - min(xs)) / w
        bh = (max(ys) - min(ys)) / h
        xc = (min(xs) + max(xs)) / 2 / w
        yc = (min(ys) + max(ys)) / 2 / h
        labels.setdefault(img_id, []).append((cls, xc, yc, bw, bh))
    return labels


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def write_fold_list(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(f"./{n}" for n in names) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare VEDAI for Y8-CFT dual training")
    parser.add_argument(
        "--src",
        required=True,
        help="Raw VEDAI folder containing Vehicules1024/, fold01.txt, annotation1024_cleaned.txt",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output root (default: <WORKSPACE>/VEDAI/Vehicules1024)",
    )
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinks")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    img_dir = src / "Vehicules1024"
    if not img_dir.is_dir():
        img_dir = src  # allow src=Vehicules1024 directly

    for req in ("fold01.txt", "fold01test.txt", "annotation1024_cleaned.txt"):
        if not (src / req).exists() and not (img_dir.parent / req).exists():
            # fold files may live next to Vehicules1024
            pass
    fold_train = src / "fold01.txt"
    fold_val = src / "fold01test.txt"
    annot = src / "annotation1024_cleaned.txt"
    if not fold_train.exists() and (img_dir.parent / "fold01.txt").exists():
        fold_train = img_dir.parent / "fold01.txt"
        fold_val = img_dir.parent / "fold01test.txt"
        annot = img_dir.parent / "annotation1024_cleaned.txt"

    missing = [p.name for p in (fold_train, fold_val, annot) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing in {src}: {missing}\n"
            "Download VEDAI from https://downloads.greyc.fr/vedai/ "
            "(need Vehicules1024, fold01.txt, fold01test.txt, annotation1024_cleaned.txt)"
        )

    out_root = Path(args.out).resolve() if args.out else (ROOT.parent / "VEDAI" / "Vehicules1024").resolve()

    color_dir = out_root / "Images" / "color"
    ir_dir = out_root / "Images" / "ir"
    color_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)

    train_ids = read_id_list(fold_train)
    val_ids = read_id_list(fold_val)
    all_ids = sorted(set(train_ids) | set(val_ids))
    print(f"fold01: {len(train_ids)} train, fold01test: {len(val_ids)} val")

    labels = load_annotations(annot, img_dir)

    for img_id in all_ids:
        co_src = img_dir / f"{img_id}_co.png"
        ir_src = img_dir / f"{img_id}_ir.png"
        if not co_src.exists() or not ir_src.exists():
            print(f"WARN skip {img_id}: missing co/ir png")
            continue
        co_dst = color_dir / f"{img_id}_co.png"
        ir_dst = ir_dir / f"{img_id}_ir.png"
        link_or_copy(co_src, co_dst, args.copy)
        link_or_copy(ir_src, ir_dst, args.copy)
        lbl = labels.get(img_id, [])
        label_path = color_dir / f"{img_id}_co.txt"
        if lbl:
            lines = [f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for c, xc, yc, bw, bh in lbl]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif not label_path.exists():
            label_path.touch()

    write_fold_list(color_dir / "fold01.txt", [f"{i}_co.png" for i in train_ids])
    write_fold_list(color_dir / "fold01test.txt", [f"{i}_co.png" for i in val_ids])
    write_fold_list(ir_dir / "fold01.txt", [f"{i}_ir.png" for i in train_ids])
    write_fold_list(ir_dir / "fold01test.txt", [f"{i}_ir.png" for i in val_ids])

    print(f"Prepared VEDAI at: {out_root}")
    print("Next:")
    print(f"  cd {ROOT}")
    print("  bash tools/setup_vedai.sh")
    print("  python tools/check_vedai_pairs.py")


if __name__ == "__main__":
    main()
