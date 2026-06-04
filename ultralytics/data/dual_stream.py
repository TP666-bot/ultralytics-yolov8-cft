# LLVIP dual-stream (RGB + IR) dataset for CFT / Add fusion training.

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.dual_augment import dual_v8_transforms
from ultralytics.utils import colorstr


def rgb_path_to_ir(rgb_path: str) -> str:
    """Map LLVIP visible path to paired infrared path."""
    p = rgb_path.replace(f"{os.sep}visible{os.sep}", f"{os.sep}infrared{os.sep}")
    if p == rgb_path:
        p = rgb_path.replace("/visible/", "/infrared/").replace("\\visible\\", "\\infrared\\")
    return p


class DualFormat(Format):
    """Format RGB annotations and IR image tensor for dual-stream training."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img2 = labels.pop("img2", None)
        out = super().__call__(labels)
        if img2 is not None:
            out["img2"] = self._format_img(img2)
        return out


class YOLODualStreamDataset(YOLODataset):
    """YOLO detect dataset with paired RGB + IR images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_files_ir = [rgb_path_to_ir(f) for f in self.im_files]

    def load_image(self, i: int, rect_mode: bool = True):
        im_rgb, hw0, hw = super().load_image(i, rect_mode=rect_mode)
        f_ir = self.im_files_ir[i]
        im_ir = cv2.imread(f_ir)
        if im_ir is None:
            raise FileNotFoundError(f"IR image not found: {f_ir}")
        if im_ir.ndim == 2:
            im_ir = im_ir[..., None]
        h0, w0 = im_ir.shape[:2]
        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                im_ir = cv2.resize(im_ir, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im_ir = cv2.resize(im_ir, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if im_ir.shape[:2] != im_rgb.shape[:2]:
            im_ir = cv2.resize(im_ir, (im_rgb.shape[1], im_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return im_rgb, im_ir, hw0, hw

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        im_rgb, im_ir, ori_shape, resized_shape = self.load_image(index)
        label["img"] = im_rgb
        label["img2"] = im_ir
        label["ori_shape"] = ori_shape
        label["resized_shape"] = resized_shape
        label["ratio_pad"] = (resized_shape[0] / ori_shape[0], resized_shape[1] / ori_shape[1])
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def build_transforms(self, hyp=None):
        """Training: synced v8 mosaic/affine/flip/HSV; val: letterbox only."""
        hyp = hyp or self.hyp
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = 0.0
            hyp.cutmix = 0.0
            transforms = dual_v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            DualFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=getattr(hyp, "mask_ratio", 4),
                mask_overlap=getattr(hyp, "overlap_mask", True),
            )
        )
        return transforms

    @staticmethod
    def collate_fn(batch):
        new_batch = YOLODataset.collate_fn(batch)
        if batch and "img2" in batch[0]:
            new_batch["img2"] = torch.stack([b["img2"] for b in batch], 0)
        return new_batch


def build_dual_yolo_dataset(
    cfg, img_path: str, batch: int, data: dict, mode: str = "train", rect: bool = False, stride: int = 32
):
    """Build paired RGB/IR YOLO dataset."""
    return YOLODualStreamDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=False,  # dual RGB/IR must share spatial size for Add/GPT
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
