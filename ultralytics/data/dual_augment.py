# Synchronized v8 augmentations for paired RGB + IR (LLVIP CFT).

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import cv2
import numpy as np

from ultralytics.data.augment import (
    Albumentations,
    Compose,
    CopyPaste,
    CutMix,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.utils.instance import Instances


class DualMosaic(Mosaic):
    """Mosaic augmentation with the same tile layout applied to ``img2`` (IR)."""

    def _mosaic4(self, labels: dict[str, Any]) -> dict[str, Any]:
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        use_ir = "img2" in labels
        img4_ir = None
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            img_ir = labels_patch.get("img2", img) if use_ir else None
            h, w = labels_patch.pop("resized_shape")

            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                if use_ir:
                    c_ir = 1 if img_ir.ndim == 2 else img_ir.shape[2]
                    img4_ir = np.full((s * 2, s * 2, c_ir), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if use_ir:
                if img_ir.ndim == 2:
                    img_ir = img_ir[..., None]
                img4_ir[y1a:y2a, x1a:x2a] = img_ir[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)

        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        if use_ir:
            final_labels["img2"] = img4_ir
        return final_labels


class DualRandomPerspective(RandomPerspective):
    """Apply the same affine warp to RGB and IR feature maps."""

    def _warp_img2(self, img2: np.ndarray, border: tuple[int, int], M: np.ndarray) -> np.ndarray:
        if img2.ndim == 2:
            img2 = img2[..., None]
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if self.perspective:
                img2 = cv2.warpPerspective(img2, M, dsize=self.size, borderValue=(114, 114, 114))
            else:
                img2 = cv2.warpAffine(img2, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        if img2.ndim == 2:
            img2 = img2[..., None]
        return np.ascontiguousarray(img2)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img2 = labels.pop("img2", None)
        if self.pre_transform and "mosaic_border" not in labels:
            if img2 is not None:
                labels["img2"] = img2
            labels = self.pre_transform(labels)
            img2 = labels.pop("img2", None)

        labels.pop("ratio_pad", None)
        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2
        img, M, scale = self.affine_transform(img, border)
        if img2 is not None:
            img2 = self._warp_img2(img2, border, M)

        bboxes = self.apply_bboxes(instances.bboxes, M)
        segments = instances.segments
        keypoints = instances.keypoints
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)
        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        new_instances.clip(*self.size)
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        if img2 is not None:
            labels["img2"] = img2
        labels["resized_shape"] = img.shape[:2]
        return labels


class DualRandomFlip(RandomFlip):
    """Flip RGB/IR with identical random decisions."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img2 = labels.pop("img2", None)
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            if img2 is not None:
                img2 = np.flipud(img2)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            if img2 is not None:
                img2 = np.fliplr(img2)
            instances.fliplr(w)

        labels["img"] = np.ascontiguousarray(img)
        if img2 is not None:
            labels["img2"] = np.ascontiguousarray(img2)
        labels["instances"] = instances
        return labels


class DualRandomHSV(RandomHSV):
    """HSV on RGB; optional identical HSV on IR (YOLOv5 multimodal applies both)."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img2 = labels.get("img2")
        labels = super().__call__(labels)
        if img2 is not None:
            labels_ir = {"img": img2}
            labels_ir = super().__call__(labels_ir)
            labels["img2"] = labels_ir["img"]
        return labels


def dual_v8_transforms(dataset, imgsz: int, hyp, stretch: bool = False):
    """YOLOv8 training augmentations with RGB/IR spatial sync (no MixUp/CutMix on dual stream)."""
    hyp = deepcopy(hyp)
    hyp.mixup = 0.0
    hyp.cutmix = 0.0

    mosaic = DualMosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = DualRandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([DualMosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )

    flip_idx = dataset.data.get("flip_idx", [])
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0

    return Compose(
        [
            pre_transform,
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None)),
            DualRandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            DualRandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            DualRandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )
