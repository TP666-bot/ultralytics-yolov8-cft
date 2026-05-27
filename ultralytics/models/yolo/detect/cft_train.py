# CFT dual-stream detection trainer and validator.

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.data.build import build_dataloader
from ultralytics.data.dual_stream import build_dual_yolo_dataset
from ultralytics.data.dual_utils import check_dual_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks_dual import DualDetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class CFTDetectionTrainer(DetectionTrainer):
    """Train YOLOv8 + CFT (RGB/IR) fusion models."""

    def get_dataset(self):
        data = check_dual_det_dataset(self.args.data)
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_dual_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train" and not getattr(dataset, "rect", False)
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch=batch_size, workers=workers, shuffle=shuffle, rank=rank)

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        if "img2" in batch:
            batch["img2"] = batch["img2"].to(self.device, non_blocking=True).float() / 255
        return batch

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = DualDetectionModel(cfg, nc=self.data["nc"], ch=3, verbose=verbose and RANK == -1)
        w = weights or (self.args.pretrained if getattr(self.args, "pretrained", None) else None)
        if w:
            model.load(w)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CFTDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class CFTDetectionValidator(DetectionValidator):
    """Validate dual-stream models (RGB + IR)."""

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            self._dual_ref = unwrap_model(trainer.ema.ema if trainer.ema else trainer.model)
        elif model is not None:
            self._dual_ref = unwrap_model(model)
        else:
            self._dual_ref = None
        return super().__call__(trainer, model)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess(batch)
        if "img2" in batch and getattr(self, "_dual_ref", None) is not None:
            batch["img2"] = (batch["img2"].half() if self.args.half else batch["img2"].float()) / 255
            self._dual_ref.ir_input = batch["img2"]
        return batch
