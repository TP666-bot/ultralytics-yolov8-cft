# CFT dual-stream detection trainer and validator.

from __future__ import annotations

from copy import copy
from pathlib import Path
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


def _on_train_epoch_start(trainer) -> None:
    """Optionally freeze RGB/IR backbone (layers 0-19) for the first ``freeze_epochs``."""
    n = int(getattr(trainer, "freeze_epochs", 0) or 0)
    if n <= 0:
        return
    freeze = trainer.epoch < n
    if trainer.epoch == n and RANK in {-1, 0}:
        LOGGER.warning(
            f"CFT backbone (layers 0–19) unfrozen at epoch {trainer.epoch + 1}. "
            "VRAM use jumps ~2×; on 8GB GPUs use batch=1 (and resume from last.pt if OOM)."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model = unwrap_model(trainer.model)
    for name, param in model.named_parameters():
        if not name.startswith("model."):
            continue
        parts = name.split(".")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            continue
        if idx <= 19:
            param.requires_grad = not freeze


class CFTDetectionTrainer(DetectionTrainer):
    """Train YOLOv8 + CFT (RGB/IR) fusion models."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None, freeze_epochs: int = 10):
        self.freeze_epochs = int(freeze_epochs or 0)
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", _on_train_epoch_start)

    def _build_train_pipeline(self):
        """Build loaders; val batch matches train batch (dual stream already loads RGB+IR)."""
        super()._build_train_pipeline()
        from ultralytics.utils import LOCAL_RANK

        batch_size = self.batch_size // max(self.world_size, 1)
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size,
            rank=LOCAL_RANK,
            mode="val",
        )

    def setup_val(self):
        """Prepare model and dataloader for standalone validation (``tools/val_cft.py``)."""
        if isinstance(self.model, torch.nn.Module) and self.validator is not None:
            return
        self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        max(int(unwrap_model(self.model).stride.max()), 32)
        self.args.imgsz = int(self.args.imgsz)
        self.test_loader = self.get_dataloader(self.data["val"], batch_size=self.batch_size, rank=-1, mode="val")
        self.validator = self.get_validator()
        LOGGER.info(f"Validation dataloader: {len(self.test_loader)} batches")

    def validate(self):
        if self.validator is None:
            self.setup_val()
        return super().validate()

    def get_dataset(self):
        data = check_dual_det_dataset(self.args.data)
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        ir_key = "train_ir" if mode == "train" else "val_ir"
        ir_path = self.data.get(ir_key)
        # LLVIP: IR derived from RGB paths; VEDAI: separate fold list files.
        if ir_path and Path(str(ir_path)).resolve() == Path(str(img_path)).resolve():
            ir_path = None
        return build_dual_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, ir_path=ir_path
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
    """Validate dual-stream models (RGB + IR).

    Standalone ``tools/val_cft.py`` must call with ``trainer=`` (not ``model=`` alone) so ``llvip_dual.yaml`` is read
    via ``check_dual_det_dataset`` and the dual dataloader is reused.
    """

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
