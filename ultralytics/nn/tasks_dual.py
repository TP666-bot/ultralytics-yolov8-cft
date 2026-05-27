# Dual-stream (RGB + IR) detection models with CFT fusion modules.

from __future__ import annotations

from copy import deepcopy

import torch

from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel, initialize_weights, parse_model, yaml_model_load
from ultralytics.utils import LOGGER


class DualDetectionModel(DetectionModel):
    """YOLOv8 detection with paired RGB/IR inputs and CFT fusion blocks (Add / GPT)."""

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        torch.nn.Module.__init__(self)
        self.dual = True
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        self.yaml["channels"] = ch
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            self.model.eval()
            m.training = True
            dummy = torch.zeros(1, ch, s, s)
            out = self._predict_once(dummy, dummy.clone())
            if isinstance(out, (list, tuple)):
                feats = out
            elif isinstance(out, dict):
                feats = out.get("feats", [out])
            else:
                feats = [out]
            m.stride = torch.tensor([s / x.shape[-2] for x in feats])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = torch.tensor([32.0])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")
        self.ir_input = None

    def __call__(self, x, augment=False, **kwargs):
        """Support validator calling model(img, augment=...)."""
        if isinstance(x, torch.Tensor) and self.ir_input is not None:
            return self.predict(x, self.ir_input)
        return self.forward(x, **kwargs)

    def _predict_once(self, x, x2=None, profile=False, visualize=False, embed=None):
        """Forward pass with RGB tensor x and IR tensor x2."""
        if x2 is None:
            if x.shape[1] == 6:
                x2 = x[:, 3:, :, :]
                x = x[:, :3, :, :]
            else:
                raise ValueError("DualDetectionModel requires img + img2 or 6-channel input")
        y = []
        for m in self.model:
            if m.f != -1 and m.f != -4:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def predict(self, x, x2=None, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            LOGGER.warning("DualDetectionModel does not support augment=True")
        return self._predict_once(x, x2, profile, visualize, embed)

    def forward(self, x, x2=None, *args, **kwargs):
        if isinstance(x, dict):
            batch = dict(x)
            img = batch["img"]
            img2 = batch.get("img2")
            if img2 is None and img.shape[1] == 6:
                img2 = img[:, 3:]
                img = img[:, :3]
            batch["img"], batch["img2"] = img, img2
            return self.loss(batch)
        return self.predict(x, x2)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self._predict_once(batch["img"], batch["img2"])
        return self.criterion(preds, batch)

    def load(self, weights, verbose=True):
        """Load weights and mirror RGB backbone weights to the IR stream when needed."""
        from ultralytics.nn.tasks import torch_safe_load

        if isinstance(weights, str):
            weights, _ = torch_safe_load(weights)
        csd = weights["model"].float().state_dict() if isinstance(weights, dict) else weights.float().state_dict()
        updated = self.load_state_dict(
            {k: v for k, v in csd.items() if k in self.state_dict() and self.state_dict()[k].shape == v.shape},
            strict=False,
        )
        n_copy = self._copy_rgb_backbone_to_ir(csd)
        if verbose:
            matched = len([k for k in csd if k in self.state_dict() and self.state_dict()[k].shape == csd[k].shape])
            LOGGER.info(
                f"Transferred {matched}/{len(self.state_dict())} items from pretrained weights"
                + (f"; mirrored {n_copy} IR backbone tensors" if n_copy else "")
            )

    def _copy_rgb_backbone_to_ir(self, csd) -> int:
        """Copy RGB stream weights (layers 0-9) to IR stream (layers 10-19) for Add/CFT yaml layout."""
        state = self.state_dict()
        n = 0
        for k, v in list(state.items()):
            if not k.startswith("model."):
                continue
            parts = k.split(".")
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            if 10 <= idx <= 19:
                src = k.replace(f"model.{idx}.", f"model.{idx - 10}.", 1)
                if src in csd and csd[src].shape == v.shape:
                    state[k] = csd[src].clone()
                    n += 1
                elif src in state and state[src].shape == v.shape and src not in csd:
                    state[k] = state[src].clone()
                    n += 1
        if n:
            self.load_state_dict(state, strict=False)
        return n
