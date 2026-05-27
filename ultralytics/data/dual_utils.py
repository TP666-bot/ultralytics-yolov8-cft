# Dual-stream dataset YAML validation (LLVIP RGB + IR).

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.data.utils import check_class_names
from ultralytics.utils import YAML, emojis
from ultralytics.utils.checks import check_file


def check_dual_det_dataset(dataset: str) -> dict[str, Any]:
    """Parse llvip_dual-style yaml with train_rgb / train_ir paths."""
    file = Path(check_file(dataset))
    data = YAML.load(file, append_filename=True)
    required = ("train_rgb", "val_rgb", "train_ir", "val_ir")
    for k in required:
        if k not in data:
            raise SyntaxError(emojis(f"{dataset} missing '{k}' for dual-stream CFT training."))
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} requires 'names' or 'nc'."))
    if "names" not in data:
        data["names"] = {i: f"class_{i}" for i in range(data["nc"])}
    else:
        data["nc"] = len(data["names"])
    data["names"] = check_class_names(data["names"])
    data["channels"] = 3
    root = Path(data.get("path") or file.parent)
    if not root.is_absolute():
        root = root.resolve()
    data["path"] = str(root)
    for k in required:
        p = (root / data[k]).resolve() if not Path(data[k]).is_absolute() else Path(data[k]).resolve()
        data[k] = str(p)
        if not p.exists():
            raise FileNotFoundError(f"Dual dataset path not found: {p}")
    # Trainer compatibility
    data["train"] = data["train_rgb"]
    data["val"] = data["val_rgb"]
    data["yaml_file"] = str(file)
    return data
