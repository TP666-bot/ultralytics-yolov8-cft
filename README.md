<div align="center">

# YOLOv8 + CFT：多光谱目标检测迁移与 LLVIP 消融实验

**Cross-Modality Fusion Transformer on Ultralytics YOLOv8**

[![GitHub](https://img.shields.io/badge/GitHub-TP666--bot-181717?logo=github)](https://github.com/TP666-bot/ultralytics-yolov8-cft)
[![Branch](https://img.shields.io/badge/branch-cft--yolov8--llvip-blue)](https://github.com/TP666-bot/ultralytics-yolov8-cft/tree/cft-yolov8-llvip)
[![Ultralytics](https://img.shields.io/badge/base-Ultralytics%20YOLOv8-111111)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/dataset-LLVIP-green)](https://github.com/bupt-ai-cz/LLVIP)
[![License](https://img.shields.io/badge/license-AGPL--3.0-lightgrey)](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

[English Abstract](#abstract) · [快速开始](#quick-start) · [实验结果](#results) · [完整文档](cft_yolov8_config.md) · [CFT 原论文](https://arxiv.org/abs/2111.00273)

</div>

---

## Abstract

本仓库在 **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** 框架上，将 **[Cross-Modality Fusion Transformer (CFT)](https://arxiv.org/abs/2111.00273)** 从 YOLOv5 双路架构迁移至 YOLOv8，并在 **LLVIP** 可见光–红外配对数据集上建立可复现的消融实验流程。

CFT 通过 Transformer 自注意力在 P3/P4/P5 多尺度特征上同时建模 **模态内** 与 **模态间** 依赖，显著增强低照度场景下的行人检测鲁棒性。本工作保留 CFT 融合思想，将 backbone 中的 `C3/SPP` 替换为 YOLOv8 的 `C2f/SPPF`，并扩展双路同步增强、融合 Trainer 与论文协议验证脚本（`val_cft.py`），便于与 YOLOv8 单模态基线及 YOLOv5+CFT 原实现进行公平对比。

**主要成果（LLVIP test，3463 图，`val_cft.py`，conf=0.001，iou=0.5，imgsz=1024）**：

| 方法 | 框架 | mAP@0.5 | mAP@0.5:0.95 |
|------|------|---------|--------------|
| CFT（论文） | YOLOv5 | 0.975 | 0.636 |
| CFT（作者权重复现） | YOLOv5 | 0.972 | 0.633 |
| **CFT（本仓库 y8_cft_v2）** | **YOLOv8** | **0.973** | **0.668** |

> **说明**：本仓库为研究与复现实验分支（`cft-yolov8-llvip`），不修改 [DocF/multispectral-object-detection](https://github.com/DocF/multispectral-object-detection) 原仓库；后者仍作为 YOLOv5+CFT 参考实现与对照实验来源。实现细节与 P1–P5 改进见 [`tp_improve.md`](tp_improve.md)。

---

## Overview

```
  RGB stream (YOLOv8-L)                IR stream (YOLOv8-L)
         │                                      │
         └──────────┬ P3 / P4 / P5 ────────────┘
                      │
            ┌─────────▼─────────┐
            │  Add  or  CFT     │  ← GPT × 3 (Cross-Modality Fusion)
            │  (feature fusion) │
            └─────────┬─────────┘
                      │
              YOLOv8 Detect Head
                      │
                 person (LLVIP)
```

CFT 原论文方法示意图见官方仓库：[cft.png](https://github.com/DocF/multispectral-object-detection/blob/main/cft.png)

---

## Highlights

| 模块 | 路径 | 说明 |
|------|------|------|
| CFT 融合层 | `ultralytics/nn/modules/cft.py` | `GPT`、`Add`、`Add2` |
| 双路模型 | `ultralytics/nn/tasks_dual.py` | RGB/IR 双流 `forward_once` |
| 双路数据与增强 | `ultralytics/data/dual_stream.py`、`dual_augment.py` | 成对 RGB+IR、同步 Mosaic/Affine |
| CFT 训练/验证 | `ultralytics/models/yolo/detect/cft_train.py` | 冻结 backbone、`val_cft` 双路推理 |
| 训练/评估入口 | `tools/train_cft.py`、`tools/val_cft.py` | SGD 默认、论文协议评估 |
| Y5→Y8 GPT 初始化 | `tools/load_cft_partial.py` | 可选方案 B |
| 融合配置 | `ultralytics/cfg/models/v8/yolov8l_fusion_*.yaml` | Add / GPT×3 |
| 实验文档 | [`cft_yolov8_config.md`](cft_yolov8_config.md) | 环境、命令、消融表、故障排查 |

---

## Results

与论文消融逻辑对齐的四组 **YOLOv8** 实验（完整表见 [`cft_yolov8_config.md` §9](cft_yolov8_config.md#九实验记录表)）：

| ID | 实验 | 模态 / 融合 | mAP@0.5 | 状态 |
|----|------|-------------|---------|------|
| 1 | **Y8-RGB** | 单路可见光 | — | 待填 |
| 2 | **Y8-IR** | 单路红外 | ~0.958 | 训练 val |
| 3 | **Y8-Add** | 双路 + 逐层 Add | — | P1 增强后待重训 |
| 4 | **Y8-CFT** | 双路 + GPT×3 | **0.973** | ✅ 论文量级（`y8_cft_v2`） |

**Y8-CFT v2 正式指标**（`best.pt`，epoch ~51 峰值）：

| P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---------|--------------|
| 0.967 | 0.936 | **0.973** | **0.668** |

**YOLOv5+CFT 参考**（原仓库作者权重，`test.py` 独立复现）：

| mAP@0.5 | mAP@0.5:0.95 |
|---------|--------------|
| 0.972 | 0.633 |

---

## Quick Start

### 1. Clone

```bash
git clone -b cft-yolov8-llvip https://github.com/TP666-bot/ultralytics-yolov8-cft.git
cd ultralytics-yolov8-cft
```

### 2. 数据集（LLVIP）

将 [LLVIP](https://github.com/bupt-ai-cz/LLVIP) 置于与仓库平级目录（推荐）：

```text
<WORKSPACE>/
├── ultralytics-yolov8-cft/    # 本仓库
└── LLVIP/
    ├── visible/train|test/    # *.jpg + 同名 *.txt
    └── infrared/train|test/
```

### 3. 一键配置路径

```bash
cp tools/machine.env.example tools/machine.env   # 编辑 WORKSPACE
bash tools/setup_new_machine.sh
```

### 4. 环境

```bash
conda create -n y8-cft python=3.10 -y
conda activate y8-cft
pip install torch torchvision   # 按 https://pytorch.org 选择 CUDA 版本
pip install -e .
```

### 5. 训练 Y8-CFT（8 GB GPU 推荐）

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt epochs=200 imgsz=1024 batch=1 nbs=16 workers=2 device=0 \
  project=runs/llvip name=y8_cft_v2 exist_ok=True
```

epoch 11 backbone 解冻 OOM 时续训：

```bash
python tools/train_cft.py \
  resume=runs/detect/runs/llvip/y8_cft_v2/weights/last.pt \
  batch=1 nbs=16 freeze_epochs=0 workers=2 device=0 \
  project=runs/llvip name=y8_cft_v2 exist_ok=True
```

### 6. 论文协议评估

```bash
python tools/val_cft.py \
  model=runs/detect/runs/llvip/y8_cft_v2/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 batch=1 device=0 conf=0.001 iou=0.5
```

> 报告指标以 **`best.pt` + `val_cft.py`** 为准；训练日志 `results.csv` 使用 iou=0.7，数值不可直接与论文对比。

单模态基线、Add 消融、Y5 GPT 初始化（方案 B）→ **[`cft_yolov8_config.md`](cft_yolov8_config.md)**

---

## Documentation

| 文档 | 内容 |
|------|------|
| [`cft_yolov8_config.md`](cft_yolov8_config.md) | 实验协议、消融设计、结果表、复现命令、故障排查 |
| [`tp_improve.md`](tp_improve.md) | P1–P5 代码级改进说明（双路增强、SGD、freeze、val 修复） |
| [`tools/setup_new_machine.sh`](tools/setup_new_machine.sh) | 新机器路径与 symlink 自动生成 |
| [Ultralytics Docs](https://docs.ultralytics.com/) | 上游 YOLOv8 通用 API |

---

## Citation

若使用 **CFT 方法**，请引用原论文：

```bibtex
@article{fang2021cross,
  title={Cross-Modality Fusion Transformer for Multispectral Object Detection},
  author={Fang, Qingyun and Han, Dapeng and Wang, Zhaokui},
  journal={arXiv preprint arXiv:2111.00273},
  year={2021}
}
```

若使用 **YOLOv8 框架**，请引用 Ultralytics：

```bibtex
@software{ultralytics_yolo,
  author = {Ultralytics},
  title = {Ultralytics YOLO},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}
```

---

## Acknowledgements

- [Cross-Modality Fusion Transformer (CFT)](https://github.com/DocF/multispectral-object-detection) — YOLOv5 双路融合原实现  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 检测框架基座  
- [LLVIP Dataset](https://github.com/bupt-ai-cz/LLVIP) — 可见光–红外配对数据  

---

<div align="center">

**Maintained by [TP666-bot](https://github.com/TP666-bot)** · 实验分支 `cft-yolov8-llvip`

</div>
