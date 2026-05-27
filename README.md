<div align="center">

# YOLOv8 + CFT：多光谱目标检测迁移与 LLVIP 消融实验

**Cross-Modality Fusion Transformer on Ultralytics YOLOv8**

[![GitHub](https://img.shields.io/badge/GitHub-TP666--bot-181717?logo=github)](https://github.com/TP666-bot/ultralytics-yolov8-cft)
[![Branch](https://img.shields.io/badge/branch-cft--yolov8--llvip-blue)](https://github.com/TP666-bot/ultralytics-yolov8-cft/tree/cft-yolov8-llvip)
[![Ultralytics](https://img.shields.io/badge/base-Ultralytics%20YOLOv8-111111)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/dataset-LLVIP-green)](https://github.com/bupt-ai-cz/LLVIP)
[![License](https://img.shields.io/badge/license-AGPL--3.0-lightgrey)](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

[English Abstract](#abstract) · [快速开始](#quick-start) · [完整文档](cft_yolov8_config.md) · [CFT 原论文](https://arxiv.org/abs/2111.00273)

</div>

---

## Abstract

本仓库在 **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** 框架上，将 **[Cross-Modality Fusion Transformer (CFT)](https://arxiv.org/abs/2111.00273)** 从 YOLOv5 双路架构迁移至 YOLOv8，并在 **LLVIP** 可见光–红外配对数据集上建立可复现的消融实验流程。

CFT 通过 Transformer 自注意力在 P3/P4/P5 多尺度特征上同时建模 **模态内** 与 **模态间** 依赖，显著增强低照度场景下的行人检测鲁棒性。本工作保留 CFT 融合思想，将 backbone 中的 `C3/SPP` 替换为 YOLOv8 的 `C2f/SPPF`，并扩展双路数据加载、融合 Trainer 与验证流程，便于与 YOLOv8 单模态基线及 YOLOv5+CFT 原实现进行公平对比。

> **说明**：本仓库为研究与复现实验分支（`cft-yolov8-llvip`），不修改 [DocF/multispectral-object-detection](https://github.com/DocF/multispectral-object-detection) 原仓库；后者仍作为 YOLOv5+CFT 参考实现与对照实验来源。

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

| 模块       | 路径                                              | 说明                       |
| ---------- | ------------------------------------------------- | -------------------------- |
| CFT 融合层 | `ultralytics/nn/modules/cft.py`                   | `GPT`、`Add`、`Add2`       |
| 双路模型   | `ultralytics/nn/tasks_dual.py`                    | RGB/IR 双流 `forward_once` |
| 双路数据   | `ultralytics/data/dual_stream.py`                 | 成对 RGB+IR 加载           |
| CFT 训练器 | `ultralytics/models/yolo/detect/cft_train.py`     | 扩展 `DetectionTrainer`    |
| 融合配置   | `ultralytics/cfg/models/v8/yolov8l_fusion_*.yaml` | Add / GPT×3                |
| 实验文档   | [`cft_yolov8_config.md`](cft_yolov8_config.md)    | 环境、命令、消融、迁移     |

---

## Experiments (LLVIP, imgsz=1024)

与论文消融逻辑对齐的四组 **YOLOv8** 实验：

| ID  | 实验       | 模态 / 融合     | 入口                                          |
| --- | ---------- | --------------- | --------------------------------------------- |
| 1   | **Y8-RGB** | 单路可见光      | `yolo train` + `llvip_rgb.yaml`               |
| 2   | **Y8-IR**  | 单路红外        | `yolo train` + `llvip_ir.yaml`                |
| 3   | **Y8-Add** | 双路 + 逐层 Add | `python tools/train_cft.py` + fusion Add yaml |
| 4   | **Y8-CFT** | 双路 + GPT×3    | `python tools/train_cft.py` + fusion CFT yaml |

**YOLOv5+CFT 对照上界**（原仓库作者权重，本机已复现）：

| 指标         | 本机复现  | 论文 README |
| ------------ | --------- | ----------- |
| mAP@0.5      | **0.972** | 0.975       |
| mAP@0.5:0.95 | **0.633** | 0.636       |

YOLOv8 结果为独立新实验，数值供迁移期参考，详见 [`cft_yolov8_config.md`](cft_yolov8_config.md) 第十节。

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
cp tools/machine.env.example tools/machine.env # 编辑 WORKSPACE
bash tools/setup_new_machine.sh
```

### 4. 环境

```bash
conda create -p ~/conda_envs/ms_cft python=3.10 -y
conda activate ~/conda_envs/ms_cft
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

### 5. 训练示例（Y8-RGB，8GB 显存）

```bash
yolo train model=yolov8l.pt data=ultralytics/cfg/datasets/llvip_rgb.yaml \
  epochs=200 imgsz=1024 batch=2 nbs=16 workers=4 device=0 \
  project=runs/llvip name=y8_rgb exist_ok=True
```

双路 CFT 融合：

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt epochs=200 imgsz=1024 batch=2 nbs=16 device=0 \
  project=runs/llvip name=y8_cft exist_ok=True
```

更多命令、显存建议、新机器 Git 迁移 → **[`cft_yolov8_config.md`](cft_yolov8_config.md)**

---

## Documentation

| 文档                                                       | 内容                                              |
| ---------------------------------------------------------- | ------------------------------------------------- |
| [`cft_yolov8_config.md`](cft_yolov8_config.md)             | YOLOv8 实验全流程、消融表、4060 Ti 显存、Git 迁移 |
| [`tools/setup_new_machine.sh`](tools/setup_new_machine.sh) | 新机器路径与 symlink 自动生成                     |
| [Ultralytics Docs](https://docs.ultralytics.com/)          | 上游 YOLOv8 通用 API                              |

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

## Acknowledgments

- [Cross-Modality Fusion Transformer (CFT)](https://github.com/DocF/multispectral-object-detection) — YOLOv5 双路融合原实现
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 检测框架基座
- [LLVIP Dataset](https://github.com/bupt-ai-cz/LLVIP) — 可见光–红外配对数据

---

<div align="center">

**Maintained by [TP666-bot](https://github.com/TP666-bot)** · 实验分支 `cft-yolov8-llvip`

</div>
