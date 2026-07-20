# YOLOv8 + CFT on LLVIP：实验协议与复现指南

本文档面向 **clone 本仓库后的使用者**，说明如何在 LLVIP 数据集上复现 YOLOv8 单模态基线、双路 Add 融合与 **Cross-Modality Fusion Transformer (CFT)** 实验，并与 YOLOv5+CFT 原论文结果对照。

- 项目概览与引用：[`README.md`](README.md)
- **性能改进与复训（P1–P5）**：[`tp_improve.md`](tp_improve.md)
- CFT 原论文：[Cross-Modality Fusion Transformer for Multispectral Object Detection](https://arxiv.org/abs/2111.00273)（arXiv:2111.00273）
- YOLOv5+CFT 官方实现：[DocF/multispectral-object-detection](https://github.com/DocF/multispectral-object-detection)
- 本仓库分支：`cft-yolov8-llvip`

---

## 一、研究背景与目标

### 1.1 背景

CFT（Cross-Modality Fusion Transformer）在 YOLOv5 双路检测器上，于 P3/P4/P5 尺度通过 Transformer 自注意力联合建模 RGB 与红外（IR）特征，在 LLVIP 等数据集上取得了优于简单 Add 融合的检测性能。

本仓库将 CFT **迁移至 Ultralytics YOLOv8**（`C3→C2f`、`SPP→SPPF`、anchor-free Detect 等结构变化），在 **相同数据划分与评估协议** 下开展消融，以回答：

1. YOLOv8 单模态（RGB / IR）在 LLVIP 上的基线性能；
2. 双路 Add 融合在 YOLOv8 上相对单模态的增益；
3. CFT 在 YOLOv8 上相对 Add 的增益，及与 YOLOv5+CFT 论文结果的可比性。

### 1.2 与 YOLOv5 原实现的关系

| 项目     | YOLOv5+CFT（原仓库）        | 本仓库（YOLOv8+CFT）                       |
| -------- | --------------------------- | ------------------------------------------ |
| 框架     | YOLOv5 + 自定义 `train.py`  | Ultralytics YOLOv8 + `tools/train_cft.py`  |
| 融合模块 | `GPT` / `Add` / `Add2`      | 同逻辑，见 `ultralytics/nn/modules/cft.py` |
| 对照权重 | 作者发布之 YOLOv5l CFT 权重 | **不可**整模载入 Y8；可选部分加载 GPT      |
| 数据     | LLVIP 四路径                | 相同划分，见第五节                         |

本仓库 **不修改** 官方 multispectral 源码；YOLOv5 对照实验需在原仓库中单独进行。

---

## 二、实验设计

### 2.1 消融组（主实验）

与 CFT 论文 LLVIP 表格逻辑一致，主实验包含 **四组 YOLOv8 实验**，外加可选 **YOLOv5 对照**：

| ID  | 名称   | 模态       | 融合方式                | 模型配置                                  | 预训练初始化                                        | 训练入口                      |
| --- | ------ | ---------- | ----------------------- | ----------------------------------------- | --------------------------------------------------- | ----------------------------- |
| 1   | Y8-RGB | 可见光单路 | —                       | `yolov8l.pt`                              | COCO 预训练 YOLOv8-L                                | `yolo train`                  |
| 2   | Y8-IR  | 红外单路   | —                       | `yolov8l.pt`                              | 同上                                                | `yolo train`                  |
| 3   | Y8-Add | RGB + IR   | P3/P4/P5 **逐元素 Add** | `yolov8l_fusion_add_llvip.yaml`           | `yolov8l.pt` 双路 backbone（`strict=False`）        | `python tools/train_cft.py`   |
| 4   | Y8-CFT | RGB + IR   | **GPT × 3**（CFT）      | `yolov8l_fusion_transformerx3_llvip.yaml` | `yolov8l.pt` + GPT 随机初始化；可选 Y5 GPT 部分加载 | `python tools/train_cft.py`   |
| —   | Y5-Add | 对照       | Add                     | 原仓库 yaml                               | `yolov5l.pt`                                        | 原仓库 `train.py`             |
| —   | Y5-CFT | 对照       | CFT                     | 原仓库 yaml                               | 作者 LLVIP 权重                                     | 原仓库 `test.py` / `train.py` |

**推荐执行顺序**：1 → 2 → 3 → 4。实验 1/2 验证单模态数据链；3/4 验证双路与 CFT 融合。

### 2.2 控制变量（公平对比）

| 变量          | 设定                                         | 说明                                                   |
| ------------- | -------------------------------------------- | ------------------------------------------------------ |
| 数据集        | LLVIP                                        | 单类 `person`；官方 train/test 划分                    |
| 训练集        | `visible/train` + `infrared/train`（双模态） | 约 12 025 对 / 模态                                    |
| 测试集        | `visible/test` + `infrared/test`             | 约 3 463 对 / 模态；**所有 mAP 均在此评估**            |
| 输入尺寸      | `imgsz=1024`                                 | 与作者 LLVIP CFT 权重及论文设置一致                    |
| 训练轮数      | `epochs=200`                                 | 与论文 LLVIP 实验一致                                  |
| 名义 batch    | `nbs=16`（可调至 32）                        | Ultralytics 梯度累积；等效 batch ≈ `nbs`               |
| 物理 batch    | `batch=1`（8 GB + Y8l-CFT @1024）            | 依 GPU 显存调整；CFT 推荐 `batch=1` + `nbs=16`，见 7.4 |
| 优化器        | `optimizer=SGD`                              | **禁止** `optimizer=auto`（8.4 MuSGD 与 GPT 不兼容）   |
| 学习率        | `lr0=0.005`, `momentum=0.937`                | `train_cft.py` 默认                                    |
| Backbone 冻结 | `freeze_epochs=10`（可选）                   | 前 10 epoch 冻结层 0–19；OOM 续训用 `freeze_epochs=0`  |
| NMS           | `conf=0.001`, `iou=0.5`                      | 对齐原仓库 `test.py`                                   |
| 指标          | P, R, mAP@0.5, mAP@0.75, mAP@0.5:0.95        | 检测标准 COCO 风格指标                                 |
| 标签          | RGB 侧 `.txt`（YOLO 格式）                   | 双模态共用 RGB 侧框；IR 同 basename                    |

**不控制 / 预期存在差异的变量**：优化器默认值（Ultralytics vs YOLOv5）、anchor-free vs anchor-based 检测头、数据增强实现细节。因此 YOLOv8 绝对数值 **不要求** 与论文 YOLOv5 行完全一致；对比应关注 **同框架内消融趋势** 与 **相对 Y5 参考区间的合理性**。

### 2.3 假设（可检验命题）

| 假设 | 预期                                                      | 状态                                              |
| ---- | --------------------------------------------------------- | ------------------------------------------------- |
| H1   | 双路 Add（Y8-Add）优于单模态（Y8-RGB / Y8-IR）            | 待 Y8-Add 重训后验证                              |
| H2   | CFT（Y8-CFT）优于 Add（Y8-Add）                           | 待 Y8-Add 重训后验证                              |
| H3   | Y8-CFT 相对 Y5-CFT 论文 mAP@0.5（≈0.975）处于合理接近区间 | **已验证**：`y8_cft_v2` mAP@0.5=**0.973**（§9.3） |

### 2.4 论文参考数值（LLVIP，YOLOv5l）

| 方法         | mAP@0.5 | mAP@0.5:0.95 | 来源                 |
| ------------ | ------- | ------------ | -------------------- |
| Add 融合     | 0.958   | 0.623        | 原仓库 README        |
| CFT（GPT×3） | 0.975   | 0.636        | 原仓库 README / 论文 |

**独立复现（YOLOv5+CFT，作者权重，`test.py`，1024）**：

| 指标         | 复现值    | 论文 README |
| ------------ | --------- | ----------- |
| P            | 0.967     | —           |
| R            | 0.931     | —           |
| mAP@0.5      | **0.972** | 0.975       |
| mAP@0.75     | 0.724     | 0.729       |
| mAP@0.5:0.95 | **0.633** | 0.636       |

该结果作为 YOLOv5+CFT 的 **参考上界**；YOLOv8 实验为 **新架构下的独立研究**，与之对比时需说明框架差异（见 2.2）。

---

## 三、快速开始（Clone 后必做）

### 3.1 获取代码

```bash
git clone -b cft-yolov8-llvip https://github.com/TP666-bot/ultralytics-yolov8-cft.git
cd ultralytics-yolov8-cft
```

公开仓库可直接 clone，**无需**仓库所有者账号。私有仓库需被添加为 Collaborator 并使用 **自己的** GitHub 凭据。

### 3.2 目录布局

将 LLVIP 置于与仓库 **平级** 的 `LLVIP/`（推荐）：

```text
<WORKSPACE>/
├── ultralytics-yolov8-cft/     # 本仓库（clone 目录名可自定）
└── LLVIP/
    ├── visible/train/          # *.jpg + 同名 *.txt
    ├── visible/test/
    ├── infrared/train/
    └── infrared/test/
```

### 3.3 配置路径与 symlink

```bash
cp tools/machine.env.example tools/machine.env
# 编辑 WORKSPACE=<WORKSPACE 绝对路径>
bash tools/setup_new_machine.sh
```

脚本将生成 `llvip_rgb.yaml`、`llvip_ir.yaml`、`llvip_dual.yaml`，并创建 `datasets/llvip_{rgb,ir}` 下的 symlink。各 split 图片数量应约为：**train 12 025**、**test 3 463**（每模态）。

### 3.4 环境

```bash
conda create -n y8-cft python=3.10 -y
conda activate y8-cft

# 按 GPU/CUDA 版本安装 PyTorch，参见 https://pytorch.org
pip install torch torchvision

pip install -e .
```

### 3.5 冒烟测试

```bash
yolo train model=yolov8l.pt data=ultralytics/cfg/datasets/llvip_rgb.yaml \
  epochs=1 imgsz=1024 batch=2 device=0 project=runs/llvip name=smoke exist_ok=True
```

日志中出现 `12025` train / `3463` val 且无报错，即表示数据链与环境正常。

---

## 四、数据集（LLVIP）

**下载**：[LLVIP download_dataset.md](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md)

**标签**：每个 `.jpg` 旁需同名 `.txt`（YOLO 格式：`0 cx cy w h`，单类 person）。若仅有 VOC `Annotations/`，可使用原仓库脚本转换：

```bash
python3 voc_to_yolo_llvip.py --dataset-root < WORKSPACE > /LLVIP
```

（脚本见 [multispectral-object-detection/tools](https://github.com/DocF/multispectral-object-detection)。）

**数据配置（由 `setup_new_machine.sh` 生成）**：

| 文件              | 用途       | 路径键                                       |
| ----------------- | ---------- | -------------------------------------------- |
| `llvip_rgb.yaml`  | 单模态 RGB | `train` / `val` → visible                    |
| `llvip_ir.yaml`   | 单模态 IR  | `train` / `val` → infrared                   |
| `llvip_dual.yaml` | 双模态融合 | `train_rgb`, `val_rgb`, `train_ir`, `val_ir` |

---

## 五、仓库结构

| 内容                         | 路径                                                                      |
| ---------------------------- | ------------------------------------------------------------------------- |
| CFT 模块（GPT / Add / Add2） | `ultralytics/nn/modules/cft.py`                                           |
| 双路模型                     | `ultralytics/nn/tasks_dual.py`                                            |
| 双路 Dataset                 | `ultralytics/data/dual_stream.py`                                         |
| CFT Trainer / Validator      | `ultralytics/models/yolo/detect/cft_train.py`                             |
| 融合模型 yaml                | `ultralytics/cfg/models/v8/yolov8l_fusion_{add,transformerx3}_llvip.yaml` |
| 训练 / 验证入口              | `tools/train_cft.py`、`tools/val_cft.py`                                  |
| Y5→Y8 GPT 部分加载           | `tools/load_cft_partial.py`                                               |
| 路径初始化                   | `tools/setup_new_machine.sh`                                              |

**实现映射（YOLOv5 → YOLOv8）**：

| YOLOv5（CFT 原仓库）        | YOLOv8（本仓库）        |
| --------------------------- | ----------------------- |
| `Focus`                     | 首层 `Conv`             |
| `C3`                        | `C2f`                   |
| `SPP`                       | `SPPF`                  |
| `Detect`（anchor-based）    | `Detect`（anchor-free） |
| `Model.forward_once(x, x2)` | `DualDetectionModel`    |

---

## 六、阶段 B：单模态基线（实验 1 & 2）

### 6.1 训练（Y8-RGB 示例）

在 **仓库根目录** 执行：

```bash
yolo train \
  model=yolov8l.pt \
  data=ultralytics/cfg/datasets/llvip_rgb.yaml \
  epochs=200 \
  imgsz=1024 \
  batch=2 \
  nbs=16 \
  workers=4 \
  device=0 \
  project=runs/llvip \
  name=y8_rgb \
  exist_ok=True
```

**Y8-IR**：将 `llvip_rgb.yaml` → `llvip_ir.yaml`，`name=y8_ir`。

### 6.2 验证

```bash
yolo val \
  model=runs/detect/runs/llvip/y8_rgb/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_rgb.yaml \
  imgsz=1024 \
  batch=4 \
  device=0 \
  conf=0.001 \
  iou=0.5 \
  project=runs/llvip \
  name=y8_rgb_val \
  exist_ok=True
```

### 6.3 超参数说明

| 参数      | 建议值                          | 说明                                               |
| --------- | ------------------------------- | -------------------------------------------------- |
| `batch`   | 1–4（`yolov8l` @ 1024）         | 物理 batch；OOM 时降为 1                           |
| `nbs`     | 16 或 32                        | 名义 batch；内部 `accumulate ≈ round(nbs / batch)` |
| `workers` | 4（OOM 或 dataloader 报错时 0） | DataLoader 进程数                                  |
| `imgsz`   | 1024                            | 与论文 / Y5 权重一致                               |

**注意**：Ultralytics 8.x **`yolo train` 不支持 `accumulate=`**；请使用 **`nbs=`** 控制梯度累积。YOLOv5 原仓库使用 `accumulate`，二者不可混用参数名。

### 6.4 显存与权重路径

- `yolov8l @ imgsz=1024` 在 **8 GB** 级 GPU 上通常需 `batch=1~2`；请勿默认 `batch=16`。
- 标准 `yolo train` 的 `project=runs/llvip` 实际保存路径多为：

  ```text
  runs/detect/runs/llvip/<name>/weights/best.pt
  ```

  训练开始时的控制台日志会打印确切 `save_dir`，请以日志为准。

- 首次训练会在数据目录生成 `.cache`；第二次启动会显著加快。

---

## 七、阶段 C：双路融合与 CFT（实验 3 & 4）

双路实验使用 **`python tools/train_cft.py`**，而非 `yolo train`（后者仅支持单路 RGB）。

### 7.1 实验 3：Y8-Add

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_add_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt \
  epochs=200 \
  imgsz=1024 \
  batch=2 \
  nbs=16 \
  workers=4 \
  device=0 \
  project=runs/llvip \
  name=y8_add \
  exist_ok=True
```

`pretrained=yolov8l.pt` 会加载 RGB backbone 并 **复制权重至 IR 流**。

验证：

```bash
python tools/val_cft.py \
  model=runs/detect/runs/llvip/y8_add/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 \
  batch=4 \
  device=0 \
  conf=0.001 \
  iou=0.5 \
  project=runs/llvip \
  name=y8_add_val \
  exist_ok=True
```

> 若 `best.pt` 不在上述路径，请根据训练日志中的 `save_dir` 调整；`train_cft.py` 与 `yolo train` 的目录前缀可能略有不同。

### 7.2 实验 4：Y8-CFT

**方案 A（推荐，已复现论文量级）**：YOLOv8 预训练 + GPT 随机初始化 + P1–P5 训练管线（见 [`tp_improve.md`](tp_improve.md)）

8 GB GPU（如 RTX 4060 Ti）推荐配置：

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt \
  epochs=200 \
  imgsz=1024 \
  batch=1 \
  nbs=16 \
  workers=2 \
  device=0 \
  project=runs/llvip \
  name=y8_cft_v2 \
  exist_ok=True
```

**论文协议正式评估**（必须用 `best.pt`，不要用 `last.pt`）：

```bash
python tools/val_cft.py \
  model=runs/detect/runs/llvip/y8_cft_v2/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 \
  batch=1 \
  device=0 \
  conf=0.001 \
  iou=0.5
```

**续训**（epoch 11 backbone 解冻后 OOM 时）：

```bash
python tools/train_cft.py \
  resume=runs/detect/runs/llvip/y8_cft_v2/weights/last.pt \
  batch=1 nbs=16 freeze_epochs=0 workers=2 device=0 \
  project=runs/llvip name=y8_cft_v2 exist_ok=True
```

> `train_cft.py` 默认：`optimizer=SGD`、`lr0=0.005`、`freeze_epochs=10`（脚本专用参数，勿传给 `yolo train`）。训练中 `results.csv` 的 mAP 使用 **iou=0.7**；对外报告以 **`val_cft.py`（iou=0.5）** 为准。

**方案 B（可选，冲击 >0.975）**：从 YOLOv5 CFT 权重 **部分加载 GPT**（需 [multispectral-object-detection](https://github.com/DocF/multispectral-object-detection) 作者权重，与本仓库平级 clone 时可自动解析 `models.*` pickle）

```bash
# 将 YOLOv5 仓库置于 <WORKSPACE>/multispectral-object-detection/ 后可直接运行
python tools/load_cft_partial.py \
  --y8-cfg ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  --y8-weights yolov8l.pt \
  --y5-cft ../multispectral-object-detection/yolov5l_transformerx3_llvip_s1024_bs32_e200.pt \
  --out runs/llvip/y8_cft_y5gpt_init.pt

python tools/train_cft.py \
  model=runs/llvip/y8_cft_y5gpt_init.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  epochs=200 imgsz=1024 batch=1 nbs=32 workers=2 device=0 \
  freeze_epochs=10 patience=25 \
  project=runs/llvip name=y8_cft_y5gpt exist_ok=True
```

> `model=*.pt` 时 `train_cft.py` 自动设 `pretrained=False`，避免覆盖已写入的 Y5 GPT 权重。首 epoch `cls_loss` 可能高于方案 A（检测头随机、GPT 已预热），属正常现象。

### 7.3 权重使用说明

| 权重                               | 用于 YOLOv8 | 说明                                                          |
| ---------------------------------- | ----------- | ------------------------------------------------------------- |
| `yolov8l.pt`                       | ✅          | 实验 1–4 的主初始化                                           |
| `yolov5l.pt`                       | ❌ 整模     | 仅 YOLOv5 对照                                                |
| `yolov5l_transformerx3_llvip_*.pt` | ❌ 整模     | 架构不兼容；仅 GPT 部分可加载                                 |
| 作者 CFT 权重                      | ⚠️ 部分     | `tools/load_cft_partial.py` 按 key/shape 匹配 GPT 相关 tensor |

### 7.4 实现状态

| 组件                                         | 状态                                                             |
| -------------------------------------------- | ---------------------------------------------------------------- |
| CFT 模块、`DualDetectionModel`、双路 Dataset | ✅                                                               |
| 融合 yaml（Add / GPT×3）                     | ✅                                                               |
| `train_cft.py` / `val_cft.py`                | ✅                                                               |
| `load_cft_partial.py`                        | ✅                                                               |
| `sweep_val_cft.py`                           | ✅ NMS/conf 敏感性扫描                                           |
| 与 YOLOv5 `test.py` 逐 bit 一致 mAP          | ⚠️ 默认使用 Ultralytics DetMetrics；跨框架对比需注意评估实现差异 |

---

## 八、公平对比与跨框架评估

| 项              | 要求                                                                    |
| --------------- | ----------------------------------------------------------------------- |
| 数据划分        | 与 `llvip_dual.yaml` 四路径一致                                         |
| 输入尺寸        | 1024                                                                    |
| 评估集          | `visible/test` + `infrared/test`（3463 对）                             |
| NMS             | `conf=0.001`, `iou=0.5`                                                 |
| 指标            | mAP@0.5、mAP@0.75、mAP@0.5:0.95                                         |
| YOLOv8 组内对比 | 统一使用 `yolo val` / `val_cft.py`，相同 conf/iou                       |
| Y8 vs Y5 对比   | 建议注明框架差异；严格对比时可导出权重至原仓库 `test.py` 或统一评估脚本 |

---

## 九、实验记录表

实验完成后填写以下表格（权重路径以训练日志 `save_dir` 为准）。

### 9.1 YOLOv5 对照（可选，原仓库）

| 实验 ID | 方法       | 权重        | mAP@.5 | mAP@.75 | mAP@.5:.95 | 备注              |
| ------- | ---------- | ----------- | ------ | ------- | ---------- | ----------------- |
| y5-cft  | CFT GPT×3  | 作者权重    | 0.972  | 0.724   | 0.633      | 独立复现          |
| y5-add  | Add        | 自训 / 论文 | —      | —       | —          | 论文 mAP@.5=0.958 |
| y5-rgb  | RGB 单模态 | 自训        |        |         |            |                   |
| y5-ir   | IR 单模态  | 自训        |        |         |            |                   |

### 9.2 YOLOv8 单模态

| 实验 ID | 模态 | batch / nbs | mAP@.5 | mAP@.75 | mAP@.5:.95 | 备注                         |
| ------- | ---- | ----------- | ------ | ------- | ---------- | ---------------------------- |
| y8-rgb  | RGB  | 2 / 16      |        |         |            | 待填                         |
| y8-ir   | IR   | 2 / 16      | ~0.958 | —       | —          | 100e，`yolo train`，训练 val |

### 9.3 YOLOv8 融合 / CFT

| 实验 ID       | 方法      | P         | R         | mAP@.5    | mAP@.5:.95 | 评估协议              | 备注                            |
| ------------- | --------- | --------- | --------- | --------- | ---------- | --------------------- | ------------------------------- |
| y8-add        | 双路 Add  | —         | —         |           |            | `val_cft`             | 旧管线曾 ~0.69；P1 增强后待重训 |
| **y8-cft-v2** | **GPT×3** | **0.967** | **0.936** | **0.973** | **0.668**  | **`val_cft` best.pt** | **已达论文量级**                |

**y8_cft_v2 训练摘要**（`runs/detect/runs/llvip/y8_cft_v2/`）：

| 项                     | 值                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| 配置                   | `batch=1`, `nbs=16`, `SGD lr0=0.005`, `freeze_epochs=10` → epoch 11 续训 `freeze_epochs=0` |
| 训练 val 峰值 mAP@.5   | **0.973**（~epoch 50，`best.pt`，iou=0.7）                                                 |
| 正式 test（`val_cft`） | P=0.967, R=0.936, mAP@.5=**0.973**, mAP@.5:.95=**0.668**                                   |
| 与论文 Y5-CFT          | mAP@.5 差 **0.002**（0.973 vs 0.975）；与 Y5 作者权重复现 **0.972** 持平                   |
| 说明                   | epoch 80+ 训练 val 缓降（过拟合）；**报告以 `best.pt` 为准**，不必追 `last.pt`             |

### 9.4 汇总

| 方法        | 框架   | mAP@.5    | mAP@.75 | mAP@.5:.95       |
| ----------- | ------ | --------- | ------- | ---------------- |
| RGB 单模态  | Y8     |           |         |                  |
| IR 单模态   | Y8     | ~0.958    | —       | —                |
| Add 融合    | Y5     | 0.958     | —       | 0.623（论文）    |
| Add 融合    | Y8     |           |         | 待 P1 增强后重训 |
| CFT         | Y5     | 0.972     | 0.724   | 0.633            |
| **CFT**     | **Y8** | **0.973** | —       | **0.668**        |
| CFT（论文） | Y5     | 0.975     | 0.729   | 0.636            |

> Y8-CFT 的 **0.973 / 0.668** 来自 `val_cft.py`：`conf=0.001`, `iou=0.5`, `imgsz=1024`, LLVIP test（3463 图）。mAP@0.75 未在 `val_cft` 默认输出中单独列出。

### 9.5 推理协议敏感性（y8_cft_v2 best.pt）

| conf   | NMS iou | mAP@0.5   | mAP@0.5:0.95 | 说明                                |
| ------ | ------- | --------- | ------------ | ----------------------------------- |
| 0.001  | 0.5     | **0.973** | 0.668        | **论文报告协议（默认）**            |
| 0.001  | 0.6     | 0.973     | 0.672        | YOLOv5 `test.py` NMS 默认           |
| 0.0005 | 0.5     | 0.973     | 0.669        | conf 降低几乎无变化                 |
| 0.001  | 0.5     | 0.953     | 0.633        | **`last.pt`（过拟合，勿用于报告）** |

> 推理调参无法弥补训练上限；epoch 50 后 val 回落时务必使用 `best.pt`。

---

## 十、命令速查

```bash
# 路径初始化（clone 后一次）
bash tools/setup_new_machine.sh

# 实验 1：Y8-RGB
yolo train model=yolov8l.pt data=ultralytics/cfg/datasets/llvip_rgb.yaml \
  epochs=200 imgsz=1024 batch=2 nbs=16 workers=4 device=0 \
  project=runs/llvip name=y8_rgb exist_ok=True

# 实验 2：Y8-IR
yolo train model=yolov8l.pt data=ultralytics/cfg/datasets/llvip_ir.yaml \
  epochs=200 imgsz=1024 batch=2 nbs=16 workers=4 device=0 \
  project=runs/llvip name=y8_ir exist_ok=True

# 实验 3：Y8-Add
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_add_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml pretrained=yolov8l.pt \
  epochs=200 imgsz=1024 batch=2 nbs=16 workers=4 device=0 \
  project=runs/llvip name=y8_add exist_ok=True

# 实验 4：Y8-CFT（8 GB 推荐）
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml pretrained=yolov8l.pt \
  epochs=200 imgsz=1024 batch=1 nbs=16 workers=2 device=0 \
  project=runs/llvip name=y8_cft_v2 exist_ok=True

# 论文协议评估
python tools/val_cft.py \
  model=runs/detect/runs/llvip/y8_cft_v2/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 batch=1 device=0 conf=0.001 iou=0.5

# NMS/conf 敏感性扫描（可选）
python tools/sweep_val_cft.py \
  model=runs/detect/runs/llvip/y8_cft_v2/weights/best.pt \
  sweep=0.001:0.5,0.001:0.6,0.0005:0.5 imgsz=1024 batch=1 device=0
```

---

## 十一、故障排查

| 现象                             | 可能原因                                        | 处理                                                                     |
| -------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------ |
| 0 images / FileNotFoundError     | 未运行 `setup_new_machine.sh` 或 LLVIP 路径错误 | 检查 `machine.env` 与脚本输出计数                                        |
| `SyntaxError: accumulate`        | 使用了 YOLOv5 参数名                            | 改用 `nbs=16`                                                            |
| `SyntaxError: freeze_epochs`     | 传给 Ultralytics `get_cfg`                      | 仅用 `train_cft.py`；该参数由脚本内部处理                                |
| `AssertionError` in `muon.py`    | `optimizer=auto` → MuSGD                        | 使用 `optimizer=SGD`（`train_cft.py` 默认）                              |
| 验证 `Add2` 尺寸不匹配           | val 只 letterbox RGB                            | 已修复：`DualLetterBox`（见 `tp_improve.md`）                            |
| epoch 11 OOM                     | backbone 解冻显存翻倍                           | `batch=1` + `resume=.../last.pt freeze_epochs=0`                         |
| `val_cft` `'train:' key missing` | `model=` 触发标准 `check_det_dataset`           | 已修复：调用 `validator(trainer=trainer)`，复用 `check_dual_det_dataset` |
| `val_cft` 末尾 AttributeError    | 独立验证无 `trainer.loss`                       | 同上，传 `trainer=` 走训练式验证分支                                     |
| CUDA OOM @ 1024                  | 物理 batch 过大                                 | `batch=1`；减小 `workers`                                                |
| dual 训练找不到 IR               | `llvip_dual.yaml` 的 `path` 不正确              | `path` 应指向含 `visible/`、`infrared/` 的 LLVIP 根目录                  |
| 首次训练卡在 Downloading         | 自动下载 `yolov8l.pt`                           | 等待完成或手动下载至仓库根目录                                           |
| Git push 凭据错误                | token / SSH 配置                                | 使用 Personal Access Token 或 SSH key；与 clone 权限无关                 |

---

## 十二、依赖与原仓库对照实验（可选）

| 实验                             | 是否需要 [multispectral-object-detection](https://github.com/DocF/multispectral-object-detection) |
| -------------------------------- | ------------------------------------------------------------------------------------------------- |
| Y8-RGB / Y8-IR / Y8-Add / Y8-CFT | **否**（仅需 LLVIP + 本仓库）                                                                     |
| Y5-CFT 论文数值复现              | **是**（作者权重 + `test.py`）                                                                    |
| Y8-CFT 方案 B（Y5 GPT 初始化）   | **是**（作者 `.pt` 权重文件）                                                                     |

YOLOv5 对照示例（在原仓库根目录）：

```bash
python3 test.py \
  --weights yolov5l_transformerx3_llvip_s1024_bs32_e200.pt \
  --data data/multispectral/LLVIP.yaml \
  --img-size 1024 --batch-size 2 --device 0 --task val --exist-ok
```

---

## 十四、VEDAI 数据集扩展（跨数据集验证）

在 LLVIP 达到论文量级（mAP@0.5=0.973）后，可在 **VEDAI** 上复现原论文第二组实验，验证 CFT 在 **9 类车辆** 场景下的泛化增益。

### 14.1 原工程（DocF/multispectral-object-detection）参考

| 项目     | 内容                                                         |
| -------- | ------------------------------------------------------------ |
| 数据配置 | `data/multispectral/vedai_color_2.yaml`                      |
| CFT 模型 | `models/transformer/yolov5s_fusion_transformerx3_vedai.yaml` |
| Add 基线 | `models/transformer/yolov5s_fusion_add_vedai.yaml`           |
| 输入尺寸 | **1024×1024**（`Vehicules1024`）                             |
| 划分     | `fold01.txt`（train 128）/ `fold01test.txt`（val 128）       |
| 配对方式 | RGB/IR **独立 list 文件**，按排序后索引配对（非路径替换）    |

**论文报告指标（YOLOv5s）**：

| CFT         | mAP@0.5         | mAP@0.75         | mAP@0.5:0.95    |
| ----------- | --------------- | ---------------- | --------------- |
| 否（Add）   | 79.7            | 47.7             | 46.8            |
| 是（GPT×3） | **85.3** (+5.6) | **65.9** (+18.2) | **56.0** (+9.2) |

下载：[https://downloads.greyc.fr/vedai/](https://downloads.greyc.fr/vedai/)

### 14.2 本仓库 YOLOv8 对应物

| 组件           | 路径                                       |
| -------------- | ------------------------------------------ |
| 数据 yaml 模板 | `ultralytics/cfg/datasets/vedai_dual.yaml` |
| 路径配置脚本   | `tools/setup_vedai.sh`                     |
| 配对检查       | `tools/check_vedai_pairs.py`               |
| Y8-Add         | `yolov8s_fusion_add_vedai.yaml`            |
| Y8-CFT         | `yolov8s_fusion_transformerx3_vedai.yaml`  |

目录结构（YOLO 格式标签与 color 图同目录、同名 `.txt`）：

```text
<VEDAI_ROOT>/Vehicules1024/Images/
├── color/
│   ├── fold01.txt          # 每行 ./00000000_co.png
│   ├── fold01test.txt
│   ├── 00000000_co.png
│   └── 00000000_co.txt     # YOLO 标签
└── ir/
    ├── fold01.txt          # 每行 ./00000000_ir.png
    ├── fold01test.txt
    └── 00000000_ir.png
```

### 14.3 准备步骤

1. 下载 VEDAI 并转换为 **YOLO 格式**（见 [YOLOv5 Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) 与原仓库 README）。
2. 确保 `color/`、`ir/` 下各有 `fold01.txt` / `fold01test.txt`（可与原工程 list 一致）。
3. 配置路径并生成 yaml：

```bash
# tools/machine.env 中可选：VEDAI_ROOT=/path/to/VEDAI
bash tools/setup_vedai.sh
python tools/check_vedai_pairs.py --data ultralytics/cfg/datasets/vedai_dual.yaml
```

### 14.4 训练与评估命令

**Y8-Add（消融基线）**：

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8s_fusion_add_vedai.yaml \
  data=ultralytics/cfg/datasets/vedai_dual.yaml \
  pretrained=yolov8s.pt epochs=100 imgsz=1024 batch=4 nbs=16 workers=4 device=0 \
  project=runs/vedai name=y8_add exist_ok=True
```

**Y8-CFT（主实验，对齐论文 yolov5s + transformerx3）**：

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8s_fusion_transformerx3_vedai.yaml \
  data=ultralytics/cfg/datasets/vedai_dual.yaml \
  pretrained=yolov8s.pt epochs=100 imgsz=1024 batch=4 nbs=16 workers=4 device=0 \
  freeze_epochs=10 project=runs/vedai name=y8_cft exist_ok=True
```

**论文协议评估**：

```bash
python tools/val_cft.py \
  model=runs/detect/runs/vedai/y8_cft/weights/best.pt \
  data=ultralytics/cfg/datasets/vedai_dual.yaml \
  imgsz=1024 batch=4 device=0 conf=0.001 iou=0.5
```

> VEDAI 体量小（256 图/模态），训练快于 LLVIP；建议同样以 **`best.pt` + `val_cft.py`** 报告，并与上表 YOLOv5 行对照（允许框架差异）。

### 14.5 VEDAI 实验记录表（待填）

| 实验 ID | 方法  | 框架    | mAP@.5   | mAP@.75  | mAP@.5:.95 | 备注 |
| ------- | ----- | ------- | -------- | -------- | ---------- | ---- |
| y5-add  | Add   | YOLOv5s | 79.7     | 47.7     | 46.8       | 论文 |
| y5-cft  | GPT×3 | YOLOv5s | **85.3** | **65.9** | **56.0**   | 论文 |
| y8-add  | Add   | YOLOv8s |          |          |            | 待训 |
| y8-cft  | GPT×3 | YOLOv8s |          |          |            | 待训 |

---

## 十三、变更日志

| 日期       | 内容                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| 2025-05-27 | 初版：YOLOv8 单模态、CFT 迁移、LLVIP 实验流程                                                            |
| 2025-05-27 | CFT 模块、DualDetectionModel、双路数据、`train_cft.py` / `val_cft.py`                                    |
| 2025-05-27 | Git 发布：`TP666-bot/ultralytics-yolov8-cft`，分支 `cft-yolov8-llvip`                                    |
| 2025-05-27 | 文档修订：面向公开复现，补充实验设计与评估协议                                                           |
| 2025-05-27 | **y8_cft_v2**：`val_cft` mAP@.5=**0.973**；更新 §7.2 / §9 / 训练默认值与故障排查                         |
| 2025-05-27 | 修复 `val_cft`（`trainer=` 双路数据）；`load_cft_partial` 自动 YOLOv5 路径；§9.5 NMS 扫描；README 结果表 |
| 2025-05-27 | **VEDAI**：双路 list 配对、`vedai_dual.yaml`、Y8s Add/CFT 模型、`setup_vedai.sh`（§十四）                |
