# CFT → YOLOv8 迁移与 LLVIP 实验记录

本文档在 **Ultralytics 仓库**（`~/my_new_space/ultralytics`）内维护 YOLOv8 单模态基线、双路融合、**YOLOv8+CFT** 迁移与完整消融对比。**不修改** `~/my_new_space/multispectral-object-detection` 源码；该仓库作为 **CFT 参考实现、LLVIP 数据、YOLOv5 对照实验与作者权重** 的只读来源。

YOLOv5+CFT 部署见：[cft_yolo_config.md](../multispectral-object-detection/cft_yolo_config.md)。

---

## 一、YOLOv5+CFT 验证结果说明（参考上界）

在 `multispectral-object-detection` 中用作者权重跑 `test.py`（LLVIP **test**，双路 RGB+IR，`imgsz=1024`，`batch=2`，`conf=0.001`，`iou=0.5`）：

| 指标 | 本机复现 | 论文 README（LLVIP CFT） |
|------|----------|--------------------------|
| P | 0.967 | — |
| R | 0.931 | — |
| **mAP@0.5** | **0.972** | **0.975** |
| mAP@0.75 | 0.724 | 0.729 |
| **mAP@0.5:0.95** | **0.633** | **0.636** |

论文 LLVIP **无 CFT 双路 Baseline（Add 融合）**：mAP@0.5 **95.8**，mAP **62.3**（见 multispectral README 表格）。完整消融应包含单模态、Add、CFT 三档。

此结果作为 **YOLOv5l + CFT + 1024** 的对比上界；YOLOv8 实验是**新架构**，数值作参考、不要求完全一致。

---

## 二、工程隔离与目录规划

| 内容 | 位置 | 说明 |
|------|------|------|
| YOLOv5 + CFT 原实现 | `~/my_new_space/multispectral-object-detection` | 只读：数据、权重、对照 train/test |
| YOLOv8 实验与 CFT 迁移 | `~/my_new_space/ultralytics` | 本文档、`yolo` CLI、扩展代码 |
| **Git 远程（你的仓库）** | https://github.com/TP666-bot/ultralytics-yolov8-cft | 分支 **`cft-yolov8-llvip`**；迁移见第十二节 |
| LLVIP 数据 | `multispectral-object-detection/LLVIP/` | 不复制；symlink 或 yaml 绝对路径引用 |
| 单模态 yaml | `ultralytics/cfg/datasets/llvip_rgb.yaml` / `llvip_ir.yaml` | 阶段 B |
| 双模态 yaml | `ultralytics/cfg/datasets/llvip_dual.yaml` | 阶段 C（四路径，需扩展 Trainer） |
| 模型 yaml | `ultralytics/cfg/models/v8/yolov8l_fusion_add_llvip.yaml` | 实验 3 |
| | `ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml` | 实验 4 |
| CFT 模块 | `ultralytics/nn/modules/cft.py` 等 | 已实现，见 6.3 |
| 训练日志 | `ultralytics/runs/detect/runs/llvip/` | `yolo train` 默认前缀 `runs/detect/`（见 5.6） |

**单模态 symlink（阶段 B，一次性）：**

```bash
# 本机已有数据时：
bash ~/my_new_space/ultralytics/tools/setup_llvip_symlinks.sh

# 新电脑解压后（必做，会重写 yaml 路径 + symlink）：
bash ~/my_new_space/ultralytics/tools/setup_new_machine.sh
```

详见 **第十二节「迁移到新电脑」**。

---

## 三、环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/my_new_space/conda_envs/ms_cft

cd ~/my_new_space/ultralytics
pip install -e .
```

大文件下载若根分区满：`source ~/my_new_space/multispectral-object-detection/tools/env_large_downloads.sh`（仅环境变量）。

---

## 四、完整消融路线图

与论文 LLVIP 表格逻辑一致，共 **4 组 YOLOv8 实验** + **YOLOv5 对照**（便于迁移期对比）：

| # | 实验 | 框架 | 模态/融合 | 模型配置 | 预训练/权重 | 训练入口 |
|---|------|------|-----------|----------|-------------|----------|
| 1 | Y8-RGB | Ultralytics | 单路 RGB | `yolov8l.pt` | `yolov8l.pt` | `yolo train` + `llvip_rgb.yaml` |
| 2 | Y8-IR | Ultralytics | 单路 IR | `yolov8l.pt` | `yolov8l.pt` | `yolo train` + `llvip_ir.yaml` |
| 3 | Y8-Add | Ultralytics + CFT 扩展 | RGB+IR，P3/P4/P5 **Add** | `yolov8l_fusion_add_llvip.yaml` | `yolov8l.pt`（双 backbone，`strict=False`） | `python tools/train_cft.py` + `llvip_dual.yaml` |
| 4 | Y8-CFT | Ultralytics + CFT 扩展 | RGB+IR，**GPT×3** | `yolov8l_fusion_transformerx3_llvip.yaml` | `yolov8l.pt` + GPT 随机初始化；可选从 Y5 CFT 权重**部分**加载 GPT | `python tools/train_cft.py` |
| — | Y5-Add（对照） | multispectral | Add 融合 | `yolov5l_fusion_add_llvip.yaml` | `yolov5l.pt` | `python3 train.py` |
| — | Y5-CFT（对照） | multispectral | CFT | `yolov5l_fusion_transformerx3_llvip.yaml` | 作者 `yolov5l_transformerx3_llvip_s1024_bs32_e200.pt` | `test.py` 直接评 |

**推荐顺序**：1 → 2 → **实现第五节迁移** → 3 → 4。1/2 验证数据链；3/4 验证融合与 CFT。

---

## 五、阶段 B：实验 1 & 2（YOLOv8 单模态）

**目的**：对应论文 YOLOv5-RGB / YOLOv5-IR；**不含**双路融合。

### 5.1 数据

| 文件 | train | val |
|------|-------|-----|
| `llvip_rgb.yaml` | `visible/train` | `visible/test` |
| `llvip_ir.yaml` | `infrared/train` | `infrared/test` |

### 5.2 训练（实验 1：RGB）

**本机推荐（RTX 4060 Ti 8GB，已实测）**：`yolov8l` + `imgsz=1024` 请 **不要用 `batch=16`**；从 **`batch=2`** 起。详见 **5.6 显存与 batch**。

```bash
cd ~/my_new_space/ultralytics
conda activate ~/my_new_space/conda_envs/ms_cft

# 首次会下载 yolov8l.pt（~84MB），GitHub 慢时进度条会停很久，属正常
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

| 参数 | 本机 4060 Ti 建议 | 说明 |
|------|-------------------|------|
| `batch` | **2**（OOM 则 **1**） | 单次 forward 的图像数 |
| `nbs` | **16** | **名义 batch**，控制梯度累积；**不是** YOLOv5 的 `accumulate` |
| `workers` | **4**（仍 OOM/报错则 **0**） | 减轻 pin_memory / 主机内存压力 |
| `imgsz` | **1024** | 与论文 / Y5 CFT 权重一致 |

**等效 batch**：Ultralytics 内部 `accumulate = round(nbs / batch)`。例如 `batch=2`、`nbs=16` → 每 8 步更新一次，等效 batch ≈ **16**（接近论文 `batch-size=32` 的一半，可改为 `nbs=32` 若显存允许）。

### 5.3 验证（实验 1）

权重实际保存在 **`runs/detect/runs/llvip/<name>/`**（见 5.6），不要写错路径：

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

### 5.4 实验 2（IR）

将 5.2 / 5.3 中 `llvip_rgb` → `llvip_ir`，`y8_rgb` → `y8_ir`。

### 5.5 阶段 B 完成标准

- [ ] RGB、IR 各跑完 200 epoch  
- [ ] 在 **test** 划分有 mAP，填入第十节表格  

### 5.6 显存、OOM 与常见报错（本机实测 2025-05-27）

**硬件**：NVIDIA GeForce RTX 4060 Ti **8188 MiB**；桌面（Xorg、Firefox、Cursor 等）常占 **~1.1 GB+**，训练可用约 **7 GB**。

#### 5.6.1 失败案例：`batch=16` @ 1024

日志现象：

1. `WARNING ⚠️ CUDA out of memory with batch=16. Reducing to batch=8...`
2. 再 OOM → 自动降为 `batch=4`
3. 仍失败：`CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED`
4. 连带：`ConnectionResetError` in `_pin_memory_loop`（dataloader worker 被杀死，**不是根因**）

**结论**：`yolov8l @ 1024` 在本机 **batch≥4 训练仍可能崩**；应 **一开始就用 `batch=2`**，不要依赖自动降 batch（每次降级会重启 dataloader，且 batch=4 仍可能 OOM）。

#### 5.6.2 无效参数：`accumulate=8`

Ultralytics **8.4.x 的 `yolo train` 不支持 `accumulate`**：

```text
SyntaxError: 'accumulate' is not a valid YOLO argument.
```

| 框架 | 梯度累积写法 |
|------|----------------|
| YOLOv5（multispectral `train.py`） | `accumulate=N` 或调 batch |
| **Ultralytics YOLOv8** | **`nbs=<名义batch>`**，内部 `accumulate ≈ nbs / batch` |

示例：想要「物理 batch=2、等效 batch=16」→ 使用 **`batch=2 nbs=16`**，**不要**写 `accumulate=8`。

#### 5.6.3 日志与权重路径

`yolo train` 的 `project=runs/llvip` 会被框架加上默认前缀，实际目录为：

```text
~/my_new_space/ultralytics/runs/detect/runs/llvip/y8_rgb/
  weights/best.pt
  weights/last.pt
  results.csv
  ...
```

第一次训练前会扫描 **12025** train / **3463** val 并生成 `.cache`（在 `LLVIP/visible/` 下），第二次启动会快很多。

#### 5.6.4 4060 Ti 8GB @ imgsz=1024 参考

| 实验 | 模型 | 建议 `batch` | 建议 `nbs` | 入口 |
|------|------|-------------|-----------|------|
| Y8-RGB / Y8-IR | `yolov8l.pt` | **2**（OOM→1） | **16** | `yolo train` |
| Y8-Add | fusion add yaml | **2** | **16** | `tools/train_cft.py` |
| Y8-CFT | fusion CFT yaml | **1~2** | **16** | `tools/train_cft.py` |
| 显存仍不足 | `yolov8m.pt` | 4 | 16 | 小模型基线 |

训练前可 `nvidia-smi` 查看占用；关闭 Firefox 等可释放数百 MB 显存。

---

## 六、阶段 C 前置：源码与权重迁移

实验 3、4 依赖第六节 CFT 扩展（已实现）；入口为 **`python tools/train_cft.py`**，不是 `yolo train`。

### 6.1 必迁源码（只读复制 + 适配 v8）

| 步骤 | multispectral 源 | ultralytics 目标 | 说明 |
|------|------------------|------------------|------|
| ① | `models/common.py`：`Add`、`Add2`、`GPT`、`myTransformerBlock` | `ultralytics/nn/modules/cft.py` | 融合核心；在 `modules/__init__.py` 导出 |
| ① | `models/yolo.py` / `parse_model` 中对上述类的分支 | `ultralytics/nn/tasks.py` `parse_model` | 能解析 yaml 里的 `GPT`/`Add`/`Add2` |
| ② | `models/transformer/yolov5l_fusion_add_llvip.yaml` | `cfg/models/v8/yolov8l_fusion_add_llvip.yaml` | **C3→C2f**，**SPP→SPPF**，**Detect→v8 Detect**；双路 backbone 索引逻辑不变 |
| ② | `models/transformer/yolov5l_fusion_transformerx3_llvip.yaml` | `cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml` | 同上 + 保留 3 处 GPT + Add2 |
| ③ | `models/yolo_test.py`：`Model.forward_once(x, x2)` | `ultralytics/nn/tasks.py`：`DualDetectionModel` | 处理 `m.f == -4` 第二路输入、多输入 `GPT`/`Add` |
| ④ | `utils/datasets.py`：`LoadMultiModalImagesAndLabels`、collate | `ultralytics/data/multimodal.py` + 改 `DetectionTrainer.get_dataloader` / `preprocess_batch` | 成对 RGB/IR；标签仍用 RGB 侧 `.txt` |
| ⑤ | `test.py` 中 mAP 计算 / 双路推理 | `tools/val_dual_llvip.py` 或自定义 `DualValidator` | **最终对比建议与 Y5 用同一套 mAP 逻辑** |
| ⑥ | — | `DetectionTrainer` 子类或 `model=yolov8l_fusion_*.yaml` 入口 | 识别 `llvip_dual.yaml` 四路径 |

**v8 结构对照（改 yaml 时）**：

| YOLOv5（CFT 仓库） | YOLOv8（Ultralytics） |
|--------------------|------------------------|
| `Focus` | 首层 `Conv`（v8 无 Focus） |
| `C3` | `C2f` |
| `SPP` | `SPPF` |
| `Detect`（anchor-based） | `Detect`（anchor-free，无 anchors 段） |

参考 v8 模板：`ultralytics/cfg/models/v8/yolov8.yaml`（scale `l`：`depth=1.0, width=1.0`）。

### 6.2 权重怎么用（重要）

| 权重文件 | 能否直接用于 YOLOv8 | 用法 |
|----------|---------------------|------|
| `yolov8l.pt` | ✅ | 实验 1–4：单模态或 **双 backbone 各加载一份**（`strict=False`，跳过 head/GPT） |
| `yolov5l.pt` | ❌ 整模 | 仅 YOLOv5 对照实验 |
| `yolov5l_transformerx3_llvip_s1024_bs32_e200.pt` | ❌ 整模载入 Y8 | 架构不同（C3/C2f、Detect、head）；**仅作 Y5-CFT 评测与论文对齐** |
| 同上（作者 CFT 权重） | ⚠️ 部分 | 迁入 **`GPT`/`Add2`/`Add` 模块且结构超参一致** 时，可用脚本按 key 匹配加载 Transformer 部分；**backbone/head 仍建议用 `yolov8l.pt` 初始化** |

**部分加载作者 CFT 权重（实验 4 可选，迁移完成后）：**

```bash
# 示例：在 ultralytics 根目录执行（需先实现 load_cft_partial.py）
python tools/load_cft_partial.py \
  --y8-cfg ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  --y8-weights yolov8l.pt \
  --y5-cft ~/my_new_space/multispectral-object-detection/yolov5l_transformerx3_llvip_s1024_bs32_e200.pt \
  --out runs/llvip/y8_cft_init.pt
```

逻辑：`yolov8l.pt` 填双路 Conv/C2f/SPPF；从 Y5 checkpoint 中提取名称含 `GPT`、`Add2`、`trans_blocks`、`pos_emb` 且 shape 一致的 tensor。

### 6.3 实施状态（2025-05-27 已实现）

| 组件 | 状态 |
|------|------|
| `llvip_dual.yaml` | ✅ |
| CFT 模块 `ultralytics/nn/modules/cft.py` | ✅ |
| `parse_model` 支持 `GPT`/`Add`/`Add2`/`f=-4` | ✅ `ultralytics/nn/tasks.py` |
| `DualDetectionModel` | ✅ `ultralytics/nn/tasks_dual.py` |
| `yolov8l_fusion_add_llvip.yaml` | ✅ |
| `yolov8l_fusion_transformerx3_llvip.yaml` | ✅ |
| 双路 Dataset | ✅ `ultralytics/data/dual_stream.py` |
| CFT Trainer / Validator | ✅ `ultralytics/models/yolo/detect/cft_train.py` |
| 训练 / 验证入口 | ✅ `tools/train_cft.py`、`tools/val_cft.py` |
| 部分加载 Y5 GPT 权重 | ✅ `tools/load_cft_partial.py` |
| 与 Y5 `test.py` 逐点一致 mAP | ⚠️ 使用 Ultralytics DetMetrics；最终对比可再导出权重用 `test.py` |

**说明**：融合实验用 **`python tools/train_cft.py`**，不用标准 `yolo train`（后者只支持单路 RGB）。

---

## 七、阶段 C：实验 3 — YOLOv8 双路 Add 融合

**对应论文**：LLVIP 无 CFT 行（mAP@0.5 **95.8**）。

### 7.1 训练（实验 3）

```bash
cd ~/my_new_space/ultralytics
conda activate ~/my_new_space/conda_envs/ms_cft

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

`pretrained=yolov8l.pt` 会自动加载 RGB backbone，并 **复制到 IR 流**（layers 10–19）。

### 7.2 验证（实验 3）

```bash
python tools/val_cft.py \
  model=runs/llvip/y8_add/weights/best.pt \
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

---

## 八、阶段 C：实验 4 — YOLOv8 + CFT（GPT×3）

### 8.1 训练（实验 4）

**方案 A — 仅 v8 预训练（推荐首轮）**

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt \
  epochs=200 \
  imgsz=1024 \
  batch=2 \
  nbs=16 \
  workers=4 \
  device=0 \
  project=runs/llvip \
  name=y8_cft \
  exist_ok=True
```

**方案 B — 从 Y5 CFT 部分初始化 GPT**

```bash
python tools/load_cft_partial.py \
  --y8-cfg ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  --y8-weights yolov8l.pt \
  --y5-cft ~/my_new_space/multispectral-object-detection/yolov5l_transformerx3_llvip_s1024_bs32_e200.pt \
  --out runs/llvip/y8_cft_init.pt

python tools/train_cft.py \
  model=runs/llvip/y8_cft_init.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  epochs=200 \
  imgsz=1024 \
  batch=2 \
  device=0 \
  project=runs/llvip \
  name=y8_cft_from_y5gpt \
  exist_ok=True
```

### 8.2 验证（实验 4）

```bash
python tools/val_cft.py \
  model=runs/llvip/y8_cft/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 \
  batch=2 \
  device=0 \
  conf=0.001 \
  iou=0.5 \
  project=runs/llvip \
  name=y8_cft_val \
  exist_ok=True
```

---

## 九、公平对比 Checklist

| 项 | 要求 |
|----|------|
| 数据划分 | 与 `LLVIP.yaml` / `llvip_dual.yaml` 四路径一致 |
| 输入尺寸 | `imgsz=1024` |
| 评估集 | 仅 `visible/test` + `infrared/test`（3463 对） |
| NMS | `conf=0.001`，`iou=0.5`（对齐 `test.py`） |
| 指标 | mAP@0.5、mAP@0.75、mAP@0.5:0.95 |
| 跨框架对比 | **同一套** `val_dual_llvip.py` 或 multispectral `test.py`（勿混用 `yolo val` 与 `test.py` 直接比） |
| 预期 | Y8 为 new experiment；论文数字是 Y5l 上的结果 |

---

## 十、实验记录表（随跑随填）

### 10.1 YOLOv5 对照（multispectral，`test.py`，1024）

| 实验 ID | 方法 | 权重 | mAP@.5 | mAP@.75 | mAP@.5:.95 | 备注 |
|---------|------|------|--------|---------|------------|------|
| y5-cft-llvip | CFT GPT×3 | `yolov5l_transformerx3_llvip_s1024_bs32_e200.pt` | 0.972 | 0.724 | 0.633 | 已复现 |
| y5-add-llvip | Add 融合 | 自训 `runs/train/y5_add_llvip/weights/best.pt` | | | | 论文参考 0.958 |
| y5-rgb | 单路 RGB | 可选自训 | | | | |
| y5-ir | 单路 IR | 可选自训 | | | | |

### 10.2 阶段 B：YOLOv8 单模态（Ultralytics）

| 实验 ID | 模态 | batch | 权重 | mAP@.5 | mAP@.75 | mAP@.5:.95 | 备注 |
|---------|------|-------|------|--------|---------|------------|------|
| y8-rgb | RGB | 2 / nbs=16 | `runs/detect/runs/llvip/y8_rgb/weights/best.pt` | | | | 4060 Ti 8GB；勿用 batch=16 |
| y8-ir | IR | | `runs/llvip/y8_ir/weights/best.pt` | | | | |

### 10.3 阶段 C：YOLOv8 融合 / CFT

| 实验 ID | 方法 | 权重 | mAP@.5 | mAP@.75 | mAP@.5:.95 | 备注 |
|---------|------|------|--------|---------|------------|------|
| y8-add | 双路 Add | `runs/llvip/y8_add/weights/best.pt` | | | | 对比 y5-add / 论文 0.958 |
| y8-cft | GPT×3 | `runs/llvip/y8_cft/weights/best.pt` | | | | 对比 y5-cft / 论文 0.975 |

### 10.4 汇总对比（实验结束后填写）

| 方法 | 框架 | mAP@.5 | mAP@.75 | mAP@.5:.95 |
|------|------|--------|---------|------------|
| RGB 单模态 | Y8 | | | |
| IR 单模态 | Y8 | | | |
| Add 融合 | Y5 | | | |
| Add 融合 | Y8 | | | |
| CFT | Y5 | 0.972 | 0.724 | 0.633 |
| CFT | Y8 | | | |
| 论文 CFT | Y5 | 0.975 | 0.729 | 0.636 |

---

## 十一、常用命令速查

```bash
# symlink（阶段 B）
bash ~/my_new_space/ultralytics/tools/setup_llvip_symlinks.sh

# 实验 1：Y8 RGB（4060 Ti 8GB：batch=2 nbs=16，勿用 accumulate=）
cd ~/my_new_space/ultralytics
yolo train model=yolov8l.pt data=ultralytics/cfg/datasets/llvip_rgb.yaml \
  epochs=200 imgsz=1024 batch=2 nbs=16 workers=4 device=0 \
  project=runs/llvip name=y8_rgb exist_ok=True

# 实验 2：Y8 IR（改 yaml 与 name）
yolo train model=yolov8l.pt data=ultralytics/cfg/datasets/llvip_ir.yaml \
  epochs=200 imgsz=1024 batch=2 nbs=16 workers=4 device=0 \
  project=runs/llvip name=y8_ir exist_ok=True

# 验证（注意权重路径含 runs/detect/）
yolo val model=runs/detect/runs/llvip/y8_rgb/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_rgb.yaml imgsz=1024 batch=4 \
  conf=0.001 iou=0.5 device=0

# 实验 3/4：CFT 融合（tools/train_cft.py，非 yolo train）
python tools/train_cft.py model=ultralytics/cfg/models/v8/yolov8l_fusion_add_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml pretrained=yolov8l.pt \
  epochs=200 imgsz=1024 batch=2 nbs=16 workers=4 device=0 \
  project=runs/llvip name=y8_add exist_ok=True

python tools/train_cft.py model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml pretrained=yolov8l.pt \
  epochs=200 imgsz=1024 batch=2 nbs=16 workers=4 device=0 \
  project=runs/llvip name=y8_cft exist_ok=True

# Y5-CFT 对照
cd ~/my_new_space/multispectral-object-detection
python3 test.py --weights yolov5l_transformerx3_llvip_s1024_bs32_e200.pt \
  --data data/multispectral/LLVIP.yaml --img-size 1024 --batch-size 2 \
  --device 0 --task val --exist-ok
```

---

## 十二、迁移到新电脑（Git 传输，推荐）

**结论：可行**，但**不能只 clone 就训练**。LLVIP 数据集、conda 环境、本机 yaml 路径需在新机器上单独准备；clone 后必须运行 `tools/setup_new_machine.sh`。

**推荐用 Git** 传到 [TP666-bot/ultralytics-yolov8-cft](https://github.com/TP666-bot/ultralytics-yolov8-cft)（比 zip 更稳，不丢权限、不夹带 `runs/` 大文件）。zip 仍可作为备选，见 **12.5**。

### 12.1 用 Git 保存与传输

#### 12.1.1 仓库说明

| 项 | 值 |
|----|-----|
| GitHub 账号 | [TP666-bot](https://github.com/TP666-bot) |
| 仓库名 | **`ultralytics-yolov8-cft`** |
| 上游（只读） | `https://github.com/ultralytics/ultralytics.git` → remote **`origin`** |
| 你的 fork 远程 | `https://github.com/TP666-bot/ultralytics-yolov8-cft.git` → remote **`github`** |
| CFT 功能分支 | **`cft-yolov8-llvip`**（含 YOLOv8+CFT 全部改动） |
| 实验文档 | 仓库根目录 **`cft_yolov8_config.md`**（本文） |

**不会进 Git 的内容**（已在 `.gitignore`）：`runs/`、`*.pt`、`/datasets` symlink、`tools/machine.env`、本机生成的 `llvip_*.yaml`（clone 后由 `setup_new_machine.sh` 生成）。

#### 12.1.2 本机：首次推送到 GitHub

**前提**：GitHub 已登录；本机 `git config user.name` / `user.email` 已设置。

**① 在 GitHub 网页创建空仓库**

1. 打开 https://github.com/new  
2. Owner 选 **TP666-bot**  
3. Repository name：**`ultralytics-yolov8-cft`**  
4. 选 **Private**（推荐，含实验配置）或 Public  
5. **不要**勾选 “Add a README” / “Add .gitignore”（保持空仓库）  
6. Create repository  

**② 在本机 ultralytics 目录提交并推送**

```bash
cd ~/my_new_space/ultralytics

# 确认 CFT 分支存在（若已在该分支可跳过 checkout）
git checkout -b cft-yolov8-llvip

# 查看将要提交的文件（不应含 runs/、*.pt、machine.env）
git status

# 添加 CFT 相关改动（yaml 数据集配置由 setup 脚本生成，已在 .gitignore）
git add .gitignore \
  cft_yolov8_config.md \
  tools/ \
  ultralytics/cfg/models/v8/yolov8l_fusion_add_llvip.yaml \
  ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  ultralytics/data/dual_stream.py \
  ultralytics/data/dual_utils.py \
  ultralytics/models/yolo/detect/cft_train.py \
  ultralytics/nn/modules/cft.py \
  ultralytics/nn/modules/__init__.py \
  ultralytics/nn/tasks.py \
  ultralytics/nn/tasks_dual.py

git commit -m "$(cat <<'EOF'
Add YOLOv8 CFT port for LLVIP dual-stream detection.

Port GPT/Add/Add2 fusion, dual dataset, CFT trainer, and experiment docs
from multispectral YOLOv5 CFT to Ultralytics YOLOv8.
EOF
)"

# 添加上游与你的远程（origin 已是 ultralytics 官方，勿改）
git remote add github https://github.com/TP666-bot/ultralytics-yolov8-cft.git 2>/dev/null || true

# 推送 CFT 分支到你的仓库（HTTPS，按提示输入 GitHub 用户名 + Personal Access Token）
git push -u github cft-yolov8-llvip

# 可选：把该分支设为 GitHub 默认分支，或再推 main：
# git push -u github cft-yolov8-llvip:main
```

**HTTPS 认证**：GitHub 已不支持账号密码 push，需 [Personal Access Token (classic)](https://github.com/settings/tokens)（勾选 `repo` 权限），密码处粘贴 token。

**SSH 方式**（若已配置 `~/.ssh/id_ed25519.pub` 并添加到 GitHub）：

```bash
git remote set-url github git@github.com:TP666-bot/ultralytics-yolov8-cft.git
git push -u github cft-yolov8-llvip
```

**③ 日常更新（改代码后同步到 GitHub）**

```bash
cd ~/my_new_space/ultralytics
git add -p          # 或 git add <文件>
git commit -m "描述本次改动"
git push github cft-yolov8-llvip
```

#### 12.1.3 新电脑：clone 并复现实验

```bash
mkdir -p ~/my_new_space
cd ~/my_new_space

# HTTPS
git clone -b cft-yolov8-llvip https://github.com/TP666-bot/ultralytics-yolov8-cft.git ultralytics

# 或 SSH
# git clone -b cft-yolov8-llvip git@github.com:TP666-bot/ultralytics-yolov8-cft.git ultralytics

cd ultralytics
cat cft_yolov8_config.md    # 完整实验说明
```

然后按 **12.2** 放置 LLVIP，**12.3** 配置环境与路径，**12.4** 冒烟测试。

**可选：跟踪官方 Ultralytics 更新**

```bash
git remote -v
# origin -> ultralytics/ultralytics（clone 时若保留了 origin）
git fetch origin
# 需要时再 merge/rebase；CFT 实验一般不必频繁同步上游
```

### 12.2 LLVIP 下载后放在哪（新机器目录规划）

推荐与 ultralytics **平级**放数据（不必克隆整个 multispectral 仓库）：

```text
<WORKSPACE>/                          # 例如 /home/you/my_new_space
├── ultralytics/                      # git clone 的仓库
└── LLVIP/                            # ← 数据集放这里（推荐）
    ├── visible/
    │   ├── train/   *.jpg + *.txt
    │   └── test/    *.jpg + *.txt
    └── infrared/
        ├── train/
        └── test/
```

也支持旧布局（与 multispectral 仓库一致）：

```text
<WORKSPACE>/multispectral-object-detection/LLVIP/...
```

**下载来源**：[LLVIP download_dataset.md](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md)（百度网盘 / Google Drive 等）。

**标签格式**：每个 `.jpg` 旁要有同名 `.txt`（YOLO 格式，`0 cx cy w h`，单类 person）。若官方包只有 VOC `Annotations/`，需要转换：

```bash
# 从本机 multispectral 仓库拷贝脚本，或在新机器 clone 该仓库仅用于转换
python3 tools/voc_to_yolo_llvip.py --dataset-root <WORKSPACE>/LLVIP
```

（脚本路径：`multispectral-object-detection/tools/voc_to_yolo_llvip.py`。）

**数量自检**（与文档一致）：每个 modality 约 **train 12025**、**test 3463** 张。

### 12.3 新机器：环境与路径配置

**① 放置 LLVIP** 到 `~/my_new_space/LLVIP/`（见 12.2）。

**② 配置路径（必做；clone 后尚无 llvip_*.yaml）**

```bash
cd ~/my_new_space/ultralytics
cp tools/machine.env.example tools/machine.env
# 编辑 machine.env，把 WORKSPACE 改成新机器实际路径，例如：
#   WORKSPACE=/home/you/my_new_space

bash tools/setup_new_machine.sh
```

该脚本会：

- 重写 `llvip_rgb.yaml` / `llvip_ir.yaml` / `llvip_dual.yaml` 中的 **绝对 path**
- 在 `datasets/llvip_rgb`、`datasets/llvip_ir` 下重建 **symlink**
- 打印各 split 图片数量

**③ 重建 Python 环境**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -p ~/my_new_space/conda_envs/ms_cft python=3.10 -y
conda activate ~/my_new_space/conda_envs/ms_cft

# 按新 GPU 选 CUDA 版 PyTorch，例如：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

cd ~/my_new_space/ultralytics
pip install -e .
```

**④ 冒烟测试**

```bash
yolo train model=yolov8l.pt data=ultralytics/cfg/datasets/llvip_rgb.yaml \
  epochs=1 imgsz=1024 batch=2 device=0 project=runs/llvip name=smoke exist_ok=True
```

能跑完 1 epoch 即说明数据链 + 环境 OK；再按第五节 / 第七节开 200 epoch 正式实验。

### 12.5 备选：zip 压缩传输

无 Git / 无网络时可用 zip（不如 Git 可靠）：

| 内容 | 是否打进 zip | 说明 |
|------|-------------|------|
| 整个 `ultralytics/` 源码（含 CFT 修改） | ✅ **必须** | 阶段 B/C 的全部代码在这里 |
| `runs/` 训练日志与权重 | ❌ 建议排除 | 体积大；新机器重新训或单独拷贝 `best.pt` |
| `datasets/` 下的 symlink | ❌ 不必依赖 | 解压后 symlink 会断；用脚本重建 |
| `*.cache` | ❌ 可排除 | 首次训练会自动生成 |
| conda 环境 `ms_cft` | ❌ **不能** zip | 在新机器按 12.3 重建 |

```bash
cd ~/my_new_space
zip -r ultralytics_y8_cft.zip ultralytics \
  -x "ultralytics/runs/*" \
  -x "ultralytics/**/__pycache__/*" \
  -x "ultralytics/**/*.pyc" \
  -x "ultralytics/datasets/*" \
  -x "ultralytics/.git/*"
```

解压后同样执行 **12.3** 的 `setup_new_machine.sh` 与环境步骤。

### 12.6 只跑 YOLOv8 还需要 multispectral 吗？

| 实验 | 是否需要 multispectral 仓库 |
|------|------------------------------|
| Y8-RGB / Y8-IR / Y8-Add / Y8-CFT | **不需要**（只要 LLVIP + ultralytics） |
| Y5-CFT 对照、`test.py` 复现论文数 | **需要**（权重 + `data/multispectral/LLVIP.yaml`） |
| 从 Y5 CFT 权重部分初始化 GPT（实验 4 方案 B） | **需要**作者 `yolov5l_transformerx3_llvip_s1024_bs32_e200.pt` |

### 12.7 常见失败原因

| 现象 | 原因 | 处理 |
|------|------|------|
| `FileNotFoundError` / 0 images | yaml 仍是旧机器 `/home/tp/...` | 运行 `bash tools/setup_new_machine.sh` |
| symlink 指向不存在路径 | zip 未重建 symlink | 同上 |
| `SyntaxError: accumulate` | 用了 YOLOv5 参数 | 改用 `nbs=16`，见 5.6 |
| CUDA OOM @ 1024 | 显存不足 | `batch=2` 或 `1`，见 5.6 |
| 训练极慢 / 卡在 Downloading | 首次下 `yolov8l.pt` | 等待或手动 wget 到 ultralytics 根目录 |
| dual 训练找不到 IR 图 | `llvip_dual.yaml` path 不对 | 确认 `path:` 指向含 `visible/`、`infrared/` 的 LLVIP 根目录 |

### 12.8 迁移 Checklist

- [ ] GitHub 仓库 `TP666-bot/ultralytics-yolov8-cft` 已 push `cft-yolov8-llvip` 分支  
- [ ] 新机器 `git clone -b cft-yolov8-llvip ...` 成功  
- [ ] LLVIP 放在 `<WORKSPACE>/LLVIP/`（或 multispectral 子目录）  
- [ ] 运行 `setup_new_machine.sh`，图片数量正常  
- [ ] `pip install -e .` + GPU 版 PyTorch  
- [ ] 冒烟 1 epoch 通过  
- [ ] 正式实验：`batch=2 nbs=16 imgsz=1024 epochs=200`  

---

## 十三、变更日志

| 日期 | 内容 |
|------|------|
| 2025-05-27 | 初版：阶段 B、symlink、YOLOv5 复现解读 |
| 2025-05-27 | **CFT 迁入完成**：cft.py、DualDetectionModel、双路数据、train/val_cft.py、融合 yaml |
| 2025-05-27 | **本机 4060 Ti 训练说明**：batch=2+nbs=16（禁用 accumulate）、OOM/cuDNN 排查、权重路径 `runs/detect/runs/llvip/` |
| 2025-05-27 | **Git 迁移**：推送到 `TP666-bot/ultralytics-yolov8-cft`，第十二节 12.1 |
