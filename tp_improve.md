# YOLOv8 + CFT 性能改进说明（tp_improve）

本文档记录另一台机器 CFT 训练结果解读、与 YOLOv5 论文差距的原因，以及仓库内已落地的 **P1–P5** 改进与复训建议。

---

## 1. 另一台机器训练结果解读

来源文件：`results (1).csv`（双路 CFT / `train_cft.py`，共 **42 epoch** 记录）。

| 阶段         | epoch  | mAP@0.5 (val) | mAP@0.5:0.95 | 备注                 |
| ------------ | ------ | ------------- | ------------ | -------------------- |
| 起步         | 1      | 0.738         | 0.308        | 预训练 backbone 起点 |
| 快速提升     | 5      | 0.881         | 0.527        |                      |
| **峰值附近** | **12** | **0.923**     | **0.594**    | 本轮日志最高点       |
| 平台期       | 20–30  | ~0.919–0.920  | ~0.587–0.590 | 增幅放缓             |
| 末段         | 42     | 0.902         | 0.567        | 略有回落，可能未收敛 |

**结论（客观）：**

1. 训练 **有效**，且明显好于本机早期「无增强 + 100 epoch」的 Add 融合（~0.69）。
2. 日志在 **42 epoch 截止**，未达到文档推荐的 **200 epoch**；末段 mAP 仍缓慢变化，**不宜判定为已训满**。
3. 训练时验证默认 **`iou=0.7`**（Ultralytics），论文 `test.py` 使用 **`iou=0.5`**；最终对外报告须用 **P5** 流程复评。
4. 与论文 YOLOv5+CFT **0.975** 相比，当前 **~0.92（val, iou=0.7）** 仍有差距，需结合未训满、增强不足、超参、框架差异综合看待。

---

## 2. 与论文 / 本机实验对照

| 设置                                  | mAP@0.5            | 说明                       |
| ------------------------------------- | ------------------ | -------------------------- |
| 论文 LLVIP CFT（YOLOv5l）             | **0.975**          | 作者权重 + 原 `test.py`    |
| 论文 LLVIP Add                        | 0.958              | 无 GPT                     |
| 独立复现 Y5-CFT                       | 0.972              | 同协议                     |
| 另一台机器 Y8-CFT（42e, val iou=0.7） | **~0.923（峰值）** | `results (1).csv`          |
| 本机 Y8-IR 单模态（100e）             | 0.958              | `yolo train`，完整 v8 增强 |
| 本机 Y8-Add（100e，旧管线）           | ~0.69              | 双路几乎无 mosaic          |

---

## 3. 差距原因（按重要性）

### 3.1 双路数据增强不足（P1，已修复）

旧版 `YOLODualStreamDataset` 训练时 **仅 LetterBox**，无 YOLOv8 标准 **mosaic / 仿射 / 翻转 / HSV**。  
YOLOv5 多光谱代码对 RGB/IR 使用 **`load_mosaic_RGB_IR` + 同步几何变换**。

→ 已在 `ultralytics/data/dual_augment.py` 实现 **同步 v8 增强**（`DualMosaic`、`DualRandomPerspective` 等）。

### 3.2 训练轮数与名义 batch（P2）

| 项         | 另一台机器日志         | 论文 / 作者权重                            |
| ---------- | ---------------------- | ------------------------------------------ |
| epochs     | 42（未完成）           | **200**                                    |
| nbs        | 未在 csv 体现，常为 16 | 文件名 **bs32**                            |
| 增强版默认 | —                      | `train_cft.py` 默认 **epochs=200, nbs=32** |

### 3.3 GPT 初始化（P3）

CFT 的 GPT 模块默认 **随机初始化**；论文使用 **充分训练的 YOLOv5 CFT 权重**。  
可选：`tools/load_cft_partial.py` 从 Y5 checkpoint **部分加载 GPT**。

### 3.4 学习率与 backbone 稳定性（P4）

双路 + 随机 GPT 对大学习率敏感；末段 mAP 回落可能与 **lr 衰减末期 + 融合层未稳定** 有关。

→ 默认 **`lr0=0.005`**，前 **`freeze_epochs=10`** 仅训练融合层与检测头（层索引 ≤19 为 backbone，见 `cft_train.py` 回调）。

### 3.5 评估协议（P5）

| 用途            | conf      | iou     | 说明                                   |
| --------------- | --------- | ------- | -------------------------------------- |
| 训练中 val      | 默认      | **0.7** | `results.csv` 中数字                   |
| 论文 / 公平对比 | **0.001** | **0.5** | 必须用 `val_cft.py` 或对齐的 `test.py` |

---

## 4. 已落地改进（P1–P5）

### P1 — 双路同步数据增强

| 文件                               | 内容                                                                          |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| `ultralytics/data/dual_augment.py` | `dual_v8_transforms()`：mosaic/affine/flip/HSV 同步 RGB+IR；禁用 MixUp/CutMix |
| `ultralytics/data/dual_stream.py`  | 训练分支调用 `dual_v8_transforms`                                             |

### P2 — 训练默认超参

| 文件                 | 默认值                                                        |
| -------------------- | ------------------------------------------------------------- |
| `tools/train_cft.py` | `epochs=200`, `nbs=32`, `batch` 仍由 CLI 指定（如 `batch=2`） |

显式覆盖示例：

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt \
  epochs=200 imgsz=1024 batch=2 nbs=32 workers=4 device=0 \
  project=runs/llvip name=y8_cft_v2 exist_ok=True
```

### P3 — YOLOv5 GPT 部分权重初始化

```bash
python tools/load_cft_partial.py \
  --y8-cfg ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  --y8-weights yolov8l.pt \
  --y5-cft /path/to/yolov5l_transformerx3_llvip_s1024_bs32_e200.pt \
  --out runs/llvip/y8_cft_init.pt

python tools/train_cft.py \
  model=runs/llvip/y8_cft_init.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  epochs=200 imgsz=1024 batch=2 nbs=32 device=0 \
  project=runs/llvip name=y8_cft_from_y5gpt exist_ok=True
```

### P4 — 更低学习率 + 前若干 epoch 冻结 backbone

| 参数            | 默认      | 含义                                                                             |
| --------------- | --------- | -------------------------------------------------------------------------------- |
| `optimizer`     | **`SGD`** | **禁止 `auto`**（8.4 会选 MuSGD/Muon，与 GPT 梯度形状冲突导致 `AssertionError`） |
| `lr0`           | `0.005`   | 较 Ultralytics 默认 `0.01` 更保守；使用 SGD 时生效                               |
| `momentum`      | `0.937`   | 与 YOLOv5 一致                                                                   |
| `freeze_epochs` | `10`      | 前 10 epoch 冻结 `model.0`–`model.19`（双路 backbone）                           |

**注意**：`freeze_epochs` 是 **`train_cft.py` 专用参数**，不是 Ultralytics 全局 cfg 字段；由脚本在内部处理，不要传给 `yolo train`。

自定义：`freeze_epochs=0` 关闭冻结；`lr0=0.01` 恢复激进学习率。

### P5 — 论文对齐评估 + 修复 `val_cft.py`

```bash
python tools/val_cft.py \
  model=runs/detect/runs/llvip/ \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 batch=4 device=0 \
  conf=0.001 iou=0.5 \
  project=runs/llvip name=y8_cft_paper_val exist_ok=True < run > /weights/best.pt
```

`CFTDetectionTrainer.setup_val()` 已修复单独验证时 `validator is None` 的问题。

---

## 5. 推荐复训流程

```bash
cd <REPO_ROOT>
cp tools/machine.env.example tools/machine.env   # 配置 WORKSPACE
bash tools/setup_new_machine.sh
pip install -e .

# 方案 A：仅 Y8 预训练 + 新增强（推荐先跑）
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt \
  epochs=200 imgsz=1024 batch=2 nbs=32 workers=4 device=0 \
  lr0=0.005 freeze_epochs=10 \
  project=runs/llvip name=y8_cft_v2 exist_ok=True

# 方案 B：在方案 A 基础上使用 Y5 GPT 初始化（见 P3）

# 论文协议复评（必做）
python tools/val_cft.py \
  model=runs/detect/runs/llvip/y8_cft_v2/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 batch=4 device=0 conf=0.001 iou=0.5
```

**验收标准：**

- [ ] `results.csv` 跑满 **200 epoch**
- [ ] 训练 val mAP@0.5 稳定 **≥ 0.93**（参考你 42e 已达 0.923）
- [ ] **P5** `conf=0.001 iou=0.5` 下 test mAP 填入 `cft_yolov8_config.md` 第九节
- [ ] 与 Y5-CFT **0.972** 对比时注明框架与评估脚本差异

---

## 6. 同步到 GitHub

```bash
git add ultralytics/data/dual_augment.py ultralytics/data/dual_stream.py \
  ultralytics/models/yolo/detect/cft_train.py tools/train_cft.py tools/val_cft.py tp_improve.md
git commit -m "Improve dual-stream training: sync augment, defaults, val fix"
git push github cft-yolov8-llvip
```

---

## 7. 故障：验证阶段 `Add2` 尺寸不匹配

现象：第 1 epoch 训练结束后验证崩溃：

```text
RuntimeError: The size of tensor a (64) must match the size of tensor b (80) at non-singleton dimension 2
```

原因：验证用的 `LetterBox` **只处理了 RGB `img`**，**IR `img2` 未做相同 letterbox**，两路特征图空间尺寸不一致，CFT `Add2` 无法相加。

处理：已加入 `DualLetterBox`，验证时对 RGB/IR 做相同缩放与 padding（`dual_augment.py` + `dual_stream.py`）。

---

## 8. 故障：第 11 epoch OOM（backbone 解冻）

现象：epoch 1–10 正常（mAP 可达 ~0.9），**第 11 epoch 训练开始**时 `CUDA out of memory`。

原因：`freeze_epochs=10` 结束后 **双路 backbone（层 0–19）全部参与反传**，显存约为冻结阶段的 2 倍；8GB 卡上 `batch=2` + `imgsz=1024` 不够。

处理：

1. **从头训练**：`batch=1 nbs=16`（有效 batch 仍为 16）
2. **从 checkpoint 续训**（推荐，保留前 10 epoch 权重）：

```bash
python tools/train_cft.py \
  resume=runs/detect/runs/llvip/y8_cft_v2/weights/last.pt \
  batch=1 nbs=16 freeze_epochs=0 workers=2 device=0
```

`freeze_epochs=0` 避免再次冻结；`last.pt` 已含优化器状态。

---

## 9. 故障：Muon / `optimizer=auto` 报错

现象：第 1 个 epoch 反向时在 `muon.py` 报 `assert len(G.shape) == 2`。

原因：Ultralytics 8.4 `optimizer=auto` 会选用 **MuSGD**，Muon 更新只支持 **2D** 梯度；CFT 的 GPT 等模块不满足。

处理：在 `train_cft.py` 中默认 **`optimizer=SGD`**。若命令行写了 `optimizer=auto` 请删掉或改为 `optimizer=SGD`。

---

## 10. 变更日志

| 日期       | 内容                                                    |
| ---------- | ------------------------------------------------------- |
| 2025-05-27 | 初版：解读 `results (1).csv`，落地 P1–P5                |
| 2025-05-27 | 默认 `optimizer=SGD`，规避 MuSGD 与 CFT 不兼容          |
| 2025-05-27 | `DualLetterBox`：验证时同步 letterbox RGB/IR            |
| 2025-05-27 | 默认 `batch=1`；验证 batch 不再 ×2；解冻 epoch 显存提示 |
