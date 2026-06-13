# ODE Robust Block 实验方案与操作记录

本文档记录在当前 `YOLOv8 + CFT` 工程中加入 **ODE Robust Block** 的最小可行实验设计。目标不是一次性大改网络，而是先用一个轻量、可回退、可消融的模块验证：

> 受稳定 ODE / 连续动力学启发的特征演化，能否提升 RGB-T 融合特征在低照、噪声、模糊、模态错位下的鲁棒性。

当前实验分支：

```bash
cd /home/tp/my_new_space/ultralytics
git switch feature/ode-robust-block
```

改动前备份点：

```bash
git switch backup/pre-ode-20260612-224208
# 或
git switch --detach pre-ode-20260612-224208
```

---

## 1. 原理

普通残差块可以写成离散更新：

```text
x_{k+1} = x_k + f(x_k)
```

ODE Block 将其看作连续动力系统：

```text
dx(t) / dt = f_theta(x(t))
```

从 `t=0` 演化到 `t=T`，得到更平滑、更稳定的特征：

```text
x(T) = x(0) + integral_0^T f_theta(x(t)) dt
```

在检测任务里，真实使用完整 ODE 求解器成本较高。因此第一阶段采用 **ODE-inspired residual solver**，用固定步长 Euler 或 RK2 近似：

Euler:

```text
x_{i+1} = x_i + h * f_theta(x_i)
```

RK2:

```text
k1 = f_theta(x_i)
k2 = f_theta(x_i + h * k1)
x_{i+1} = x_i + h/2 * (k1 + k2)
```

为了和前面几篇文章关联：

- `稳定平衡点 ODE`：强调稳定特征演化、抑制扰动。
- `FROND / 分数连续动力学`：强调连续动力学、历史记忆、非瞬时更新。
- `拓扑扰动图神经扩散`：强调扩散过程的抗扰动和边缘保持。

本文档第一阶段只做 **稳定 ODE 特征演化**，后续如果有效，再扩展分数阶记忆版本。

---

## 2. 实验思路

不要一开始替换大量 Backbone `C2f`。推荐先在 CFT 融合后的 P3/P4/P5 特征上加 ODE Robust Block。

当前 CFT 模型关键层：

```text
P3 fused: layer 29
P4 fused: layer 30
P5 fused: layer 31
Detect 输入: layer 37, 40, 43
```

第一阶段做四组：

| 实验 | 说明 |
|------|------|
| `y8_cft` | 原始 YOLOv8-CFT baseline |
| `y8_cft_ode_p4` | 只在 P4 融合后加 ODE |
| `y8_cft_ode_p5` | 只在 P5 融合后加 ODE |
| `y8_cft_ode_p345` | P3/P4/P5 融合后都加 ODE |

优先推荐：

```text
P4 -> P5 -> P3/P4/P5
```

理由：

- P3 更偏小目标和边缘，过强平滑可能伤害细节。
- P4/P5 语义更强，更适合稳定演化和抗噪。
- 融合后加 ODE 不破坏 YOLOv8 backbone 预训练迁移。

---

## 3. 代码实现

### 3.1 新增模块文件

新增文件：

```bash
ultralytics/nn/modules/ode.py
```

建议代码：

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ODE-inspired robust feature evolution blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class ODERobustBlock(nn.Module):
    """Lightweight ODE-inspired residual block.

    This block approximates continuous feature evolution with fixed-step Euler or RK2 updates:
        dx/dt = f_theta(x)

    Args:
        c1: input/output channels.
        steps: number of solver steps.
        step_size: integration step size h.
        solver: "euler" or "rk2".
        expansion: hidden channel expansion ratio.
        gamma: residual output scale.
    """

    def __init__(self, c1, steps=2, step_size=0.5, solver="euler", expansion=0.5, gamma=1.0):
        super().__init__()
        hidden = max(16, int(c1 * expansion))
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.solver = str(solver).lower()
        self.gamma = float(gamma)
        self.f = nn.Sequential(
            Conv(c1, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1),
            Conv(hidden, c1, 1, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        h = self.step_size
        for _ in range(self.steps):
            if self.solver == "rk2":
                k1 = self.f(z)
                k2 = self.f(z + h * k1)
                z = z + 0.5 * h * (k1 + k2)
            else:
                z = z + h * self.f(z)
        return x + self.gamma * (z - x)
```

第一版参数建议：

```text
steps=2
step_size=0.5
solver=euler
expansion=0.5
gamma=1.0
```

如果训练不稳定，降低：

```text
step_size=0.25
gamma=0.5
```

---

### 3.2 注册模块

修改：

```bash
ultralytics/nn/modules/__init__.py
```

加入导入：

```python
from .ode import ODERobustBlock
```

并在 `__all__` 中加入：

```python
"ODERobustBlock",
```

修改：

```bash
ultralytics/nn/tasks.py
```

在模块导入区加入：

```python
ODERobustBlock,
```

并在 `parse_model()` 的 `base_modules` 中加入：

```python
ODERobustBlock,
```

不要加入 `repeat_modules`，第一阶段每个 YAML 层只放一个 `ODERobustBlock`。

---

## 4. YAML 配置

### 4.1 P4 版本

复制原配置：

```bash
cp ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
   ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_ode_p4_llvip.yaml
```

在 backbone 中 `stream fusion` 位置，把 P4 融合输出从 layer 30 后接 ODE。

原始：

```yaml
  - [[11, 12], 1, Add, [1]] # 29 P3
  - [[18, 19], 1, Add, [1]] # 30 P4
  - [[27, 28], 1, Add, [1]] # 31 P5
```

改为：

```yaml
  - [[11, 12], 1, Add, [1]] # 29 P3
  - [[18, 19], 1, Add, [1]] # 30 P4
  - [-1, 1, ODERobustBlock, [512, 2, 0.5, "euler", 0.5, 1.0]] # 31 P4 ODE
  - [[27, 28], 1, Add, [1]] # 32 P5
```

对应 head 的引用也要改：

原始：

```yaml
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 32
  - [[-1, 30], 1, Concat, [1]] # 33
  - [-1, 3, C2f, [512]] # 34
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 35
  - [[-1, 29], 1, Concat, [1]] # 36
  - [-1, 3, C2f, [256]] # 37 P3
  - [-1, 1, Conv, [256, 3, 2]] # 38
  - [[-1, 34], 1, Concat, [1]] # 39
  - [-1, 3, C2f, [512]] # 40 P4
  - [-1, 1, Conv, [512, 3, 2]] # 41
  - [[-1, 31], 1, Concat, [1]] # 42 cat fused P5 backbone
  - [-1, 3, C2f, [1024]] # 43 P5
  - [[37, 40, 43], 1, Detect, [nc]] # 44
```

P4 ODE 后：

```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 33
  - [[-1, 31], 1, Concat, [1]] # 34 cat P4 ODE
  - [-1, 3, C2f, [512]] # 35
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 36
  - [[-1, 29], 1, Concat, [1]] # 37
  - [-1, 3, C2f, [256]] # 38 P3
  - [-1, 1, Conv, [256, 3, 2]] # 39
  - [[-1, 35], 1, Concat, [1]] # 40
  - [-1, 3, C2f, [512]] # 41 P4
  - [-1, 1, Conv, [512, 3, 2]] # 42
  - [[-1, 32], 1, Concat, [1]] # 43 cat fused P5 backbone
  - [-1, 3, C2f, [1024]] # 44 P5
  - [[38, 41, 44], 1, Detect, [nc]] # 45
```

### 4.2 P5 版本

复制：

```bash
cp ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
   ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_ode_p5_llvip.yaml
```

只在 P5 fusion 后加入：

```yaml
  - [[11, 12], 1, Add, [1]] # 29 P3
  - [[18, 19], 1, Add, [1]] # 30 P4
  - [[27, 28], 1, Add, [1]] # 31 P5
  - [-1, 1, ODERobustBlock, [1024, 2, 0.5, "euler", 0.5, 1.0]] # 32 P5 ODE
```

head 中 P5 引用由 `31` 改成 `32`，后续层号整体顺延。

### 4.3 P3/P4/P5 版本

复制：

```bash
cp ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_llvip.yaml \
   ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_ode_p345_llvip.yaml
```

融合后分别加入：

```yaml
  - [[11, 12], 1, Add, [1]] # P3
  - [-1, 1, ODERobustBlock, [256, 2, 0.5, "euler", 0.5, 1.0]]
  - [[18, 19], 1, Add, [1]] # P4
  - [-1, 1, ODERobustBlock, [512, 2, 0.5, "euler", 0.5, 1.0]]
  - [[27, 28], 1, Add, [1]] # P5
  - [-1, 1, ODERobustBlock, [1024, 2, 0.5, "euler", 0.5, 1.0]]
```

注意同步修改 head 中 P3/P4/P5 的引用层号。

---

## 5. 冒烟测试

先确认模型能构建：

```bash
cd /home/tp/my_new_space/ultralytics
python - <<'PY'
from ultralytics.nn.tasks_dual import DualDetectionModel

cfg = "ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_ode_p4_llvip.yaml"
model = DualDetectionModel(cfg, ch=3, nc=1, verbose=True)
print("OK", type(model).__name__, "stride=", model.stride)
PY
```

再做 1 epoch 训练冒烟：

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_ode_p4_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt \
  epochs=1 imgsz=1024 batch=1 nbs=16 workers=2 device=0 \
  project=runs/llvip name=smoke_ode_p4 exist_ok=True
```

若显存不足：

```bash
workers=0
batch=1
```

---

## 6. 正式训练

### 6.1 Baseline 复评

先复评当前 CFT baseline：

```bash
python tools/val_cft.py \
  model=runs/detect/runs/llvip/y8_cft_v2/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 batch=1 device=0 conf=0.001 iou=0.5 \
  project=runs/llvip name=y8_cft_v2_paper_val exist_ok=True
```

### 6.2 P4 ODE 正式训练

```bash
python tools/train_cft.py \
  model=ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_ode_p4_llvip.yaml \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  pretrained=yolov8l.pt \
  epochs=200 imgsz=1024 batch=1 nbs=16 workers=2 device=0 \
  optimizer=SGD lr0=0.005 momentum=0.937 freeze_epochs=10 \
  project=runs/llvip name=y8_cft_ode_p4 exist_ok=True
```

如果第 11 epoch 解冻后 OOM：

```bash
python tools/train_cft.py \
  resume=runs/detect/runs/llvip/y8_cft_ode_p4/weights/last.pt \
  batch=1 nbs=16 freeze_epochs=0 workers=2 device=0 \
  project=runs/llvip name=y8_cft_ode_p4 exist_ok=True
```

正式评估：

```bash
python tools/val_cft.py \
  model=runs/detect/runs/llvip/y8_cft_ode_p4/weights/best.pt \
  data=ultralytics/cfg/datasets/llvip_dual.yaml \
  imgsz=1024 batch=1 device=0 conf=0.001 iou=0.5 \
  project=runs/llvip name=y8_cft_ode_p4_paper_val exist_ok=True
```

---

## 7. 鲁棒性评估

ODE 模块的价值重点不是只看 clean mAP，而是看扰动下是否更稳。

建议构造以下测试集变体：

| 扰动 | 模拟问题 |
|------|----------|
| RGB brightness down | 夜间可见光退化 |
| RGB Gaussian noise | 低照噪声 |
| RGB blur | 模糊边界 |
| IR Gaussian noise | 热成像噪声 |
| RGB/IR shift 2/4/8 px | 模态配准误差 |
| random occlusion | 遮挡 |

记录：

```text
clean mAP
corrupted mAP
mAP drop = clean mAP - corrupted mAP
```

论文表格：

| Method | Clean | RGB Noise | RGB Blur | IR Noise | Shift 4px | Avg Drop |
|--------|-------|-----------|----------|----------|-----------|----------|
| Y8-CFT |       |           |          |          |           |          |
| + ODE  |       |           |          |          |           |          |

如果 clean mAP 只提升很小，但 `Avg Drop` 明显降低，也能支撑 ODE Robust Block 的科研价值。

---

## 8. 消融实验

### 8.1 插入位置

| 实验名 | YAML |
|--------|------|
| P4 | `yolov8l_fusion_transformerx3_ode_p4_llvip.yaml` |
| P5 | `yolov8l_fusion_transformerx3_ode_p5_llvip.yaml` |
| P3/P4/P5 | `yolov8l_fusion_transformerx3_ode_p345_llvip.yaml` |

### 8.2 求解器

修改 YAML 中参数：

```yaml
ODERobustBlock, [512, 2, 0.5, "euler", 0.5, 1.0]
ODERobustBlock, [512, 2, 0.5, "rk2", 0.5, 1.0]
```

### 8.3 步数

```yaml
steps=1
steps=2
steps=3
```

建议不要超过 3，避免速度下降明显。

### 8.4 稳定强度

```yaml
step_size=0.25 / 0.5 / 0.75
gamma=0.5 / 1.0
```

若出现 mAP50-95 下降明显，优先降低 `step_size` 和 `gamma`。

---

## 9. 结果记录模板

主结果：

| Method | ODE Pos | Solver | Steps | mAP50 | mAP50-95 | Params | FLOPs | FPS |
|--------|---------|--------|-------|-------|----------|--------|-------|-----|
| Y8-CFT | - | - | - | | | | | |
| ODE-P4 | P4 | Euler | 2 | | | | | |
| ODE-P5 | P5 | Euler | 2 | | | | | |
| ODE-P345 | P3/P4/P5 | Euler | 2 | | | | | |

鲁棒性：

| Method | Clean | Noise | Blur | Shift | Occlusion | Avg Drop |
|--------|-------|-------|------|-------|-----------|----------|
| Y8-CFT | | | | | | |
| ODE-P4 | | | | | | |

---

## 10. 回退与备份

查看当前状态：

```bash
git status --short --branch
```

提交实验改动：

```bash
git add ultralytics/nn/modules/ode.py \
        ultralytics/nn/modules/__init__.py \
        ultralytics/nn/tasks.py \
        ultralytics/cfg/models/v8/yolov8l_fusion_transformerx3_ode_p4_llvip.yaml \
        add_ode.md
git commit -m "Add ODE robust block experiment plan and module"
git push github feature/ode-robust-block
```

回到 ODE 前备份：

```bash
git switch backup/pre-ode-20260612-224208
```

如果只想查看备份，不切分支：

```bash
git show pre-ode-20260612-224208 --stat
```

---

## 11. 第一阶段结论标准

认为 ODE Robust Block 有继续价值，需要至少满足一条：

1. clean `mAP50` 或 `mAP50-95` 高于 Y8-CFT；
2. clean mAP 基本持平，但扰动下 `Avg Drop` 明显降低；
3. P/R 中 Precision 提升，误检案例减少；
4. 可视化中 ODE 后目标区域更稳定，背景噪声被抑制。

如果 clean mAP 和鲁棒 mAP 都下降，则停止 ODE 路线，回退到备份点或只保留文档记录。
