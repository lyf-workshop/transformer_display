"""
象限分类神经网络
================
给定平面直角坐标系中的点 (x, y)，判断该点属于第几象限。

5 分类标签：
  0 -> 第一象限  (x > 0, y > 0)
  1 -> 第二象限  (x < 0, y > 0)
  2 -> 第三象限  (x < 0, y < 0)
  3 -> 第四象限  (x > 0, y < 0)
  4 -> 坐标轴上  (x == 0 或 y == 0)

网络结构：
  输入层  : 2  (x, y)
  隐藏层1 : 16  ReLU
  隐藏层2 : 8   ReLU
  输出层  : 5   Softmax (CrossEntropyLoss 内置)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 Windows 上多份 OpenMP 库冲突

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Windows 中文字体支持（SimHei 黑体；若无则退回英文标签）
_CN_FONTS = ["SimHei", "Microsoft YaHei", "STHeiti", "WenQuanYi Micro Hei"]
_available = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
_cn_font = next((f for f in _CN_FONTS if f in _available), None)
if _cn_font:
    plt.rcParams["font.family"] = _cn_font
plt.rcParams["axes.unicode_minus"] = False

# ────────────────────────────────────────────────
# 0. 超参数
# ────────────────────────────────────────────────
SEED          = 42
N_QUADRANT    = 2000   # 每个象限的样本数
N_AXIS        = 400    # 坐标轴上的样本数
NOISE         = 0.0    # 坐标轴类别附近的噪声幅度（0 = 恰好在轴上）
COORD_RANGE   = 5.0    # 坐标范围 [-5, 5]
HIDDEN1       = 16
HIDDEN2       = 8
NUM_CLASSES   = 5      # 改为 4 可去掉坐标轴类别
BATCH_SIZE    = 128
LR            = 1e-3
EPOCHS        = 200

torch.manual_seed(SEED)
np.random.seed(SEED)

# ────────────────────────────────────────────────
# 1. 数据生成
# ────────────────────────────────────────────────

def get_label(x: float, y: float) -> int:
    """根据坐标返回 5 分类标签（纯 Python 用于单点预测）。"""
    if x == 0 or y == 0:
        return 4
    if x > 0 and y > 0:
        return 0
    if x < 0 and y > 0:
        return 1
    if x < 0 and y < 0:
        return 2
    return 3   # x > 0, y < 0


def generate_dataset(n_quadrant: int, n_axis: int, coord_range: float):
    """生成含四象限 + 坐标轴五类样本的数据集。"""
    xs, ys, labels = [], [], []

    # 四个象限：在各自半平面内均匀采样，排除紧邻轴的极小区域
    eps = 0.05
    signs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    for label, (sx, sy) in enumerate(signs):
        x = sx * np.random.uniform(eps, coord_range, n_quadrant)
        y = sy * np.random.uniform(eps, coord_range, n_quadrant)
        xs.append(x)
        ys.append(y)
        labels.append(np.full(n_quadrant, label, dtype=np.int64))

    # 坐标轴：x 轴 (y=0) + y 轴 (x=0)
    half = n_axis // 2
    ax_x = np.random.uniform(-coord_range, coord_range, half)
    ax_y = np.zeros(half)
    ay_x = np.zeros(n_axis - half)
    ay_y = np.random.uniform(-coord_range, coord_range, n_axis - half)
    xs.append(np.concatenate([ax_x, ay_x]))
    ys.append(np.concatenate([ax_y, ay_y]))
    labels.append(np.full(n_axis, 4, dtype=np.int64))

    X = np.stack([np.concatenate(xs), np.concatenate(ys)], axis=1).astype(np.float32)
    y = np.concatenate(labels)

    # 打乱
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


X_all, y_all = generate_dataset(N_QUADRANT, N_AXIS, COORD_RANGE)

# 训练 / 测试 8:2 分割
split = int(0.8 * len(y_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t  = torch.from_numpy(X_test)
y_test_t  = torch.from_numpy(y_test)

train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

print(f"训练集: {len(X_train)} 样本  测试集: {len(X_test)} 样本")

# ────────────────────────────────────────────────
# 2. 网络定义
# ────────────────────────────────────────────────

class QuadrantNet(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = QuadrantNet(num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}\n")

# ────────────────────────────────────────────────
# 3. 训练循环
# ────────────────────────────────────────────────

loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(X_train)
    loss_history.append(epoch_loss)

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t)
            preds = logits_test.argmax(dim=1)
            acc = (preds == y_test_t).float().mean().item()
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.4f}  test_acc={acc:.4f}")

# ────────────────────────────────────────────────
# 4. 最终评估
# ────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    logits_test = model(X_test_t)
    preds = logits_test.argmax(dim=1)
    final_acc = (preds == y_test_t).float().mean().item()

print(f"\n最终测试准确率: {final_acc * 100:.2f}%")

# 每类准确率
class_names = ["第一象限", "第二象限", "第三象限", "第四象限", "坐标轴"]
for c in range(NUM_CLASSES):
    mask = y_test_t == c
    if mask.sum() == 0:
        continue
    c_acc = (preds[mask] == c).float().mean().item()
    print(f"  {class_names[c]}: {c_acc * 100:.2f}%  (样本数={mask.sum().item()})")

# ────────────────────────────────────────────────
# 5. 单点预测函数
# ────────────────────────────────────────────────

def predict_point(x: float, y: float) -> dict:
    """对单个坐标点进行象限预测，返回预测标签和各类概率。"""
    model.eval()
    with torch.no_grad():
        inp = torch.tensor([[x, y]], dtype=torch.float32)
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_label = int(np.argmax(probs))
    return {
        "坐标": (x, y),
        "预测象限": class_names[pred_label],
        "各类概率": {class_names[i]: f"{probs[i]:.4f}" for i in range(NUM_CLASSES)},
    }


# 演示
test_points = [(3.0, 2.0), (-1.5, 4.0), (-2.0, -3.0), (1.0, -1.0), (0.0, 2.5)]
print("\n──── 单点预测演示 ────")
for pt in test_points:
    result = predict_point(*pt)
    print(f"  ({pt[0]:5.1f}, {pt[1]:5.1f})  ->  {result['预测象限']}")
    print(f"    概率: { {k: v for k, v in result['各类概率'].items()} }")

# ────────────────────────────────────────────────
# 6. 可视化
# ────────────────────────────────────────────────

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
CMAP   = ListedColormap(COLORS[:NUM_CLASSES])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("象限分类神经网络", fontsize=14, fontweight="bold")

# 子图1: 训练损失曲线
ax = axes[0]
ax.plot(loss_history, color="#4ECDC4", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("CrossEntropy Loss")
ax.set_title("训练损失曲线")
ax.grid(True, alpha=0.3)

# 子图2: 测试集散点图（真实标签）
ax = axes[1]
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                     cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1,
                     s=10, alpha=0.6)
ax.axhline(0, color="k", linewidth=0.8)
ax.axvline(0, color="k", linewidth=0.8)
ax.set_title("测试集（真实标签）")
ax.set_xlabel("x")
ax.set_ylabel("y")
patches = [mpatches.Patch(color=COLORS[i], label=class_names[i]) for i in range(NUM_CLASSES)]
ax.legend(handles=patches, fontsize=7, loc="upper left")

# 子图3: 决策边界
ax = axes[2]
res = 200
xi = np.linspace(-COORD_RANGE, COORD_RANGE, res)
yi = np.linspace(-COORD_RANGE, COORD_RANGE, res)
xx, yy = np.meshgrid(xi, yi)
grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

model.eval()
with torch.no_grad():
    grid_logits = model(torch.from_numpy(grid))
    grid_preds  = grid_logits.argmax(dim=1).numpy().reshape(res, res)

ax.contourf(xx, yy, grid_preds, levels=np.arange(-0.5, NUM_CLASSES, 1),
            cmap=CMAP, alpha=0.4)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
           cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1,
           s=8, alpha=0.5, edgecolors="none")
ax.axhline(0, color="k", linewidth=0.8)
ax.axvline(0, color="k", linewidth=0.8)
ax.set_title(f"决策边界  (测试准确率={final_acc*100:.1f}%)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(handles=patches, fontsize=7, loc="upper left")

plt.tight_layout()
out_path = "quadrant_result.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n可视化结果已保存至: {out_path}")
plt.show()
