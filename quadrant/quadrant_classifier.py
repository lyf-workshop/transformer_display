"""
象限分类神经网络 —— 纯 Python 从零实现
=====================================
不依赖任何第三方库（numpy / torch 等）
仅使用 Python 内置标准库：
  · math   —— exp / log / sqrt / cos / pi
  · random —— 均匀随机数（配合 Box-Muller 生成正态分布）
可视化传入普通 Python list，不在代码中调用 numpy

网络结构：
  Linear(2→16) → ReLU → Linear(16→8) → ReLU → Linear(8→5) → SoftmaxCE
"""

import math
import random
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_avail = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
_cn    = next((f for f in ["SimHei", "Microsoft YaHei", "STHeiti"] if f in _avail), None)
if _cn:
    plt.rcParams["font.family"] = _cn
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
random.seed(SEED)


# ════════════════════════════════════════════════════════════
# 1. 基础矩阵工具
#    矩阵用 list[list[float]] 表示，行主序
# ════════════════════════════════════════════════════════════

def _randn() -> float:
    """
    Box-Muller 变换：从均匀分布生成标准正态随机数
        Z = sqrt(-2·ln U₁) · cos(2π U₂),   U₁,U₂ ~ Uniform(0,1)
    """
    u1 = random.random() or 1e-12   # 防止 log(0)
    u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def zeros(rows: int, cols: int) -> list:
    return [[0.0] * cols for _ in range(rows)]


def he_init(fan_in: int, fan_out: int) -> list:
    """
    He 初始化：W ~ N(0, sqrt(2/fan_in))
    避免 ReLU 激活后信号逐层衰减或爆炸
    """
    std = math.sqrt(2.0 / fan_in)
    return [[_randn() * std for _ in range(fan_out)] for _ in range(fan_in)]


def matmul(A: list, B: list) -> list:
    """
    矩阵乘法  A(m,k) @ B(k,n) → C(m,n)
        C[i][j] = Σ_p  A[i][p] · B[p][j]
    """
    k = len(B)
    n = len(B[0])
    return [
        [sum(A[i][p] * B[p][j] for p in range(k)) for j in range(n)]
        for i in range(len(A))
    ]


def T(A: list) -> list:
    """矩阵转置  A(m,n) → A^T(n,m)"""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def add_bias(A: list, b: list) -> list:
    """每行加上 1D 偏置向量 b（广播）"""
    return [[A[i][j] + b[j] for j in range(len(b))] for i in range(len(A))]


def elem_mul(A: list, B: list) -> list:
    """逐元素乘法  A ⊙ B"""
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def col_sum(A: list) -> list:
    """沿行方向求和 → 1D list，形状 (cols,)"""
    return [sum(A[i][j] for i in range(len(A))) for j in range(len(A[0]))]


def mat_scale(A: list, s: float) -> list:
    """矩阵数乘"""
    return [[v * s for v in row] for row in A]


# ════════════════════════════════════════════════════════════
# 2. 网络层
# ════════════════════════════════════════════════════════════

class Linear:
    """
    全连接层   out = X @ W + b

    前向：缓存输入 X，供反向传播使用
    反向（链式法则）：
        dL/dW = X^T  @ dout       (fan_in, fan_out)
        dL/db = Σ_batch dout      (fan_out,)
        dL/dX = dout  @ W^T       (batch, fan_in)  ← 传给上一层
    """
    def __init__(self, fan_in: int, fan_out: int):
        self.W  = he_init(fan_in, fan_out)
        self.b  = [0.0] * fan_out
        self._x = None
        self.dW = None
        self.db = None

    def forward(self, x: list) -> list:
        self._x = x
        return add_bias(matmul(x, self.W), self.b)

    def backward(self, dout: list) -> list:
        self.dW = matmul(T(self._x), dout)     # (fan_in, fan_out)
        self.db = col_sum(dout)                 # (fan_out,)
        return matmul(dout, T(self.W))          # (batch, fan_in)

    def params(self) -> list:
        """返回 [(参数, 梯度), ...] 供优化器遍历"""
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """
    ReLU(x) = max(0, x)

    导数：
        d ReLU / dx = 1  if x > 0
                      0  otherwise
    反向：逐元素乘以前向时记录的 0/1 掩码
    """
    def __init__(self):
        self._mask = None

    def forward(self, x: list) -> list:
        self._mask = [[1.0 if x[i][j] > 0 else 0.0
                       for j in range(len(x[0]))]
                      for i in range(len(x))]
        return elem_mul(x, self._mask)

    def backward(self, dout: list) -> list:
        return elem_mul(dout, self._mask)


class SoftmaxCrossEntropy:
    """
    Softmax + 交叉熵损失（合并求导，结果极简）

    前向：
        p_k = exp(z_k - max z) / Σ exp(z_j - max z)    ← 减去 max 保持数值稳定
        L   = -1/N · Σᵢ log p_{i, yᵢ}

    反向（Softmax 与 CrossEntropy 联合推导）：
        dL/dz_k = (p_k - 1[k == y]) / N
    整个梯度就是"预测概率减去 one-hot 标签"再除以批大小。
    """
    def __init__(self):
        self._probs = None
        self._y     = None

    def _softmax(self, logits: list) -> list:
        result = []
        for row in logits:
            m    = max(row)
            exps = [math.exp(v - m) for v in row]
            s    = sum(exps)
            result.append([e / s for e in exps])
        return result

    def forward(self, logits: list, y: list) -> float:
        self._probs = self._softmax(logits)
        self._y     = y
        loss = -sum(
            math.log(max(self._probs[i][y[i]], 1e-12))
            for i in range(len(y))
        ) / len(y)
        return loss

    def backward(self) -> list:
        N  = len(self._y)
        dz = [row[:] for row in self._probs]    # 深拷贝
        for i in range(N):
            dz[i][self._y[i]] -= 1.0            # 正确类别减 1
        return mat_scale(dz, 1.0 / N)

    def predict(self, logits: list) -> list:
        return self._softmax(logits)


# ════════════════════════════════════════════════════════════
# 3. Adam 优化器
# ════════════════════════════════════════════════════════════

class Adam:
    """
    Adam（Adaptive Moment Estimation）

    对每个参数独立维护一阶矩（梯度均值）和二阶矩（梯度方差）：

        m_t = β₁·m_{t-1} + (1-β₁)·g          ← 平滑梯度
        v_t = β₂·v_{t-1} + (1-β₂)·g²          ← 平滑梯度平方

    偏差修正（早期 m, v 被 0 初始化拉低）：
        m̂ = m_t / (1 - β₁ᵗ)
        v̂ = v_t / (1 - β₂ᵗ)

    参数更新：
        θ ← θ - lr · m̂ / (√v̂ + ε)
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m: dict = {}
        self.v: dict = {}

    @staticmethod
    def _zeros_like(p: list) -> list:
        if isinstance(p[0], list):
            return [[0.0] * len(p[0]) for _ in range(len(p))]
        return [0.0] * len(p)

    def step(self, layers: list):
        self.t += 1
        b1t = self.beta1 ** self.t
        b2t = self.beta2 ** self.t

        for layer in layers:
            if not hasattr(layer, "params"):
                continue
            for param, grad in layer.params():
                if grad is None:
                    continue
                pid = id(param)
                if pid not in self.m:
                    self.m[pid] = self._zeros_like(param)
                    self.v[pid] = self._zeros_like(param)
                M, V = self.m[pid], self.v[pid]

                if isinstance(param[0], list):          # 2D 权重矩阵
                    for i in range(len(param)):
                        for j in range(len(param[0])):
                            g           = grad[i][j]
                            M[i][j]     = self.beta1 * M[i][j] + (1 - self.beta1) * g
                            V[i][j]     = self.beta2 * V[i][j] + (1 - self.beta2) * g * g
                            m_hat       = M[i][j] / (1 - b1t)
                            v_hat       = V[i][j] / (1 - b2t)
                            param[i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                else:                                   # 1D 偏置向量
                    for j in range(len(param)):
                        g        = grad[j]
                        M[j]     = self.beta1 * M[j] + (1 - self.beta1) * g
                        V[j]     = self.beta2 * V[j] + (1 - self.beta2) * g * g
                        m_hat    = M[j] / (1 - b1t)
                        v_hat    = V[j] / (1 - b2t)
                        param[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


# ════════════════════════════════════════════════════════════
# 4. 网络组装
# ════════════════════════════════════════════════════════════

layers = [
    Linear(2, 16),
    ReLU(),
    Linear(16, 8),
    ReLU(),
    Linear(8, 5),
]
loss_fn   = SoftmaxCrossEntropy()
optimizer = Adam(lr=1e-3)

total_params = sum(
    len(p) * len(p[0]) if isinstance(p[0], list) else len(p)
    for layer in layers if hasattr(layer, "params")
    for p, _ in layer.params()
)
print(f"网络结构: 2 → 16 → 8 → 5   总参数量: {total_params}")


def forward(x: list) -> list:
    out = x
    for layer in layers:
        out = layer.forward(out)
    return out


def backward(dout: list):
    for layer in reversed(layers):
        dout = layer.backward(dout)


def predict_proba(x: list) -> list:
    return loss_fn.predict(forward(x))


def argmax_1d(lst: list) -> int:
    return max(range(len(lst)), key=lambda i: lst[i])


# ════════════════════════════════════════════════════════════
# 5. 数据生成（仅用 random 标准库）
# ════════════════════════════════════════════════════════════

CLASS_NAMES = ["第一象限", "第二象限", "第三象限", "第四象限", "坐标轴"]
NUM_CLASSES = 5
N_QUADRANT  = 300       # 每个象限的样本数（纯 Python 训练较慢，适当减小）
N_AXIS      = 60        # 坐标轴样本数
COORD_RANGE = 5.0
EPS         = 0.05      # 避免采样到紧邻轴的点

def generate_dataset():
    data, labels = [], []
    for sx, sy, label in [(1,1,0), (-1,1,1), (-1,-1,2), (1,-1,3)]:
        for _ in range(N_QUADRANT):
            data.append([sx * random.uniform(EPS, COORD_RANGE),
                         sy * random.uniform(EPS, COORD_RANGE)])
            labels.append(label)
    half = N_AXIS // 2
    for _ in range(half):                                  # x 轴 (y=0)
        data.append([random.uniform(-COORD_RANGE, COORD_RANGE), 0.0])
        labels.append(4)
    for _ in range(N_AXIS - half):                         # y 轴 (x=0)
        data.append([0.0, random.uniform(-COORD_RANGE, COORD_RANGE)])
        labels.append(4)
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return list(data), list(labels)


X_all, y_all = generate_dataset()
split   = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]
print(f"训练集: {len(X_train)} 样本   测试集: {len(X_test)} 样本\n")

# ════════════════════════════════════════════════════════════
# 6. 训练循环
# ════════════════════════════════════════════════════════════

EPOCHS     = 150
BATCH_SIZE = 64
loss_history: list = []

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    indices = list(range(len(X_train)))
    random.shuffle(indices)

    epoch_loss, n_batches = 0.0, 0
    for start in range(0, len(X_train), BATCH_SIZE):
        idx = indices[start : start + BATCH_SIZE]
        xb  = [X_train[i] for i in idx]
        yb  = [y_train[i] for i in idx]

        # ── 前向 ──────────────────────────
        logits  = forward(xb)
        loss    = loss_fn.forward(logits, yb)

        # ── 反向 ──────────────────────────
        dlogits = loss_fn.backward()
        backward(dlogits)

        # ── Adam 更新 ─────────────────────
        optimizer.step(layers)

        epoch_loss += loss
        n_batches  += 1

    epoch_loss /= n_batches
    loss_history.append(epoch_loss)

    if epoch % 20 == 0:
        probs = predict_proba(X_test)
        preds = [argmax_1d(p) for p in probs]
        acc   = sum(p == y for p, y in zip(preds, y_test)) / len(y_test)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.4f}  "
              f"test_acc={acc:.4f}  elapsed={elapsed:.1f}s")

print(f"\n训练总耗时: {time.time() - t0:.1f}s")

# ════════════════════════════════════════════════════════════
# 7. 评估
# ════════════════════════════════════════════════════════════

probs_test = predict_proba(X_test)
preds_test = [argmax_1d(p) for p in probs_test]
final_acc  = sum(p == y for p, y in zip(preds_test, y_test)) / len(y_test)

print(f"\n最终测试准确率: {final_acc * 100:.2f}%")
for c in range(NUM_CLASSES):
    mask  = [i for i, y in enumerate(y_test) if y == c]
    if not mask:
        continue
    c_acc = sum(1 for i in mask if preds_test[i] == c) / len(mask)
    print(f"  {CLASS_NAMES[c]}: {c_acc * 100:.2f}%  (样本数={len(mask)})")

# ════════════════════════════════════════════════════════════
# 8. 单点预测
# ════════════════════════════════════════════════════════════

def predict_point(x: float, y: float) -> dict:
    probs = predict_proba([[x, y]])[0]
    label = argmax_1d(probs)
    return {
        "坐标":    (x, y),
        "预测象限": CLASS_NAMES[label],
        "各类概率": {CLASS_NAMES[i]: f"{probs[i]:.4f}" for i in range(NUM_CLASSES)},
    }

print("\n──── 单点预测演示 ────")
for pt in [(3.0, 2.0), (-1.5, 4.0), (-2.0, -3.0), (1.0, -1.0), (0.0, 2.5)]:
    r = predict_point(*pt)
    print(f"  ({pt[0]:5.1f}, {pt[1]:5.1f})  ->  {r['预测象限']}")
    print(f"    概率: {r['各类概率']}")

# ════════════════════════════════════════════════════════════
# 9. 可视化（matplotlib 接受普通 Python list，无需 numpy）
# ════════════════════════════════════════════════════════════

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("象限分类神经网络（纯 Python）", fontsize=14, fontweight="bold")

# ── 子图1：训练损失曲线 ─────────────────────────────
ax = axes[0]
ax.plot(list(range(1, EPOCHS + 1)), loss_history, color="#4ECDC4", lw=2)
ax.set_xlabel("Epoch");  ax.set_ylabel("CrossEntropy Loss")
ax.set_title("训练损失曲线");  ax.grid(True, alpha=0.3)

# ── 子图2：测试集真实标签 ───────────────────────────
ax = axes[1]
for c in range(NUM_CLASSES):
    xs_c = [X_test[i][0] for i in range(len(X_test)) if y_test[i] == c]
    ys_c = [X_test[i][1] for i in range(len(X_test)) if y_test[i] == c]
    ax.scatter(xs_c, ys_c, color=COLORS[c], s=12, alpha=0.7, label=CLASS_NAMES[c])
ax.axhline(0, color="k", lw=0.8);  ax.axvline(0, color="k", lw=0.8)
ax.set_title("测试集（真实标签）");  ax.set_xlabel("x");  ax.set_ylabel("y")
ax.legend(fontsize=7, loc="upper left")

# ── 子图3：决策边界（网格散点，无需 contourf / numpy）─
ax  = axes[2]
res  = 55                                     # 格点数，55×55 = 3025 点
step = 2 * COORD_RANGE / res
grid_pts = [
    [(-COORD_RANGE + j * step), (-COORD_RANGE + i * step)]
    for i in range(res + 1)
    for j in range(res + 1)
]
grid_pr  = predict_proba(grid_pts)
grid_pd  = [argmax_1d(p) for p in grid_pr]

for c in range(NUM_CLASSES):
    gx = [grid_pts[i][0] for i in range(len(grid_pts)) if grid_pd[i] == c]
    gy = [grid_pts[i][1] for i in range(len(grid_pts)) if grid_pd[i] == c]
    if gx:
        ax.scatter(gx, gy, color=COLORS[c], s=20, alpha=0.25, linewidths=0)

for c in range(NUM_CLASSES):
    xs_c = [X_test[i][0] for i in range(len(X_test)) if y_test[i] == c]
    ys_c = [X_test[i][1] for i in range(len(X_test)) if y_test[i] == c]
    ax.scatter(xs_c, ys_c, color=COLORS[c], s=10, alpha=0.9, linewidths=0)

ax.axhline(0, color="k", lw=0.8);  ax.axvline(0, color="k", lw=0.8)
ax.set_title(f"决策边界  (准确率={final_acc*100:.1f}%)")
ax.set_xlabel("x");  ax.set_ylabel("y")
patches = [mpatches.Patch(color=COLORS[i], label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
ax.legend(handles=patches, fontsize=7, loc="upper left")

plt.tight_layout()
plt.savefig("quadrant_result.png", dpi=150, bbox_inches="tight")
print("\n可视化已保存至: quadrant_result.png")
plt.show()
