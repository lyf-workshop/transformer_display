# Transformer 可视化

> 一个纯前端的 Transformer 交互式演示页面 —— 从矩阵到注意力，从训练到推理，每一步都看得见。

![架构](https://img.shields.io/badge/架构-Encoder--Decoder-6aa6ff) ![依赖](https://img.shields.io/badge/依赖-无-34d399) ![语言](https://img.shields.io/badge/语言-原生_JS-a78bfa)

## 特性

- **架构总览** — 完整的 Encoder-Decoder 结构图，点击任意模块直达该组件的矩阵变换。
- **矩阵变换逐步演示** — 15 个步骤覆盖 Transformer 全部核心组件：分词 → Embedding → 位置编码 → Q/K/V 投影 → 多头切分 → 缩放点积注意力 → Softmax → 残差 + LayerNorm → FFN → 因果掩码 → Cross-Attention → 输出 Linear + Softmax。每一步都用带热力色彩的矩阵展示真实数值。
- **训练过程** — 实时展示前向传播 → 交叉熵损失 → 反向传播 → SGD 参数更新的完整链路，带 Loss 曲线。**是真实的反向传播算法, 不是动画**。
- **推理过程** — 自回归生成 + Top-5 概率分布可视化 + KV Cache 示意。
- **零依赖** — 纯 HTML + CSS + JavaScript, 所有矩阵运算在浏览器中实时计算。

## 快速开始

```bash
# 方式 1: 任意静态服务器
python -m http.server 8765
# 然后打开 http://localhost:8765

# 方式 2: 直接用浏览器打开 index.html (某些浏览器对 file:// 有限制)
```

## 模型配置

为了在浏览器里跑得动、且数字能看清楚, 采用极简规模 (定义见 `js/model.js`):

| 参数 | 值 | 说明 |
|------|----|----|
| `d_model` | 8 | 隐藏维度 |
| `n_heads` | 2 | 注意力头数 |
| `d_k`, `d_v` | 4 | 每头维度 |
| `d_ff` | 16 | FFN 中间层维度 |
| `n_enc_layers` / `n_dec_layers` | 1 / 1 | Encoder/Decoder 层数 |
| Src Vocab | 7 | `[<pad>, 我, 爱, 你, 好, 是, 学生]` |
| Tgt Vocab | 9 | `[<pad>, <bos>, <eos>, I, love, you, hello, am, student]` |

训练样本 (中→英翻译):

1. 我 爱 你 → I love you
2. 你 好 → hello  
3. 我 是 学生 → I am student

200 步后 Loss 降到 ≈0.06, 推理输出 "I love you" 概率 >90%。

## 目录结构

```
transformer_display/
├── index.html                     # 主页 (4 个视图 tab)
├── styles.css                     # 全部样式
├── js/
│   ├── matrix.js                  # 矩阵运算 + 热力图可视化
│   ├── model.js                   # 小型 Transformer (前向 + 反向 + 训练 + 推理)
│   ├── main.js                    # 应用入口 / Tab 切换
│   └── views/
│       ├── architecture.js        # 架构总览视图
│       ├── components.js          # 矩阵变换逐步视图 (核心)
│       ├── training.js            # 训练过程视图
│       └── inference.js           # 推理过程视图
├── test-model.js                  # Node.js 下的模型自测脚本 (可选)
└── README.md
```

## 实现细节

### 前向传播

`model.js` 完整实现了 Transformer 的每一层：

- Embedding + 正弦位置编码
- Multi-Head Attention (Q/K/V 投影 → 切分多头 → scaled dot-product → 拼接 → Wo 投影)
- 因果掩码 (Causal Mask) 用于 decoder 的 masked self-attention
- Cross-Attention (Q 来自 decoder, K/V 来自 encoder 输出)
- Position-wise Feed Forward: `ReLU(x·W1+b1)·W2+b2`
- 残差连接 + LayerNorm (Post-LN 版本)

### 反向传播

**完整解析梯度**, 而不是动画效果。每次点击"执行一步训练"都会做真实的反向传播:

- 交叉熵对 logits 的梯度: `softmax(logits) - onehot(target)`
- 线性层反向: `dX = dY·Wᵀ`, `dW = Xᵀ·dY`
- Softmax 反向: `dS[i,j] = A[i,j]·(dA[i,j] - Σₖ A[i,k]·dA[i,k])`
- LayerNorm 反向: 用标准解析式
- ReLU 反向: `grad · (x > 0)`
- 梯度从 output 一路回到 src/tgt Embedding

### 可视化

- 每个矩阵渲染为热力图: 正值偏红, 负值偏蓝, 数值直接印在格子里。
- 共用一套小型模型的参数, 所以各视图展示的数字是自洽的 (在"训练过程"里训练后, 同一参数的变化会反映到"矩阵变换"和"推理"视图中)。

## 使用建议

1. **第一次打开**: 先到"架构总览"了解宏观结构, 然后进入"矩阵变换"逐步浏览。
2. **想看 Transformer 怎么学习**: 切到"训练过程", 点"连续训练 20 步"观察 Loss 曲线下降。
3. **想看 Transformer 怎么生成**: 切到"推理过程", 点"自动生成全部", 观察每一步 Top-5 的概率分布和 KV Cache 的积累。
4. 在训练过程页面训练到 Loss <0.1 后, 去推理页可以看到模型真的学会了翻译。

## 常见问题

**Q: 为什么用 `d_model=8` 这么小?**  
A: 为了让每个矩阵都能在一屏内完整显示所有数值, 保留教学可读性。真实的 Transformer (例如 GPT-3) 的 `d_model` 是 12288, 矩阵不可能直接可视化。

**Q: 为什么选中英翻译这种任务?**  
A: 原始 Transformer 论文就是做机器翻译的, 可以完整展示 Encoder-Decoder 结构和 Cross-Attention。

**Q: 可以拿这个代码真的训练更大的模型吗?**  
A: 不建议。JavaScript 的矩阵运算比 NumPy/PyTorch 慢好几个数量级, 且这里是教学实现, 没有优化 batching/GPU/Adam 等。真正训练请用 PyTorch。

## 授权

MIT
