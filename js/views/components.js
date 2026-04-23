// ============================================================
// components.js — 矩阵变换逐步视图
//
// 使用固定样本 "我 爱 你" -> "I love you" 展示 Transformer
// 内部每一步的矩阵变换
// ============================================================

const ComponentsView = (function () {
  let params = null;
  let ctx = null; // 保存所有中间结果
  let currentStep = 0;

  // ------- 步骤定义 -------
  const STEPS = [
    { id: "tokenize",  phase: "enc", title: "① 分词 Tokenization",              render: stepTokenize },
    { id: "embedding", phase: "enc", title: "② 词嵌入 Embedding",                render: stepEmbedding },
    { id: "posenc",    phase: "enc", title: "③ 位置编码 Positional Encoding",    render: stepPosEnc },
    { id: "encinput",  phase: "enc", title: "④ Encoder 输入 X = E + PE",         render: stepEncInput },
    { id: "qkv",       phase: "enc", title: "⑤ Q/K/V 线性投影",                  render: stepQKV },
    { id: "heads",     phase: "enc", title: "⑥ 多头切分 Multi-Head Split",       render: stepHeads },
    { id: "attention", phase: "enc", title: "⑦ 注意力分数 QK^T / √d_k",          render: stepAttention },
    { id: "addnorm",   phase: "enc", title: "⑧ 注意力输出 & Add&Norm (编码器)",  render: stepAttnOutEnc },
    { id: "ffn",       phase: "enc", title: "⑨ 前馈网络 FFN",                    render: stepFFN },
    { id: "encout",    phase: "enc", title: "⑩ Encoder 输出",                    render: stepEncOut },

    { id: "decinput",  phase: "dec", title: "⑪ Decoder 输入 (含 <bos>)",          render: stepDecInput },
    { id: "masked",    phase: "dec", title: "⑫ Masked Self-Attention (因果掩码)", render: stepMasked },
    { id: "cross",     phase: "dec", title: "⑬ Cross-Attention (Q 来自 Dec, K/V 来自 Enc)", render: stepCross },
    { id: "decffn",    phase: "dec", title: "⑭ Decoder FFN + Add&Norm",           render: stepDecFFN },

    { id: "output",    phase: "out", title: "⑮ 输出 Linear + Softmax → 预测",     render: stepOutput },
  ];

  // ============================================================
  // 初始化: 预计算所有中间结果
  // ============================================================
  function currentSampleIdx() {
    const el = document.getElementById("comp-sample");
    return el ? +el.value : 0;
  }

  function setup(globalParams) {
    params = globalParams;
    const sample = MODEL.makeTrainPair(currentSampleIdx());
    const encFwd = MODEL.encodeFwd(sample.srcIds, params);
    const decFwd = MODEL.decodeFwd(sample.tgtIn, encFwd.out, params);

    ctx = {
      sample,
      srcIds: sample.srcIds,
      srcTokens: sample.srcTokens,
      tgtTokens: ["<bos>", ...sample.tgtTokens],
      tgtIds: sample.tgtIn,
      encEmb: encFwd.cache.emb,
      encPE: encFwd.cache.PE,
      encX: MX.add(encFwd.cache.emb, encFwd.cache.PE),
      encLayer0: encFwd.cache.layerCaches[0],
      encOut: encFwd.out,
      decEmb: decFwd.cache.emb,
      decPE: decFwd.cache.PE,
      decX: MX.add(decFwd.cache.emb, decFwd.cache.PE),
      decLayer0: decFwd.cache.layerCaches[0],
      decLogits: decFwd.logits,
    };
  }

  function renderStepNav() {
    const html = STEPS.map((s, i) => {
      const phaseColor = { enc: "#60a5fa", dec: "#f472b6", out: "#fbbf24" }[s.phase];
      return `<button class="step-pill${i === currentStep ? " active" : ""}" data-idx="${i}" style="${i === currentStep ? "" : `border-left: 3px solid ${phaseColor}`}">
        <span class="pill-num">${i + 1}</span>${s.title.replace(/^[①-⑳]\s*/, "")}
      </button>`;
    }).join("");
    document.getElementById("step-nav").innerHTML = html;
    document.querySelectorAll("#step-nav .step-pill").forEach((el) => {
      el.addEventListener("click", () => jumpTo(+el.dataset.idx));
    });
  }

  function jumpTo(idx) {
    currentStep = Math.max(0, Math.min(STEPS.length - 1, idx));
    renderStepNav();
    document.getElementById("step-indicator").textContent = `${currentStep + 1} / ${STEPS.length}`;
    document.getElementById("step-prev").disabled = currentStep === 0;
    document.getElementById("step-next").disabled = currentStep === STEPS.length - 1;
    const s = STEPS[currentStep];
    document.getElementById("step-container").innerHTML = s.render();
  }

  // 从架构图点击跳转的映射
  function jumpFromArch(target) {
    const map = {
      embedding: "embedding",
      posenc:    "posenc",
      attention: "attention",
      addnorm:   "addnorm",
      ffn:       "ffn",
      masked:    "masked",
      cross:     "cross",
      output:    "output",
    };
    const id = map[target];
    if (!id) return;
    const idx = STEPS.findIndex((s) => s.id === id);
    if (idx >= 0) jumpTo(idx);
  }

  // ============================================================
  // 各步骤渲染函数
  // ============================================================

  function stepTokenize() {
    const { srcTokens, srcIds, tgtTokens, tgtIds } = ctx;
    return `
      <div class="step-title">分词: 把文字映射成整数 ID</div>
      <div class="step-desc">每个词在词表中有一个唯一编号, 这些编号稍后会被用来查 embedding 表。</div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div>
          <h3 style="color:#60a5fa;font-size:14px;margin-bottom:8px">源句子 (输入 Encoder)</h3>
          ${tokenTable(srcTokens, srcIds, "src")}
        </div>
        <div>
          <h3 style="color:#f472b6;font-size:14px;margin-bottom:8px">目标句子 (输入 Decoder, 加 &lt;bos&gt;)</h3>
          ${tokenTable(tgtTokens, tgtIds, "tgt")}
        </div>
      </div>

      <div class="step-formula" style="margin-top:16px">
        src = [${srcTokens.map(MX.escapeHtml).join(", ")}]  →  [${srcIds.join(", ")}]<br/>
        tgt = [${tgtTokens.map(MX.escapeHtml).join(", ")}]  →  [${tgtIds.join(", ")}]
      </div>
    `;
  }

  function tokenTable(tokens, ids, which) {
    const vocab = which === "src" ? MODEL.VOCAB_SRC : MODEL.VOCAB_TGT;
    let html = `<table class="param-table">`;
    html += `<tr><th>位置</th><th>Token</th><th>ID</th></tr>`;
    tokens.forEach((t, i) => {
      html += `<tr><td>${i}</td><td>${MX.escapeHtml(t)}</td><td>${ids[i]}</td></tr>`;
    });
    html += `</table>`;
    html += `<div style="font-size:11px;color:#6f7cac;margin-top:8px">词表大小: ${vocab.length}</div>`;
    return html;
  }

  function stepEmbedding() {
    const { srcIds, srcTokens, encEmb } = ctx;
    const E = params.embedSrc;
    // 高亮被查询的行
    const highlight = srcIds.map((id) => [id, 0]); // 只是标记
    return `
      <div class="step-title">词嵌入: 查表 (Embedding Lookup)</div>
      <div class="step-desc">用 token ID 查 Embedding 矩阵 E ∈ ℝ<sup>V × d_model</sup>, 得到每个 token 的向量表示。</div>
      <div class="step-formula">E<sub>src</sub>[id<sub>i</sub>] = x<sub>i</sub>   (x<sub>i</sub> ∈ ℝ<sup>d_model</sup>)</div>

      <div style="display:flex;flex-wrap:wrap;align-items:flex-start;gap:12px">
        ${MX.render(E, {
          label: "Src Embedding 表 E_src",
          rowLabels: MODEL.VOCAB_SRC,
          colLabels: Array.from({ length: MX.shape(E)[1] }, (_, i) => `d${i}`),
          highlight: srcIds.flatMap((id) => Array.from({ length: MX.shape(E)[1] }, (_, j) => [id, j])),
        })}
        <div class="op-arrow"><div class="op">lookup(${srcIds.join(",")})</div><div class="arr">→</div></div>
        ${MX.render(encEmb, {
          label: "查出的嵌入矩阵 E",
          rowLabels: srcTokens,
          colLabels: Array.from({ length: MX.shape(encEmb)[1] }, (_, i) => `d${i}`),
        })}
      </div>
      <p style="color:#97a3cc;font-size:13px;margin-top:14px">
        高亮的行是源 token 对应 ID 在 Embedding 表中的位置。这些向量是<b>可学习的参数</b>。
      </p>
    `;
  }

  function stepPosEnc() {
    const { srcTokens, encPE } = ctx;
    return `
      <div class="step-title">位置编码: 注入位置信息</div>
      <div class="step-desc">自注意力本身对位置无感知, 因此需要加入<b>位置编码 PE</b>。经典实现使用正弦/余弦:</div>
      <div class="step-formula">
        PE(pos, 2i)&nbsp;&nbsp;= sin(pos / 10000<sup>2i/d_model</sup>)<br/>
        PE(pos, 2i+1) = cos(pos / 10000<sup>2i/d_model</sup>)
      </div>

      ${MX.render(encPE, {
        label: "位置编码矩阵 PE",
        rowLabels: srcTokens.map((t, i) => `pos=${i}`),
        colLabels: Array.from({ length: MX.shape(encPE)[1] }, (_, i) => `d${i}`),
      })}

      <p style="color:#97a3cc;font-size:13px;margin-top:14px">
        不同维度对应不同波长, 模型通过这些周期信号识别相对位置。PE 与 d_model 同维度, 可以直接与 embedding 相加。
      </p>
    `;
  }

  function stepEncInput() {
    const { encEmb, encPE, encX, srcTokens } = ctx;
    return `
      <div class="step-title">Encoder 输入: X = E + PE</div>
      <div class="step-desc">把 embedding 和位置编码<b>逐元素相加</b>作为 Encoder 第一层的输入。</div>
      <div class="step-formula">X = E + PE ∈ ℝ<sup>T × d_model</sup></div>
      ${MX.renderOp(encEmb, "+", encPE, encX, {
        labelA: "Embedding E", labelB: "PE", labelC: "X = E + PE",
        rowLabelsA: srcTokens, rowLabelsB: srcTokens, rowLabelsC: srcTokens,
      })}
    `;
  }

  function stepQKV() {
    const { encX, srcTokens, encLayer0 } = ctx;
    const L = params.enc[0];
    const H = MODEL.CONFIG.n_heads;
    const dk = MODEL.CONFIG.d_k;
    const dm = MODEL.CONFIG.d_model;
    // 把大 W_Q [d_model × d_model] 拆成 H 个 W_Q^h [d_model × d_k]
    const WqHeads = [], WkHeads = [], WvHeads = [];
    for (let h = 0; h < H; h++) {
      WqHeads.push(MX.fromShape(dm, dk, (i, j) => L.Wq[i][h * dk + j]));
      WkHeads.push(MX.fromShape(dm, dk, (i, j) => L.Wk[i][h * dk + j]));
      WvHeads.push(MX.fromShape(dm, dk, (i, j) => L.Wv[i][h * dk + j]));
    }
    const Qh = encLayer0.saCache.Qh;
    const Kh = encLayer0.saCache.Kh;
    const Vh = encLayer0.saCache.Vh;

    let html = `
      <div class="step-title">Q, K, V 投影 — 每个头有独立参数</div>
      <div class="step-desc">
        每个注意力头有自己独立的投影矩阵 W<sub>Q</sub><sup>h</sup>、W<sub>K</sub><sup>h</sup>、W<sub>V</sub><sup>h</sup>，
        尺寸都是 <code>[d_model × d_k]</code>。<br/>
        每个头把输入 X 投影到自己的 Q、K、V 子空间, 各头<b>独立学习不同的注意力模式</b>。
      </div>
      <div class="step-formula">
        Q<sub>h</sub> = X · W<sub>Q</sub><sup>(h)</sup> ∈ ℝ<sup>T × d_k</sup>&nbsp;&nbsp;(h = 1..${H})
      </div>
      <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:12px">
        <div style="color:#f472b6;font-size:13px;font-weight:600;margin-bottom:10px">
          ⚠ 概念视角: 每个头有独立的参数矩阵
        </div>
        <div style="color:#97a3cc;font-size:12px;line-height:1.6;margin-bottom:12px">
          论文原始描述: 每个头 h 用<b>自己的</b> W<sub>Q</sub><sup>(h)</sup> ∈ ℝ<sup>${dm} × ${dk}</sup>
          投影 X → Q<sub>h</sub>，而不是先算一个大 Q 再切分。<br/>
          实现上这两种方式数学等价 — 把 ${H} 个 W<sub>Q</sub><sup>(h)</sup> 水平拼接就等于一个大 W<sub>Q</sub> ∈ ℝ<sup>${dm} × ${dm}</sup>，
          先乘再切分和分别乘结果相同。但"每头独立参数"的视角更清晰。
        </div>
    `;

    for (let h = 0; h < H; h++) {
      html += `<div style="margin-bottom:16px">
        <div style="color:#60a5fa;font-size:13px;font-weight:600;margin-bottom:6px">第 ${h + 1} 头 (head ${h + 1})</div>`;
      html += MX.renderOp(encX, "·", WqHeads[h], Qh[h], {
        labelA: "X", labelB: `W_Q^(${h+1}) [${dm}×${dk}]`, labelC: `Q_${h+1} [T×${dk}]`,
        rowLabelsA: srcTokens, rowLabelsC: srcTokens,
      });
      html += MX.renderOp(encX, "·", WkHeads[h], Kh[h], {
        labelA: "X", labelB: `W_K^(${h+1}) [${dm}×${dk}]`, labelC: `K_${h+1} [T×${dk}]`,
        rowLabelsA: srcTokens, rowLabelsC: srcTokens,
      });
      html += MX.renderOp(encX, "·", WvHeads[h], Vh[h], {
        labelA: "X", labelB: `W_V^(${h+1}) [${dm}×${dk}]`, labelC: `V_${h+1} [T×${dk}]`,
        rowLabelsA: srcTokens, rowLabelsC: srcTokens,
      });
      html += `</div>`;
    }

    html += `</div>`;

    html += `
      <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:14px;margin-top:12px">
        <div style="color:#a78bfa;font-size:13px;font-weight:600;margin-bottom:8px">
          ≡ 等价的实现视角 (工程优化)
        </div>
        <div style="color:#97a3cc;font-size:12px;line-height:1.6;margin-bottom:10px">
          实际工程中 (PyTorch、Hugging Face 等), 会把 ${H} 个小矩阵拼成一个大矩阵
          W<sub>Q</sub> = [W<sub>Q</sub><sup>(1)</sup> | W<sub>Q</sub><sup>(2)</sup>] ∈ ℝ<sup>${dm} × ${dm}</sup>,
          一次大矩阵乘法算出全部 Q, 然后再切分给各头。这样 GPU 更高效, 但数学上完全等价。
        </div>
        ${MX.renderOp(encX, "·", L.Wq, encLayer0.saCache.Q, {
          labelA: "X", labelB: `W_Q [${dm}×${dm}] (拼接)`, labelC: `Q [T×${dm}] (完整)`,
          rowLabelsA: srcTokens, rowLabelsC: srcTokens,
        })}
        <div style="color:#97a3cc;font-size:12px;margin-top:8px">
          ↑ 然后切分 Q ∈ ℝ<sup>T×${dm}</sup> → Q<sub>1</sub>, Q<sub>2</sub> ∈ ℝ<sup>T×${dk}</sup>,
          结果与上面完全一致。
        </div>
      </div>
    `;

    return html;
  }

  function stepHeads() {
    const { srcTokens, encLayer0 } = ctx;
    const Qh = encLayer0.saCache.Qh;
    const Kh = encLayer0.saCache.Kh;
    const Vh = encLayer0.saCache.Vh;
    const H = MODEL.CONFIG.n_heads;
    const dk = MODEL.CONFIG.d_k;
    let html = `
      <div class="step-title">各头的 Q, K, V 汇总</div>
      <div class="step-desc">上一步已经用每个头独立的参数矩阵得到了各头的 Q<sub>h</sub>, K<sub>h</sub>, V<sub>h</sub>。
      下面把它们列在一起, 准备分头计算注意力。</div>
      <div class="step-formula">
        每个头: Q<sub>h</sub>, K<sub>h</sub>, V<sub>h</sub> ∈ ℝ<sup>T × ${dk}</sup>&nbsp;&nbsp;(h = 1..${H})
      </div>
    `;
    for (let h = 0; h < H; h++) {
      html += `<div style="margin-bottom:16px">
        <div style="color:#60a5fa;font-size:13px;font-weight:600;margin-bottom:4px">Head ${h + 1}</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px">`;
      html += MX.render(Qh[h], {
        label: `Q_${h+1}`,
        rowLabels: srcTokens,
        colLabels: Array.from({ length: dk }, (_, i) => `d${i}`),
      });
      html += MX.render(Kh[h], {
        label: `K_${h+1}`,
        rowLabels: srcTokens,
        colLabels: Array.from({ length: dk }, (_, i) => `d${i}`),
      });
      html += MX.render(Vh[h], {
        label: `V_${h+1}`,
        rowLabels: srcTokens,
        colLabels: Array.from({ length: dk }, (_, i) => `d${i}`),
      });
      html += `</div></div>`;
    }
    html += `<p style="color:#97a3cc;font-size:13px;margin-top:10px">
      注意: 每个头只看 d_k=${dk} 维的子空间。不同头的参数独立, 所以可以学到不同的注意力模式。
      例如一个头可能关注局部相邻词, 另一个头关注句法依赖。
    </p>`;
    return html;
  }

  function stepAttention() {
    const { srcTokens, encLayer0 } = ctx;
    const sdpa = encLayer0.saCache.sdpaCaches[0]; // head 0
    const Q = sdpa.Q, K = sdpa.K;
    const scores0 = MX.scale(MX.matmul(Q, MX.transpose(K)), 1 / Math.sqrt(sdpa.dk));
    const attn = sdpa.attn;
    return `
      <div class="step-title">缩放点积注意力: Scaled Dot-Product Attention (Head 1)</div>
      <div class="step-desc">计算每个 Query 对所有 Key 的相似度, 除以 √d_k 以稳定梯度, 然后做 row-wise softmax 得到注意力权重。</div>
      <div class="step-formula">
        scores = Q · K<sup>T</sup> / √d_k  ∈ ℝ<sup>T × T</sup><br/>
        A = softmax(scores)  (每行和为 1)
      </div>
      <div>
        ${MX.renderOp(Q, "·K^T /√d_k", MX.transpose(K), scores0, {
          labelA:"Q (head 1)", labelB:"Kᵀ", labelC:"Scores",
          rowLabelsA: srcTokens, colLabelsC: srcTokens, rowLabelsC: srcTokens,
        })}
      </div>
      <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-top:14px">
        ${MX.render(scores0, { label:"Scores", rowLabels: srcTokens, colLabels: srcTokens })}
        <div class="op-arrow"><div class="op">softmax<br/>(row-wise)</div><div class="arr">→</div></div>
        ${MX.render(attn, { label:"Attention A", rowLabels: srcTokens, colLabels: srcTokens, digits: 3 })}
      </div>
      <p style="color:#97a3cc;font-size:13px;margin-top:12px">
        Scores[i,j] 衡量第 i 个 token 应该多关注第 j 个 token。经过 softmax 后, 每一行是一个概率分布。
      </p>
    `;
  }

  function stepAttnOutEnc() {
    const { srcTokens, encLayer0 } = ctx;
    const sdpa = encLayer0.saCache.sdpaCaches[0];
    const A = sdpa.attn, V = sdpa.V;
    const headOut = MX.matmul(A, V);
    const concat = encLayer0.saCache.O;
    return `
      <div class="step-title">注意力输出 → 拼接 → 残差 → LayerNorm</div>
      <div class="step-desc">
        ① 每头输出 Z<sub>h</sub> = A<sub>h</sub> · V<sub>h</sub><br/>
        ② 拼接所有头, 再乘输出投影 W<sub>O</sub>: <b>Output = Concat(Z) · W<sub>O</sub></b><br/>
        ③ 加残差: X + Output<br/>
        ④ LayerNorm: 对每一行归一化 (使数值稳定)
      </div>
      <div class="step-formula">
        Z<sub>h</sub> = A<sub>h</sub> · V<sub>h</sub>&nbsp;;&nbsp;&nbsp; 
        Y = Concat(Z<sub>1</sub>..Z<sub>H</sub>) · W<sub>O</sub>&nbsp;;&nbsp;&nbsp; 
        X' = LayerNorm(X + Y)
      </div>
      <div>
        ${MX.renderOp(A, "·", V, headOut, {
          labelA:"A (head 1)", labelB:"V (head 1)", labelC:"Z (head 1)",
          rowLabelsA: srcTokens, rowLabelsC: srcTokens,
        })}
        <div style="margin-top:14px">
          <b style="color:#60a5fa;font-size:13px">拼接所有头 + W_O:</b>
          ${MX.render(concat, { label:"Concat(Z₁..Z_H)", rowLabels: srcTokens })}
          ${MX.render(encLayer0.resid1, { label:"X + Y (残差)", rowLabels: srcTokens })}
          ${MX.render(encLayer0.inp, { label:"LayerNorm 后 (x1)", rowLabels: srcTokens })}
        </div>
      </div>
      <p style="color:#97a3cc;font-size:13px;margin-top:10px">
        LayerNorm 的作用: 把每个 token 的向量标准化为均值 0 方差 1, 再用可学习的 γ, β 放缩平移。
      </p>
    `;
  }

  function stepFFN() {
    const { srcTokens, encLayer0 } = ctx;
    const L = params.enc[0];
    const x1 = encLayer0.inp;
    const h = encLayer0.h;
    const hRelu = encLayer0.hRelu;
    const ffRaw = MX.matmul(hRelu, L.W2);
    const [fr, fc] = MX.shape(ffRaw);
    const ff = MX.fromShape(fr, fc, (i, j) => ffRaw[i][j] + L.b2[j]);
    return `
      <div class="step-title">前馈网络 FFN (Position-wise Feed Forward)</div>
      <div class="step-desc">对每个位置独立应用一个两层 MLP, 扩展到更高维 <code>d_ff</code> 再回到 <code>d_model</code>。</div>
      <div class="step-formula">
        FFN(x) = ReLU(x · W<sub>1</sub> + b<sub>1</sub>) · W<sub>2</sub> + b<sub>2</sub>
      </div>
      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px">
        ${MX.render(x1, { label:"输入 x1", rowLabels: srcTokens })}
        <div class="op-arrow"><div class="op">·W₁+b₁<br/>ReLU</div><div class="arr">→</div></div>
        ${MX.render(hRelu, { label:`ReLU(x·W₁+b₁) [d_ff=${MODEL.CONFIG.d_ff}]`, rowLabels: srcTokens })}
        <div class="op-arrow"><div class="op">·W₂+b₂</div><div class="arr">→</div></div>
        ${MX.render(ff, { label:"FFN 输出", rowLabels: srcTokens })}
      </div>
      <p style="color:#97a3cc;font-size:13px;margin-top:10px">
        FFN 可以看作把每个 token 向量通过一个非线性变换, 增强表达力。它对每个位置<b>独立</b>应用 (无跨位置交互)。
      </p>
    `;
  }

  function stepEncOut() {
    const { srcTokens, encOut } = ctx;
    return `
      <div class="step-title">Encoder 最终输出</div>
      <div class="step-desc">经过 N 层 Encoder (这里 N=1) 之后, 我们得到源句子的上下文表示, 供 Decoder 中的 Cross-Attention 使用。</div>
      ${MX.render(encOut, {
        label: `Encoder Output (将传给 Decoder 的 Cross-Attention 作 K, V)`,
        rowLabels: srcTokens,
        colLabels: Array.from({ length: MX.shape(encOut)[1] }, (_, i) => `d${i}`),
      })}
      <p style="color:#97a3cc;font-size:13px;margin-top:10px">
        形状 <code>[src_len, d_model] = [${MX.shape(encOut).join(", ")}]</code>。
        这同时也是 Encoder 训练的"副产品"——所有可学习参数已在前面的矩阵中。
      </p>
    `;
  }

  function stepDecInput() {
    const { tgtTokens, tgtIds, decEmb, decPE, decX } = ctx;
    return `
      <div class="step-title">Decoder 输入构建</div>
      <div class="step-desc">Decoder 的输入是 <code>&lt;bos&gt;</code> + 目标句子 (训练时用<b>右移</b>的 target 作为输入, 目标输出是向左移一位, 以便预测下一个词)。</div>
      <div style="font-family:ui-monospace,monospace;color:#97a3cc;margin-bottom:10px">
        decoder input: [${tgtTokens.map(MX.escapeHtml).join(", ")}]   →   ids: [${tgtIds.join(", ")}]
      </div>
      <div>
        ${MX.renderOp(decEmb, "+", decPE, decX, {
          labelA:"Embedding E", labelB:"PE", labelC:"X = E + PE",
          rowLabelsA: tgtTokens, rowLabelsB: tgtTokens, rowLabelsC: tgtTokens,
        })}
      </div>
    `;
  }

  function stepMasked() {
    const { tgtTokens, decLayer0 } = ctx;
    const sdpa = decLayer0.saCache.sdpaCaches[0];
    const dk = sdpa.dk;
    const Q = sdpa.Q, K = sdpa.K;
    // scores 带 mask
    const rawScores = MX.scale(MX.matmul(Q, MX.transpose(K)), 1 / Math.sqrt(dk));
    const mask = MX.causalMask(tgtTokens.length);
    const maskedScores = MX.clone(rawScores);
    for (let i = 0; i < MX.shape(maskedScores)[0]; i++)
      for (let j = 0; j < MX.shape(maskedScores)[1]; j++)
        if (mask[i][j] === 0) maskedScores[i][j] = -Infinity;
    const attn = sdpa.attn;
    return `
      <div class="step-title">Masked Self-Attention (Decoder 第一层)</div>
      <div class="step-desc">
        训练时, Decoder 一次性接收整个目标序列, 但为了模拟"一个一个生成"的过程,
        我们用<b>因果掩码 (Causal Mask)</b> 阻止每个位置看到未来。
      </div>
      <div class="step-formula">
        scores<sub>ij</sub> = (Q · K<sup>T</sup>)<sub>ij</sub> / √d_k  , &nbsp;&nbsp;
        若 j &gt; i: 置为 -∞ ⇒ softmax 后为 0
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
        ${MX.render(rawScores, { label:"原始 scores", rowLabels: tgtTokens, colLabels: tgtTokens })}
        <div class="op-arrow"><div class="op">+ mask</div><div class="arr">→</div></div>
        ${MX.render(maskedScores, { label:"加上 causal mask", rowLabels: tgtTokens, colLabels: tgtTokens })}
        <div class="op-arrow"><div class="op">softmax</div><div class="arr">→</div></div>
        ${MX.render(attn, { label:"Attention A (上三角为 0)", rowLabels: tgtTokens, colLabels: tgtTokens, digits: 3 })}
      </div>
      <p style="color:#97a3cc;font-size:13px;margin-top:10px">
        A 中的上三角全为 0: 说明位置 i 只能看到 &le; i 的位置。这让 Decoder 具有自回归性质。
      </p>
    `;
  }

  function stepCross() {
    const { tgtTokens, srcTokens, decLayer0, encOut } = ctx;
    const caSdpa = decLayer0.caCache.sdpaCaches[0];
    const attn = caSdpa.attn;
    return `
      <div class="step-title">Cross-Attention: Decoder ↔ Encoder 对齐</div>
      <div class="step-desc">
        Q 来自 Decoder 当前状态, K/V 来自 Encoder 输出。 
        它决定: 为了生成目标句子的第 i 个位置, 应该<b>从源句子哪里</b>取信息 (类似机器翻译中的对齐)。
      </div>
      <div class="step-formula">
        Q = X<sub>dec</sub> · W<sub>Q</sub><sup>cross</sup>&nbsp;;&nbsp;&nbsp;
        K = X<sub>enc</sub> · W<sub>K</sub><sup>cross</sup>&nbsp;;&nbsp;&nbsp;
        V = X<sub>enc</sub> · W<sub>V</sub><sup>cross</sup>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:10px">
        ${MX.render(attn, {
          label: "Cross-Attention A (row=目标位, col=源位)",
          rowLabels: tgtTokens,
          colLabels: srcTokens,
          digits: 3,
        })}
        ${MX.render(encOut, {
          label: "X_enc (作为 K/V 来源)",
          rowLabels: srcTokens,
        })}
      </div>
      <p style="color:#97a3cc;font-size:13px;margin-top:10px">
        每一行是某个目标位对所有源位的注意力分布, 其和为 1。例如对应 "love" 的行, 理想情况下会集中关注源句中的 "爱"。
      </p>
    `;
  }

  function stepDecFFN() {
    const { tgtTokens, decLayer0 } = ctx;
    return `
      <div class="step-title">Decoder FFN + 最后的 Add &amp; Norm</div>
      <div class="step-desc">和 Encoder 中的 FFN 相同: 逐位置两层 MLP。完成后得到 decoder 该层的输出 X'。</div>
      ${MX.render(decLayer0.resid3, { label:"残差 x2 + FFN(x2)", rowLabels: tgtTokens })}
      ${MX.render(MX.layerNorm(decLayer0.resid3, params.dec[0].ln3_g, params.dec[0].ln3_b), {
        label:"最后 LayerNorm 后", rowLabels: tgtTokens,
      })}
    `;
  }

  function stepOutput() {
    const { tgtTokens, decLogits } = ctx;
    const probs = MX.softmax(decLogits);
    return `
      <div class="step-title">输出: Linear + Softmax</div>
      <div class="step-desc">
        把 Decoder 最终隐状态 h 投影到词表大小, 再 softmax 得到每个位置下一个词的概率分布。
      </div>
      <div class="step-formula">
        logits = h · W<sub>out</sub> + b<sub>out</sub>   ∈ ℝ<sup>T × V</sup><br/>
        P(y<sub>t</sub>) = softmax(logits<sub>t</sub>)
      </div>
      ${MX.render(decLogits, {
        label:"Logits [T × vocab]",
        rowLabels: tgtTokens,
        colLabels: MODEL.VOCAB_TGT,
      })}
      ${MX.render(probs, {
        label:"Softmax 后的概率",
        rowLabels: tgtTokens,
        colLabels: MODEL.VOCAB_TGT,
        digits: 3,
      })}
      <p style="color:#97a3cc;font-size:13px;margin-top:10px">
        训练时, 每个位置 t 对应的<b>真实下一个词</b>是目标概率的参考, 用交叉熵作为损失。
        推理时, 取 argmax (贪心) 或按分布采样即可。
      </p>
    `;
  }

  // ============================================================
  // 初始化
  // ============================================================
  function init(globalParams) {
    setup(globalParams);
    renderStepNav();
    document.getElementById("step-prev").addEventListener("click", () => jumpTo(currentStep - 1));
    document.getElementById("step-next").addEventListener("click", () => jumpTo(currentStep + 1));
    const sampleSel = document.getElementById("comp-sample");
    if (sampleSel) {
      sampleSel.addEventListener("change", () => {
        setup(params);
        jumpTo(currentStep);  // 刷新当前步骤, 保持位置
      });
    }
    jumpTo(0);
  }

  return { init, jumpFromArch };
})();
