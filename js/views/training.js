// ============================================================
// training.js — 训练过程可视化
// 展示前向传播 → 交叉熵 → 反向传播 → SGD 更新 → Loss 曲线
// ============================================================

const TrainingView = (function () {
  let params = null;
  let step = 0;
  let lossHistory = [];
  let gradNormHistory = [];   // [{step, Wout, decW1, decWqs, encWq, embedTgt}]
  let lastInfo = null;

  function getLR() {
    return parseFloat(document.getElementById("train-lr").value);
  }
  function getRunSteps() {
    return Math.max(1, Math.min(200, parseInt(document.getElementById("train-steps").value) || 20));
  }

  function init(globalParams) {
    params = globalParams;
    bindControls();
    render();
  }

  function bindControls() {
    document.getElementById("train-step").addEventListener("click", () => {
      doTrainStep(); render();
    });
    document.getElementById("train-run").addEventListener("click", () => {
      runMany(getRunSteps());
    });
    document.getElementById("train-reset").addEventListener("click", () => {
      window.APP.resetParams();
      params = window.APP.getParams();
      step = 0; lossHistory = []; gradNormHistory = []; lastInfo = null;
      render();
    });
    document.getElementById("train-sample").addEventListener("change", render);
    document.getElementById("train-lr").addEventListener("input", () => {
      document.getElementById("train-lr-val").textContent = getLR().toFixed(3);
    });
  }

  function currentSampleIdx() {
    return +document.getElementById("train-sample").value;
  }

  function doTrainStep() {
    const pair = MODEL.makeTrainPair(currentSampleIdx());
    const fwd = MODEL.forward(pair.srcIds, pair.tgtIn, params);
    const bwd = MODEL.backward(fwd, pair.tgtOut, params);
    MODEL.sgdStep(params, bwd.grads, getLR());
    step++;
    lossHistory.push(bwd.loss);
    const g = bwd.grads;
    gradNormHistory.push({
      step,
      Wout:    norm(g.Wout),
      decW1:   norm(g.dec[0].W1),
      decWqs:  norm(g.dec[0].Wqs),
      encWq:   norm(g.enc[0].Wq),
      embedTgt: norm(g.embedTgt),
    });
    lastInfo = { pair, fwd, bwd };
  }

  async function runMany(n) {
    const btn = document.getElementById("train-run");
    btn.disabled = true;
    btn.textContent = "训练中...";
    const label = document.getElementById("train-run-label");
    for (let i = 0; i < n; i++) {
      const idx = i % MODEL.SAMPLES.length;
      document.getElementById("train-sample").value = String(idx);
      doTrainStep();
      if (i % 3 === 0) {
        if (label) label.textContent = `${i + 1}/${n}`;
        render();
        await new Promise((r) => setTimeout(r, 40));
      }
    }
    render();
    if (label) label.textContent = "";
    btn.disabled = false;
    btn.textContent = "连续训练";
  }

  function render() {
    document.getElementById("train-stats").textContent =
      `Step: ${step} · Loss: ${lastInfo ? lastInfo.bwd.loss.toFixed(4) : "—"} · LR: ${getLR().toFixed(3)}`;
    renderForward();
    renderLoss();
    renderBackward();
    renderUpdate();
    renderLossChart();
    renderGradChart();
  }

  // ------------------- ① 前向传播 -------------------
  function renderForward() {
    const el = document.getElementById("train-forward");
    const pair = MODEL.makeTrainPair(currentSampleIdx());
    const fwd = lastInfo ? lastInfo.fwd : MODEL.forward(pair.srcIds, pair.tgtIn, params);
    const tgtTokens = ["<bos>", ...pair.tgtTokens];
    const expected = [...pair.tgtTokens, "<eos>"];
    const probs = MX.softmax(fwd.logits);
    const pred = probs.map((row) => {
      let mx = -1, idx = 0;
      for (let i = 0; i < row.length; i++) if (row[i] > mx) { mx = row[i]; idx = i; }
      return { word: MODEL.VOCAB_TGT[idx], p: mx };
    });

    const flowNodes = [
      { name: `源: [${pair.srcTokens.map(MX.escapeHtml).join(", ")}]`, shape: `[${pair.srcIds.length}]` },
      { name: "Src Embedding + PE",      shape: `[${pair.srcIds.length} × ${MODEL.CONFIG.d_model}]` },
      { name: "Encoder Self-Attn + FFN", shape: `[${pair.srcIds.length} × ${MODEL.CONFIG.d_model}]` },
      { name: `目标输入: [${tgtTokens.map(MX.escapeHtml).join(", ")}]`, shape: `[${tgtTokens.length}]` },
      { name: "Tgt Embedding + PE",      shape: `[${tgtTokens.length} × ${MODEL.CONFIG.d_model}]` },
      { name: "Masked Self-Attn",        shape: `[${tgtTokens.length} × ${MODEL.CONFIG.d_model}]` },
      { name: "Cross-Attn (看 Encoder)", shape: `[${tgtTokens.length} × ${MODEL.CONFIG.d_model}]` },
      { name: "FFN + LayerNorm",         shape: `[${tgtTokens.length} × ${MODEL.CONFIG.d_model}]` },
      { name: "Linear + Softmax",        shape: `[${tgtTokens.length} × ${MODEL.VOCAB_TGT.length}]` },
    ];
    let html = flowNodes.map((n, i) =>
      `<div class="flow-node">
        <span class="fn-name">${n.name}</span>
        <span class="fn-shape">${n.shape}</span>
      </div>${i < flowNodes.length - 1 ? '<div class="flow-arrow">↓</div>' : ''}`
    ).join("");

    html += `<div style="margin-top:12px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:8px">
      <div style="font-size:12px;color:#97a3cc;margin-bottom:6px">当前预测 vs 期望:</div>`;
    for (let i = 0; i < tgtTokens.length; i++) {
      const ok = pred[i].word === expected[i];
      html += `<div style="display:flex;justify-content:space-between;font-family:ui-monospace,monospace;font-size:12px;padding:2px 0">
        <span>pos ${i}: <b style="color:${ok ? 'var(--ok)' : 'var(--err)'}">${MX.escapeHtml(pred[i].word)}</b> (p=${pred[i].p.toFixed(3)})</span>
        <span style="color:#97a3cc">期望: ${MX.escapeHtml(expected[i])}</span>
      </div>`;
    }
    html += `</div>`;
    el.innerHTML = html;
  }

  // ------------------- ② 损失 -------------------
  function renderLoss() {
    const el = document.getElementById("train-loss");
    const pair = MODEL.makeTrainPair(currentSampleIdx());
    const fwd = lastInfo ? lastInfo.fwd : MODEL.forward(pair.srcIds, pair.tgtIn, params);
    const tgtIn = ["<bos>", ...pair.tgtTokens];
    const target = [...pair.tgtTokens, "<eos>"];
    const probs = MX.softmax(fwd.logits);
    let total = 0;
    let html = `<div class="loss-detail">`;
    html += `<div style="margin-bottom:10px;color:#97a3cc">L = -(1/T) Σ log P(y<sub>t</sub> | y<sub>&lt;t</sub>, x)</div>`;
    for (let i = 0; i < target.length; i++) {
      const tId = MODEL.TGT_ID[target[i]];
      const p = probs[i][tId];
      const nll = -Math.log(Math.max(1e-9, p));
      total += nll;
      html += `<div class="loss-row">
        <span>t=${i} 输入=<span class="token">${MX.escapeHtml(tgtIn[i])}</span> 期望=<span class="token">${MX.escapeHtml(target[i])}</span></span>
        <span class="prob">P(${MX.escapeHtml(target[i])}) = ${p.toFixed(4)}</span>
        <span class="val">-log P = ${nll.toFixed(3)}</span>
      </div>`;
    }
    const avg = total / target.length;
    html += `<div class="loss-total">总损失 L = ${avg.toFixed(4)} (= ${total.toFixed(3)} / ${target.length})</div>`;
    html += `</div>`;
    el.innerHTML = html;
  }

  // ------------------- ③ 反向传播 -------------------
  function renderBackward() {
    const el = document.getElementById("train-backward");
    if (!lastInfo) {
      el.innerHTML = `<div style="color:#97a3cc;font-size:13px">点击 "执行一步训练" 触发反向传播</div>`;
      return;
    }
    const g = lastInfo.bwd.grads;
    const nodes = [
      { name: "∂L/∂logits",             val: "= softmax − onehot" },
      { name: "∂L/∂W_out",              val: `‖·‖ = ${norm(g.Wout).toFixed(4)}` },
      { name: "∂L/∂ (Dec FFN)",         val: `‖∂W₁‖ = ${norm(g.dec[0].W1).toFixed(4)}` },
      { name: "∂L/∂ (Cross-Attn)",      val: `‖∂Wqc‖ = ${norm(g.dec[0].Wqc).toFixed(4)}` },
      { name: "∂L/∂ (Masked Self-Attn)",val: `‖∂Wqs‖ = ${norm(g.dec[0].Wqs).toFixed(4)}` },
      { name: "∂L/∂ (Tgt Embedding)",   val: `‖·‖ = ${norm(g.embedTgt).toFixed(4)}` },
      { name: "∂L/∂ (Encoder)",         val: `‖∂Wq‖ = ${norm(g.enc[0].Wq).toFixed(4)}` },
      { name: "∂L/∂ (Src Embedding)",   val: `‖·‖ = ${norm(g.embedSrc).toFixed(4)}` },
    ];
    let html = nodes.map((n, i) =>
      `<div class="flow-node">
        <span class="fn-name">${n.name}</span>
        <span class="fn-shape">${n.val}</span>
      </div>${i < nodes.length - 1 ? '<div class="flow-arrow">·</div>' : ''}`
    ).join("");
    html += `<div style="color:#97a3cc;font-size:12px;margin-top:10px;line-height:1.6">
      梯度通过链式法则从输出层向输入层流动。‖·‖ 为 RMS 范数。
    </div>`;
    el.innerHTML = html;
  }

  // ------------------- ④ 参数更新 -------------------
  function renderUpdate() {
    const el = document.getElementById("train-update");
    const lr = getLR();
    if (!lastInfo) {
      el.innerHTML = `<div style="color:#97a3cc;font-size:13px">参数更新规则: θ ← θ − η · ∂L/∂θ&nbsp;&nbsp;(η = ${lr})</div>`;
      return;
    }
    const g = lastInfo.bwd.grads;
    const rows = [
      { k: "W_out",          dim: `[${MX.shape(params.Wout).join("×")}]`,          gn: norm(g.Wout) },
      { k: "b_out",          dim: `[${params.bout.length}]`,                        gn: vnorm(g.bout) },
      { k: "Dec W₁ (FFN)",   dim: `[${MX.shape(params.dec[0].W1).join("×")}]`,     gn: norm(g.dec[0].W1) },
      { k: "Dec W₂ (FFN)",   dim: `[${MX.shape(params.dec[0].W2).join("×")}]`,     gn: norm(g.dec[0].W2) },
      { k: "Dec W_Q (self)", dim: `[${MX.shape(params.dec[0].Wqs).join("×")}]`,    gn: norm(g.dec[0].Wqs) },
      { k: "Dec W_K (self)", dim: `[${MX.shape(params.dec[0].Wks).join("×")}]`,    gn: norm(g.dec[0].Wks) },
      { k: "Dec W_V (self)", dim: `[${MX.shape(params.dec[0].Wvs).join("×")}]`,    gn: norm(g.dec[0].Wvs) },
      { k: "Dec W_Q (cross)",dim: `[${MX.shape(params.dec[0].Wqc).join("×")}]`,    gn: norm(g.dec[0].Wqc) },
      { k: "Enc W_Q",        dim: `[${MX.shape(params.enc[0].Wq).join("×")}]`,     gn: norm(g.enc[0].Wq) },
      { k: "Enc W₁ (FFN)",   dim: `[${MX.shape(params.enc[0].W1).join("×")}]`,     gn: norm(g.enc[0].W1) },
      { k: "Src Embedding",  dim: `[${MX.shape(params.embedSrc).join("×")}]`,       gn: norm(g.embedSrc) },
      { k: "Tgt Embedding",  dim: `[${MX.shape(params.embedTgt).join("×")}]`,       gn: norm(g.embedTgt) },
    ];
    let html = `
      <div style="font-size:12px;color:#97a3cc;margin-bottom:10px;font-family:ui-monospace,monospace">
        η = ${lr}  ·  W ← W − η · ∂L/∂W
      </div>
      <table class="param-table">
        <tr><th>参数</th><th>形状</th><th>‖梯度‖ (RMS)</th><th>变化量 η·‖g‖</th></tr>`;
    for (const r of rows) {
      const dw = lr * r.gn;
      html += `<tr>
        <td>${r.k}</td><td>${r.dim}</td>
        <td>${r.gn.toFixed(4)}</td>
        <td class="delta neg">${dw.toFixed(4)}</td>
      </tr>`;
    }
    html += `</table>`;
    el.innerHTML = html;
  }

  function norm(M) {
    let s = 0, n = 0;
    for (const r of M) for (const v of r) { s += v * v; n++; }
    return Math.sqrt(s / Math.max(1, n));
  }
  function vnorm(v) {
    let s = 0;
    for (const x of v) s += x * x;
    return Math.sqrt(s / Math.max(1, v.length));
  }

  // ------------------- Loss 曲线 -------------------
  function renderLossChart() {
    const canvas = document.getElementById("loss-chart");
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = "#0b1020";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#2a3566"; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, 20); ctx.lineTo(50, H - 30); ctx.lineTo(W - 20, H - 30);
    ctx.stroke();

    if (lossHistory.length < 1) {
      ctx.fillStyle = "#6f7cac"; ctx.font = "13px ui-monospace, monospace";
      ctx.fillText("尚未训练. 点击上方 '执行一步训练' 开始.", 60, H / 2);
      return;
    }

    const mn = Math.min(...lossHistory);
    const mx = Math.max(...lossHistory);
    const pad = (mx - mn) * 0.15 + 0.05;
    const ymin = Math.max(0, mn - pad), ymax = mx + pad;

    ctx.fillStyle = "#6f7cac"; ctx.font = "10px ui-monospace, monospace";
    for (let i = 0; i <= 4; i++) {
      const y = 20 + ((H - 50) * i) / 4;
      const v = ymax - ((ymax - ymin) * i) / 4;
      ctx.fillText(v.toFixed(2), 10, y + 4);
      ctx.strokeStyle = "#1a2244"; ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(W - 20, y); ctx.stroke();
    }

    const stepsShown = lossHistory.length;
    for (let i = 0; i <= 5; i++) {
      const x = 50 + ((W - 70) * i) / 5;
      const v = Math.round((stepsShown * i) / 5);
      ctx.fillStyle = "#6f7cac";
      ctx.fillText(v.toString(), x - 6, H - 12);
    }
    // X 轴标签
    ctx.fillStyle = "#6f7cac"; ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("Step", W - 40, H - 12);

    // 画折线
    ctx.strokeStyle = "#6aa6ff"; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < lossHistory.length; i++) {
      const x = 50 + ((W - 70) * i) / Math.max(1, stepsShown - 1);
      const y = 20 + ((H - 50) * (ymax - lossHistory[i])) / (ymax - ymin);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 最后一个点高亮
    if (lossHistory.length > 0) {
      const li = lossHistory.length - 1;
      const x = 50 + ((W - 70) * li) / Math.max(1, stepsShown - 1);
      const y = 20 + ((H - 50) * (ymax - lossHistory[li])) / (ymax - ymin);
      ctx.fillStyle = "#fbbf24";
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#fbbf24"; ctx.font = "11px ui-monospace,monospace";
      ctx.fillText(lossHistory[li].toFixed(3), x + 6, y - 4);
    }

    // 普通点
    ctx.fillStyle = "#a78bfa";
    for (let i = 0; i < lossHistory.length - 1; i++) {
      const x = 50 + ((W - 70) * i) / Math.max(1, stepsShown - 1);
      const y = 20 + ((H - 50) * (ymax - lossHistory[i])) / (ymax - ymin);
      ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
    }

    ctx.fillStyle = "#97a3cc"; ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(`Loss (min=${mn.toFixed(3)}, cur=${lossHistory[lossHistory.length-1].toFixed(3)})`, 60, 15);
  }

  // ------------------- 梯度范数历史图 -------------------
  function renderGradChart() {
    const canvas = document.getElementById("grad-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = "#0b1020";
    ctx.fillRect(0, 0, W, H);

    if (gradNormHistory.length < 2) {
      ctx.fillStyle = "#6f7cac"; ctx.font = "12px ui-monospace, monospace";
      ctx.fillText("训练两步后显示梯度范数曲线", 60, H / 2);
      return;
    }

    const keys = ["Wout", "decW1", "decWqs", "encWq", "embedTgt"];
    const colors = ["#6aa6ff", "#f472b6", "#fbbf24", "#34d399", "#a78bfa"];
    const labels = ["W_out", "Dec W₁", "Dec Wqs", "Enc Wq", "Emb_tgt"];

    const allVals = gradNormHistory.flatMap((h) => keys.map((k) => h[k]));
    const mxV = Math.max(...allVals, 0.001);

    ctx.strokeStyle = "#2a3566"; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, 10); ctx.lineTo(50, H - 24); ctx.lineTo(W - 20, H - 24);
    ctx.stroke();

    ctx.fillStyle = "#6f7cac"; ctx.font = "9px ui-monospace, monospace";
    for (let i = 0; i <= 3; i++) {
      const y = 10 + ((H - 34) * i) / 3;
      const v = mxV * (1 - i / 3);
      ctx.fillText(v.toFixed(3), 2, y + 3);
      ctx.strokeStyle = "#1a2244";
      ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(W - 20, y); ctx.stroke();
    }

    const N = gradNormHistory.length;
    for (let ki = 0; ki < keys.length; ki++) {
      ctx.strokeStyle = colors[ki]; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const x = 50 + ((W - 70) * i) / Math.max(1, N - 1);
        const y = 10 + ((H - 34) * (1 - gradNormHistory[i][keys[ki]] / mxV));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // 图例
    let lx = 55;
    for (let ki = 0; ki < keys.length; ki++) {
      ctx.fillStyle = colors[ki];
      ctx.fillRect(lx, H - 20, 8, 8);
      ctx.fillStyle = "#97a3cc"; ctx.font = "9px ui-monospace,monospace";
      ctx.fillText(labels[ki], lx + 10, H - 13);
      lx += 70;
    }

    ctx.fillStyle = "#97a3cc"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("梯度范数 (RMS)", 55, 10);
  }

  return { init, refresh: render };
})();
