// ============================================================
// inference.js — 推理 (自回归生成) 可视化
// 使用真实 KV Cache 增量计算, 支持 temperature 和采样模式
// ============================================================

const InferenceView = (function () {
  let params = null;
  let state = null;

  function getTemperature() {
    return parseFloat(document.getElementById("inf-temperature").value);
  }
  function getUseSampling() {
    return document.getElementById("inf-mode").value === "sample";
  }
  function currentSampleIdx() {
    return +document.getElementById("inf-source").value;
  }

  function init(globalParams) {
    params = globalParams;
    bind();
    reset();
  }

  function bind() {
    document.getElementById("inf-source").addEventListener("change", () => reset());
    document.getElementById("inf-step").addEventListener("click", stepOnce);
    document.getElementById("inf-auto").addEventListener("click", runAuto);
    document.getElementById("inf-reset").addEventListener("click", reset);
    document.getElementById("inf-temperature").addEventListener("input", () => {
      document.getElementById("inf-temp-val").textContent =
        parseFloat(document.getElementById("inf-temperature").value).toFixed(2);
    });
  }

  function reset() {
    const pair = MODEL.makeTrainPair(currentSampleIdx());
    // 真实 KV Cache: encoder 只算一次, cross-attn K/V 预计算
    const kvCache = MODEL.initInferenceKVCache(pair.srcIds, params);
    state = {
      srcIds: pair.srcIds,
      srcTokens: pair.srcTokens,
      kvCache,
      generated: [MODEL.CONFIG.bos_id],     // 从 <bos> 开始作为首个输入
      generatedTokens: ["<bos>"],
      done: false,
      stepIdx: 0,
      lastStep: null,
    };
    // 预处理 <bos>: 把 <bos> 送入 decoder 更新 KV cache, 准备预测 y1
    const r = MODEL.generateStepWithKVCache(MODEL.CONFIG.bos_id, kvCache, params, 1.0, false);
    state.kvCache = r.newKVCache;
    state.pendingResult = r;  // 第一步的预测已就绪, 等用户触发
    render();
  }

  function stepOnce() {
    if (!state || state.done) return;

    const temp = getTemperature();
    const sampling = getUseSampling();

    // 使用上一步预计算的结果 (或首次结果)
    const result = state.pendingResult;
    const chosen = result.chosen;

    state.generated.push(chosen.i);
    state.generatedTokens.push(chosen.token);
    state.stepIdx++;
    state.lastStep = { result, temp, sampling };

    if (chosen.i === MODEL.CONFIG.eos_id || state.generated.length >= MODEL.CONFIG.max_len + 1) {
      state.done = true;
      state.pendingResult = null;
    } else {
      // 预先计算下一步, 更新 KV cache
      const next = MODEL.generateStepWithKVCache(chosen.i, state.kvCache, params, temp, sampling);
      state.kvCache = next.newKVCache;
      state.pendingResult = next;
    }

    render();
  }

  async function runAuto() {
    const btn = document.getElementById("inf-auto");
    btn.disabled = true; btn.textContent = "生成中...";
    while (!state.done) {
      stepOnce();
      await new Promise((r) => setTimeout(r, 450));
    }
    btn.disabled = false; btn.textContent = "自动生成全部";
  }

  // ============================================================
  // 渲染
  // ============================================================
  function render() {
    renderEncoder();
    renderTokens();
    renderStepDetail();
    renderKVCache();
  }

  function renderEncoder() {
    const el = document.getElementById("inf-encoder");
    const encOut = state.kvCache.encOut;
    el.innerHTML = `
      <p class="hint">源句子: <b>${state.srcTokens.join(" ")}</b> · 只需在推理开始时计算一次</p>
      ${MX.render(encOut, {
        label: `Encoder Output [${MX.shape(encOut).join(" × ")}]`,
        rowLabels: state.srcTokens,
      })}
    `;
  }

  function renderTokens() {
    const el = document.getElementById("inf-tokens");
    let html = "";
    state.generatedTokens.forEach((t, i) => {
      let cls = "token-chip";
      if (i === 0) cls += " given";
      else if (i === state.generatedTokens.length - 1 && state.lastStep) cls += " new";
      if (t === "<eos>") cls += " eos";
      html += `<div class="${cls}">${MX.escapeHtml(t)}</div>`;
      if (i < state.generatedTokens.length - 1) {
        html += `<div style="color:#6f7cac;align-self:center">→</div>`;
      }
    });
    if (!state.done) {
      html += `<div style="color:#6f7cac;align-self:center;font-family:ui-monospace,monospace">→ ?</div>`;
    }
    el.innerHTML = html;
  }

  function renderStepDetail() {
    const el = document.getElementById("inf-step-detail");
    if (!state.lastStep) {
      el.innerHTML = `<p class="hint">点击 "生成下一个 token" 查看每一步详情.</p>
        <div style="color:#97a3cc;font-size:13px;line-height:1.8;margin-top:8px">
          <b>自回归生成 (每步仅计算新 token, KV Cache 复用历史):</b><br/>
          1. 把新 token 的 embedding + PE 送入 Decoder<br/>
          2. Self-Attn: 仅算新 Q, 与 <b>Cache 中的全部 K</b> 做点积 → 无需重跑历史<br/>
          3. Cross-Attn: Q 来自当前状态, K/V 来自 Encoder (固定不变)<br/>
          4. FFN → logits → 按温度缩放后取 argmax 或采样<br/>
          5. 若生成 &lt;eos&gt; 则停止
        </div>
      `;
      return;
    }

    const { result, temp, sampling } = state.lastStep;
    const top = result.topk;
    const maxP = top[0].p;
    const modeLabel = sampling ? "随机采样" : "贪心";

    let html = `<div style="font-size:13px;color:#97a3cc;margin-bottom:8px">
      第 ${state.stepIdx} 步 (T=${temp.toFixed(2)}, ${modeLabel}):
      当前输入 = [${state.generatedTokens.slice(0, -1).map(MX.escapeHtml).join(", ")}],
      预测 → <b style="color:var(--warn)">${MX.escapeHtml(top[0].token)}</b>
    </div>`;

    html += `<div class="topk-list">`;
    top.forEach((t, i) => {
      const pct = (t.p * 100).toFixed(1);
      const barW = (t.p / maxP) * 100;
      const isChosen = t.i === result.chosen.i;
      html += `<div class="topk-row${isChosen ? " chosen" : ""}">
        <span class="rank">#${i + 1}</span>
        <span class="word">${MX.escapeHtml(t.token)}</span>
        <span class="bar"><div class="bar-fill" style="width:${barW}%"></div></span>
        <span class="pct">${pct}%</span>
      </div>`;
    });
    html += `</div>`;
    el.innerHTML = html;
  }

  function renderKVCache() {
    const el = document.getElementById("inf-kvcache");
    const kv = state.kvCache.selfKV[0];

    if (!kv.K || kv.K.length === 0) {
      el.innerHTML = `<p class="hint" style="margin-top:10px">生成第一个 token 后, 把对应的 K/V 缓存下来。</p>`;
      return;
    }

    // kv.K: [T, d_model] — 真实的累计 K 矩阵, 每行对应一个已处理 token
    const T = kv.K.length;
    // 已处理的 token: <bos> 和之前生成的 (不含最新预测)
    const processedTokens = state.generatedTokens.slice(0, T);

    let html = `<div style="font-size:12px;color:#97a3cc;margin-bottom:6px">
      Decoder Self-Attn 层 1 · 真实 K/V Cache ∈ ℝ<sup>${T} × ${MX.shape(kv.K)[1]}</sup><br/>
      <span style="color:#34d399">每行 = 一个已处理 token 的 Key 向量 (推理时只需计算新 token 的 Q, 无需重跑历史)</span>
    </div>`;

    html += MX.render(kv.K, {
      label: `K Cache [${T} × ${MX.shape(kv.K)[1]}] (随生成逐行增长)`,
      rowLabels: processedTokens,
    });
    html += MX.render(kv.V, {
      label: `V Cache [${T} × ${MX.shape(kv.V)[1]}]`,
      rowLabels: processedTokens,
    });

    html += `<p class="hint" style="margin-top:8px">
      下一步新 token 的 Q 与这 ${T} 行 K 做点积即可, 无需重算历史行。
      复杂度从 O(t²·d) 降到 O(t·d)。
    </p>`;
    el.innerHTML = html;
  }

  return { init, refresh: reset };
})();
