// ============================================================
// model.js — 小型 Transformer 实现 (可训练, 可推理)
//
// 为了教学展示, 采用极小规模:
//   d_model = 8, n_heads = 2, d_k = 4, d_ff = 16
//   encoder: 1 层, decoder: 1 层
//
// 包含完整前向传播 + 完整反向传播 + SGD 更新
// ============================================================

const MODEL = (function () {
  // -------- 配置 --------
  const CONFIG = {
    d_model: 8,
    n_heads: 2,
    d_k: 4, // = d_model / n_heads
    d_ff: 16,
    n_enc_layers: 1,
    n_dec_layers: 1,
    max_len: 8,
    pad_id: 0,
    bos_id: 1,
    eos_id: 2,
  };

  // 源 / 目标词表
  const VOCAB_SRC = ["<pad>", "我", "爱", "你", "好", "是", "学生"];
  const VOCAB_TGT = ["<pad>", "<bos>", "<eos>", "I", "love", "you", "hello", "am", "student"];

  const SRC_ID = Object.fromEntries(VOCAB_SRC.map((w, i) => [w, i]));
  const TGT_ID = Object.fromEntries(VOCAB_TGT.map((w, i) => [w, i]));

  // 训练样本 (src tokens -> tgt tokens, 不含 <bos>/<eos>)
  const SAMPLES = [
    { src: ["我", "爱", "你"],     tgt: ["I", "love", "you"] },
    { src: ["你", "好"],           tgt: ["hello"] },
    { src: ["我", "是", "学生"],   tgt: ["I", "am", "student"] },
  ];

  function tokensToIds(tokens, vocab) {
    return tokens.map((t) => vocab[t]);
  }

  function makeTrainPair(sampleIdx) {
    const s = SAMPLES[sampleIdx];
    const srcIds = tokensToIds(s.src, SRC_ID);
    // decoder input: <bos> + tgt   output(label): tgt + <eos>
    const tgtTokens = s.tgt;
    const tgtIn = [CONFIG.bos_id, ...tokensToIds(tgtTokens, TGT_ID)];
    const tgtOut = [...tokensToIds(tgtTokens, TGT_ID), CONFIG.eos_id];
    return { srcIds, tgtIn, tgtOut, srcTokens: s.src, tgtTokens };
  }

  // ============================================================
  // 参数初始化
  // ============================================================
  function initParams(seed = 7) {
    const rng = MX.mulberry32(seed);
    const randN = (r, c, s = 0.3) => MX.fromShape(r, c, () => MX.randn(rng) * s);
    const zeros1 = (n) => new Array(n).fill(0);
    const ones1 = (n) => new Array(n).fill(1);

    const { d_model, d_ff } = CONFIG;

    const mkEncLayer = () => ({
      Wq: randN(d_model, d_model), bq: zeros1(d_model),
      Wk: randN(d_model, d_model), bk: zeros1(d_model),
      Wv: randN(d_model, d_model), bv: zeros1(d_model),
      Wo: randN(d_model, d_model), bo: zeros1(d_model),
      W1: randN(d_model, d_ff),    b1: zeros1(d_ff),
      W2: randN(d_ff, d_model),    b2: zeros1(d_model),
      ln1_g: ones1(d_model), ln1_b: zeros1(d_model),
      ln2_g: ones1(d_model), ln2_b: zeros1(d_model),
    });
    const mkDecLayer = () => ({
      // self attention
      Wqs: randN(d_model, d_model), bqs: zeros1(d_model),
      Wks: randN(d_model, d_model), bks: zeros1(d_model),
      Wvs: randN(d_model, d_model), bvs: zeros1(d_model),
      Wos: randN(d_model, d_model), bos: zeros1(d_model),
      // cross attention
      Wqc: randN(d_model, d_model), bqc: zeros1(d_model),
      Wkc: randN(d_model, d_model), bkc: zeros1(d_model),
      Wvc: randN(d_model, d_model), bvc: zeros1(d_model),
      Woc: randN(d_model, d_model), boc: zeros1(d_model),
      // ffn
      W1: randN(d_model, d_ff),    b1: zeros1(d_ff),
      W2: randN(d_ff, d_model),    b2: zeros1(d_model),
      // layernorms
      ln1_g: ones1(d_model), ln1_b: zeros1(d_model),
      ln2_g: ones1(d_model), ln2_b: zeros1(d_model),
      ln3_g: ones1(d_model), ln3_b: zeros1(d_model),
    });

    return {
      embedSrc: randN(VOCAB_SRC.length, d_model, 0.4),
      embedTgt: randN(VOCAB_TGT.length, d_model, 0.4),
      enc: Array.from({ length: CONFIG.n_enc_layers }, () => mkEncLayer()),
      dec: Array.from({ length: CONFIG.n_dec_layers }, () => mkDecLayer()),
      Wout: randN(d_model, VOCAB_TGT.length, 0.3),
      bout: zeros1(VOCAB_TGT.length),
    };
  }

  // ============================================================
  // 辅助: linear + bias, with bias broadcast
  // ============================================================
  function linear(X, W, b) {
    // X: [T, in],  W: [in, out], b: [out]
    const Y = MX.matmul(X, W);
    const [T, out] = MX.shape(Y);
    for (let i = 0; i < T; i++) for (let j = 0; j < out; j++) Y[i][j] += b[j];
    return Y;
  }
  function linearBackward(X, W, dY) {
    // returns { dX, dW, db }
    const dX = MX.matmul(dY, MX.transpose(W));
    const dW = MX.matmul(MX.transpose(X), dY);
    const [T, out] = MX.shape(dY);
    const db = new Array(out).fill(0);
    for (let i = 0; i < T; i++) for (let j = 0; j < out; j++) db[j] += dY[i][j];
    return { dX, dW, db };
  }

  // ============================================================
  // LayerNorm (row-wise)   out = (x-mu)/sigma * gamma + beta
  // ============================================================
  function layerNormFwd(X, gamma, beta, eps = 1e-5) {
    const [T, D] = MX.shape(X);
    const out = MX.zeros(T, D);
    const cache = { X, gamma, beta, eps, mu: [], sigma: [], xhat: MX.zeros(T, D) };
    for (let i = 0; i < T; i++) {
      let mu = 0; for (let j = 0; j < D; j++) mu += X[i][j]; mu /= D;
      let v = 0;  for (let j = 0; j < D; j++) v += (X[i][j] - mu) ** 2; v /= D;
      const sigma = Math.sqrt(v + eps);
      cache.mu.push(mu); cache.sigma.push(sigma);
      for (let j = 0; j < D; j++) {
        const xh = (X[i][j] - mu) / sigma;
        cache.xhat[i][j] = xh;
        out[i][j] = xh * gamma[j] + beta[j];
      }
    }
    return { out, cache };
  }
  function layerNormBwd(dY, cache) {
    const { xhat, sigma, gamma } = cache;
    const [T, D] = MX.shape(dY);
    const dgamma = new Array(D).fill(0);
    const dbeta = new Array(D).fill(0);
    const dX = MX.zeros(T, D);
    for (let i = 0; i < T; i++) {
      // d xhat = dY * gamma
      const dxhat = new Array(D);
      let sum_dxhat = 0, sum_dxhat_xhat = 0;
      for (let j = 0; j < D; j++) {
        dbeta[j] += dY[i][j];
        dgamma[j] += dY[i][j] * xhat[i][j];
        dxhat[j] = dY[i][j] * gamma[j];
        sum_dxhat += dxhat[j];
        sum_dxhat_xhat += dxhat[j] * xhat[i][j];
      }
      const inv = 1 / sigma[i];
      for (let j = 0; j < D; j++) {
        dX[i][j] = (inv / D) * (D * dxhat[j] - sum_dxhat - xhat[i][j] * sum_dxhat_xhat);
      }
    }
    return { dX, dgamma, dbeta };
  }

  // ============================================================
  // Multi-Head Attention
  // ============================================================
  function splitH(X, H) {
    // X: [T, D] -> array of [T, d_k]
    return MX.splitHeads(X, H);
  }
  function concatH(heads) { return MX.concatHeads(heads); }

  // single-head scaled dot product with full cache
  function sdpaFwd(Q, K, V, mask) {
    const dk = MX.shape(Q)[1];
    const scores0 = MX.matmul(Q, MX.transpose(K));
    const scale = 1 / Math.sqrt(dk);
    const [nq, nk] = MX.shape(scores0);
    const scores = MX.zeros(nq, nk);
    for (let i = 0; i < nq; i++)
      for (let j = 0; j < nk; j++)
        scores[i][j] = scores0[i][j] * scale + (mask && mask[i][j] === 0 ? -1e9 : 0);
    const attn = MX.softmax(scores);
    const out = MX.matmul(attn, V);
    return { out, attn, scores, cache: { Q, K, V, attn, dk, mask } };
  }
  function sdpaBwd(dOut, cache) {
    const { Q, K, V, attn, dk } = cache;
    const [nq, dV] = MX.shape(V);
    // dV = attn^T @ dOut
    const dV_ = MX.matmul(MX.transpose(attn), dOut);
    // dAttn = dOut @ V^T
    const dAttn = MX.matmul(dOut, MX.transpose(V));
    // dScores = softmax_bwd(dAttn, attn)
    const [nq2, nk] = MX.shape(attn);
    const dScores = MX.zeros(nq2, nk);
    for (let i = 0; i < nq2; i++) {
      let s = 0; for (let j = 0; j < nk; j++) s += dAttn[i][j] * attn[i][j];
      for (let j = 0; j < nk; j++) dScores[i][j] = attn[i][j] * (dAttn[i][j] - s);
    }
    // scale
    const inv = 1 / Math.sqrt(dk);
    for (let i = 0; i < nq2; i++) for (let j = 0; j < nk; j++) dScores[i][j] *= inv;
    // dQ = dScores @ K,  dK = dScores^T @ Q
    const dQ = MX.matmul(dScores, K);
    const dK = MX.matmul(MX.transpose(dScores), Q);
    return { dQ, dK, dV: dV_ };
  }

  // multi-head attention forward
  // Xq source of Q (self: Xq=X, cross: Xq=decoder state)
  // Xkv source of K,V
  function mhaFwd(Xq, Xkv, W, mask, H) {
    // W: { Wq, bq, Wk, bk, Wv, bv, Wo, bo }
    const Q = linear(Xq, W.Wq, W.bq);
    const K = linear(Xkv, W.Wk, W.bk);
    const V = linear(Xkv, W.Wv, W.bv);
    const Qh = splitH(Q, H);
    const Kh = splitH(K, H);
    const Vh = splitH(V, H);
    const outs = [];
    const attns = [];
    const caches = [];
    for (let h = 0; h < H; h++) {
      const r = sdpaFwd(Qh[h], Kh[h], Vh[h], mask);
      outs.push(r.out); attns.push(r.attn); caches.push(r.cache);
    }
    const O = concatH(outs);
    const Y = linear(O, W.Wo, W.bo);
    return {
      out: Y, attns,
      cache: { Xq, Xkv, Q, K, V, Qh, Kh, Vh, O, sdpaCaches: caches, H },
    };
  }
  function mhaBwd(dY, cache, W) {
    const { Xq, Xkv, Q, K, V, O, sdpaCaches, H } = cache;
    // back through output linear
    const bo = linearBackward(O, W.Wo, dY);
    const dO = bo.dX;
    // split dO into heads
    const dOh = splitH(dO, H);
    // sum gradient dQ, dK, dV per head, then concat
    const dQhArr = [], dKhArr = [], dVhArr = [];
    for (let h = 0; h < H; h++) {
      const r = sdpaBwd(dOh[h], sdpaCaches[h]);
      dQhArr.push(r.dQ); dKhArr.push(r.dK); dVhArr.push(r.dV);
    }
    const dQ = concatH(dQhArr), dK = concatH(dKhArr), dV = concatH(dVhArr);
    // back through Q=X_q@Wq+bq, K=X_kv@Wk+bk, V=X_kv@Wv+bv
    const bq = linearBackward(Xq, W.Wq, dQ);
    const bk = linearBackward(Xkv, W.Wk, dK);
    const bv = linearBackward(Xkv, W.Wv, dV);
    const dXq = bq.dX;
    const dXkv = MX.add(bk.dX, bv.dX);
    return {
      dXq, dXkv,
      grads: {
        Wq: bq.dW, bq: bq.db,
        Wk: bk.dW, bk: bk.db,
        Wv: bv.dW, bv: bv.db,
        Wo: bo.dW, bo: bo.db,
      },
    };
  }

  // ============================================================
  // Embedding lookup
  // ============================================================
  function embedLookup(tokenIds, E) {
    const [V, D] = MX.shape(E);
    return tokenIds.map((id) => E[id].slice());
  }
  function embedBackward(tokenIds, dX, vocabSize, dModel) {
    const dE = MX.zeros(vocabSize, dModel);
    for (let i = 0; i < tokenIds.length; i++) {
      for (let j = 0; j < dModel; j++) dE[tokenIds[i]][j] += dX[i][j];
    }
    return dE;
  }

  // ============================================================
  // Encoder 前向
  // ============================================================
  function encodeFwd(srcIds, params) {
    const { d_model, n_heads } = CONFIG;
    const emb = embedLookup(srcIds, params.embedSrc); // [T, D]
    const PE = MX.positionalEncoding(srcIds.length, d_model);
    let x = MX.add(emb, PE);
    const layerCaches = [];
    for (const L of params.enc) {
      // self attention
      const mhaQKV = { Wq: L.Wq, bq: L.bq, Wk: L.Wk, bk: L.bk, Wv: L.Wv, bv: L.bv, Wo: L.Wo, bo: L.bo };
      const sa = mhaFwd(x, x, mhaQKV, null, n_heads);
      const resid1 = MX.add(x, sa.out);
      const ln1 = layerNormFwd(resid1, L.ln1_g, L.ln1_b);
      const x1 = ln1.out;
      // ffn
      const h = linear(x1, L.W1, L.b1);
      const hRelu = MX.relu(h);
      const ff = linear(hRelu, L.W2, L.b2);
      const resid2 = MX.add(x1, ff);
      const ln2 = layerNormFwd(resid2, L.ln2_g, L.ln2_b);
      x = ln2.out;
      layerCaches.push({ inp: x1, resid1, saCache: sa.cache, ln1Cache: ln1.cache,
                         h, hRelu, ffIn: x1, resid2, ln2Cache: ln2.cache,
                         saAttns: sa.attns });
    }
    return { out: x, cache: { srcIds, emb, PE, layerCaches } };
  }

  // ============================================================
  // Decoder 前向
  // ============================================================
  function decodeFwd(tgtIds, encOut, params) {
    const { d_model, n_heads } = CONFIG;
    const emb = embedLookup(tgtIds, params.embedTgt);
    const PE = MX.positionalEncoding(tgtIds.length, d_model);
    let x = MX.add(emb, PE);
    const causal = MX.causalMask(tgtIds.length);
    const layerCaches = [];
    for (const L of params.dec) {
      // masked self-attention
      const mhaSelf = { Wq: L.Wqs, bq: L.bqs, Wk: L.Wks, bk: L.bks, Wv: L.Wvs, bv: L.bvs, Wo: L.Wos, bo: L.bos };
      const sa = mhaFwd(x, x, mhaSelf, causal, n_heads);
      const resid1 = MX.add(x, sa.out);
      const ln1 = layerNormFwd(resid1, L.ln1_g, L.ln1_b);
      const x1 = ln1.out;
      // cross-attention (Q from dec, K/V from enc)
      const mhaCross = { Wq: L.Wqc, bq: L.bqc, Wk: L.Wkc, bk: L.bkc, Wv: L.Wvc, bv: L.bvc, Wo: L.Woc, bo: L.boc };
      const ca = mhaFwd(x1, encOut, mhaCross, null, n_heads);
      const resid2 = MX.add(x1, ca.out);
      const ln2 = layerNormFwd(resid2, L.ln2_g, L.ln2_b);
      const x2 = ln2.out;
      // ffn
      const h = linear(x2, L.W1, L.b1);
      const hRelu = MX.relu(h);
      const ff = linear(hRelu, L.W2, L.b2);
      const resid3 = MX.add(x2, ff);
      const ln3 = layerNormFwd(resid3, L.ln3_g, L.ln3_b);
      x = ln3.out;
      layerCaches.push({
        inp: x1, resid1, ln1Cache: ln1.cache, saCache: sa.cache, saAttns: sa.attns,
        ca_in: x1, resid2, ln2Cache: ln2.cache, caCache: ca.cache, caAttns: ca.attns,
        ff_in: x2, h, hRelu, resid3, ln3Cache: ln3.cache,
      });
    }
    // logits
    const logits = linear(x, params.Wout, params.bout);
    return { logits, out: x, cache: { tgtIds, emb, PE, layerCaches, finalX: x } };
  }

  // ============================================================
  // 完整前向
  // ============================================================
  function forward(srcIds, tgtIds, params) {
    const enc = encodeFwd(srcIds, params);
    const dec = decodeFwd(tgtIds, enc.out, params);
    return { logits: dec.logits, encOut: enc.out, decOut: dec.out,
             encCache: enc.cache, decCache: dec.cache };
  }

  // ============================================================
  // 损失: cross-entropy
  // ============================================================
  function crossEntropyLoss(logits, targetIds) {
    const [T, V] = MX.shape(logits);
    let total = 0;
    const probs = MX.softmax(logits);
    const dLogits = MX.zeros(T, V);
    for (let i = 0; i < T; i++) {
      const t = targetIds[i];
      total += -Math.log(Math.max(1e-9, probs[i][t]));
      for (let j = 0; j < V; j++) dLogits[i][j] = probs[i][j];
      dLogits[i][t] -= 1;
    }
    const loss = total / T;
    // normalize gradient by T
    for (let i = 0; i < T; i++) for (let j = 0; j < V; j++) dLogits[i][j] /= T;
    return { loss, probs, dLogits };
  }

  // ============================================================
  // 完整反向
  // ============================================================
  function backward(fwdResult, targetIds, params) {
    const { logits, encCache, decCache, encOut } = fwdResult;
    const { loss, probs, dLogits } = crossEntropyLoss(logits, targetIds);

    const grads = {
      embedSrc: MX.zeros(VOCAB_SRC.length, CONFIG.d_model),
      embedTgt: MX.zeros(VOCAB_TGT.length, CONFIG.d_model),
      enc: params.enc.map(() => ({})),
      dec: params.dec.map(() => ({})),
      Wout: null, bout: null,
    };

    // logits = finalX @ Wout + bout
    const boutBack = linearBackward(decCache.finalX, params.Wout, dLogits);
    grads.Wout = boutBack.dW;
    grads.bout = boutBack.db;
    let dX = boutBack.dX; // [T_tgt, d_model]

    // 反向穿过 decoder 层
    let dEncOut = MX.zeros(MX.shape(encOut)[0], CONFIG.d_model);
    for (let li = params.dec.length - 1; li >= 0; li--) {
      const L = params.dec[li];
      const c = decCache.layerCaches[li];

      // --- ln3 + resid3 ---
      const ln3b = layerNormBwd(dX, c.ln3Cache);
      const dResid3 = ln3b.dX;            // [T, D]
      const dX2 = MX.clone(dResid3);      // 对 x2 的直接贡献 (skip connection)
      const dFF = MX.clone(dResid3);      // 对 ff 的贡献

      // --- ffn 反向: ff = relu(x2@W1+b1)@W2+b2 ---
      const b2back = linearBackward(c.hRelu, L.W2, dFF);
      const dW2 = b2back.dW, db2 = b2back.db;
      const dHrelu = b2back.dX;
      const dH = MX.elemMul(dHrelu, MX.reluGrad(c.h));
      const b1back = linearBackward(c.ff_in, L.W1, dH);
      const dW1 = b1back.dW, db1 = b1back.db;
      // 累加到 x2
      const dx2 = MX.add(dX2, b1back.dX);

      // --- ln2 + resid2 ---
      const ln2b = layerNormBwd(dx2, c.ln2Cache);
      const dResid2 = ln2b.dX;
      const dX1a = MX.clone(dResid2);
      const dCA = MX.clone(dResid2);

      // --- cross-attn 反向 ---
      const crossW = { Wq: L.Wqc, bq: L.bqc, Wk: L.Wkc, bk: L.bkc, Wv: L.Wvc, bv: L.bvc, Wo: L.Woc, bo: L.boc };
      const cab = mhaBwd(dCA, c.caCache, crossW);
      // cab.dXq -> x1, cab.dXkv -> encOut
      const dx1_from_ca = cab.dXq;
      dEncOut = MX.add(dEncOut, cab.dXkv);

      const dx1 = MX.add(dX1a, dx1_from_ca);

      // --- ln1 + resid1 ---
      const ln1b = layerNormBwd(dx1, c.ln1Cache);
      const dResid1 = ln1b.dX;
      const dXin = MX.clone(dResid1);
      const dSA = MX.clone(dResid1);

      // --- masked self-attn 反向 ---
      const selfW = { Wq: L.Wqs, bq: L.bqs, Wk: L.Wks, bk: L.bks, Wv: L.Wvs, bv: L.bvs, Wo: L.Wos, bo: L.bos };
      const sab = mhaBwd(dSA, c.saCache, selfW);
      // self: Xq=Xkv=x, so combine
      const dx_from_sa = MX.add(sab.dXq, sab.dXkv);
      dX = MX.add(dXin, dx_from_sa);

      // 保存梯度
      grads.dec[li] = {
        Wqs: sab.grads.Wq, bqs: sab.grads.bq,
        Wks: sab.grads.Wk, bks: sab.grads.bk,
        Wvs: sab.grads.Wv, bvs: sab.grads.bv,
        Wos: sab.grads.Wo, bos: sab.grads.bo,
        Wqc: cab.grads.Wq, bqc: cab.grads.bq,
        Wkc: cab.grads.Wk, bkc: cab.grads.bk,
        Wvc: cab.grads.Wv, bvc: cab.grads.bv,
        Woc: cab.grads.Wo, boc: cab.grads.bo,
        W1: dW1, b1: db1,
        W2: dW2, b2: db2,
        ln1_g: ln1b.dgamma, ln1_b: ln1b.dbeta,
        ln2_g: ln2b.dgamma, ln2_b: ln2b.dbeta,
        ln3_g: ln3b.dgamma, ln3_b: ln3b.dbeta,
      };
    }

    // dX 此时是对 decoder 输入 (emb + PE) 的梯度
    grads.embedTgt = embedBackward(decCache.tgtIds, dX, VOCAB_TGT.length, CONFIG.d_model);

    // 反向穿过 encoder 层 (起点 = dEncOut)
    let dE = dEncOut;
    for (let li = params.enc.length - 1; li >= 0; li--) {
      const L = params.enc[li];
      const c = encCache.layerCaches[li];
      // ln2 + resid2
      const ln2b = layerNormBwd(dE, c.ln2Cache);
      const dR2 = ln2b.dX;
      const dX1skip = MX.clone(dR2);
      const dFF = MX.clone(dR2);
      const b2back = linearBackward(c.hRelu, L.W2, dFF);
      const dHrelu = b2back.dX;
      const dH = MX.elemMul(dHrelu, MX.reluGrad(c.h));
      const b1back = linearBackward(c.ffIn, L.W1, dH);
      const dx1 = MX.add(dX1skip, b1back.dX);
      // ln1 + resid1
      const ln1b = layerNormBwd(dx1, c.ln1Cache);
      const dR1 = ln1b.dX;
      const dXin = MX.clone(dR1);
      const dSA = MX.clone(dR1);
      const selfW = { Wq: L.Wq, bq: L.bq, Wk: L.Wk, bk: L.bk, Wv: L.Wv, bv: L.bv, Wo: L.Wo, bo: L.bo };
      const sab = mhaBwd(dSA, c.saCache, selfW);
      const dx_from_sa = MX.add(sab.dXq, sab.dXkv);
      dE = MX.add(dXin, dx_from_sa);
      grads.enc[li] = {
        Wq: sab.grads.Wq, bq: sab.grads.bq,
        Wk: sab.grads.Wk, bk: sab.grads.bk,
        Wv: sab.grads.Wv, bv: sab.grads.bv,
        Wo: sab.grads.Wo, bo: sab.grads.bo,
        W1: b1back.dW, b1: b1back.db,
        W2: b2back.dW, b2: b2back.db,
        ln1_g: ln1b.dgamma, ln1_b: ln1b.dbeta,
        ln2_g: ln2b.dgamma, ln2_b: ln2b.dbeta,
      };
    }
    grads.embedSrc = embedBackward(encCache.srcIds, dE, VOCAB_SRC.length, CONFIG.d_model);

    return { loss, probs, grads };
  }

  // ============================================================
  // SGD 更新
  // ============================================================
  function sgdStep(params, grads, lr) {
    const update = (P, G) => {
      const [r, c] = MX.shape(P);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) P[i][j] -= lr * G[i][j];
    };
    const updateVec = (P, G) => { for (let i = 0; i < P.length; i++) P[i] -= lr * G[i]; };

    update(params.embedSrc, grads.embedSrc);
    update(params.embedTgt, grads.embedTgt);
    for (let li = 0; li < params.enc.length; li++) {
      const L = params.enc[li], G = grads.enc[li];
      for (const k of ["Wq","Wk","Wv","Wo","W1","W2"]) update(L[k], G[k]);
      for (const k of ["bq","bk","bv","bo","b1","b2","ln1_g","ln1_b","ln2_g","ln2_b"]) updateVec(L[k], G[k]);
    }
    for (let li = 0; li < params.dec.length; li++) {
      const L = params.dec[li], G = grads.dec[li];
      for (const k of ["Wqs","Wks","Wvs","Wos","Wqc","Wkc","Wvc","Woc","W1","W2"]) update(L[k], G[k]);
      for (const k of ["bqs","bks","bvs","bos","bqc","bkc","bvc","boc","b1","b2",
                       "ln1_g","ln1_b","ln2_g","ln2_b","ln3_g","ln3_b"]) updateVec(L[k], G[k]);
    }
    update(params.Wout, grads.Wout);
    updateVec(params.bout, grads.bout);
  }

  // ============================================================
  // 推理: 贪心/Top-k 自回归生成 (旧版, 保留兼容)
  // ============================================================
  function generateStep(srcIds, tgtIdsSoFar, params) {
    const enc = encodeFwd(srcIds, params);
    const dec = decodeFwd(tgtIdsSoFar, enc.out, params);
    const lastLogits = dec.logits[dec.logits.length - 1];
    const probs = MX.softmaxVec(lastLogits);
    const topk = probs
      .map((p, i) => ({ p, i, token: VOCAB_TGT[i] }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 5);
    const chosen = topk[0];
    return { chosen, topk, logits: lastLogits, probs, encOut: enc.out, decOut: dec.out, decCache: dec.cache };
  }

  // ============================================================
  // 真实 KV Cache 推理
  // ============================================================

  // 增量自注意力: 仅对新 token 计算 Q, 扩展 K/V cache 后做注意力
  function mhaFwdIncremental(xNewMat, cachedK, cachedV, W, H) {
    const Q_new = linear(xNewMat, W.Wq, W.bq);   // [1, d_model]
    const K_new = linear(xNewMat, W.Wk, W.bk);   // [1, d_model]
    const V_new = linear(xNewMat, W.Wv, W.bv);   // [1, d_model]

    const K_full = cachedK ? [...cachedK, K_new[0]] : [K_new[0]];
    const V_full = cachedV ? [...cachedV, V_new[0]] : [V_new[0]];

    const Qh = splitH(Q_new, H);
    const Kh = splitH(K_full, H);
    const Vh = splitH(V_full, H);

    const outs = [], attns = [];
    for (let h = 0; h < H; h++) {
      const r = sdpaFwd(Qh[h], Kh[h], Vh[h], null);
      outs.push(r.out); attns.push(r.attn);
    }
    const O = concatH(outs);
    const Y = linear(O, W.Wo, W.bo);
    return { out: Y, newK: K_full, newV: V_full, attns };
  }

  // 增量 Cross-Attention: Q 来自当前 decoder 状态, K/V 来自预计算的 encoder
  function mhaFwdIncrementalCross(xNewMat, K_cross, V_cross, W, H) {
    const Q_new = linear(xNewMat, W.Wq, W.bq);   // [1, d_model]
    const Qh = splitH(Q_new, H);
    const Kh = splitH(K_cross, H);
    const Vh = splitH(V_cross, H);

    const outs = [], attns = [];
    for (let h = 0; h < H; h++) {
      const r = sdpaFwd(Qh[h], Kh[h], Vh[h], null);
      outs.push(r.out); attns.push(r.attn);
    }
    const O = concatH(outs);
    const Y = linear(O, W.Wo, W.bo);
    return { out: Y, attns };
  }

  // 初始化推理 KV Cache: 只计算一次 encoder, 预计算 cross-attn 的 K/V
  function initInferenceKVCache(srcIds, params) {
    const enc = encodeFwd(srcIds, params);
    const crossKV = params.dec.map((L) => ({
      K: linear(enc.out, L.Wkc, L.bkc),
      V: linear(enc.out, L.Wvc, L.bvc),
    }));
    return {
      encOut: enc.out,
      crossKV,
      selfKV: params.dec.map(() => ({ K: null, V: null })),
    };
  }

  // 用 KV Cache 做单步增量生成
  // newTokenId: 当前输入 token (上一步生成的, 或 <bos>)
  // temperature: 温度缩放 (1.0 = 不变, <1 更确定, >1 更随机)
  // useSampling: true=按分布采样, false=贪心
  function generateStepWithKVCache(newTokenId, kvCache, params, temperature = 1.0, useSampling = false) {
    const { d_model, n_heads: H } = CONFIG;
    const pos = kvCache.selfKV[0].K ? kvCache.selfKV[0].K.length : 0;

    const embVec = params.embedTgt[newTokenId].slice();
    const peRow = MX.positionalEncoding(pos + 1, d_model)[pos];
    let xNew = [embVec.map((v, i) => v + peRow[i])];   // [1, d_model]

    const newSelfKV = [];
    const layerAttns = [];

    for (let li = 0; li < params.dec.length; li++) {
      const L = params.dec[li];
      const { K: prevK, V: prevV } = kvCache.selfKV[li];

      const selfW = { Wq: L.Wqs, bq: L.bqs, Wk: L.Wks, bk: L.bks,
                      Wv: L.Wvs, bv: L.bvs, Wo: L.Wos, bo: L.bos };
      const sa = mhaFwdIncremental(xNew, prevK, prevV, selfW, H);
      const resid1 = MX.add(xNew, sa.out);
      const ln1 = layerNormFwd(resid1, L.ln1_g, L.ln1_b);
      const x1 = ln1.out;

      const crossW = { Wq: L.Wqc, bq: L.bqc, Wo: L.Woc, bo: L.boc };
      const { K: K_cross, V: V_cross } = kvCache.crossKV[li];
      const ca = mhaFwdIncrementalCross(x1, K_cross, V_cross, crossW, H);
      const resid2 = MX.add(x1, ca.out);
      const ln2 = layerNormFwd(resid2, L.ln2_g, L.ln2_b);
      const x2 = ln2.out;

      const h = linear(x2, L.W1, L.b1);
      const hRelu = MX.relu(h);
      const ff = linear(hRelu, L.W2, L.b2);
      const resid3 = MX.add(x2, ff);
      const ln3 = layerNormFwd(resid3, L.ln3_g, L.ln3_b);
      xNew = ln3.out;

      newSelfKV.push({ K: sa.newK, V: sa.newV });
      layerAttns.push({ selfAttns: sa.attns, crossAttns: ca.attns });
    }

    const lastLogits = linear(xNew, params.Wout, params.bout)[0];

    // 温度缩放
    const scaledLogits = temperature === 1.0
      ? lastLogits
      : lastLogits.map((v) => v / temperature);

    const probs = MX.softmaxVec(scaledLogits);
    const ranked = probs
      .map((p, i) => ({ p, i, token: VOCAB_TGT[i] }))
      .sort((a, b) => b.p - a.p);
    const topk = ranked.slice(0, 5);

    let chosen;
    if (useSampling) {
      let r = Math.random(), cumsum = 0;
      chosen = ranked[ranked.length - 1];
      for (const item of ranked) { cumsum += item.p; if (r <= cumsum) { chosen = item; break; } }
    } else {
      chosen = topk[0];
    }

    const newKVCache = {
      encOut: kvCache.encOut,
      crossKV: kvCache.crossKV,
      selfKV: newSelfKV,
    };

    return { chosen, topk, probs, lastLogits, newKVCache, layerAttns };
  }

  // ============================================================
  // 导出
  // ============================================================
  return {
    CONFIG, VOCAB_SRC, VOCAB_TGT, SRC_ID, TGT_ID, SAMPLES,
    tokensToIds, makeTrainPair,
    initParams,
    linear, linearBackward,
    layerNormFwd, layerNormBwd,
    mhaFwd, mhaBwd, sdpaFwd,
    embedLookup,
    encodeFwd, decodeFwd, forward,
    crossEntropyLoss, backward, sgdStep,
    generateStep,
    initInferenceKVCache, generateStepWithKVCache,
  };
})();

if (typeof module !== "undefined") module.exports = MODEL;
