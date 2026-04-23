// ============================================================
// matrix.js — 矩阵运算 + 可视化工具
// 所有矩阵都用二维数组 number[rows][cols] 表示
// ============================================================

const MX = (function () {
  // ---------- 构造 ----------
  function zeros(r, c) {
    return Array.from({ length: r }, () => new Array(c).fill(0));
  }
  function fromShape(r, c, fn) {
    const m = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m[i][j] = fn(i, j);
    return m;
  }
  // 伪随机 (可复现)
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(rng) {
    // Box-Muller
    const u1 = Math.max(1e-9, rng());
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  function randomMatrix(r, c, seed = 42, scale = 0.3) {
    const rng = mulberry32(seed);
    return fromShape(r, c, () => randn(rng) * scale);
  }

  // ---------- 形状 ----------
  function shape(m) { return [m.length, (m[0] && m[0].length) || 0]; }
  function clone(m) { return m.map((r) => r.slice()); }

  // ---------- 基础运算 ----------
  function matmul(A, B) {
    const [ra, ca] = shape(A);
    const [rb, cb] = shape(B);
    if (ca !== rb) throw new Error(`matmul shape mismatch: [${ra},${ca}] x [${rb},${cb}]`);
    const C = zeros(ra, cb);
    for (let i = 0; i < ra; i++) {
      for (let k = 0; k < ca; k++) {
        const aik = A[i][k];
        for (let j = 0; j < cb; j++) C[i][j] += aik * B[k][j];
      }
    }
    return C;
  }
  function transpose(A) {
    const [r, c] = shape(A);
    const T = zeros(c, r);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
    return T;
  }
  function add(A, B) {
    const [r, c] = shape(A);
    return fromShape(r, c, (i, j) => A[i][j] + B[i][j]);
  }
  function sub(A, B) {
    const [r, c] = shape(A);
    return fromShape(r, c, (i, j) => A[i][j] - B[i][j]);
  }
  function scale(A, s) {
    const [r, c] = shape(A);
    return fromShape(r, c, (i, j) => A[i][j] * s);
  }
  function elemMul(A, B) {
    const [r, c] = shape(A);
    return fromShape(r, c, (i, j) => A[i][j] * B[i][j]);
  }

  // ---------- 激活 ----------
  function relu(A) {
    const [r, c] = shape(A);
    return fromShape(r, c, (i, j) => Math.max(0, A[i][j]));
  }
  function reluGrad(A) {
    const [r, c] = shape(A);
    return fromShape(r, c, (i, j) => (A[i][j] > 0 ? 1 : 0));
  }

  // row-wise softmax (matrix)
  function softmax(A) {
    const [r, c] = shape(A);
    const out = zeros(r, c);
    for (let i = 0; i < r; i++) {
      let mx = -Infinity;
      for (let j = 0; j < c; j++) if (A[i][j] > mx) mx = A[i][j];
      let sum = 0;
      for (let j = 0; j < c; j++) { out[i][j] = Math.exp(A[i][j] - mx); sum += out[i][j]; }
      for (let j = 0; j < c; j++) out[i][j] /= sum;
    }
    return out;
  }
  function softmaxVec(v) {
    let mx = -Infinity;
    for (const x of v) if (x > mx) mx = x;
    const out = new Array(v.length);
    let sum = 0;
    for (let i = 0; i < v.length; i++) { out[i] = Math.exp(v[i] - mx); sum += out[i]; }
    for (let i = 0; i < v.length; i++) out[i] /= sum;
    return out;
  }

  // ---------- LayerNorm ----------
  // 对每一行做 (x - mean) / sqrt(var + eps) * gamma + beta
  function layerNorm(X, gamma, beta, eps = 1e-5) {
    const [r, c] = shape(X);
    const out = zeros(r, c);
    for (let i = 0; i < r; i++) {
      let mean = 0; for (let j = 0; j < c; j++) mean += X[i][j]; mean /= c;
      let varv = 0; for (let j = 0; j < c; j++) varv += (X[i][j] - mean) ** 2; varv /= c;
      const denom = Math.sqrt(varv + eps);
      for (let j = 0; j < c; j++) out[i][j] = ((X[i][j] - mean) / denom) * gamma[j] + beta[j];
    }
    return out;
  }

  // ---------- 位置编码 ----------
  function positionalEncoding(maxLen, dModel) {
    const pe = zeros(maxLen, dModel);
    for (let pos = 0; pos < maxLen; pos++) {
      for (let i = 0; i < dModel; i++) {
        const exponent = (2 * Math.floor(i / 2)) / dModel;
        const angle = pos / Math.pow(10000, exponent);
        pe[pos][i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      }
    }
    return pe;
  }

  // ---------- Attention ----------
  // Scaled dot-product attention
  // Q: [n_q, d_k], K: [n_k, d_k], V: [n_k, d_v], mask: [n_q, n_k] (0/1, 1=可见)
  function scaledDotProductAttention(Q, K, V, mask = null) {
    const [nq, dk] = shape(Q);
    const scores = scale(matmul(Q, transpose(K)), 1 / Math.sqrt(dk)); // [nq, nk]
    const [, nk] = shape(scores);
    if (mask) {
      for (let i = 0; i < nq; i++)
        for (let j = 0; j < nk; j++)
          if (mask[i][j] === 0) scores[i][j] = -1e9;
    }
    const attn = softmax(scores);
    const output = matmul(attn, V);
    return { scores, attn, output };
  }
  function causalMask(n) {
    const m = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) m[i][j] = 1;
    return m;
  }

  // ---------- 切分多头 ----------
  // 把 [seq, d_model] 切分成 numHeads 个 [seq, d_k]
  function splitHeads(X, numHeads) {
    const [seq, dm] = shape(X);
    const dk = dm / numHeads;
    const heads = [];
    for (let h = 0; h < numHeads; h++) {
      const head = zeros(seq, dk);
      for (let i = 0; i < seq; i++)
        for (let j = 0; j < dk; j++) head[i][j] = X[i][h * dk + j];
      heads.push(head);
    }
    return heads;
  }
  function concatHeads(heads) {
    const [seq, dk] = shape(heads[0]);
    const H = heads.length;
    const out = zeros(seq, dk * H);
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < seq; i++)
        for (let j = 0; j < dk; j++) out[i][h * dk + j] = heads[h][i][j];
    }
    return out;
  }

  // ---------- 统计 ----------
  function min(A) { let x = Infinity; for (const r of A) for (const v of r) if (v < x) x = v; return x; }
  function max(A) { let x = -Infinity; for (const r of A) for (const v of r) if (v > x) x = v; return x; }
  function abs(A) { const [r, c] = shape(A); return fromShape(r, c, (i, j) => Math.abs(A[i][j])); }

  // ========================================================
  // ============== 可视化 =====================================
  // ========================================================

  // 把值映射到颜色 (对称 diverging colormap)
  function valueToColor(v, absMax) {
    if (!absMax) absMax = 1;
    const t = Math.max(-1, Math.min(1, v / absMax));
    // -1 -> 蓝, 0 -> 灰, 1 -> 红
    const rC = Math.round(t > 0 ? 239 * t + 30 * (1 - t) : 30);
    const gC = Math.round(30 * (1 - Math.abs(t)) + 30 * Math.abs(t));
    const bC = Math.round(t < 0 ? 180 * -t + 30 * (1 + t) : 30);
    return `rgb(${Math.min(255, Math.max(20, rC))}, ${gC}, ${Math.min(255, Math.max(20, bC))})`;
  }

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function fmt(v, digits = 2) {
    if (!isFinite(v)) return v > 0 ? "∞" : "-∞";
    if (Math.abs(v) < 0.005 && v !== 0) return v.toExponential(1);
    return v.toFixed(digits);
  }

  // 渲染矩阵为 HTML (带颜色热力)
  // opts: { label, rowLabels, colLabels, highlight: [[r,c]...], digits, maxCols }
  function render(M, opts = {}) {
    if (!M || !M.length) return `<div class="matrix-wrap">(empty)</div>`;
    const [r, c] = shape(M);
    const absMax = Math.max(Math.abs(min(M)), Math.abs(max(M)), 0.01);
    const digits = (opts.digits != null) ? opts.digits : 2;
    const rowLbl = opts.rowLabels || [];
    const colLbl = opts.colLabels || [];
    const highlight = new Set((opts.highlight || []).map(([i, j]) => `${i},${j}`));

    const showRowLbl = rowLbl.length > 0;
    const showColLbl = colLbl.length > 0;

    // 限制列数显示
    const maxCols = opts.maxCols || 16;
    const truncC = c > maxCols;
    const cShow = truncC ? maxCols : c;

    const cols = (showRowLbl ? 1 : 0) + cShow + (truncC ? 1 : 0);

    let html = `<div class="matrix-wrap">`;
    if (opts.label) {
      html += `<div class="matrix-label"><span>${escapeHtml(opts.label)}</span><span class="shape">[${r} × ${c}]</span></div>`;
    }
    html += `<div class="matrix-grid" style="grid-template-columns: repeat(${cols}, auto);">`;

    if (showColLbl) {
      if (showRowLbl) html += `<div class="matrix-cell row-label"></div>`;
      for (let j = 0; j < cShow; j++) {
        html += `<div class="matrix-cell col-label">${escapeHtml(colLbl[j] != null ? colLbl[j] : j)}</div>`;
      }
      if (truncC) html += `<div class="matrix-cell col-label">…</div>`;
    }

    for (let i = 0; i < r; i++) {
      if (showRowLbl) html += `<div class="matrix-cell row-label">${escapeHtml(rowLbl[i] != null ? rowLbl[i] : i)}</div>`;
      for (let j = 0; j < cShow; j++) {
        const v = M[i][j];
        const color = valueToColor(v, absMax);
        const hl = highlight.has(`${i},${j}`) ? " highlight" : "";
        html += `<div class="matrix-cell${hl}" style="background:${color};" title="[${i},${j}] = ${v.toFixed(4)}">${fmt(v, digits)}</div>`;
      }
      if (truncC) html += `<div class="matrix-cell" style="background:transparent;color:#6f7cac">…</div>`;
    }
    html += `</div></div>`;
    return html;
  }

  // 渲染一个算式：  A [op] B = C
  function renderOp(A, op, B, C, opts = {}) {
    const la = opts.labelA || "A";
    const lb = opts.labelB || "B";
    const lc = opts.labelC || "C";
    const ra = opts.rowLabelsA, rb = opts.rowLabelsB, rc = opts.rowLabelsC;
    const ca = opts.colLabelsA, cb = opts.colLabelsB, cc = opts.colLabelsC;
    let html = `<div class="matrix-row-flex">`;
    html += render(A, { label: la, rowLabels: ra, colLabels: ca, digits: opts.digits });
    html += `<div class="op-arrow"><div class="op">${op}</div></div>`;
    html += render(B, { label: lb, rowLabels: rb, colLabels: cb, digits: opts.digits });
    html += `<div class="op-arrow"><div class="op">=</div><div class="arr">→</div></div>`;
    html += render(C, { label: lc, rowLabels: rc, colLabels: cc, digits: opts.digits });
    html += `</div>`;
    return html;
  }

  return {
    zeros, fromShape, randomMatrix, mulberry32, randn,
    shape, clone,
    matmul, transpose, add, sub, scale, elemMul,
    relu, reluGrad, softmax, softmaxVec,
    layerNorm, positionalEncoding,
    scaledDotProductAttention, causalMask,
    splitHeads, concatHeads,
    min, max, abs,
    valueToColor, fmt, render, renderOp, escapeHtml,
  };
})();

if (typeof module !== "undefined") module.exports = MX;
