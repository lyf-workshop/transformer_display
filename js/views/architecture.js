// ============================================================
// architecture.js — 架构总览视图
// ============================================================
const ArchitectureView = (function () {
  function renderConfig() {
    const cfg = MODEL.CONFIG;
    const items = [
      { k: "d_model",     v: cfg.d_model,          desc: "模型隐藏维度" },
      { k: "n_heads",     v: cfg.n_heads,          desc: "注意力头数" },
      { k: "d_k (d_v)",   v: cfg.d_k,              desc: "每个头的维度" },
      { k: "d_ff",        v: cfg.d_ff,             desc: "FFN 中间维度" },
      { k: "n_enc / n_dec", v: `${cfg.n_enc_layers} / ${cfg.n_dec_layers}`, desc: "Encoder / Decoder 层数" },
      { k: "src vocab",   v: MODEL.VOCAB_SRC.length,  desc: "源词表大小" },
      { k: "tgt vocab",   v: MODEL.VOCAB_TGT.length,  desc: "目标词表大小" },
      { k: "max_len",     v: cfg.max_len,          desc: "最大序列长度" },
    ];
    const html = items.map(
      (it) => `<div class="config-item">
        <div class="config-key">${it.k}</div>
        <div class="config-val">${it.v}</div>
        <div class="config-key" style="margin-top:4px;text-transform:none">${it.desc}</div>
      </div>`
    ).join("");
    document.getElementById("config-display").innerHTML = html;
  }

  function bindJumps(onJump) {
    document.querySelectorAll(".arch-block.clickable").forEach((el) => {
      el.addEventListener("click", () => {
        const target = el.dataset.jump;
        onJump(target);
      });
    });
  }

  function init(onJump) {
    renderConfig();
    bindJumps(onJump);
  }

  return { init };
})();
