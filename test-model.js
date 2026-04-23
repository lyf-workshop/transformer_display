// 快速自测: 验证模型前向 + 反向 + 训练能正常工作 (Node.js)
const MX = require("./js/matrix.js");
global.MX = MX;
const MODEL = require("./js/model.js");

const params = MODEL.initParams(7);
console.log("=== 样本 0: 我 爱 你 -> I love you ===");
const pair = MODEL.makeTrainPair(0);
console.log("src ids:", pair.srcIds);
console.log("tgt in:", pair.tgtIn);
console.log("tgt out:", pair.tgtOut);

console.log("\n=== 前向传播 ===");
const fwd = MODEL.forward(pair.srcIds, pair.tgtIn, params);
console.log("encOut shape:", MX.shape(fwd.encOut));
console.log("logits shape:", MX.shape(fwd.logits));

console.log("\n=== 反向 + 训练 200 步 ===");
const losses = [];
for (let i = 0; i < 200; i++) {
  const sampleIdx = i % MODEL.SAMPLES.length;
  const p = MODEL.makeTrainPair(sampleIdx);
  const f = MODEL.forward(p.srcIds, p.tgtIn, params);
  const b = MODEL.backward(f, p.tgtOut, params);
  MODEL.sgdStep(params, b.grads, 0.05);
  losses.push(b.loss);
  if (i < 5 || i >= 195 || i % 20 === 0) console.log(`step ${i+1} (sample ${sampleIdx}): loss = ${b.loss.toFixed(4)}`);
}
console.log("\nfinal 10 losses:", losses.slice(-10).map((x)=>x.toFixed(3)).join(" -> "));

console.log("\n=== 推理示例 ===");
const genPair = MODEL.makeTrainPair(0);
let genIds = [MODEL.CONFIG.bos_id];
for (let i = 0; i < 6; i++) {
  const r = MODEL.generateStep(genPair.srcIds, genIds, params);
  console.log(`step ${i}: top token = ${r.chosen.token} (p=${r.chosen.p.toFixed(3)})`);
  genIds.push(r.chosen.i);
  if (r.chosen.i === MODEL.CONFIG.eos_id) break;
}
console.log("generated:", genIds.map((id) => MODEL.VOCAB_TGT[id]).join(" "));
