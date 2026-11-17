import assert from "node:assert/strict";
import test from "node:test";

import { parsePartition, partitionLayers, ppEligibility, resolveStageCounts, stageRanges } from "../app/pp-plan.ts";
import { calculateWeightFromTensors, stageOfTensor } from "../app/weight-calc.ts";

test("自动分区遵循 vLLM 规则：余数从倒数第二级往前分", () => {
  assert.deepEqual(partitionLayers(61, 4), [15, 15, 16, 15]);
  assert.deepEqual(partitionLayers(61, 2), [31, 30]);
  assert.deepEqual(partitionLayers(40, 3), [13, 14, 13]);
  assert.deepEqual(partitionLayers(40, 1), [40]);
  assert.deepEqual(partitionLayers(40, 40), Array(40).fill(1));
  assert.deepEqual(stageRanges([15, 15, 16, 15]), [[0, 15], [15, 30], [30, 46], [46, 61]]);
});

test("PP 支持级别：Ascend 已验证 / 上游通用 / 不开放", () => {
  assert.equal(ppEligibility("DeepSeek-V4.1-Flash").level, "ascend");
  assert.equal(ppEligibility("GLM-5.2").level, "ascend");
  assert.equal(ppEligibility("MiniMax-M3.1-Flash").level, "ascend");
  assert.equal(ppEligibility("Qwen3-32B").level, "upstream");
  assert.equal(ppEligibility("DeepSeek-V3.1").level, "upstream");
  assert.equal(ppEligibility("Step-3.7-Flash").level, "upstream");
  assert.equal(ppEligibility("Hy3").level, "none");
  assert.equal(ppEligibility("Ling-3.1-flash").level, "none");
  assert.equal(ppEligibility("DeepSeek-Math-V2").level, "none");
});

test("张量归属：层内按区间，embedding 首级、LM head/draft 末级、其余复本", () => {
  const ranges = stageRanges(partitionLayers(4, 2));
  assert.equal(stageOfTensor("model.layers.0.self_attn.q_proj.weight", ranges), 0);
  assert.equal(stageOfTensor("model.layers.3.self_attn.q_proj.weight", ranges), 1);
  assert.equal(stageOfTensor("model.embed_tokens.weight", ranges), 0);
  assert.equal(stageOfTensor("lm_head.weight", ranges), 1);
  assert.equal(stageOfTensor("model.layers.0.mtp.1.fc.weight", ranges), 1);
  assert.equal(stageOfTensor("vision.blocks.0.attn.q.weight", ranges), "replicated");
});

test("逐 stage 权重：首级含 embedding、末级含 LM head、复本两级都计", () => {
  const tensors = [
    { name: "model.layers.0.mlp.gate_proj.weight", dtype: "BF16", shape: [2, 2], bytes: 100 },
    { name: "model.layers.1.mlp.gate_proj.weight", dtype: "BF16", shape: [2, 2], bytes: 200 },
    { name: "model.embed_tokens.weight", dtype: "BF16", shape: [2, 2], bytes: 1000 },
    { name: "lm_head.weight", dtype: "BF16", shape: [2, 2], bytes: 2000 },
    { name: "vision.blocks.0.attn.q.weight", dtype: "BF16", shape: [2, 2], bytes: 50 },
  ];
  const ranges = stageRanges(partitionLayers(2, 2));
  const stage0 = calculateWeightFromTensors(tensors, { tp: 1, ep: 1 }, { ranges, stage: 0 });
  const stage1 = calculateWeightFromTensors(tensors, { tp: 1, ep: 1 }, { ranges, stage: 1 });
  assert.equal(stage0.perDeviceWeight, 100 + 1000 + 50);
  assert.equal(stage1.perDeviceWeight, 200 + 2000 + 50);
  // 全 stage 之和 = 全部层 + 边缘模块 + 复本×stage 数
  assert.equal(stage0.perDeviceWeight + stage1.perDeviceWeight, 100 + 200 + 1000 + 2000 + 50 * 2);
});

test("手动层数分布：合法则采用，非法回退自动并标记", () => {
  assert.deepEqual(parsePartition("15,15,16,15"), [15, 15, 16, 15]);
  assert.deepEqual(parsePartition(" 15 15 16 15 "), [15, 15, 16, 15]);
  assert.equal(parsePartition(""), null);
  assert.equal(parsePartition("15,0,46"), null);

  const ok = resolveStageCounts(61, 4, "32,29,0,0".replace(",0,0", "").padEnd(0) || "32,29");
  // 个数不足 → 回退自动
  assert.equal(ok.source, "auto");
  assert.equal(ok.valid, false);

  const manual = resolveStageCounts(61, 2, "32,29");
  assert.deepEqual(manual, { counts: [32, 29], source: "manual", valid: true });

  const auto = resolveStageCounts(61, 4, "");
  assert.deepEqual(auto.counts, [15, 15, 16, 15]);
  assert.equal(auto.source, "auto");
});
