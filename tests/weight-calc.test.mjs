import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyTensor,
  calculateWeightFromTensors,
} from "../app/weight-calc.ts";

test("classifyTensor 覆盖各模块命名", () => {
  assert.equal(classifyTensor("model.embed_tokens.weight"), "embedding");
  assert.equal(classifyTensor("lm_head.weight"), "lm_head");
  assert.equal(classifyTensor("model.layers.0.self_attn.q_proj.weight"), "attention");
  assert.equal(classifyTensor("model.layers.0.self_attn.o_proj.weight"), "attention");
  assert.equal(classifyTensor("model.layers.0.mlp.gate_proj.weight"), "dense_mlp");
  assert.equal(classifyTensor("model.layers.0.mlp.up_proj.weight_scale"), "dense_mlp");
  assert.equal(classifyTensor("model.layers.0.mlp.experts.0.w1.weight"), "routed_expert");
  assert.equal(classifyTensor("block_sparse_moe.gate.weight"), "router");
  assert.equal(classifyTensor("model.norm.weight"), "norm");
  assert.equal(classifyTensor("model.layers.0.input_layernorm.weight"), "norm");
  assert.equal(classifyTensor("model.visual.blocks.0.attn.proj.weight"), "vision");
  assert.equal(classifyTensor("model.layers.0.self_attn.kv_a_proj_with_mqa.weight"), "attention");
  assert.equal(classifyTensor("model.layers.0.mlp.shared_experts.gate_proj.weight"), "shared_expert");
});

test("classifyTensor 覆盖 Step 系列（moe.*_proj 路由专家 / share_expert / MTP 头）", () => {
  // 路由专家矩阵合并在 moe.*_proj 下，命名没有 expert 字样，必须按 .moe. 判定。
  assert.equal(classifyTensor("model.layers.3.moe.gate_proj.weight"), "routed_expert");
  assert.equal(classifyTensor("model.layers.3.moe.up_proj.weight"), "routed_expert");
  assert.equal(classifyTensor("model.layers.3.moe.down_proj.weight"), "routed_expert");
  assert.equal(classifyTensor("model.layers.3.moe.gate_proj.weight_scale"), "routed_expert");
  assert.equal(classifyTensor("model.layers.3.moe.gate_proj.weight_scale_inv"), "routed_expert");
  // 路由 gate 与 router bias 仍归 router，不能被上面的规则吞掉。
  assert.equal(classifyTensor("model.layers.3.moe.gate.weight"), "router");
  assert.equal(classifyTensor("model.layers.3.moe.router_bias"), "router");
  // 共享专家写作 share_expert（无 d）。
  assert.equal(classifyTensor("model.layers.3.share_expert.gate_proj.weight"), "shared_expert");
  assert.equal(classifyTensor("model.layers.3.share_expert.down_proj.weight"), "shared_expert");
  // MTP 投机层：输出头按 LM Head 切分，norm 仍归 norm，eh_proj 保持 other（复制）。
  assert.equal(classifyTensor("model.layers.92.transformer.shared_head.output.weight"), "lm_head");
  assert.equal(classifyTensor("model.layers.92.transformer.shared_head.norm.weight"), "norm");
  assert.equal(classifyTensor("model.layers.92.eh_proj.weight"), "other");
  // 普通 dense MLP 与注意力门控保持原归类。
  assert.equal(classifyTensor("model.layers.0.mlp.gate_proj.weight"), "dense_mlp");
  assert.equal(classifyTensor("model.layers.0.self_attn.g_proj.weight"), "attention");
  assert.equal(classifyTensor("model.layers.3.self_attn.sparse_indexer_q.weight"), "attention");
});

test("calculateWeightFromTensors 全量精确求和且单卡按模块切分", () => {
  const tensors = [
    { name: "model.embed_tokens.weight", dtype: "BF16", shape: [100, 64], bytes: 12800 },
    { name: "model.layers.0.self_attn.q_proj.weight", dtype: "I8", shape: [64, 64], bytes: 4096 },
    { name: "model.layers.0.self_attn.o_proj.weight", dtype: "I8", shape: [64, 64], bytes: 4096 },
    { name: "model.layers.0.mlp.gate_proj.weight", dtype: "I8", shape: [128, 64], bytes: 8192 },
    { name: "model.layers.0.mlp.experts.0.w1.weight", dtype: "I8", shape: [128, 64], bytes: 8192 },
    { name: "model.layers.0.input_layernorm.weight", dtype: "F32", shape: [64], bytes: 256 },
  ];
  const p = {
    tp: 4,
    ep: 8,
    attentionTp: 4,
    oprojTp: 2,
    embeddingTp: 4,
    lmHeadTp: 4,
    sharedExpertTp: 1,
  };

  const r = calculateWeightFromTensors(tensors, p);

  // 全量 = 12800+4096+4096+8192+8192+256 = 37632
  assert.equal(r.fullWeight, 37632);

  // 单卡：embed/4 + q(attentionTp)/4 + o(oprojTp)/2 + mlp/tp4 + expert/ep8 + norm/1
  assert.equal(r.breakdown.embedding.perDevice, 3200);
  assert.equal(r.breakdown.attention.perDevice, 1024 + 2048);
  assert.equal(r.breakdown.dense_mlp.perDevice, 2048);
  assert.equal(r.breakdown.routed_expert.perDevice, 1024);
  assert.equal(r.breakdown.norm.perDevice, 256);
  assert.equal(r.perDeviceWeight, 3200 + 1024 + 2048 + 2048 + 1024 + 256);
});

test("TP=EP=1 时单卡权重等于全量权重", () => {
  const tensors = [
    { name: "model.embed_tokens.weight", dtype: "BF16", shape: [100, 64], bytes: 12800 },
    { name: "model.layers.0.mlp.experts.0.w1.weight", dtype: "I8", shape: [128, 64], bytes: 8192 },
  ];
  const p = { tp: 1, ep: 1, attentionTp: 1, oprojTp: 1, embeddingTp: 1, lmHeadTp: 1, sharedExpertTp: 1 };
  const r = calculateWeightFromTensors(tensors, p);
  assert.equal(r.perDeviceWeight, r.fullWeight);
  assert.equal(r.fullWeight, 12800 + 8192);
});