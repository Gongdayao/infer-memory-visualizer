import assert from "node:assert/strict";
import test from "node:test";

import { extractModelStructure } from "../app/config-model.ts";

test("MiniMax 风格嵌套 text_config 提取", () => {
  const cfg = {
    model_type: "minimax_m3_vl",
    architectures: ["MiniMaxM3VLForCausalLM"],
    text_config: {
      model_type: "minimax_m3",
      hidden_size: 6144,
      num_hidden_layers: 64,
      num_attention_heads: 48,
      num_key_value_heads: 8,
      head_dim: 128,
      vocab_size: 151936,
      intermediate_size: 28672,
      moe_intermediate_size: 1024,
      num_experts: 256,
      num_experts_per_tok: 12,
      n_shared_experts: 1,
    },
    vision_config: { hidden_size: 1152 },
  };
  const s = extractModelStructure(cfg);
  assert.equal(s.hiddenSize, 6144);
  assert.equal(s.numLayers, 64);
  assert.equal(s.attentionHeads, 48);
  assert.equal(s.kvHeads, 8);
  assert.equal(s.headDim, 128);
  assert.equal(s.vocabSize, 151936);
  assert.equal(s.expertCount, 256);
  assert.equal(s.topK, 12);
  assert.equal(s.sharedExperts, 1);
  assert.equal(s.intermediateSize, 28672);
  assert.equal(s.moeIntermediateSize, 1024);
  assert.equal(s.hasVision, true);
  assert.deepEqual(s.architectures, ["MiniMaxM3VLForCausalLM"]);
});

test("DeepSeek 风格平铺字段提取（num_experts 别名）", () => {
  const cfg = {
    model_type: "deepseek_v4",
    hidden_size: 7168,
    num_hidden_layers: 61,
    num_attention_heads: 128,
    num_key_value_heads: 8,
    head_dim: 128,
    vocab_size: 129280,
    intermediate_size: 18432,
    moe_intermediate_size: 2048,
    n_routed_experts: 256,
    num_experts_per_tok: 8,
    n_shared_experts: 1,
  };
  const s = extractModelStructure(cfg);
  assert.equal(s.hiddenSize, 7168);
  assert.equal(s.numLayers, 61);
  assert.equal(s.expertCount, 256);
  assert.equal(s.topK, 8);
  assert.equal(s.hasVision, false);
});

test("dense 模型无 expert 字段时 expertCount=0", () => {
  const cfg = {
    model_type: "qwen3",
    hidden_size: 4096,
    num_hidden_layers: 36,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    intermediate_size: 10240,
  };
  const s = extractModelStructure(cfg);
  assert.equal(s.expertCount, 0);
  assert.equal(s.topK, 0);
});

test("Step 风格 text_config 提取（moe_num_experts / num_attention_groups）", () => {
  // 字段全部嵌在 text_config 下，且 GQA 的 KV 分组用 num_attention_groups 表示。
  const cfg = {
    model_type: "step5_vl",
    architectures: ["Step5ForConditionalGeneration"],
    vision_config: { model_type: "perception_encoder", width: 1536, layers: 47 },
    text_config: {
      model_type: "step5",
      hidden_size: 4096,
      num_hidden_layers: 92,
      num_attention_heads: 64,
      num_attention_groups: 4,
      head_dim: 192,
      vocab_size: 128896,
      intermediate_size: 13824,
      moe_intermediate_size: 1536,
      moe_num_experts: 352,
      moe_top_k: 8,
      num_experts_per_tok: 8,
      num_nextn_predict_layers: 3,
    },
  };
  const s = extractModelStructure(cfg);
  assert.equal(s.hiddenSize, 4096);
  assert.equal(s.numLayers, 92);
  assert.equal(s.attentionHeads, 64);
  assert.equal(s.kvHeads, 4);
  assert.equal(s.headDim, 192);
  assert.equal(s.vocabSize, 128896);
  assert.equal(s.intermediateSize, 13824);
  assert.equal(s.moeIntermediateSize, 1536);
  assert.equal(s.expertCount, 352);
  assert.equal(s.topK, 8);
  assert.equal(s.hasVision, true);
});

test("Step-3.x 风格：只提供 moe_top_k 时也能取到 topK", () => {
  const cfg = {
    model_type: "step3p5_vl",
    vision_config: { layers: 47 },
    text_config: {
      hidden_size: 4096,
      num_hidden_layers: 45,
      num_attention_heads: 64,
      num_attention_groups: 8,
      head_dim: 128,
      vocab_size: 128896,
      moe_intermediate_size: 1280,
      moe_num_experts: 288,
      moe_top_k: 8,
      use_moe: true,
    },
  };
  const s = extractModelStructure(cfg);
  assert.equal(s.numLayers, 45);
  assert.equal(s.kvHeads, 8);
  assert.equal(s.expertCount, 288);
  assert.equal(s.topK, 8);
});