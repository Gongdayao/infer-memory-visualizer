import assert from "node:assert/strict";
import test from "node:test";

import { cachePrecisionOptions, calculateKvCache, getKvCacheModelInfo } from "../app/kv-cache-model.ts";

test("MiniMax M3 stores K and V separately and supports FP4 index cache", () => {
  const info = getKvCacheModelInfo("minimax-m3");
  assert.ok(info);
  assert.equal(info.layers, 60);
  assert.equal(info.sparseLayers, 57);
  const result = calculateKvCache({
    modelId: "minimax-m3",
    tokens: 1024,
    sequences: 1,
    kvPrecision: "bf16_fp16",
    indexPrecision: "fp4_int4",
  });

  assert.ok(result);
  assert.equal(result.kvCopies, 2);
  assert.equal(result.kvCache, 125_829_120);
  assert.equal(result.indexCache, 3_735_552);
  assert.equal(result.total, 129_564_672);
});

test("KV and index precision can be selected independently", () => {
  const bf16 = calculateKvCache({
    modelId: "minimax-m3",
    tokens: 4096,
    sequences: 8,
    kvPrecision: "bf16_fp16",
    indexPrecision: "bf16_fp16",
  });
  const compressed = calculateKvCache({
    modelId: "minimax-m3",
    tokens: 4096,
    sequences: 8,
    kvPrecision: "fp8_int8",
    indexPrecision: "fp4_int4",
  });

  assert.ok(bf16 && compressed);
  assert.equal(compressed.kvCache, bf16.kvCache / 2);
  assert.equal(compressed.indexCache, bf16.indexCache / 4);
});

test("each MTP layer adds one KV layer and one sparse Index layer", () => {
  const common = {
    modelId: "minimax-m3",
    tokens: 1024,
    sequences: 1,
    kvPrecision: "fp8_int8",
    indexPrecision: "fp4_int4",
  };
  const baseline = calculateKvCache({ ...common, mtpLayers: 0 });
  const threeMtp = calculateKvCache({ ...common, mtpLayers: 3 });

  assert.ok(baseline && threeMtp);
  assert.equal(threeMtp.effectiveKvLayers, 63);
  assert.equal(threeMtp.effectiveIndexLayers, 60);
  assert.equal(
    threeMtp.kvCache - baseline.kvCache,
    3 * 1024 * 2 * 4 * 128,
  );
  assert.equal(
    threeMtp.indexCache - baseline.indexCache,
    3 * 1024 * 128 * 0.5,
  );
});

test("KV cache shards across TP while the index cache stays replicated", () => {
  const common = {
    modelId: "minimax-m3",
    tokens: 4096,
    sequences: 8,
    kvPrecision: "fp8_int8",
    indexPrecision: "fp8_int8",
  };
  const tp1 = calculateKvCache({ ...common, tpSize: 1 });
  const tp4 = calculateKvCache({ ...common, tpSize: 4 });

  assert.ok(tp1 && tp4);
  assert.equal(tp4.kvCache, tp1.kvCache / 4);
  assert.equal(tp4.indexCache, tp1.indexCache);
});

test("Step 5 Preview：23 层带索引器、KV 头 4 个、索引器单头 256 维", () => {
  const info = getKvCacheModelInfo("step-5-preview");
  assert.ok(info);
  assert.equal(info.layers, 92);
  assert.equal(info.sparseLayers, 23);
  assert.equal(info.kvHeads, 4);
  assert.equal(info.headDim, 192);
  assert.equal(info.indexHeadDim, 256);
});

test("Step 5 Preview 索引器缓存按 8 token 块压缩，字节数精确", () => {
  const result = calculateKvCache({
    modelId: "step-5-preview",
    tokens: 1_000_000,
    sequences: 1,
    kvPrecision: "bf16_fp16",
    indexPrecision: "fp8_int8",
  });
  assert.ok(result);
  // full: 1e6 x 23 x 4 x (192+192) x 2 B；sliding: 512 x 69 x 4 x 384 x 2 B
  const full = 1_000_000 * 23 * 4 * 384 * 2;
  const sliding = 512 * 69 * 4 * 384 * 2;
  // 索引器：ceil(1e6/8) = 125000 个摘要 x 23 层 x 1 个 K 头 x 256 维 x 1 B
  const indexer = 125_000 * 23 * 256;
  assert.equal(result.kvCache, full + sliding);
  assert.equal(result.indexCache, indexer);
  assert.equal(result.total, full + sliding + indexer);
  assert.equal(result.total, 71_500_527_616);
  assert.equal(result.effectiveKvLayers, 92);
  assert.equal(result.effectiveIndexLayers, 23);
});

test("Step 5 Preview：TP=8 时 4 个 KV 头在卡对间重复，索引器每卡完整复制", () => {
  const common = {
    modelId: "step-5-preview",
    tokens: 1_000_000,
    sequences: 1,
    kvPrecision: "bf16_fp16",
    indexPrecision: "fp8_int8",
  };
  const tp1 = calculateKvCache({ ...common, tpSize: 1 });
  const tp4 = calculateKvCache({ ...common, tpSize: 4 });
  const tp8 = calculateKvCache({ ...common, tpSize: 8 });
  assert.ok(tp1 && tp4 && tp8);
  assert.equal(tp4.kvCache, tp1.kvCache / 4);
  assert.equal(tp4.indexCache, tp1.indexCache);
  // 官方部署 TP=8：每张卡仍只持有 1/4 KV，索引器不切分。
  assert.equal(tp8.kvCache, tp4.kvCache);
  assert.equal(tp8.indexCache, tp4.indexCache);
  assert.equal(tp8.total, 18_427_131_904);
});

test("Step 5 Preview：MTP 的 3 层是滑动窗口层，不增加索引器缓存", () => {
  const common = {
    modelId: "step-5-preview",
    tokens: 1_000_000,
    sequences: 1,
    kvPrecision: "bf16_fp16",
    indexPrecision: "fp8_int8",
    tpSize: 8,
  };
  const baseline = calculateKvCache({ ...common, mtpLayers: 0 });
  const withMtp = calculateKvCache({ ...common, mtpLayers: 3 });
  assert.ok(baseline && withMtp);
  assert.equal(withMtp.effectiveKvLayers, 95);
  assert.equal(withMtp.effectiveIndexLayers, 23);
  // 每层每卡：min(1e6, 512) x 1 个本地 KV 组 x (192+192) x 2 B
  assert.equal(withMtp.kvCache - baseline.kvCache, 3 * 512 * 384 * 2);
  assert.equal(withMtp.indexCache, baseline.indexCache);
});

test("Step 5 Preview：TP 无法整除 KV 头数时退回线性均分", () => {
  const common = {
    modelId: "step-5-preview",
    tokens: 1_000_000,
    sequences: 1,
    kvPrecision: "bf16_fp16",
    indexPrecision: "fp8_int8",
  };
  const tp1 = calculateKvCache({ ...common, tpSize: 1 });
  const tp3 = calculateKvCache({ ...common, tpSize: 3 });
  assert.ok(tp1 && tp3);
  assert.equal(tp3.kvCache, tp1.kvCache / 3);
  assert.equal(tp3.indexCache, tp1.indexCache);
});

test("精度档位按权重形态（mxfp）与 c 位推导，不受支持的档位不返回", () => {
  // A2/A3 形态：c8 预选 INT8
  const plain = cachePrecisionOptions("GLM-5.3-w8a8c8", "glm-5.3", "kv");
  assert.deepEqual(plain.options, ["bf16_fp16", "fp8_int8"]);
  assert.equal(plain.defaultValue, "fp8_int8");

  // A5 形态：c8 预选 MXFP8，4-bit 未在权重中声明则不给
  const a5 = cachePrecisionOptions("GLM-5.3-w4a4c8-mxfp4", "glm-5.3", "kv");
  assert.deepEqual(a5.options, ["bf16_fp16", "mxfp8"]);
  assert.equal(a5.defaultValue, "mxfp8");

  // 无 c 位 → 默认 BF16，仍可切同形态 8bit
  const raw = cachePrecisionOptions("Qwen3-32B-w8a8", "qwen3-32b", "kv");
  assert.deepEqual(raw.options, ["bf16_fp16", "fp8_int8"]);
  assert.equal(raw.defaultValue, "bf16_fp16");

  // 打包页模型（V4.1）额外提供 4bit
  const v41 = cachePrecisionOptions("DeepSeek-V4.1-Flash-w8a8", "deepseek-v4.1-flash", "kv");
  assert.deepEqual(v41.options, ["bf16_fp16", "fp8_int8"]);
  const v41a5 = cachePrecisionOptions("DeepSeek-V4.1-Flash-w8a8-mxfp8", "deepseek-v4.1-flash", "kv");
  assert.deepEqual(v41a5.options, ["bf16_fp16", "mxfp8", "mxfp4"]);
  assert.deepEqual(cachePrecisionOptions("DeepSeek-V4.1-Flash-w8a8-mxfp8", "deepseek-v4.1-flash", "index").options, ["mxfp4"]);
  assert.deepEqual(cachePrecisionOptions("GLM-5.3-w4a4c8-mxfp4", "glm-5.3", "index").options, ["mxfp8"]);

  // Index：无 indexer 返回空（前端隐藏）
  assert.deepEqual(cachePrecisionOptions("Kimi-K2.5-w4a8", "kimi-k2.5", "index").options, []);
  // Index：固定值的模型只给一个只读档
  assert.deepEqual(cachePrecisionOptions("MiniMax-M3-w8a8c8", "minimax-m3", "index").options, ["bf16_fp16"]);
  assert.deepEqual(cachePrecisionOptions("Step-5-Preview-w8a8-mxfp8", "step-5-preview", "index").options, ["fp8_int8"]);
});
