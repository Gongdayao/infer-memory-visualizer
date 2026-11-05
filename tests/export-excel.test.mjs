import assert from "node:assert/strict";
import test from "node:test";

import { buildExportSheet, EXPORT_SHEET_NAME, exportFileName } from "../app/export-excel.ts";

const GIB = 1024 ** 3;

function rowIndex(rows, label) {
  return rows.findIndex((row) => row[0] === label);
}

function findRow(rows, label) {
  const index = rowIndex(rows, label);
  assert.notEqual(index, -1, `应包含行 ${label}`);
  return rows[index];
}

/** 真实的 Step-5-Preview 分类汇总（TP=8 / EP=32，attention/oproj/embedding/lmHead 均为 8）。 */
const STEP5_CATEGORIES = [
  { category: "routed_expert", label: "路由专家（Routed Expert）", parallel: "ep", divisor: 32, full: 604_721_627_136, perDevice: 18_897_550_848, tensors: 519 },
  { category: "attention_qkv", label: "Attention QKV / Gate", parallel: "attentionTp", divisor: 8, full: 10_808_203_008, perDevice: 1_351_025_376, tensors: 403 },
  { category: "vision", label: "视觉塔（Vision Tower）", parallel: "replicated", divisor: 1, full: 3_962_029_568, perDevice: 3_962_029_568, tensors: 667 },
  { category: "shared_expert", label: "共享专家（Share Expert）", parallel: "sharedExpertTp", divisor: 1, full: 3_321_888_768, perDevice: 3_321_888_768, tensors: 264 },
  { category: "attention_oproj", label: "Attention O-Proj", parallel: "oprojTp", divisor: 8, full: 9_563_013_120, perDevice: 1_195_376_640, tensors: 95 },
  { category: "lm_head", label: "LM Head / MTP 输出头", parallel: "lmHeadTp", divisor: 8, full: 4_223_664_128, perDevice: 527_958_016, tensors: 4 },
  { category: "dense_mlp", label: "Dense MLP", parallel: "tp", divisor: 8, full: 2_378_170_368, perDevice: 297_271_296, tensors: 21 },
  { category: "router", label: "Router Gate", parallel: "replicated", divisor: 1, full: 253_879_296, perDevice: 253_879_296, tensors: 176 },
  { category: "other", label: "其他（MTP 融合等）", parallel: "replicated", divisor: 1, full: 201_326_592, perDevice: 201_326_592, tensors: 3 },
  { category: "embedding", label: "Embedding（词嵌入）", parallel: "embeddingTp", divisor: 8, full: 1_055_916_032, perDevice: 131_989_504, tensors: 1 },
  { category: "sparse_indexer", label: "Sparse Indexer", parallel: "attentionTp", divisor: 8, full: 874_250_240, perDevice: 109_281_280, tensors: 92 },
  { category: "norm", label: "Norm（含 q/k norm）", parallel: "replicated", divisor: 1, full: 3_493_376, perDevice: 3_493_376, tensors: 459 },
];
const FULL_TOTAL = STEP5_CATEGORIES.reduce((sum, row) => sum + row.full, 0);
const PER_TOTAL = STEP5_CATEGORIES.reduce((sum, row) => sum + row.perDevice, 0);

function step5Context(overrides = {}) {
  return {
    weight: {
      owner: "stepfun-ai",
      name: "Step-5-Preview",
      kind: "raw",
      rawName: "Step-5-Preview",
    },
    struct: {
      hiddenSize: 4096,
      numLayers: 92,
      attentionHeads: 64,
      kvHeads: 4,
      headDim: 192,
      vocabSize: 128896,
      expertCount: 352,
      topK: 8,
      sharedExperts: 1,
      intermediateSize: 13824,
      moeIntermediateSize: 1536,
      architectures: ["Step5ForConditionalGeneration"],
      modelType: "step5_vl",
      hasVision: true,
    },
    inputs: {
      maxBatchedTokens: 4096,
      maxBS: 128,
      tpSize: 8,
      dpSize: 4,
      attentionTpSize: null,
      oprojTpSize: null,
      embeddingTpSize: null,
      lmHeadTpSize: null,
      sharedExpertTpSize: 1,
      mtpLayers: 0,
      kvCacheTokens: 1048576,
      kvCacheSequences: 1,
      kvPrecision: "bf16_fp16",
      indexCachePrecision: "fp8_int8",
      graphCount: 5,
      cannGB: 1,
    },
    result: {
      total: PER_TOTAL + 18_427_131_904,
      weight: PER_TOTAL,
      fullWeight: FULL_TOTAL,
      kvCache: 18_427_131_904,
      activation: 1_073_741_824,
      hiddenResidual: 536_870_912,
      moeBuffers: 536_870_912,
      hccl: 4_000_000_000,
      hcclDP: 100_000_000,
      hcclTP: 400_000_000,
      hcclEP: 400_000_000,
      hcclMC2: 3_100_000_000,
      epDispatch: 1_500_000_000,
      epCombine: 50_000_000,
      alignedDispatch: 8256,
      alignedCombine: 8704,
      graph: 270_000_000,
      cann: 1_073_741_824,
      deviceOS: 4.25 * GIB,
      localExperts: 11,
      attentionTp: 8,
      kvCacheBreakdown: {
        kvCache: 17_691_131_904,
        indexCache: 736_000_000,
        total: 18_427_131_904,
        kvBytesPerElement: 2,
        indexBytesPerElement: 1,
        kvCopies: 2,
        effectiveKvLayers: 92,
        effectiveIndexLayers: 23,
      },
      kvCacheInfo: { layers: 92, sparseLayers: 23, kvHeads: 4, headDim: 192, indexHeadDim: 256 },
      weightResult: {
        fullWeight: FULL_TOTAL,
        perDeviceWeight: PER_TOTAL,
        categories: STEP5_CATEGORIES,
      },
    },
    epSize: 32,
    kvCacheModelId: "step-5-preview",
    generatedAt: new Date("2026-11-05T15:20:00"),
    ...overrides,
  };
}

test("工作簿只有一张表，主表列头固定为五列", () => {
  const sheet = buildExportSheet(step5Context());
  assert.equal(sheet.name, EXPORT_SHEET_NAME);
  assert.deepEqual(sheet.rows[0], [
    "权重（张量）类别",
    "切分方式",
    "全模型 bytes",
    "单卡 bytes",
    "单卡 GiB",
  ]);
  assert.deepEqual(sheet.columnWidths, [30, 24, 20, 20, 14]);
  assert.equal(sheet.rows[0].length, 5, "严格五列");
});

test("类别行按单卡 GiB 降序，切分方式按类别给出 TP / EP / 复本", () => {
  const rows = buildExportSheet(step5Context()).rows;
  const first = rows[1];
  assert.equal(first[0], "路由专家（Routed Expert）");
  assert.equal(first[1], "EP ÷32");
  assert.equal(first[2], 604_721_627_136);
  assert.equal(first[3], 18_897_550_848);
  assert.equal(first[4], 17.6);

  const perCards = rows.slice(1, 1 + STEP5_CATEGORIES.length).map((row) => row[3]);
  const sorted = [...perCards].sort((a, b) => b - a);
  assert.deepEqual(perCards, sorted, "应按单卡占用降序");

  assert.equal(findRow(rows, "视觉塔（Vision Tower）")[1], "复本（÷1）");
  assert.equal(findRow(rows, "Embedding（词嵌入）")[1], "embeddingTp ÷8");
  assert.equal(findRow(rows, "共享专家（Share Expert）")[1], "sharedExpertTp ÷1（复本）");
  assert.equal(findRow(rows, "Attention O-Proj")[1], "oprojTp ÷8");
  assert.equal(findRow(rows, "Sparse Indexer")[1], "attentionTp ÷8");
});

test("合计行汇总单卡，末行给出全模型 bytes 与 GiB", () => {
  const rows = buildExportSheet(step5Context()).rows;
  const lastCategoryIndex = 1 + STEP5_CATEGORIES.length - 1;
  const perRow = rows[lastCategoryIndex + 1];
  assert.equal(perRow.length, 5);
  assert.equal(perRow[0], "合计（单卡权重）");
  assert.equal(perRow[3], Math.round(PER_TOTAL));
  assert.equal(perRow[4], Math.round((PER_TOTAL / GIB) * 100) / 100);
  assert.equal(perRow[2], "");

  // 末行：两组标签-数值横向铺开，全模型 bytes 落在全模型列，全模型 GiB 紧邻其后
  const fullRow = rows[lastCategoryIndex + 2];
  assert.equal(fullRow.length, 5);
  assert.equal(fullRow[0], "全模型权重合计");
  assert.equal(fullRow[1], "全模型 bytes");
  assert.equal(fullRow[2], FULL_TOTAL);
  assert.equal(fullRow[3], "全模型 GiB");
  assert.equal(fullRow[4], Math.round((FULL_TOTAL / GIB) * 100) / 100);
});

test("主表下方隔行给出基础信息，且横向多列排布", () => {
  const rows = buildExportSheet(step5Context()).rows;
  const section = rowIndex(rows, "基础信息");
  assert.ok(section > 1, "基础信息应在主表之下");
  assert.deepEqual(rows[section - 1], [], "基础信息前应有一行空行");
  assert.ok(rows[section - 2].length === 0, "基础信息前应空两行");

  const body = rows.slice(section + 1);
  assert.ok(body.every((row) => row.length === 5), "基础信息每行应铺满五列");
  const pairs = body.flatMap((row) => [
    [row[0], row[1]],
    [row[2], row[3]],
  ]).filter(([label]) => label !== "");
  const lookup = Object.fromEntries(pairs);
  assert.equal(lookup["导出时间"], "2026-11-05 15:20:00");
  assert.equal(lookup["厂商 owner"], "stepfun-ai");
  assert.match(lookup["ModelScope"], /modelscope\.cn\/models\/stepfun-ai\/Step-5-Preview/);
  assert.equal(lookup["专家总数"], 352);
  assert.equal(lookup["EP size"], 32);
  assert.equal(lookup["单卡权重"], `${(PER_TOTAL / GIB).toFixed(2)} GiB`);
  assert.equal(lookup["有效 KV / Index 层"], "92 / 23");
  assert.equal(lookup["KV 精度"], "BF16 / FP16");
  assert.equal(lookup["Index 精度"], "FP8 / INT8");
});

test("无 KV 公式或无权重时不报错，主表仍保留合计行", () => {
  const ctx = step5Context();
  const noKv = buildExportSheet({
    ...ctx,
    result: { ...ctx.result, kvCache: 0, kvCacheBreakdown: null, kvCacheInfo: undefined },
  });
  assert.equal(rowIndex(noKv.rows, "KV 精度"), -1, "无 KV 时不写精度行");
  assert.equal(findRow(noKv.rows, "合计（单卡权重）")[3], Math.round(PER_TOTAL));

  const noWeights = buildExportSheet({
    ...ctx,
    result: { ...ctx.result, weightResult: null, weight: 0, fullWeight: 0 },
  });
  assert.equal(noWeights.rows[1][0], "合计（单卡权重）");
  assert.equal(noWeights.rows[2][0], "全模型权重合计");
  assert.equal(noWeights.rows[1][3], 0);
});

test("文件名包含 owner、权重名与时间戳，并去掉非法字符", () => {
  const date = new Date("2026-11-05T12:30:00");
  assert.equal(
    exportFileName({ owner: "Eco-Tech", name: "Step-5-Preview-w8a8-mxfp8", kind: "quantized", rawName: "Step-5-Preview" }, date),
    "infer-memory-Eco-Tech-Step-5-Preview-w8a8-mxfp8-20261105-1230.xlsx",
  );
  assert.equal(
    exportFileName({ owner: "moonshotai", name: "Kimi K2.5/a", kind: "raw", rawName: "Kimi-K2.5" }, date),
    "infer-memory-moonshotai-Kimi-K2.5-a-20261105-1230.xlsx",
  );
});
