import assert from "node:assert/strict";
import test from "node:test";

import { buildExportSheets, exportFileName } from "../app/export-excel.ts";

const GIB = 1024 ** 3;

function findRow(rows, label) {
  return rows.find((row) => row[0] === label);
}

function sheetNamed(sheets, name) {
  const sheet = sheets.find((item) => item.name === name);
  assert.ok(sheet, `应包含工作表 ${name}`);
  return sheet;
}

function step5Context(overrides = {}) {
  const breakdown = {
    routed_expert: { full: 604788334592, perDevice: 18899635456, tensors: 519 },
    shared_expert: { full: 3321888768, perDevice: 3321888768, tensors: 264 },
    dense_mlp: { full: 2579496960, perDevice: 322437120, tensors: 24 },
    attention: { full: 21243667456, perDevice: 2655458432, tensors: 590 },
    embedding: { full: 1055916032, perDevice: 131989504, tensors: 1 },
    lm_head: { full: 4223664128, perDevice: 527958016, tensors: 4 },
    router: { full: 253468672, perDevice: 253468672, tensors: 176 },
    norm: { full: 3211264, perDevice: 3211264, tensors: 459 },
    vision: { full: 3962113720, perDevice: 3962113720, tensors: 667 },
    other: { full: 0, perDevice: 0, tensors: 0 },
  };
  const weight = Object.values(breakdown).reduce((sum, item) => sum + item.perDevice, 0);
  const fullWeight = Object.values(breakdown).reduce((sum, item) => sum + item.full, 0);
  const kvCache = 18_427_131_904;
  const activation = 1_073_741_824;
  const hccl = 4_000_000_000;
  const graph = 270_000_000;
  const cann = 1_073_741_824;
  const deviceOS = 4.25 * GIB;
  const total = weight + kvCache + activation + hccl + graph + cann + deviceOS;

  return {
    weight: {
      owner: "Eco-Tech",
      name: "Step-5-Preview-w8a8-mxfp8",
      kind: "quantized",
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
      mtpLayers: 3,
      kvCacheTokens: 1048576,
      kvCacheSequences: 1,
      kvPrecision: "bf16_fp16",
      indexCachePrecision: "fp8_int8",
      graphCount: 5,
      cannGB: 1,
    },
    result: {
      total,
      weight,
      fullWeight,
      kvCache,
      activation,
      hiddenResidual: 536_870_912,
      moeBuffers: 536_870_912,
      hccl,
      hcclDP: 100_000_000,
      hcclTP: 400_000_000,
      hcclEP: 400_000_000,
      hcclMC2: 3_100_000_000,
      epDispatch: 1_500_000_000,
      epCombine: 50_000_000,
      alignedDispatch: 8256,
      alignedCombine: 8704,
      graph,
      cann,
      deviceOS,
      localExperts: 11,
      attentionTp: 8,
      kvCacheBreakdown: {
        kvCache: 17_691_131_904,
        indexCache: 736_000_000,
        total: 18_427_131_904,
        kvBytesPerElement: 2,
        indexBytesPerElement: 1,
        kvCopies: 2,
        effectiveKvLayers: 95,
        effectiveIndexLayers: 23,
      },
      kvCacheInfo: { layers: 92, sparseLayers: 23, kvHeads: 4, headDim: 192, indexHeadDim: 256 },
      weightResult: { fullWeight, perDeviceWeight: weight, breakdown },
    },
    epSize: 32,
    kvCacheModelId: "step-5-preview",
    generatedAt: new Date("2026-11-05T12:30:00"),
    ...overrides,
  };
}

test("导出工作簿包含五个工作表，顺序固定", () => {
  const sheets = buildExportSheets(step5Context());
  assert.deepEqual(
    sheets.map((sheet) => sheet.name),
    ["概要", "显存构成", "权重模块", "KV Cache", "明细"],
  );
});

test("概要表记录权重信息、结构与关键结果", () => {
  const ctx = step5Context();
  const summary = sheetNamed(buildExportSheets(ctx), "概要");
  assert.equal(findRow(summary.rows, "当前权重")[1], "Eco-Tech/Step-5-Preview-w8a8-mxfp8");
  assert.equal(findRow(summary.rows, "原始权重")[1], "Step-5-Preview");
  assert.equal(findRow(summary.rows, "权重类型")[1], "Eco-Tech 量化权重");
  assert.match(findRow(summary.rows, "ModelScope")[1], /modelscope\.cn\/models\/Eco-Tech\/Step-5-Preview-w8a8-mxfp8/);
  assert.equal(findRow(summary.rows, "Hidden size")[1], 4096);
  assert.equal(findRow(summary.rows, "专家总数")[1], 352);
  assert.equal(findRow(summary.rows, "EP size")[1], 32);
  assert.equal(findRow(summary.rows, "Attention TP")[1], 8);
  assert.equal(findRow(summary.rows, "本地专家数")[1], 11);
  assert.equal(findRow(summary.rows, "含视觉塔")[1], "是");
  assert.equal(findRow(summary.rows, "单卡总显存 (GiB)")[1], Math.round((ctx.result.total / GIB) * 1000) / 1000);
  assert.equal(findRow(summary.rows, "单卡总显存 (字节)")[1], ctx.result.total);
  assert.equal(findRow(summary.rows, "导出时间")[1], "2026-11-05 12:30:00");
});

test("显存构成表各项相加等于单卡总计，占比合计 100%", () => {
  const ctx = step5Context();
  const sheet = sheetNamed(buildExportSheets(ctx), "显存构成");
  const total = findRow(sheet.rows, "单卡合计");
  assert.equal(total[1], ctx.result.total);
  assert.equal(total[3], 100);
  const weightRow = findRow(sheet.rows, "权重占用");
  assert.equal(weightRow[1], ctx.result.weight);
  const sum = sheet.rows
    .filter((row, index) => index > 0 && row[0] !== "单卡合计")
    .reduce((acc, row) => acc + row[1], 0);
  assert.equal(sum, ctx.result.total);
});

test("权重模块表逐模块给出全量与单卡，并合计一致", () => {
  const ctx = step5Context();
  const sheet = sheetNamed(buildExportSheets(ctx), "权重模块");
  const header = sheet.rows[0];
  assert.deepEqual(header, ["模块", "全量字节", "全量 (GiB)", "单卡字节", "单卡 (GiB)", "张量数"]);
  assert.equal(findRow(sheet.rows, "路由专家")[1], 604788334592);
  assert.equal(findRow(sheet.rows, "路由专家")[3], 18899635456);
  const total = findRow(sheet.rows, "合计");
  assert.equal(total[1], ctx.result.fullWeight);
  assert.equal(total[3], ctx.result.weight);
  assert.equal(total[5], 519 + 264 + 24 + 590 + 1 + 4 + 176 + 459 + 667);
});

test("KV Cache 表反映索引器块压缩后的层数与精度", () => {
  const ctx = step5Context();
  const sheet = sheetNamed(buildExportSheets(ctx), "KV Cache");
  assert.equal(findRow(sheet.rows, "上游 model id")[1], "step-5-preview");
  assert.equal(findRow(sheet.rows, "有效 KV 层数")[1], 95);
  assert.equal(findRow(sheet.rows, "有效 Index 层数")[1], 23);
  assert.equal(findRow(sheet.rows, "Index head_dim")[1], 256);
  assert.equal(findRow(sheet.rows, "KV Cache")[1], 17_691_131_904);
  assert.equal(findRow(sheet.rows, "Index Cache")[1], 736_000_000);
  assert.equal(findRow(sheet.rows, "单卡合计 (GiB)")[1], Math.round((18_427_131_904 / GIB) * 1000) / 1000);
});

test("明细表列出激活/HCCL/运行时各项", () => {
  const ctx = step5Context();
  const sheet = sheetNamed(buildExportSheets(ctx), "明细");
  assert.equal(findRow(sheet.rows, "HCCL 合计")[1], ctx.result.hccl);
  assert.equal(findRow(sheet.rows, "ACLGraph")[1], ctx.result.graph);
  assert.equal(findRow(sheet.rows, "Device OS")[1], ctx.result.deviceOS);
  assert.equal(findRow(sheet.rows, "对齐后 dispatch 行字节")[1], ctx.result.alignedDispatch);
  assert.match(findRow(sheet.rows, "MoE 激活 buffer")[3], /topK/);
});

test("没有 KV 公式或权重未加载时优雅退化", () => {
  const noKv = step5Context({ result: { ...step5Context().result, kvCacheBreakdown: null, kvCacheInfo: undefined, kvCache: 0 } });
  assert.deepEqual(
    buildExportSheets(noKv).map((sheet) => sheet.name),
    ["概要", "显存构成", "权重模块", "明细"],
  );

  const noWeights = step5Context({ result: { ...step5Context().result, weightResult: null, weight: 0, fullWeight: 0 } });
  const breakdownSheet = sheetNamed(buildExportSheets(noWeights), "权重模块");
  assert.equal(breakdownSheet.rows[1][0], "（未加载权重）");
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
