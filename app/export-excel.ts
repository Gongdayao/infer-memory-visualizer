// 把当前页面的计算结果整理成多张工作表，供「导出 Excel」按钮使用。
// 纯函数部分（buildExportSheets）可在 Node 测试中直接断言，浏览器侧只负责下载。

import type { ModelStructure } from "./config-model";
import type { KvCacheBreakdown, KvCacheModelInfo, CachePrecision } from "./kv-cache-model";
import type { WeightResult, TensorModule } from "./weight-calc";
import type { Cell, SheetSpec } from "./xlsx";

const GIB = 1024 ** 3;

export type ExportWeight = {
  owner: string;
  name: string;
  /** raw = 官方原始权重，quantized = Eco-Tech 量化权重 */
  kind: "raw" | "quantized";
  /** 对应的原始权重名（KV Cache 等映射以它为准） */
  rawName: string;
};

export type ExportInputs = {
  maxBatchedTokens: number;
  maxBS: number;
  tpSize: number;
  dpSize: number;
  attentionTpSize: number | null;
  oprojTpSize: number | null;
  embeddingTpSize: number | null;
  lmHeadTpSize: number | null;
  sharedExpertTpSize: number;
  mtpLayers: number;
  kvCacheTokens: number;
  kvCacheSequences: number;
  kvPrecision: CachePrecision;
  indexCachePrecision: CachePrecision;
  graphCount: number;
  cannGB: number;
};

export type ExportResult = {
  total: number;
  weight: number;
  fullWeight: number;
  kvCache: number;
  activation: number;
  hiddenResidual: number;
  moeBuffers: number;
  hccl: number;
  hcclDP: number;
  hcclTP: number;
  hcclEP: number;
  hcclMC2: number;
  epDispatch: number;
  epCombine: number;
  alignedDispatch: number;
  alignedCombine: number;
  graph: number;
  cann: number;
  deviceOS: number;
  localExperts: number;
  attentionTp: number;
  kvCacheBreakdown: KvCacheBreakdown | null;
  kvCacheInfo?: KvCacheModelInfo;
  weightResult: WeightResult | null;
};

export type ExportContext = {
  weight: ExportWeight;
  struct: ModelStructure;
  inputs: ExportInputs;
  result: ExportResult;
  epSize: number;
  /** 当前 KV Cache 使用到的上游 model id（未收录时为空） */
  kvCacheModelId?: string;
  /** 生成时间（默认取当前时间，测试可固定） */
  generatedAt?: Date;
};

const MODULE_LABEL: Record<TensorModule, string> = {
  routed_expert: "路由专家",
  shared_expert: "共享专家",
  dense_mlp: "Dense MLP",
  attention: "Attention",
  embedding: "Embedding",
  lm_head: "LM Head",
  router: "Router",
  norm: "Norm",
  vision: "视觉塔",
  other: "其他",
};

const PRECISION_LABEL: Record<CachePrecision, string> = {
  bf16_fp16: "BF16 / FP16",
  fp8_int8: "FP8 / INT8",
  fp4_int4: "FP4 / INT4",
};

function round(value: number, digits = 2): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function gib(bytes: number): number {
  return round(bytes / GIB, 3);
}

function timestamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
  );
}

function percent(value: number, total: number): number {
  return total > 0 ? round((value / total) * 100, 2) : 0;
}

function kvRows(ctx: ExportContext): Cell[][] {
  const { result } = ctx;
  const breakdown = result.kvCacheBreakdown;
  const info = result.kvCacheInfo;
  const rows: Cell[][] = [["项目", "值", "单位"]];
  if (!breakdown || !info) {
    rows.push(["KV Cache", "该模型未收录上游 KV Cache 公式", ""]);
    return rows;
  }
  rows.push(
    ["上游 model id", ctx.kvCacheModelId ?? "", ""],
    ["KV 精度", PRECISION_LABEL[ctx.inputs.kvPrecision] ?? ctx.inputs.kvPrecision, ""],
    ["Index 精度", PRECISION_LABEL[ctx.inputs.indexCachePrecision] ?? ctx.inputs.indexCachePrecision, ""],
    ["缓存 tokens", ctx.inputs.kvCacheTokens, "token"],
    ["并发序列数", ctx.inputs.kvCacheSequences, "seq"],
    ["有效 KV 层数", breakdown.effectiveKvLayers, "层"],
    ["有效 Index 层数", breakdown.effectiveIndexLayers, "层"],
    ["KV heads", info.kvHeads, "个"],
    ["head_dim", info.headDim, "维"],
    ["Index head_dim", info.indexHeadDim, "维"],
    ["每元素 KV 字节", breakdown.kvBytesPerElement, "B"],
    ["每元素 Index 字节", breakdown.indexBytesPerElement, "B"],
    ["KV Cache", breakdown.kvCache, "B"],
    ["KV Cache (GiB)", gib(breakdown.kvCache), "GiB"],
    ["Index Cache", breakdown.indexCache, "B"],
    ["Index Cache (GiB)", gib(breakdown.indexCache), "GiB"],
    ["单卡合计", breakdown.total, "B"],
    ["单卡合计 (GiB)", gib(breakdown.total), "GiB"],
  );
  return rows;
}

function breakdownRows(ctx: ExportContext): Cell[][] {
  const rows: Cell[][] = [["模块", "全量字节", "全量 (GiB)", "单卡字节", "单卡 (GiB)", "张量数"]];
  const breakdown = ctx.result.weightResult?.breakdown;
  if (!breakdown) {
    rows.push(["（未加载权重）", 0, 0, 0, 0, 0]);
    return rows;
  }
  for (const [module, item] of Object.entries(breakdown) as [TensorModule, { full: number; perDevice: number; tensors: number }][]) {
    rows.push([MODULE_LABEL[module] ?? module, item.full, gib(item.full), item.perDevice, gib(item.perDevice), item.tensors]);
  }
  rows.push([
    "合计",
    ctx.result.fullWeight,
    gib(ctx.result.fullWeight),
    ctx.result.weight,
    gib(ctx.result.weight),
    Object.values(breakdown).reduce((sum, item) => sum + item.tensors, 0),
  ]);
  return rows;
}

function memoryRows(ctx: ExportContext): Cell[][] {
  const { result } = ctx;
  const total = result.total;
  const items: { label: string; bytes: number; note: string }[] = [
    { label: "权重占用", bytes: result.weight, note: "逐张量按模块切分后的单卡权重" },
    { label: "KV + Index Cache", bytes: result.kvCache, note: "上游 kv-cache-calculator 公式" },
    { label: "激活占用", bytes: result.activation, note: "hidden states + residual + MoE buffer" },
    { label: "HCCL buffer", bytes: result.hccl, note: "DP / TP / EP / MC2" },
    { label: "ACLGraph 占用", bytes: result.graph, note: "按图数量估算" },
    { label: "CANN + PTA + 算子", bytes: result.cann, note: "手动预留" },
    { label: "Device OS", bytes: result.deviceOS, note: "固定 4.25 GiB" },
  ];
  const rows: Cell[][] = [["项目", "字节数", "GiB", "占比(%)", "说明"]];
  for (const item of items) {
    if (item.bytes === 0 && item.label.startsWith("权重")) continue;
    rows.push([item.label, item.bytes, gib(item.bytes), percent(item.bytes, total), item.note]);
  }
  rows.push(["单卡合计", total, gib(total), 100, "含以上全部项目"]);
  return rows;
}

function detailRows(ctx: ExportContext): Cell[][] {
  const { result, inputs } = ctx;
  const rows: Cell[][] = [["明细项", "字节数", "MiB", "公式/说明"]];
  const push = (label: string, bytes: number, formula: string) => {
    rows.push([label, bytes, round(bytes / 1024 ** 2, 2), formula]);
  };
  push("Hidden states + residual", result.hiddenResidual, `2 × 2 B × maxBatchedTokens(${inputs.maxBatchedTokens}) × hidden × dp`);
  push("MoE 激活 buffer", result.moeBuffers, "4 × 2 B × dp × tokens × topK ÷ EP × hidden");
  push("HCCL DP buffer", result.hcclDP, `max(ceil((dp+1)×4 ÷ 1024²), 50) × 2 MB`);
  push("HCCL TP buffer", result.hcclTP, "200 MB × 2");
  push("HCCL EP buffer", result.hcclEP, "200 MB × 2");
  push("EP Dispatch", result.epDispatch, "2 × 本地专家数 × maxBS × EP × alignedDispatch");
  push("EP Combine", result.epCombine, "2 × topK × maxBS × alignedCombine");
  push("HCCL MC2 小计", result.hcclMC2, "EP Dispatch + EP Combine");
  push("HCCL 合计", result.hccl, "DP + TP + EP + MC2");
  push("ACLGraph", result.graph, `${inputs.graphCount} 张图 ÷ 5 × 0.27 GB`);
  push("CANN + PTA + 算子", result.cann, `${inputs.cannGB} GiB 预留`);
  push("Device OS", result.deviceOS, "固定 4.25 GiB");
  rows.push(["对齐后 dispatch 行字节", result.alignedDispatch, "", "align(2 × hidden, 32) + 64 后再按 480→512 对齐"]);
  rows.push(["对齐后 combine 行字节", result.alignedCombine, "", "align(2 × hidden, 512)"]);
  return rows;
}

function summaryRows(ctx: ExportContext): Cell[][] {
  const { weight, struct, inputs, result, epSize } = ctx;
  const rows: Cell[][] = [["项目", "值"]];
  const push = (label: string, value: Cell) => rows.push([label, value]);

  push("导出时间", timestamp(ctx.generatedAt ?? new Date()));
  push("厂商 owner", weight.owner);
  push("原始权重", weight.rawName);
  push("当前权重", `${weight.owner}/${weight.name}`);
  push("权重类型", weight.kind === "raw" ? "官方原始权重" : "Eco-Tech 量化权重");
  push("ModelScope", `https://modelscope.cn/models/${weight.owner}/${weight.name}`);
  push("", "");
  push("Hidden size", struct.hiddenSize);
  push("层数", struct.numLayers);
  push("注意力头数", struct.attentionHeads);
  push("KV 头数", struct.kvHeads);
  push("head_dim", struct.headDim);
  push("词表大小", struct.vocabSize);
  push("专家总数", struct.expertCount);
  push("TopK 专家", struct.topK);
  push("共享专家", struct.sharedExperts);
  push("Dense intermediate", struct.intermediateSize);
  push("MoE intermediate", struct.moeIntermediateSize);
  push("含视觉塔", struct.hasVision ? "是" : "否");
  push("architectures", struct.architectures.join(", "));
  push("model_type", struct.modelType);
  push("", "");
  push("TP", inputs.tpSize);
  push("DP", inputs.dpSize);
  push("EP size", epSize);
  push("Attention TP", inputs.attentionTpSize ?? inputs.tpSize);
  push("O-Proj TP", inputs.oprojTpSize ?? inputs.tpSize);
  push("Embedding TP", inputs.embeddingTpSize ?? inputs.tpSize);
  push("LM Head TP", inputs.lmHeadTpSize ?? inputs.tpSize);
  push("Shared Expert TP", inputs.sharedExpertTpSize);
  push("本地专家数", round(result.localExperts, 4));
  push("", "");
  push("Max batched tokens", inputs.maxBatchedTokens);
  push("Max BS", inputs.maxBS);
  push("MTP 层数", inputs.mtpLayers);
  push("", "");
  push("单卡权重 (GiB)", gib(result.weight));
  push("全量权重 (GiB)", gib(result.fullWeight));
  push("KV + Index Cache (GiB)", gib(result.kvCache));
  push("激活占用 (GiB)", gib(result.activation));
  push("HCCL buffer (GiB)", gib(result.hccl));
  push("ACLGraph (GiB)", gib(result.graph));
  push("CANN + PTA + 算子 (GiB)", gib(result.cann));
  push("Device OS (GiB)", gib(result.deviceOS));
  push("单卡总显存 (GiB)", gib(result.total));
  push("单卡总显存 (字节)", result.total);
  return rows;
}

/** 组装工作簿：概要 / 显存构成 / 权重模块 / KV Cache / 明细。 */
export function buildExportSheets(ctx: ExportContext): SheetSpec[] {
  const sheets: SheetSpec[] = [
    { name: "概要", columnWidths: [26, 46], rows: summaryRows(ctx) },
    { name: "显存构成", columnWidths: [24, 18, 12, 12, 46], rows: memoryRows(ctx) },
    { name: "权重模块", columnWidths: [18, 18, 14, 18, 14, 10], rows: breakdownRows(ctx) },
  ];
  if (ctx.result.kvCacheBreakdown) {
    sheets.push({ name: "KV Cache", columnWidths: [22, 30, 10], rows: kvRows(ctx) });
  }
  sheets.push({ name: "明细", columnWidths: [26, 18, 12, 60], rows: detailRows(ctx) });
  return sheets;
}

/** 形如 infer-memory-Step-5-Preview-20261105-1220.xlsx 的文件名。 */
export function exportFileName(weight: ExportWeight, date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const safeName = `${weight.owner}-${weight.name}`.replace(/[^A-Za-z0-9._-]+/g, "-");
  const stamp =
    `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}` +
    `-${pad(date.getHours())}${pad(date.getMinutes())}`;
  return `infer-memory-${safeName}-${stamp}.xlsx`;
}
