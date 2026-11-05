// 把当前页面的计算结果整理成一张工作表，供「导出 Excel」按钮使用。
// 主表按「权重（张量）类别 × 切分方式」汇总全模型/单卡占用，表格下方隔行给出基础信息。
// 纯函数部分（buildExportSheet）可在 Node 测试中直接断言，浏览器侧只负责下载。

import type { ModelStructure } from "./config-model";
import type { KvCacheBreakdown, KvCacheModelInfo, CachePrecision } from "./kv-cache-model";
import type { WeightCategoryRow } from "./weight-calc";
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
  weightResult: {
    fullWeight: number;
    perDeviceWeight: number;
    categories: WeightCategoryRow[];
  } | null;
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

export const EXPORT_SHEET_NAME = "权重与显存";

const PRECISION_LABEL: Record<CachePrecision, string> = {
  bf16_fp16: "BF16 / FP16",
  fp8_int8: "FP8 / INT8",
  fp4_int4: "FP4 / INT4",
};

const PARALLEL_LABEL: Record<string, string> = {
  tp: "TP",
  ep: "EP",
  attentionTp: "attentionTp",
  oprojTp: "oprojTp",
  embeddingTp: "embeddingTp",
  lmHeadTp: "lmHeadTp",
  sharedExpertTp: "sharedExpertTp",
  replicated: "复本",
};

function round(value: number, digits = 2): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function gib(bytes: number): number {
  return round(bytes / GIB, 2);
}

function gibText(bytes: number): string {
  return `${gib(bytes).toFixed(2)} GiB`;
}

function timestamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
  );
}

/** 「切分方式」列：复本或具体切分维度 ÷ 除数。 */
function splitLabel(row: WeightCategoryRow): string {
  if (row.parallel === "replicated") return "复本（÷1）";
  const name = PARALLEL_LABEL[row.parallel] ?? row.parallel;
  return row.divisor === 1 ? `${name} ÷1（复本）` : `${name} ÷${row.divisor}`;
}

/** 每组是一行里的若干 [标题, 值]，按主表 5 列铺开（空位补空字符串）。 */
function pairRows(groups: [string, Cell][][]): Cell[][] {
  return groups.map((group) => {
    const row: Cell[] = [];
    for (const [label, value] of group) row.push(label, value);
    while (row.length < 5) row.push("");
    return row;
  });
}

function mainTableRows(categories: WeightCategoryRow[]): Cell[][] {
  const rows: Cell[][] = [["权重（张量）类别", "切分方式", "全模型 bytes", "单卡 bytes", "单卡 GiB"]];
  const sorted = [...categories].sort((a, b) => b.perDevice - a.perDevice);
  for (const row of sorted) {
    rows.push([row.label, splitLabel(row), row.full, Math.round(row.perDevice), gib(row.perDevice)]);
  }
  const fullTotal = sorted.reduce((sum, row) => sum + row.full, 0);
  const perTotal = sorted.reduce((sum, row) => sum + row.perDevice, 0);
  rows.push(["合计（单卡权重）", "—", "", Math.round(perTotal), gib(perTotal)]);
  // 末行把「全模型 bytes / 全模型 GiB」两组标签-数值横向铺在 5 列里，不额外占列。
  rows.push(["全模型权重合计", "全模型 bytes", fullTotal, "全模型 GiB", gib(fullTotal)]);
  return rows;
}

function metaRows(ctx: ExportContext, tensors: number): Cell[][] {
  const { weight, struct, inputs, result } = ctx;
  const pairs: [string, Cell][] = [
    ["导出时间", timestamp(ctx.generatedAt ?? new Date())],
    ["厂商 owner", weight.owner],
    ["原始权重", weight.rawName],
    ["当前权重", `${weight.owner}/${weight.name}`],
    ["权重类型", weight.kind === "raw" ? "官方原始权重" : "Eco-Tech 量化权重"],
    ["张量总数", tensors],
    ["ModelScope", `https://modelscope.cn/models/${weight.owner}/${weight.name}`],
    ["Hidden size", struct.hiddenSize],
    ["层数", struct.numLayers],
    ["专家总数", struct.expertCount],
    ["TopK 专家", struct.topK],
    ["KV 头数", struct.kvHeads],
    ["head_dim", struct.headDim],
    ["词表大小", struct.vocabSize],
    ["MoE intermediate", struct.moeIntermediateSize],
    ["含视觉塔", struct.hasVision ? "是" : "否"],
    ["TP", inputs.tpSize],
    ["DP", inputs.dpSize],
    ["EP size", ctx.epSize],
    ["attentionTp", inputs.attentionTpSize ?? inputs.tpSize],
    ["oprojTp", inputs.oprojTpSize ?? inputs.tpSize],
    ["embeddingTp", inputs.embeddingTpSize ?? inputs.tpSize],
    ["lmHeadTp", inputs.lmHeadTpSize ?? inputs.tpSize],
    ["sharedExpertTp", inputs.sharedExpertTpSize],
    ["本地专家数", round(result.localExperts, 2)],
    ["Max batched tokens", inputs.maxBatchedTokens],
    ["Max BS", inputs.maxBS],
    ["MTP 层数", inputs.mtpLayers],
    ["单卡权重", gibText(result.weight)],
    ["KV + Index Cache", gibText(result.kvCache)],
    ["激活占用", gibText(result.activation)],
    ["HCCL buffer", gibText(result.hccl)],
    ["ACLGraph", gibText(result.graph)],
    ["CANN + PTA + 算子", gibText(result.cann)],
    ["Device OS", gibText(result.deviceOS)],
    ["单卡总显存", gibText(result.total)],
  ];
  if (result.kvCacheBreakdown) {
    pairs.push(
      ["KV 精度", PRECISION_LABEL[inputs.kvPrecision] ?? inputs.kvPrecision],
      ["Index 精度", PRECISION_LABEL[inputs.indexCachePrecision] ?? inputs.indexCachePrecision],
      [
        "有效 KV / Index 层",
        `${result.kvCacheBreakdown.effectiveKvLayers} / ${result.kvCacheBreakdown.effectiveIndexLayers}`,
      ],
    );
  }
  // 分组：URL 独占一行，其余每行 3 组，横向铺满，避免只占两列却拉很多行。
  const groups: [string, Cell][][] = [];
  let current: [string, Cell][] = [];
  for (const pair of pairs) {
    const isLink = pair[0] === "ModelScope";
    if (isLink) {
      if (current.length > 0) groups.push(current);
      groups.push([pair]);
      current = [];
      continue;
    }
    current.push(pair);
    if (current.length === 2) {
      groups.push(current);
      current = [];
    }
  }
  if (current.length > 0) groups.push(current);
  return pairRows(groups);
}

/**
 * 生成导出用的单张工作表：
 * 1. 主表：类别 / 切分方式 / 全模型 bytes,GiB / 单卡 bytes,GiB，末尾两行给出单卡合计与全模型合计；
 * 2. 空两行后给出基础信息（横向多列对齐，避免只占两列却拉很多行）。
 */
export function buildExportSheet(ctx: ExportContext): SheetSpec {
  const categories = ctx.result.weightResult?.categories ?? [];
  const rows = mainTableRows(categories);
  const tensors = categories.reduce((sum, row) => sum + row.tensors, 0);
  rows.push([], []);
  const sectionRow = rows.length;
  rows.push(["基础信息", "", "", "", ""]);
  rows.push(...metaRows(ctx, tensors));
  return {
    name: EXPORT_SHEET_NAME,
    columnWidths: [30, 24, 20, 20, 14],
    rows,
    boldRows: [sectionRow],
  };
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
