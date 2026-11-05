import kvCacheCalculator from "kv-cache-calculator";
import kvModelData from "kv-cache-calculator/models";

export type CachePrecision = "bf16_fp16" | "fp8_int8" | "fp4_int4";

export const CACHE_PRECISIONS: Record<CachePrecision, { label: string; bytes: number }> = {
  bf16_fp16: { label: "BF16 / FP16", bytes: 2 },
  fp8_int8: { label: "FP8 / INT8", bytes: 1 },
  fp4_int4: { label: "FP4 / INT4", bytes: 0.5 },
};

export type KvCacheModelInfo = {
  layers: number;
  sparseLayers: number;
  kvHeads: number;
  headDim: number;
  indexHeadDim: number;
};

/**
 * 上游模型里 MTP / next-N 层在缓存中的归属：
 * - "indexer"：MTP 层同时计入 KV 层与索引层（MiniMax M3 这类按稀疏层处理）；
 * - "sliding"：MTP 层是滑动窗口层，只增加 KV，不增加索引器缓存（Step-5）。
 */
const MTP_LAYER_KIND: Record<string, "indexer" | "sliding"> = {
  "step-5-preview": "sliding",
};

function findUpstreamModel(modelId: string) {
  return kvModelData.models.find((model) => model.id === modelId);
}

export function getKvCacheModelInfo(modelId: string): KvCacheModelInfo | undefined {
  const upstream = findUpstreamModel(modelId);
  if (!upstream) return undefined;
  const fields = upstream.fields;
  const layers = Number(fields.num_hidden_layers);
  return {
    layers,
    // 索引器层数：Step 系列记在 indexer_full_layers，MiniMax 记在 sparse_attention_layers。
    sparseLayers: Number(fields.indexer_full_layers ?? fields.sparse_attention_layers ?? layers),
    kvHeads: Number(fields.num_key_value_heads),
    headDim: Number(fields.head_dim),
    indexHeadDim: Number(fields.index_head_dim),
  };
}

type KvCacheInput = {
  modelId: string;
  tokens: number;
  sequences: number;
  kvPrecision: CachePrecision;
  indexPrecision: CachePrecision;
  mtpLayers?: number;
  tpSize?: number;
};

export type KvCacheBreakdown = {
  kvCache: number;
  indexCache: number;
  total: number;
  kvBytesPerElement: number;
  indexBytesPerElement: number;
  kvCopies: number;
  effectiveKvLayers: number;
  effectiveIndexLayers: number;
};

/** 按界面输入的 MTP 层数改写上游模型字段。 */
function fieldsWithMtpLayers(
  upstream: { id: string; fields: Record<string, unknown> },
  info: KvCacheModelInfo,
  mtpLayers: number,
): Record<string, unknown> {
  const fields: Record<string, unknown> = {
    ...upstream.fields,
    num_hidden_layers: info.layers + mtpLayers,
  };
  if (mtpLayers > 0) {
    const slidingLayers = Number(fields.sliding_attention_layers);
    if (MTP_LAYER_KIND[upstream.id] === "sliding" && Number.isFinite(slidingLayers)) {
      fields.sliding_attention_layers = slidingLayers + mtpLayers;
    } else {
      fields.sparse_attention_layers = info.sparseLayers + mtpLayers;
    }
  }
  // 界面允许独立选择索引器精度，模型自带固定精度只作为默认值（与上游网页的行为一致）。
  delete fields.indexer_fixed_precision_id;
  return fields;
}

export function calculateKvCache({
  modelId,
  tokens,
  sequences,
  kvPrecision,
  indexPrecision,
  mtpLayers = 0,
  tpSize = 1,
}: KvCacheInput): KvCacheBreakdown | null {
  const upstream = findUpstreamModel(modelId);
  const info = getKvCacheModelInfo(modelId);
  if (!upstream || !info) return null;
  const safeTokens = Math.max(0, Math.floor(tokens));
  const safeSequences = Math.max(0, Math.floor(sequences));
  const safeMtpLayers = Math.max(0, Math.floor(mtpLayers));
  const safeTp = Math.max(1, Math.floor(tpSize));
  const effectiveKvLayers = info.layers + safeMtpLayers;
  // sliding 型 MTP 层不携带索引器，索引层数不随 MTP 增长。
  const effectiveIndexLayers =
    MTP_LAYER_KIND[upstream.id] === "sliding"
      ? info.sparseLayers
      : info.sparseLayers + safeMtpLayers;
  const kvBytesPerElement = (CACHE_PRECISIONS[kvPrecision] ?? CACHE_PRECISIONS.bf16_fp16).bytes;
  const indexBytesPerElement = (CACHE_PRECISIONS[indexPrecision] ?? CACHE_PRECISIONS.fp4_int4).bytes;
  // Standard GQA stores K and V separately and shards KV heads across TP ranks.
  const kvCopies = 2;

  if (safeTokens === 0 || safeSequences === 0) {
    return {
      kvCache: 0,
      indexCache: 0,
      total: 0,
      kvBytesPerElement,
      indexBytesPerElement,
      kvCopies,
      effectiveKvLayers,
      effectiveIndexLayers,
    };
  }

  // Delegate the per-token math to the upstream kv-cache-calculator engine.
  const model = { ...upstream, fields: fieldsWithMtpLayers(upstream, info, safeMtpLayers) };
  const input = {
    tokens: safeTokens,
    sequences: safeSequences,
    precision: kvPrecision,
    indexerPrecision: indexPrecision,
  };

  let kvCache: number;
  let indexCache: number;
  try {
    // 上游按每个缓存组的头数把开销分摊到 TP 卡：KV 组按 KV 头数（头数少于 TP 时在卡间重复），
    // 索引器只有单个 K 头，因此每张卡都持有完整索引器缓存。
    const result = kvCacheCalculator.calculate(model, { ...input, tensorParallel: safeTp });
    kvCache = 0;
    indexCache = 0;
    for (const group of result.cacheGroups) {
      if (group.role === "indexer") indexCache += group.perDeviceBytes;
      else kvCache += group.perDeviceBytes;
    }
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    // TP 与头数无法整除时（例如 4 个 KV 头配 TP=3），退回按 TP 线性均分。
    const result = kvCacheCalculator.calculate(model, { ...input, tensorParallel: 1 });
    kvCache = result.kvBytes / safeTp;
    indexCache = result.indexerBytes;
  }

  return {
    kvCache,
    indexCache,
    total: kvCache + indexCache,
    kvBytesPerElement,
    indexBytesPerElement,
    kvCopies,
    effectiveKvLayers,
    effectiveIndexLayers,
  };
}
