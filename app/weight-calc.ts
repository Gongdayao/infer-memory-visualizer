// 张量级权重显存计算：从 safetensors header 解析出的张量清单，按模块归类后
// 计算全量权重与单卡（按 TP/EP 切分后）权重。
//
// 全量权重（TP=EP=1）为 numel × dtype 字节数之和，与模块分类无关，绝对精确；
// 单卡权重按张量所属模块应用通用切分规则（expert 按 EP、embedding/LM Head 按 TP、
// attention 的 QKV 按 attentionTp、O-Proj 按 oprojTp、dense 按 TP、其余复制），
// 面向不同架构为通用近似，后续可逐架构细化。

import type { SafetensorInfo } from "./safetensors";

export type WeightTensor = Pick<SafetensorInfo, "name" | "dtype" | "shape" | "bytes">;

export type TensorModule =
  | "host_memory"
  | "routed_expert"
  | "shared_expert"
  | "dense_mlp"
  | "attention"
  | "embedding"
  | "lm_head"
  | "router"
  | "norm"
  | "vision"
  | "other";

export type WeightParallel = {
  tp: number;
  ep: number;
  attentionTp: number;
  oprojTp: number;
  embeddingTp: number;
  lmHeadTp: number;
  sharedExpertTp: number;
};

export type ModuleBreakdown = {
  full: number;
  perDevice: number;
  tensors: number;
};

/**
 * 比 TensorModule 更细的类别：把 attention 拆成 QKV / O-Proj / 稀疏索引器，
 * 便于按「类别 × 切分方式」汇总单卡占用。
 */
export type WeightCategory =
  | "host_memory"
  | "routed_expert"
  | "shared_expert"
  | "attention_qkv"
  | "attention_oproj"
  | "sparse_indexer"
  | "dense_mlp"
  | "embedding"
  | "lm_head"
  | "router"
  | "norm"
  | "vision"
  | "other";

/** 类别 → 切分维度（replicated 表示每张卡保留完整副本）。 */
export type CategoryParallel = keyof WeightParallel | "replicated" | "host";

export type WeightCategoryRow = {
  category: WeightCategory;
  label: string;
  parallel: CategoryParallel;
  /** 实际除数：replicated 为 1 */
  divisor: number;
  full: number;
  perDevice: number;
  tensors: number;
};

export type WeightResult = {
  fullWeight: number;
  perDeviceWeight: number;
  breakdown: Record<TensorModule, ModuleBreakdown>;
  /** 更细的类别汇总（按固定顺序，数值与 breakdown 完全对账） */
  categories: WeightCategoryRow[];
};

const VISION = ["vision", "visual", "projector", "patch_merge", "vae", "vision_model"];
const EMBEDDING = ["embed_tokens", "word_embedding", "wte."];
// shared_head.output 是 MTP 投机层的输出头（词表 × hidden），切分行为与 LM Head 一致。
// shared_head.output（Step）与 shared_head.head（GLM）都是 MTP 投机层的输出头，与 LM Head 同切分。
const LM_HEAD = ["lm_head", "final_fc", "score_head", "shared_head.output", "shared_head.head"];
// Step 系列把共享专家写作 share_expert（无 d），需与 shared_experts 同时覆盖。
const SHARED_EXPERT = ["shared_expert", "shared_experts", "share_expert"];
const ROUTER = ["router", "e_score", "tid2eid", ".gate.weight", ".gate.bias"];
const NORM = [
  "layernorm", "rmsnorm", "norm.weight", "norm.", "norm1", "norm2", "q_norm",
  "k_norm", "kv_norm", "ln_1", "ln_2", "final_norm",
];
const ATTENTION = ["engram.q_weight", "engram.k_weight", "ple.key_proj", 
  "attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj", "wq", "wkv",
  "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "index", "linear_attn",
  "mamba", "a_log", "conv1d", "dt_bias", "mixer",
];
const DENSE_MLP = [
  "gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h",
  "fc1", "fc2", "linear_fc1", "linear_fc2", "mlp", "ffn",
  // MTP 的输入/主干投影同样是线性层，按 TP 切分
  "mtp.fc.", "main_proj", "mtp.pre_fc",
];
const OPROJ = ["o_proj", "out_proj", "wo_a", "wo_b"];

// Step 系列把路由专家的三个 MLP 矩阵合并在 moe.*_proj 下（形状含专家维），
// 命名里没有 expert 字样，因此显式按「.moe. + MLP 矩阵」判定为路由专家权重。
const MOE_EXPERT_MATRIX = ["gate_proj", "up_proj", "down_proj", "weight_scale", "weight_scale_inv"];

function includes(name: string, words: string[]): boolean {
  const n = name.toLowerCase();
  return words.some((w) => n.includes(w));
}

function isMoERoutedExpert(name: string): boolean {
  const n = name.toLowerCase();
  return n.includes(".moe.") && MOE_EXPERT_MATRIX.some((w) => n.includes(w));
}

const CATEGORY_LABEL: Record<WeightCategory, string> = {
  host_memory: "Engram / PLE（主机内存）",
  routed_expert: "路由专家（Routed Expert）",
  shared_expert: "共享专家（Share Expert）",
  attention_qkv: "Attention QKV / Gate",
  attention_oproj: "Attention O-Proj",
  sparse_indexer: "Sparse Indexer",
  dense_mlp: "Dense MLP",
  embedding: "Embedding（词嵌入）",
  lm_head: "LM Head / MTP 输出头",
  router: "Router Gate",
  norm: "Norm（含 q/k norm）",
  vision: "视觉塔（Vision Tower）",
  other: "其他（MTP 融合等）",
};

/** 类别展示顺序（导出时再按单卡占用降序重排）。 */
export const WEIGHT_CATEGORY_ORDER: WeightCategory[] = [
  "host_memory",
  "routed_expert",
  "attention_qkv",
  "vision",
  "shared_expert",
  "attention_oproj",
  "lm_head",
  "dense_mlp",
  "router",
  "other",
  "embedding",
  "sparse_indexer",
  "norm",
];

const CATEGORY_PARALLEL: Record<WeightCategory, CategoryParallel> = {
  // Engram 表与 Qwen4Exp PLE 表默认 CPU offload 到主机内存（vLLM --engram-config），不占显存。
  host_memory: "host",
  routed_expert: "ep",
  shared_expert: "sharedExpertTp",
  attention_qkv: "attentionTp",
  attention_oproj: "oprojTp",
  sparse_indexer: "attentionTp",
  dense_mlp: "tp",
  embedding: "embeddingTp",
  lm_head: "lmHeadTp",
  router: "replicated",
  norm: "replicated",
  vision: "replicated",
  other: "replicated",
};

const CATEGORY_MODULE: Record<WeightCategory, TensorModule> = {
  host_memory: "host_memory",
  routed_expert: "routed_expert",
  shared_expert: "shared_expert",
  attention_qkv: "attention",
  attention_oproj: "attention",
  sparse_indexer: "attention",
  dense_mlp: "dense_mlp",
  embedding: "embedding",
  lm_head: "lm_head",
  router: "router",
  norm: "norm",
  vision: "vision",
  other: "other",
};

/** 将张量名归类到细分权重类别。量化辅助张量（weight_scale 等）跟随主权重归类。 */
export function categorizeTensor(name: string): WeightCategory {
  // Engram（DeepSeek-V4.1）与 PLE（Qwen4Exp）是只查表、不参与计算的静态表，
  // vLLM 默认把它们 CPU offload 到主机内存（--engram-config.cpu_offload=true），
  // 因此不计入单卡显存；仍需在全量权重里体现。
  if (name.includes("engram.embed") || name.includes("ple_embedding") || name.includes("ngram_embedding")) {
    return "host_memory";
  }
  if (includes(name, VISION)) return "vision";
  // 裸名 embed.weight / head.weight（DeepSeek-V4.1 的顶层词嵌入与输出头）按词表并行。
  if (/(^|\.)embed\.weight$/.test(name)) return "embedding";
  if (/(^|\.)head\.weight$/.test(name)) return "lm_head";
  if (includes(name, EMBEDDING)) return "embedding";
  if (includes(name, LM_HEAD)) return "lm_head";
  if (includes(name, SHARED_EXPERT)) return "shared_expert";
  if (name.toLowerCase().includes("expert")) return "routed_expert";
  if (includes(name, ROUTER)) return "router";
  if (isMoERoutedExpert(name)) return "routed_expert";
  if (includes(name, NORM)) return "norm";
  // 稀疏索引器单独成类，但仍按注意力 TP 切分。
  if (name.toLowerCase().includes("indexer")) return "sparse_indexer";
  if (includes(name, ATTENTION)) return isOproj(name) ? "attention_oproj" : "attention_qkv";
  if (includes(name, DENSE_MLP)) return "dense_mlp";
  return "other";
}

export function weightCategoryLabel(category: WeightCategory): string {
  return CATEGORY_LABEL[category];
}

/** 将张量名归类到模块类型（与细分类别一一对应）。 */
export function classifyTensor(name: string): TensorModule {
  return CATEGORY_MODULE[categorizeTensor(name)];
}

function isOproj(name: string): boolean {
  return includes(name, OPROJ);
}

/** 细分类别对应的切分除数。 */
export function divisorForCategory(
  category: WeightCategory,
  p: WeightParallel,
): number {
  // 主机内存类别不占显存：单卡为 0。
  if (category === "host_memory") return Number.POSITIVE_INFINITY;
  const safe = (v: number, dflt: number) => (Number.isFinite(v) && v > 0 ? v : dflt);
  const tp = safe(p.tp, 1);
  switch (CATEGORY_PARALLEL[category]) {
    case "ep":
      return safe(p.ep, 1);
    case "sharedExpertTp":
      return safe(p.sharedExpertTp, 1);
    case "attentionTp":
      return safe(p.attentionTp, tp);
    case "oprojTp":
      return safe(p.oprojTp, tp);
    case "embeddingTp":
      return safe(p.embeddingTp, tp);
    case "lmHeadTp":
      return safe(p.lmHeadTp, tp);
    case "tp":
      return tp;
    default:
      return 1;
  }
}

export function categoryParallelOf(category: WeightCategory): CategoryParallel {
  return CATEGORY_PARALLEL[category];
}

/** 按「类别 × 切分方式」汇总全量与单卡字节。 */
export function calculateWeightCategories(
  tensors: WeightTensor[],
  p: WeightParallel,
): WeightCategoryRow[] {
  const rows = new Map<WeightCategory, WeightCategoryRow>();
  for (const category of WEIGHT_CATEGORY_ORDER) {
    rows.set(category, {
      category,
      label: CATEGORY_LABEL[category],
      parallel: CATEGORY_PARALLEL[category],
      divisor: divisorForCategory(category, p),
      full: 0,
      perDevice: 0,
      tensors: 0,
    });
  }
  for (const t of tensors) {
    const row = rows.get(categorizeTensor(t.name)) ?? rows.get("other")!;
    const bytes = t.bytes || 0;
    row.full += bytes;
    row.perDevice += bytes / row.divisor;
    row.tensors += 1;
  }
  return WEIGHT_CATEGORY_ORDER.map((category) => rows.get(category)!).filter(
    (row) => row.tensors > 0,
  );
}

function newBreakdown(): Record<TensorModule, ModuleBreakdown> {
  const empty: ModuleBreakdown = { full: 0, perDevice: 0, tensors: 0 };
  return {
    host_memory: { ...empty },
    routed_expert: { ...empty },
    shared_expert: { ...empty },
    dense_mlp: { ...empty },
    attention: { ...empty },
    embedding: { ...empty },
    lm_head: { ...empty },
    router: { ...empty },
    norm: { ...empty },
    vision: { ...empty },
    other: { ...empty },
  };
}

export function calculateWeightFromTensors(
  tensors: WeightTensor[],
  p: WeightParallel,
): WeightResult {
  const safe = (v: number, dflt: number) => (Number.isFinite(v) && v > 0 ? v : dflt);
  const tp = safe(p.tp, 1);
  const ep = safe(p.ep, 1);
  const attentionTp = safe(p.attentionTp, tp);
  const oprojTp = safe(p.oprojTp, tp);
  const embeddingTp = safe(p.embeddingTp, tp);
  const lmHeadTp = safe(p.lmHeadTp, tp);
  const sharedExpertTp = safe(p.sharedExpertTp, 1);

  const divisorFor = (module: TensorModule, name: string): number => {
    switch (module) {
      case "host_memory":
        return Number.POSITIVE_INFINITY;
      case "routed_expert":
        return ep;
      case "shared_expert":
        return sharedExpertTp;
      case "attention":
        return isOproj(name) ? oprojTp : attentionTp;
      case "dense_mlp":
        return tp;
      case "embedding":
        return embeddingTp;
      case "lm_head":
        return lmHeadTp;
      default:
        return 1;
    }
  };

  const breakdown = newBreakdown();
  let fullWeight = 0;

  for (const t of tensors) {
    const tensorModule = classifyTensor(t.name);
    const bytes = t.bytes || 0;
    fullWeight += bytes;
    const b = breakdown[tensorModule];
    b.full += bytes;
    b.tensors += 1;
    b.perDevice += bytes / divisorFor(tensorModule, t.name);
  }

  const perDeviceWeight = Object.values(breakdown).reduce((s, b) => s + b.perDevice, 0);
  return {
    fullWeight,
    perDeviceWeight,
    breakdown,
    categories: calculateWeightCategories(tensors, p),
  };
}