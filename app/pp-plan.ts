// PP（Pipeline Parallel）计划：层分区、权重 eligibility、张量归属。
//
// 依据 vllm-ascend 官方文档（feature_guide/pipeline_parallel）：
// - 只有实现了 SupportsPP 的模型才能开 PP；
// - 只切「目标模型的 Transformer hidden layers」，embedding / final norm / LM head / 投机 draft 层
//   不参与切分（首 stage 拿 embedding，末 stage 拿输出侧模块与 drafter）；
// - 层数不要求整除：默认均分，余数从倒数第二级往前分。

export type PpLevel = "ascend" | "upstream";

/** 模型 → PP 支持级别；未列出的模型不开放 PP 选项。 */
const ASCEND_VERIFIED = [
  /^DeepSeek-V4/, // V4 / V4.1（含 1017、Vision-Exp）均有 Ascend SupportsPP 实现
  /^GLM-5\.2$/,
  /^MiniMax-M3/,
  /^Qwen3\.6-/,
  /^Kimi-K3/,
];

const UPSTREAM_SUPPORTED = [
  /^DeepSeek-V3/, /^DeepSeek-R1/,
  /^GLM-(4\.7|5(\.1|\.3)?)$/,
  /^kimi-k2/i,
  /^MiniMax-M2/,
  /^Qwen2\.5-/, /^Qwen3-/, /^Qwen3-Next-/, /^Qwen3\.(5|6|8)-/,
  /^Step-3\.[57]-Flash$/,
  /^gemma-4-/,
];

/** 明确不支持/未验证 PP 的家族（用于给出禁用原因）。 */
const NOT_SUPPORTED = [
  /^DeepSeek-Math/, /^GLM-5\.3-Flash$/, /^Hy[34]/, /^InternVL/, /^Ling-/, /^Ring-/,
  /^Step-5-Preview$/, /^Qwen3\.8-Flash-Next$/,
];

export function ppEligibility(rawName: string): { level: PpLevel } | { level: "none"; reason: string } {
  if (ASCEND_VERIFIED.some((re) => re.test(rawName))) return { level: "ascend" };
  if (UPSTREAM_SUPPORTED.some((re) => re.test(rawName))) return { level: "upstream" };
  if (NOT_SUPPORTED.some((re) => re.test(rawName))) {
    return { level: "none", reason: "该家族没有 PP 实现或尚未在 Ascend 验证" };
  }
  return { level: "none", reason: "未登记 PP 支持" };
}

/**
 * 按 vLLM 的自动分区规则切层：先均分，余数从倒数第二级开始向前分配，
 * 避免把余数压到承担 LM head / drafter 的末级。
 * 例：61 层 PP4 -> [15, 15, 16, 15]
 */
export function partitionLayers(layers: number, pp: number): number[] {
  const total = Math.max(1, Math.floor(layers));
  const size = Math.max(1, Math.floor(pp));
  const base = Math.floor(total / size);
  const rem = total - base * size;
  const counts = Array.from({ length: size }, () => base);
  let idx = size - 2;
  for (let left = rem; left > 0; left--) {
    if (idx < 0) idx = size - 1;
    counts[idx] += 1;
    idx -= 1;
  }
  return counts;
}

/** 把分区计数展开成 [start, end) 区间。 */
export function stageRanges(counts: number[]): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  let cursor = 0;
  for (const n of counts) {
    ranges.push([cursor, cursor + n]);
    cursor += n;
  }
  return ranges;
}

/** 解析用户输入的层数分布（如 "15,15,16,15"）；非法返回 null。 */
export function parsePartition(text: string): number[] | null {
  const trimmed = (text ?? "").trim();
  if (!trimmed) return null;
  const parts = trimmed.split(/[\s,，]+/).filter(Boolean);
  if (parts.length === 0) return null;
  const counts = parts.map((part) => Number(part));
  if (counts.some((n) => !Number.isInteger(n) || n < 1)) return null;
  return counts;
}

/**
 * 决定最终使用哪一组 stage 层数：
 * - 手动输入合法（个数 = pp、每级 ≥1、总和 = 层数）→ 用手动值；
 * - 否则回退到自动分区，并标记 valid=false 供 UI 提示。
 */
export function resolveStageCounts(
  layers: number,
  pp: number,
  text: string,
): { counts: number[]; source: "manual" | "auto"; valid: boolean } {
  const auto = partitionLayers(layers, pp);
  const manual = parsePartition(text);
  if (!manual) return { counts: auto, source: "auto", valid: true };
  const sum = manual.reduce((a, b) => a + b, 0);
  const valid = manual.length === pp && sum === layers;
  return valid
    ? { counts: manual, source: "manual", valid: true }
    : { counts: auto, source: "auto", valid: false };
}
