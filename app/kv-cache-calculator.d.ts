declare module "kv-cache-calculator" {
  export type UpstreamCacheGroup = {
    role: string;
    label: string;
    elements?: number;
    /** 该缓存组的总字节数（全部卡、全部序列）。 */
    bytes: number;
    /** 分摊到每张卡后的字节数（已乘上序列数）。 */
    perDeviceBytes: number;
    /** 全部卡合计的物理占用。 */
    allDeviceBytes: number;
    heads: number;
    localHeads: number;
    replicated: boolean;
    replicationFactor: number;
  };

  export type UpstreamCalculateResult = {
    totalBytes: number;
    kvBytes: number;
    indexerBytes: number;
    cacheGroups: UpstreamCacheGroup[];
  };

  const core: {
    calculate: (
      model: unknown,
      input: Record<string, unknown>,
      options?: unknown,
    ) => UpstreamCalculateResult;
    formatBytes: (bytes: number) => string;
  };
  export default core;
}

declare module "kv-cache-calculator/models" {
  export type UpstreamModel = {
    id: string;
    label: string;
    family: string;
    formula: string;
    default_tokens?: number;
    fields: Record<string, unknown>;
  };

  const data: {
    models: UpstreamModel[];
  };
  export default data;
}
