export type Shape = {
  n_layers: number;
  d_model: number;
  n_heads: number;
  kv_heads?: number;
  attention?: string;
  attention_window?: number;
  kv_compress_block?: number;
  mla_rank?: number;
  mla_rope_dim?: number;
  ff_multiplier?: number;
  feed_forward?: string;
};

export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction !== 0.5) return Math.round(value);
  return floor % 2 === 0 ? floor : floor + 1;
}

export function hiddenSize(shape: Shape): number {
  const full = shape.d_model * (shape.ff_multiplier ?? 4);
  if ((shape.feed_forward ?? "gelu") === "gelu") return full;
  return Math.max(8, roundHalfEven((full * 2) / 3 / 8) * 8);
}

export function cacheDims(shape: Shape): number {
  if (shape.attention === "mla") return (shape.mla_rank ?? 0) + (shape.mla_rope_dim ?? 0);
  const heads = shape.kv_heads || shape.n_heads;
  return 2 * heads * (shape.d_model / shape.n_heads);
}

export function cachedPositions(shape: Shape, length: number): number {
  const block = shape.kv_compress_block ?? 0;
  if (!block) return length;
  const window = Math.min(shape.attention_window ?? 0, length);
  return window + Math.floor((length - window) / block);
}

export function kvCacheBytes(shape: Shape, length: number, elementBytes = 2): number {
  return cacheDims(shape) * elementBytes * shape.n_layers * cachedPositions(shape, length);
}

export type Config = Shape & {
  vocab_size: number;
  context_length: number;
  position?: string;
  norm?: string;
};

export function parameterBreakdown(c: Config) {
  const head = c.d_model / c.n_heads;
  const kv = (c.kv_heads || c.n_heads) * head;
  const norm = (c.norm ?? "layernorm") === "layernorm" ? 2 * c.d_model : c.d_model;
  const matrices = (c.feed_forward ?? "gelu") === "gelu" ? 2 : 3;
  const attention = 2 * c.d_model * c.d_model + 2 * c.d_model * kv;
  const feedForward = matrices * c.d_model * hiddenSize(c);
  return {
    embedding: c.vocab_size * c.d_model,
    positions: (c.position ?? "learned") === "learned" ? c.context_length * c.d_model : 0,
    attention: c.n_layers * attention,
    feedForward: c.n_layers * feedForward,
    norms: c.n_layers * 2 * norm + norm,
  };
}

export function parameterCount(c: Config): number {
  return Object.values(parameterBreakdown(c)).reduce((sum, value) => sum + value, 0);
}
