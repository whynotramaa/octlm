import { type Matrix, linear, matmul } from "./matrix.ts";

export type QwenShape = { d_model: number; n_heads: number; kv_heads: number; head_size: number; ff_hidden: number; n_layers: number };
export type Projection = { name: string; part: "attention" | "feed_forward"; shape: [number, number] };

export function qwenProjections(c: QwenShape): Projection[] {
  const q = c.n_heads * c.head_size;
  const kv = c.kv_heads * c.head_size;
  return [
    { name: "query", part: "attention", shape: [c.d_model, q] },
    { name: "key", part: "attention", shape: [c.d_model, kv] },
    { name: "value", part: "attention", shape: [c.d_model, kv] },
    { name: "output", part: "attention", shape: [q, c.d_model] },
    { name: "gate", part: "feed_forward", shape: [c.d_model, c.ff_hidden] },
    { name: "up", part: "feed_forward", shape: [c.d_model, c.ff_hidden] },
    { name: "down", part: "feed_forward", shape: [c.ff_hidden, c.d_model] },
  ];
}

export const loraParameters = (shapes: [number, number][], rank: number, layers: number) =>
  layers * shapes.reduce((sum, [input, output]) => sum + rank * (input + output), 0);

export function mergeLora(base: Matrix, a: Matrix, b: Matrix, scale: number): Matrix {
  const delta = matmul(b, a);
  return base.map((row, i) => row.map((w, j) => w + delta[i][j] * scale));
}

export function loraForward(x: Matrix, base: Matrix, a: Matrix, b: Matrix, scale: number): Matrix {
  const update = linear(linear(x, a), b);
  return linear(x, base).map((row, i) => row.map((v, j) => v + update[i][j] * scale));
}
