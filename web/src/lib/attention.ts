import { type Matrix, columns, linear, matmul, softmax, transpose } from "./matrix.ts";

export type Weights = { query: Matrix; key: Matrix; value: Matrix; output: Matrix };

export type Head = { q: Matrix; k: Matrix; v: Matrix; scores: Matrix; masked: Matrix; weights: Matrix; out: Matrix };

export function causalMask(length: number, causal = true): boolean[][] {
  return Array.from({ length }, (_, row) => Array.from({ length }, (_, column) => !causal || column <= row));
}

export function attendHead(q: Matrix, k: Matrix, v: Matrix, mask: boolean[][]): Head {
  const scale = Math.sqrt(q[0].length);
  const scores = matmul(q, transpose(k)).map((row) => row.map((value) => value / scale));
  const masked = scores.map((row, i) => row.map((value, j) => (mask[i][j] ? value : -Infinity)));
  const weights = masked.map(softmax);
  return { q, k, v, scores, masked, weights, out: matmul(weights, v) };
}

export function causalSelfAttention(x: Matrix, weights: Weights, heads: number, causal = true) {
  const width = x[0].length / heads;
  const q = linear(x, weights.query);
  const k = linear(x, weights.key);
  const v = linear(x, weights.value);
  const mask = causalMask(x.length, causal);
  const perHead = Array.from({ length: heads }, (_, h) => {
    const slice = (m: Matrix) => columns(m, h * width, (h + 1) * width);
    return attendHead(slice(q), slice(k), slice(v), mask);
  });
  const joined = x.map((_, row) => perHead.flatMap((head) => head.out[row]));
  return { heads: perHead, joined, output: linear(joined, weights.output) };
}
