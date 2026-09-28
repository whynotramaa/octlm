import type { Matrix } from "./matrix.ts";

export type TileStep = { start: number; stop: number; runningMax: number[]; runningSum: number[]; output: Matrix };

export function tiledAttention(query: Matrix, key: Matrix, value: Matrix, block: number) {
  const length = query.length;
  const width = value[0].length;
  const scale = Math.sqrt(query[0].length);
  let runningMax = query.map(() => -Infinity);
  let runningSum = query.map(() => 0);
  let output: Matrix = query.map(() => new Array(width).fill(0));
  const steps: TileStep[] = [];
  for (let start = 0; start < length; start += block) {
    const stop = Math.min(start + block, length);
    const next = query.map((q, row) => {
      const scores = key.slice(start, stop).map((k, offset) => (start + offset > row ? -Infinity : q.reduce((s, x, i) => s + x * k[i], 0) / scale));
      const seen = Math.max(runningMax[row], ...scores);
      const blockMax = seen === -Infinity ? 0 : seen;
      const correction = Math.exp(runningMax[row] - blockMax);
      const weights = scores.map((score) => Math.exp(score - blockMax));
      const sum = correction * runningSum[row] + weights.reduce((s, w) => s + w, 0);
      const out = output[row].map((o, c) => correction * o + weights.reduce((s, w, j) => s + w * value[start + j][c], 0));
      return { blockMax, sum, out };
    });
    runningMax = next.map((n) => n.blockMax);
    runningSum = next.map((n) => n.sum);
    output = next.map((n) => n.out);
    steps.push({ start, stop, runningMax, runningSum, output });
  }
  return { output: output.map((row, i) => row.map((o) => o / runningSum[i])), steps };
}
