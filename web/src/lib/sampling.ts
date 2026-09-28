import { softmax } from "./matrix.ts";

export function samplingDistribution(logits: number[], temperature: number, topK: number): number[] {
  let kept = logits;
  if (topK) {
    const cutoff = [...logits].sort((a, b) => b - a)[Math.min(topK, logits.length) - 1];
    kept = logits.map((value) => (value < cutoff ? -Infinity : value));
  }
  return softmax(kept.map((value) => value / temperature));
}

export function draw(probabilities: number[], uniform: number): number {
  let total = 0;
  for (let index = 0; index < probabilities.length; index++) {
    total += probabilities[index];
    if (uniform < total) return index;
  }
  return probabilities.length - 1;
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
