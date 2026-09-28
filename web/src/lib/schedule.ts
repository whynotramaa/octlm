export type Schedule = { learning_rate: number; min_learning_rate: number; warmup_steps: number; steps: number };

export function learningRate(step: number, s: Schedule): number {
  if (step < s.warmup_steps) return (s.learning_rate * (step + 1)) / Math.max(1, s.warmup_steps);
  const progress = (step - s.warmup_steps) / Math.max(1, s.steps - s.warmup_steps);
  const cosine = 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1)));
  return s.min_learning_rate + cosine * (s.learning_rate - s.min_learning_rate);
}
