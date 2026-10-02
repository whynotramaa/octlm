export function advantages(rewards: number[], scaleByStd: boolean): number[] {
  const mean = rewards.reduce((sum, r) => sum + r, 0) / rewards.length;
  const centered = rewards.map((r) => r - mean);
  if (!scaleByStd) return centered;
  const std = Math.sqrt(centered.reduce((sum, c) => sum + c * c, 0) / (rewards.length - 1));
  return centered.map((c) => (std === 0 ? 0 : c / std));
}
