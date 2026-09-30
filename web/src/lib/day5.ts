export function seedSpread(values: number[]) {
  if (!values.length) throw new Error("at least one seed is required");
  return { mean: values.reduce((sum, value) => sum + value, 0) / values.length, spread: Math.max(...values) - Math.min(...values) };
}

export function quantizeRow(weights: number[]) {
  const scale = Math.max(Math.max(...weights.map(Math.abs)), 1e-8) / 127;
  const roundEven = (value: number) => {
    const lower = Math.floor(value);
    const fraction = value - lower;
    return fraction === 0.5 ? (lower % 2 === 0 ? lower : lower + 1) : Math.round(value);
  };
  const codes = weights.map((weight) => Math.max(-127, Math.min(127, roundEven(weight / scale))));
  return { scale, codes, restored: codes.map((code) => code * scale) };
}
