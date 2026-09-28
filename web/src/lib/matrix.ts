export type Matrix = number[][];

export function matmul(a: Matrix, b: Matrix): Matrix {
  return a.map((row) => b[0].map((_, column) => row.reduce((sum, value, k) => sum + value * b[k][column], 0)));
}

export function transpose(a: Matrix): Matrix {
  return a[0].map((_, column) => a.map((row) => row[column]));
}

export function linear(x: Matrix, weight: Matrix): Matrix {
  return matmul(x, transpose(weight));
}

export function columns(a: Matrix, start: number, stop: number): Matrix {
  return a.map((row) => row.slice(start, stop));
}

export function softmax(row: number[]): number[] {
  const peak = Math.max(...row);
  if (peak === -Infinity) return row.map(() => 0);
  const weights = row.map((value) => Math.exp(value - peak));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map((value) => value / total);
}

export function maxAbsDifference(a: Matrix, b: Matrix): number {
  return Math.max(...a.flatMap((row, i) => row.map((value, j) => Math.abs(value - b[i][j]))));
}
