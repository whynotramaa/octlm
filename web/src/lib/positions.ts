import type { Matrix } from "./matrix.ts";

export const ROPE_BASE = 10000;

export function sinusoidal(length: number, width: number): Matrix {
  return Array.from({ length }, (_, position) =>
    Array.from({ length: width }, (_, column) => {
      const angle = position / Math.pow(ROPE_BASE, (column - (column % 2)) / width);
      return column % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }),
  );
}

export function ropeAngles(length: number, headSize: number, scale = 1): Matrix {
  const frequencies = Array.from({ length: headSize / 2 }, (_, i) => 1 / Math.pow(ROPE_BASE, (2 * i) / headSize));
  return Array.from({ length }, (_, position) => frequencies.map((f) => (position / scale) * f));
}

export function ropeTables(length: number, headSize: number, scale = 1) {
  const angles = ropeAngles(length, headSize, scale);
  return { cos: angles.map((row) => row.map(Math.cos)), sin: angles.map((row) => row.map(Math.sin)) };
}

export function rotatePair(left: number, right: number, angle: number): [number, number] {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [left * c - right * s, left * s + right * c];
}

export function applyRope(x: Matrix, angles: Matrix): Matrix {
  return x.map((row, position) =>
    row.flatMap((_, column) => {
      if (column % 2) return [];
      return rotatePair(row[column], row[column + 1], angles[position][column / 2]);
    }),
  );
}
