export function attentionMask(length: number, window: number, stride: number): boolean[][] {
  return Array.from({ length }, (_, row) =>
    Array.from({ length }, (_, column) => {
      const distance = row - column;
      if (distance < 0) return false;
      if (!window) return true;
      return distance < window || (stride > 0 && column % stride === 0);
    }),
  );
}

export function compressedMask(length: number, window: number, block: number): boolean[][] {
  const exact = attentionMask(length, window, 0);
  const blocks = Math.floor(length / block);
  return exact.map((row, position) => [
    ...Array.from({ length: blocks }, (_, b) => (b + 1) * block <= position - window + 1),
    ...row,
  ]);
}

function count(mask: boolean[][]): number {
  return mask.reduce((sum, row) => sum + row.filter(Boolean).length, 0);
}

export function maskDensity(length: number, window: number, stride: number): number {
  return count(attentionMask(length, window, stride)) / count(attentionMask(length, 0, 0));
}
