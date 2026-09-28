import { useId, useState } from "preact/hooks";
import type { Matrix } from "../lib/matrix.ts";

type Props = {
  matrix: Matrix;
  mode?: "signed" | "weight";
  rowLabel?: (row: number) => string;
  columnLabel?: (column: number) => string;
  showValues?: boolean;
  highlightRow?: number | null;
  onRow?: (row: number | null) => void;
  title?: string;
  maxAbs?: number;
};

const CELL = 28;
const GUTTER = 26;

function fill(value: number, mode: "signed" | "weight", scale: number, pattern: string): string {
  if (value === -Infinity) return `url(#${pattern})`;
  const strength = Math.min(1, Math.abs(value) / scale);
  const tone = mode === "weight" ? "--heat-weight" : value < 0 ? "--heat-negative" : "--heat-positive";
  return `color-mix(in oklch, var(${tone}) ${Math.round(strength * 88)}%, var(--heat-empty))`;
}

function format(value: number): string {
  if (value === -Infinity) return "−∞";
  const text = Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(2);
  return text.replace("-", "−").replace(/^0\./, ".").replace(/^−0\./, "−.");
}

export default function Heatmap(props: Props) {
  const { matrix, mode = "signed", showValues = false, highlightRow = null, onRow, title } = props;
  const [hover, setHover] = useState<[number, number] | null>(null);
  const pattern = `masked-${useId()}`;
  const rows = matrix.length;
  const cols = matrix[0].length;
  const finite = matrix.flat().filter(Number.isFinite);
  const scale = props.maxAbs ?? (mode === "weight" ? 1 : Math.max(1e-9, ...finite.map(Math.abs)));
  const width = GUTTER + cols * CELL;
  const height = GUTTER + rows * CELL;
  const readout = hover ? `[${hover[0]}, ${hover[1]}] = ${format(matrix[hover[0]][hover[1]])}` : `${rows} × ${cols}`;
  return (
    <div class="heatmap" style={{ maxWidth: `${Math.round(width * 1.35)}px` }}>
      <div class="heatmap-head">
        {title && <span class="heatmap-title">{title}</span>}
        <span class="heatmap-readout">{readout}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title ?? "matrix"}, ${rows} by ${cols}`} onMouseLeave={() => { setHover(null); onRow?.(null); }}>
        <defs>
          <pattern id={pattern} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="var(--surface-2)" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--line-2)" stroke-width="2" />
          </pattern>
        </defs>
        {Array.from({ length: cols }, (_, c) => (
          <text x={GUTTER + c * CELL + CELL / 2} y={GUTTER - 9} class="axis" text-anchor="middle">{props.columnLabel?.(c) ?? c}</text>
        ))}
        {matrix.map((row, r) => (
          <g key={r} class={highlightRow !== null && highlightRow !== r ? "dim" : undefined}>
            <text x={GUTTER - 8} y={GUTTER + r * CELL + CELL / 2 + 4} class="axis" text-anchor="end">{props.rowLabel?.(r) ?? r}</text>
            {row.map((value, c) => (
              <g key={c} onMouseEnter={() => { setHover([r, c]); onRow?.(r); }}>
                <rect x={GUTTER + c * CELL + 1} y={GUTTER + r * CELL + 1} width={CELL - 2} height={CELL - 2} rx="4" style={{ fill: fill(value, mode, scale, pattern) }} />
                {showValues && (
                  <text x={GUTTER + c * CELL + CELL / 2} y={GUTTER + r * CELL + CELL / 2 + 3.5} class="value" text-anchor="middle">{format(value)}</text>
                )}
              </g>
            ))}
          </g>
        ))}
      </svg>
    </div>
  );
}
