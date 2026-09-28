import { scaleLinear } from "d3-scale";
import { useState } from "preact/hooks";
import { format } from "./format.ts";

export type RangeRow = { name: string; value: number; spread: number; note?: string };
type Props = { rows: RangeRow[]; label: string; highlight?: string[]; baseline?: string };

const WIDTH = 680;
const ROW = 38;
const M = { top: 12, right: 24, bottom: 44, left: 108 };

export default function RangeChart({ rows, label, highlight = [], baseline }: Props) {
  const [hover, setHover] = useState<string | null>(null);
  const height = M.top + rows.length * ROW + M.bottom;
  const lows = rows.map((r) => r.value - r.spread / 2);
  const highs = rows.map((r) => r.value + r.spread / 2);
  const sx = scaleLinear().domain([Math.min(...lows), Math.max(...highs)]).nice().range([M.left, WIDTH - M.right]);
  const base = rows.find((r) => r.name === baseline);
  const active = rows.find((r) => r.name === hover);
  return (
    <div class="chart">
      <div class="chart-scroll">
        <svg viewBox={`0 0 ${WIDTH} ${height}`} role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
          {base && (
            <g>
              <rect class="band" x={sx(base.value - base.spread / 2)} width={sx(base.value + base.spread / 2) - sx(base.value - base.spread / 2)} y={M.top} height={rows.length * ROW} />
              <text class="band-label" x={sx(base.value + base.spread / 2) + 4} y={M.top + 10}>{baseline} seed spread</text>
            </g>
          )}
          {sx.ticks(6).map((t) => (
            <g>
              <line class="grid" x1={sx(t)} x2={sx(t)} y1={M.top} y2={height - M.bottom} />
              <text class="tick" x={sx(t)} y={height - M.bottom + 18} text-anchor="middle">{format(t, "fixed2")}</text>
            </g>
          ))}
          <text class="axis-label" x={(M.left + WIDTH - M.right) / 2} y={height - 6} text-anchor="middle">{label}</text>
          {rows.map((r, i) => {
            const cy = M.top + i * ROW + ROW / 2;
            const on = highlight.length === 0 || highlight.includes(r.name);
            return (
              <g class={on ? "range-row" : "range-row dim"} onMouseEnter={() => setHover(r.name)}>
                <rect class="hit" x="0" y={cy - ROW / 2} width={WIDTH} height={ROW} />
                <text class="row-label" x={M.left - 12} y={cy + 4} text-anchor="end">{r.name}</text>
                <line class="range s1" x1={sx(r.value - r.spread / 2)} x2={sx(r.value + r.spread / 2)} y1={cy} y2={cy} />
                <circle class="marker s1" cx={sx(r.value)} cy={cy} r="5" />
              </g>
            );
          })}
        </svg>
      </div>
      <div class="chart-readout" aria-live="polite">
        {active ? (
          <>
            <b>{active.name}</b>
            <span>mean {format(active.value, "fixed4")}</span>
            <span>seed spread {format(active.spread, "fixed4")}</span>
            {base && active !== base && <span>vs {baseline} {format(active.value - base.value, "fixed4")}</span>}
            {active.note && <span>{active.note}</span>}
          </>
        ) : <span class="muted">Hover a row to read its numbers.</span>}
      </div>
    </div>
  );
}
