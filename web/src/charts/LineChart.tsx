import { scaleLinear, scaleLog } from "d3-scale";
import { line } from "d3-shape";
import { useState } from "preact/hooks";
import { type FormatKey, format } from "./format.ts";

export type Series = { name: string; points: [number, number][]; slot?: 1 | 2 | 3 | 4; dashed?: boolean };
type Axis = { label: string; log?: boolean; format?: FormatKey; domain?: [number, number]; ticks?: number[] };
type Props = { series: Series[]; x: Axis; y: Axis; height?: number; markers?: boolean; title?: string; bands?: { from: number; to: number; label: string }[] };

const WIDTH = 680;
const M = { top: 20, right: 116, bottom: 46, left: 64 };

function scale(axis: Axis, values: number[], range: [number, number]) {
  const domain = axis.domain ?? [Math.min(...values), Math.max(...values)];
  const s = axis.log ? scaleLog().base(2).domain(domain) : scaleLinear().domain(domain);
  return (axis.log || axis.domain ? s : s.nice()).range(range);
}

function nearest(points: [number, number][], x: number) {
  return points.reduce((best, p) => (Math.abs(p[0] - x) < Math.abs(best[0] - x) ? p : best), points[0]);
}

export default function LineChart({ series, x, y, height = 320, markers = false, title, bands = [] }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const all = series.flatMap((s) => s.points);
  const sx = scale(x, all.map((p) => p[0]), [M.left, WIDTH - M.right]);
  const sy = scale(y, all.map((p) => p[1]), [height - M.bottom, M.top]);
  const xs = [...new Set(all.map((p) => p[0]))].sort((a, b) => a - b);
  const xTicks = x.ticks ?? (x.log ? xs : sx.ticks(6));
  const yTicks = y.ticks ?? sy.ticks(5);
  const path = line<[number, number]>().x((p) => sx(p[0])).y((p) => sy(p[1]));
  const move = (event: MouseEvent) => {
    const svg = event.currentTarget as SVGSVGElement;
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * WIDTH;
    setHover(nearest(xs.map((v) => [v, 0]), sx.invert(px))[0]);
  };
  const labels = series.map((s) => ({ s, end: s.points[s.points.length - 1] }));
  return (
    <div class="chart">
      {title && <div class="chart-title">{title}</div>}
      {series.length > 1 && (
        <div class="legend">
          {series.map((s) => <span class="legend-item"><i class={`swatch s${s.slot ?? 1}${s.dashed ? " dashed" : ""}`} />{s.name}</span>)}
        </div>
      )}
      <div class="chart-scroll">
        <svg viewBox={`0 0 ${WIDTH} ${height}`} role="img" aria-label={title ?? `${y.label} against ${x.label}`} onMouseMove={move} onMouseLeave={() => setHover(null)}>
          {bands.map((b) => (
            <g>
              <rect class="band" x={M.left} width={WIDTH - M.left - M.right} y={sy(Math.max(b.from, b.to))} height={Math.abs(sy(b.from) - sy(b.to))} />
              <text class="band-label" x={M.left + 6} y={sy(Math.max(b.from, b.to)) + 12}>{b.label}</text>
            </g>
          ))}
          {yTicks.map((t) => (
            <g>
              <line class="grid" x1={M.left} x2={WIDTH - M.right} y1={sy(t)} y2={sy(t)} />
              <text class="tick" x={M.left - 8} y={sy(t) + 4} text-anchor="end">{format(t, y.format)}</text>
            </g>
          ))}
          {xTicks.map((t) => <text class="tick" x={sx(t)} y={height - M.bottom + 18} text-anchor="middle">{format(t, x.format ?? "int")}</text>)}
          <text class="axis-label" x={(M.left + WIDTH - M.right) / 2} y={height - 6} text-anchor="middle">{x.label}</text>
          <text class="axis-label" transform={`translate(14 ${(M.top + height - M.bottom) / 2}) rotate(-90)`} text-anchor="middle">{y.label}</text>
          {series.map((s) => <path class={`series s${s.slot ?? 1}${s.dashed ? " dashed" : ""}`} d={path(s.points) ?? ""} />)}
          {markers && series.flatMap((s) => s.points.map((p) => <circle class={`marker s${s.slot ?? 1}`} cx={sx(p[0])} cy={sy(p[1])} r="4" />))}
          {series.length <= 4 && labels.map(({ s, end }) => (
            <text class="direct" x={sx(end[0]) + 8} y={sy(end[1]) + 4}>{s.name}</text>
          ))}
          {hover !== null && (
            <g class="crosshair">
              <line x1={sx(hover)} x2={sx(hover)} y1={M.top} y2={height - M.bottom} />
              {series.map((s) => {
                const p = nearest(s.points, hover);
                return p[0] === hover ? <circle class={`marker s${s.slot ?? 1}`} cx={sx(p[0])} cy={sy(p[1])} r="5" /> : null;
              })}
            </g>
          )}
        </svg>
      </div>
      <div class="chart-readout" aria-live="polite">
        {hover === null ? <span class="muted">Hover the chart to read values.</span> : (
          <>
            <b>{x.label} {format(hover, x.format ?? "int")}</b>
            {series.map((s) => {
              const p = nearest(s.points, hover);
              return p[0] === hover ? <span><i class={`swatch s${s.slot ?? 1}`} />{s.name} {format(p[1], y.format)}</span> : null;
            })}
          </>
        )}
      </div>
    </div>
  );
}
