import { useState } from "preact/hooks";
import { Slider, Stat, int } from "./ui.tsx";

const S = 1.1;

function Block({ x, y, w, h, label, dims, cls }: { x: number; y: number; w: number; h: number; label: string; dims: string; cls: string }) {
  return (
    <g>
      <rect class={`box ${cls}`} x={x} y={y} width={w} height={h} rx="4" />
      <text class="t" x={x + w / 2} y={y - 18} text-anchor="middle">{label}</text>
      <text class="t mono" x={x + w / 2} y={y - 5} text-anchor="middle">{dims}</text>
    </g>
  );
}

export default function AttentionShapes() {
  const [t, setT] = useState(64);
  const [d, setD] = useState(32);
  const T = t * S;
  const D = d * S;
  const top = 40;
  const qx = 10;
  const kx = qx + D + 30;
  const sx = kx + T + 44;
  const vx = sx + T + 30;
  const ox = vx + D + 44;
  const width = ox + D + 10;
  return (
    <div class="widget">
      <div class="diagram" style={{ overflowX: "auto" }}>
        <svg viewBox={`0 0 ${width} ${top + T + 10}`} style={{ minWidth: "420px" }} role="img" aria-label="Shapes of one attention head">
          <Block x={qx} y={top} w={D} h={T} label="Q" dims={`${t}×${d}`} cls="on" />
          <text class="t" x={kx - 15} y={top + T / 2} text-anchor="middle">×</text>
          <Block x={kx} y={top} w={T} h={D} label="Kᵀ" dims={`${d}×${t}`} cls="on" />
          <text class="t" x={sx - 22} y={top + T / 2} text-anchor="middle">→</text>
          <Block x={sx} y={top} w={T} h={T} label="scores, masked" dims={`${t}×${t}`} cls="warn" />
          <path class="box muted" d={`M ${sx + 1} ${top + 1} L ${sx + T - 1} ${top + 1} L ${sx + T - 1} ${top + T - 1} Z`} style={{ stroke: "none" }} />
          <text class="t" x={vx - 15} y={top + T / 2} text-anchor="middle">×</text>
          <Block x={vx} y={top} w={D} h={T} label="V" dims={`${t}×${d}`} cls="on" />
          <text class="t" x={ox - 22} y={top + T / 2} text-anchor="middle">→</text>
          <Block x={ox} y={top} w={D} h={T} label="out" dims={`${t}×${d}`} cls="good" />
        </svg>
      </div>
      <div class="grid-controls widget-controls">
        <Slider label="Sequence length T" value={t} min={8} max={160} step={8} onInput={setT} />
        <Slider label="Head width d" value={d} min={8} max={64} step={8} onInput={setD} />
      </div>
      <div class="readout-grid">
        <Stat label="Score entries" value={int(t * t)} note="T², the orange square" />
        <Stat label="Q, K, V entries" value={int(3 * t * d)} note="3 × T × d" />
        <Stat label="Multiply-adds" value={int(2 * t * t * d)} note="QKᵀ plus weights × V" />
      </div>
      <p class="widget-note">Q, K, V and the output grow in a straight line with T. The score square grows with T². Double the length and the square quadruples, whatever the head width. The gray triangle above the diagonal is the causal mask: positions it covers are computed and then thrown away.</p>
    </div>
  );
}
