import { useState } from "preact/hooks";
import { ROPE_BASE, rotatePair } from "../lib/positions.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

const HEAD = 8;
const R = 80;
const frequency = (pair: number) => 1 / Math.pow(ROPE_BASE, (2 * pair) / HEAD);
const Q: [number, number] = [0.9, 0.35];
const K: [number, number] = [0.6, -0.7];

function Dial({ vectors }: { vectors: { v: [number, number]; slot: number; name: string }[] }) {
  return (
    <svg viewBox="-110 -110 220 220" class="dial" role="img" aria-label="Rotated vectors">
      <circle r={R} class="dial-ring" />
      <line x1={-R - 10} x2={R + 10} y1="0" y2="0" class="dial-axis" />
      <line y1={-R - 10} y2={R + 10} x1="0" x2="0" class="dial-axis" />
      {vectors.map(({ v, slot, name }) => (
        <g class={`s${slot}`}>
          <line x1="0" y1="0" x2={v[0] * R} y2={-v[1] * R} class="dial-vec" />
          <circle cx={v[0] * R} cy={-v[1] * R} r="4" class="dial-dot" />
          <text x={v[0] * R * 1.18} y={-v[1] * R * 1.18 + 4} text-anchor="middle" class="dial-text">{name}</text>
        </g>
      ))}
    </svg>
  );
}

export default function RopeRotate() {
  const [pair, setPair] = useState(0);
  const [m, setM] = useState(3);
  const [n, setN] = useState(1);
  const [shift, setShift] = useState(0);
  const theta = frequency(pair);
  const q = rotatePair(...Q, (m + shift) * theta);
  const k = rotatePair(...K, (n + shift) * theta);
  const dot = q[0] * k[0] + q[1] * k[1];
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Channel pair i" value={pair} min={0} max={HEAD / 2 - 1} shown={`${pair}, θ = ${theta.toPrecision(3)}`} onInput={setPair} />
        <Slider label="Query position m" value={m} min={0} max={32} onInput={setM} />
        <Slider label="Key position n" value={n} min={0} max={32} onInput={setN} />
        <Slider label="Shift both by" value={shift} min={0} max={64} onInput={setShift} />
      </div>
      <div class="two-col">
        <Dial vectors={[{ v: q, slot: 1, name: "q" }, { v: k, slot: 2, name: "k" }]} />
        <div class="readout-grid" style={{ alignContent: "start" }}>
          <Stat label="Query angle" value={`${fixed((m + shift) * theta)} rad`} note="(m + shift) × θ" />
          <Stat label="Key angle" value={`${fixed((n + shift) * theta)} rad`} note="(n + shift) × θ" />
          <Stat label="Distance m − n" value={m - n} />
          <Stat label="q · k after rotation" value={fixed(dot, 4)} note="moves with m − n only" />
          <Stat label="|q|, |k|" value={`${fixed(Math.hypot(...q), 3)}, ${fixed(Math.hypot(...k), 3)}`} note="rotation keeps length" />
        </div>
      </div>
      <p class="widget-note">Drag "Shift both by". Both arrows turn, and the dot product does not change, because it depends on the angle between them. Change m or n and it does change. Pair 0 turns 1 radian per position. Pair 3 turns {frequency(3).toPrecision(2)} radians per position, so it separates far positions that pair 0 would wrap around.</p>
    </div>
  );
}
