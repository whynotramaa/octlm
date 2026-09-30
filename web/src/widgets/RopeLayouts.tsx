import { useState } from "preact/hooks";
import { applyRope, applyRopeSplit, ropeAngles } from "../lib/positions.ts";
import { Segmented, Slider, Stat, fixed } from "./ui.tsx";

const HEAD = 8;
const Q = [0.9, -0.2, 0.4, 0.7, -0.5, 0.3, 0.8, -0.1];
const K = [0.6, 0.5, -0.3, 0.4, 0.2, -0.7, 0.5, 0.3];
const BOX = 56;
const GAP = 12;

const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const pairs = (split: boolean) => Array.from({ length: HEAD / 2 }, (_, i) => (split ? [i, i + HEAD / 2] : [2 * i, 2 * i + 1]));

function score(split: boolean, base: number, m: number, n: number) {
  const angles = ropeAngles(Math.max(m, n) + 1, HEAD, 1, base);
  const rotate = split ? applyRopeSplit : applyRope;
  return dot(rotate([Q], [angles[m]])[0], rotate([K], [angles[n]])[0]);
}

function Pairing({ split }: { split: boolean }) {
  const x = (i: number) => i * (BOX + GAP) + BOX / 2;
  return (
    <div class="diagram">
      <svg viewBox={`0 0 ${HEAD * (BOX + GAP) - GAP} 120`} role="img" aria-label={split ? "Channel i pairs with channel i plus 4" : "Channel 2i pairs with channel 2i plus 1"}>
        {pairs(split).map(([a, b], pair) => (
          <path class="link" d={`M ${x(a)} 70 Q ${(x(a) + x(b)) / 2} ${70 - (x(b) - x(a)) / 3 - 14} ${x(b)} 70`} fill="none" stroke={`var(--series-${(pair % 4) + 1})`} stroke-width="2" />
        ))}
        {Array.from({ length: HEAD }, (_, i) => (
          <g>
            <rect class="box" x={i * (BOX + GAP)} y="72" width={BOX} height="40" rx="8" />
            <text x={x(i)} y="97" text-anchor="middle" font-size="14">c{i}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export default function RopeLayouts() {
  const [split, setSplit] = useState(true);
  const [base, setBase] = useState(1e6);
  const [m, setM] = useState(12);
  const [n, setN] = useState(4);
  const trained = score(true, base, m, n);
  const ours = score(split, base, m, n);
  const slowest = 1 / Math.pow(base, (HEAD - 2) / HEAD);
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Segmented label="Layout octlm applies" value={split ? "split" : "adjacent"} options={[{ value: "adjacent", text: "Adjacent pairs" }, { value: "split", text: "Split halves" }]} onChange={(v) => setSplit(v === "split")} />
        <Segmented label="RoPE base" value={base} options={[{ value: 1e4, text: "10,000 (Day 4)" }, { value: 1e6, text: "1,000,000 (Qwen)" }]} onChange={setBase} />
        <Slider label="Query position m" value={m} min={0} max={64} onInput={setM} />
        <Slider label="Key position n" value={n} min={0} max={64} onInput={setN} />
      </div>
      <Pairing split={split} />
      <div class="readout-grid">
        <Stat label="Score Qwen was trained with" value={fixed(trained, 4)} note="split halves, same base" />
        <Stat label="Score octlm computes" value={fixed(ours, 4)} note={split ? "same layout" : "wrong channels paired"} />
        <Stat label="Difference" value={fixed(Math.abs(ours - trained), 4)} note={split ? "exact match" : "the model reads garbage"} />
        <Stat label="Slowest pair turns" value={`${slowest.toPrecision(2)} rad`} note="per position, last pair" />
      </div>
      <p class="widget-note">One 8-channel head with made-up query and key values. Both layouts rotate 4 pairs by the same angles. They differ only in which channels form a pair. Qwen's projection weights were trained with split halves, so adjacent pairing changes the attention scores even though every angle is right. A larger base makes the last pairs turn more slowly, which keeps positions tens of thousands of tokens apart distinguishable.</p>
    </div>
  );
}
