import { useState } from "preact/hooks";
import { mulberry32 } from "../lib/sampling.ts";
import { Slider, Stat } from "./ui.tsx";

const T = 24;
const random = mulberry32(9);
const KEYS = Array.from({ length: T }, (_, i) => Math.sin(i / 2.5) * 0.6 + (random() - 0.5) * 0.8);
const W = 640;
const H = 80;

export default function PoolBlocks() {
  const [block, setBlock] = useState(4);
  const [window, setWindow] = useState(6);
  const cut = T - window;
  const full = Math.floor(cut / block);
  const pooled = Array.from({ length: full }, (_, b) => KEYS.slice(b * block, b * block + block).reduce((s, v) => s + v, 0) / block);
  const bar = W / T;
  const y = (v: number) => H / 2 - v * (H / 2.4);
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Block size" value={block} min={2} max={8} onInput={setBlock} />
        <Slider label="Exact window" value={window} min={2} max={16} onInput={setWindow} />
      </div>
      <div class="diagram">
        <svg viewBox={`0 0 ${W} ${2 * H + 50}`} role="img" aria-label="Keys pooled into block means">
          <text class="t small" x="0" y="12">keys, one bar per position</text>
          {KEYS.map((v, i) => (
            <rect class={i >= cut ? "fill-1" : "fill-muted"} x={i * bar + 2} width={bar - 4} y={20 + Math.min(y(v), H / 2)} height={Math.abs(y(v) - H / 2)} rx="2" opacity={i >= cut || i < full * block ? 1 : 0.35} />
          ))}
          <text class="t small" x="0" y={H + 42}>what the last query reads</text>
          {pooled.map((v, b) => (
            <g>
              <rect class="fill-2" x={b * block * bar + 2} width={block * bar - 4} y={H + 50 + Math.min(y(v), H / 2)} height={Math.max(1, Math.abs(y(v) - H / 2))} rx="2" />
              <line class="dial-axis" x1={b * block * bar} x2={b * block * bar} y1="20" y2={2 * H + 50} />
            </g>
          ))}
          {KEYS.map((v, i) => i >= cut && <rect class="fill-1" x={i * bar + 2} width={bar - 4} y={H + 50 + Math.min(y(v), H / 2)} height={Math.abs(y(v) - H / 2)} rx="2" />)}
        </svg>
      </div>
      <div class="readout-grid">
        <Stat label="Columns the last query reads" value={full + window} note={`full causal: ${T}`} />
        <Stat label="Pooled blocks" value={full} note="orange, one mean each" />
        <Stat label="Positions lost" value={cut - full * block} note="a partial block is dropped" />
      </div>
      <p class="widget-note">Old keys are averaged in blocks, so the query sees their rough shape but not any single position. Recent keys, in blue, stay exact. A copy task that needs one old token exactly is the case pooling can blur, which is what the needle probe was built to test. The cache holds window + ⌊(T − window) / block⌋ positions instead of T.</p>
    </div>
  );
}
