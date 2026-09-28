import { useState } from "preact/hooks";
import { mulberry32 } from "../lib/sampling.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

const random = mulberry32(21);
const NORMS = Array.from({ length: 40 }, (_, i) => (i === 26 ? 6.4 : i === 11 ? 2.3 : 0.4 + random() * 0.9));
const TOP = 7;
const W = 640;
const H = 200;

export default function GradClip() {
  const [clip, setClip] = useState(1);
  const bar = W / NORMS.length;
  const y = (v: number) => H - (v / TOP) * H;
  const clipped = NORMS.filter((n) => n > clip).length;
  return (
    <div class="widget">
      <div class="diagram">
        <svg viewBox={`0 0 ${W} ${H + 24}`} role="img" aria-label="Gradient norm per step, before and after clipping">
          {NORMS.map((n, i) => (
            <g>
              <rect class="fill-muted" x={i * bar + 2} width={bar - 4} y={y(n)} height={H - y(n)} rx="2" />
              <rect class={n > clip ? "fill-2" : "fill-1"} x={i * bar + 2} width={bar - 4} y={y(Math.min(n, clip))} height={H - y(Math.min(n, clip))} rx="2" />
            </g>
          ))}
          <line class="rule" x1="0" x2={W} y1={y(clip)} y2={y(clip)} />
          <text class="t small" x={W - 4} y={y(clip) - 6} text-anchor="end">max norm {fixed(clip, 1)}</text>
          <text class="t small" x="0" y={H + 18}>step 1</text>
          <text class="t small" x={W} y={H + 18} text-anchor="end">step {NORMS.length}</text>
        </svg>
      </div>
      <Slider label="grad_clip" value={clip} min={0.2} max={7} step={0.1} shown={fixed(clip, 1)} onInput={setClip} />
      <div class="readout-grid">
        <Stat label="Steps clipped" value={`${clipped} of ${NORMS.length}`} />
        <Stat label="Spike at step 27" value={fixed(NORMS[26], 1)} note={`scaled by ${fixed(Math.min(1, clip / NORMS[26]), 2)}`} />
      </div>
      <p class="widget-note">Each bar is one step's global gradient norm. Made-up values with one spike. Clipping rescales the whole gradient so its norm is at most the threshold, which keeps its direction and only shortens it. Gray is what was cut off. Set the threshold too low and every step is shortened, which acts like a smaller learning rate.</p>
    </div>
  );
}
