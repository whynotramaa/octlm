import { useState } from "preact/hooks";
import { seedSpread } from "../lib/day5.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

export default function SpreadDecision() {
  const [gap, setGap] = useState(0.02);
  const [noise, setNoise] = useState(0.04);
  const baseline = [0.5 - noise / 2, 0.5, 0.5 + noise / 2];
  const modern = baseline.map((value) => value - gap);
  const control = seedSpread(baseline);
  const candidate = seedSpread(modern);
  const threshold = Math.max(control.spread, candidate.spread);
  const clear = control.mean - candidate.mean > threshold;
  const x = (value: number) => 30 + ((value - 0.37) / 0.18) * 540;
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Modern improvement" value={gap} min={0} max={0.08} step={0.005} shown={fixed(gap, 3)} onInput={setGap} />
        <Slider label="Three-seed spread" value={noise} min={0.005} max={0.08} step={0.005} shown={fixed(noise, 3)} onInput={setNoise} />
      </div>
      <div class="diagram" style={{ overflowX: "auto" }}>
        <svg viewBox="0 0 630 184" style={{ minWidth: "480px" }} role="img" aria-label="Illustrative baseline and modern validation scores across three seeds">
          {[baseline, modern].map((values, row) => (
            <g>
              <text class="t" x="30" y={25 + row * 60}>{row ? "modern" : "baseline"}</text>
              <line class="link" x1={x(Math.min(...values))} x2={x(Math.max(...values))} y1={48 + row * 60} y2={48 + row * 60} />
              {values.map((value) => <circle cx={x(value)} cy={48 + row * 60} r="6" fill={row ? "var(--series-2)" : "var(--series-1)"} />)}
              <text class="t small" x="590" y={25 + row * 60} text-anchor="end">mean {fixed(row ? candidate.mean : control.mean, 3)}</text>
            </g>
          ))}
          <line class="link" x1="30" x2="570" y1="139" y2="139" />
          {[0.38, 0.46, 0.54].map((value) => (
            <g><line class="link" x1={x(value)} x2={x(value)} y1="135" y2="143" /><text class="t small mono" x={x(value)} y="160" text-anchor="middle">{fixed(value, 2)}</text></g>
          ))}
          <text class="t small" x="30" y="180">← better bits per byte</text>
        </svg>
      </div>
      <div class="readout-grid">
        <Stat label="Gap between means" value={fixed(control.mean - candidate.mean, 3)} note="lower is better" />
        <Stat label="Minimum detectable effect" value={fixed(threshold, 3)} note="larger three-seed range" />
        <Stat label="Decision rule" value={clear ? "Gap clears spread" : "Unresolved"} />
      </div>
      <p class="widget-note">Illustrative scores, not Day 5 results. The six trained runs below use the same rule. Change the spread while holding the mean gap fixed: a smaller gap can only be called a win once it rises above the observed seed range.</p>
    </div>
  );
}
