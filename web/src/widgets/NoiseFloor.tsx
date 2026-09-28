import { useState } from "preact/hooks";
import { Segmented, Stat, fixed } from "./ui.tsx";

type Row = { variant: string; code: number; codeSpread: number; prose: number; proseSpread: number };

export default function NoiseFloor({ rows }: { rows: Row[] }) {
  const [a, setA] = useState(rows[0].variant);
  const [b, setB] = useState(rows.find((r) => r.variant === "swiglu")?.variant ?? rows[1].variant);
  const [metric, setMetric] = useState<"code" | "prose">("code");
  const pick = (name: string) => rows.find((r) => r.variant === name)!;
  const span = (r: Row) => ({ mean: r[metric], half: r[metric === "code" ? "codeSpread" : "proseSpread"] / 2 });
  const [x, y] = [span(pick(a)), span(pick(b))];
  const lo = Math.min(x.mean - x.half, y.mean - y.half);
  const hi = Math.max(x.mean + x.half, y.mean + y.half);
  const pad = (hi - lo) * 0.25 || 0.01;
  const px = (v: number) => 20 + ((v - lo + pad) / (hi - lo + 2 * pad)) * 600;
  const gap = Math.abs(x.mean - y.mean);
  const clear = gap > x.half + y.half;
  const select = (value: string, set: (v: string) => void, label: string) => (
    <label class="control">
      <span class="control-label">{label}</span>
      <select value={value} onChange={(e) => set(e.currentTarget.value)}>{rows.map((r) => <option value={r.variant}>{r.variant}</option>)}</select>
    </label>
  );
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        {select(a, setA, "Variant A")}
        {select(b, setB, "Variant B")}
        <Segmented label="Held-out text" value={metric} options={[{ value: "code", text: "Code" }, { value: "prose", text: "Prose" }]} onChange={setMetric} />
      </div>
      <div class="diagram">
        <svg viewBox="0 0 640 110" role="img" aria-label="Seed ranges of the two variants">
          {[x, y].map((s, i) => (
            <g>
              <rect class={i ? "fill-2" : "fill-1"} x={px(s.mean - s.half)} y={20 + i * 40} width={Math.max(2, px(s.mean + s.half) - px(s.mean - s.half))} height="14" rx="7" opacity="0.45" />
              <circle class={i ? "fill-2" : "fill-1"} cx={px(s.mean)} cy={27 + i * 40} r="6" />
              <text class="t small" x={px(s.mean)} y={14 + i * 40} text-anchor="middle">{i ? b : a} {fixed(s.mean, 4)}</text>
            </g>
          ))}
          <line class="dial-axis" x1="20" x2="620" y1="100" y2="100" />
          <text class="t small" x="20" y="96">better</text>
          <text class="t small" x="620" y="96" text-anchor="end">worse</text>
        </svg>
      </div>
      <div class="readout-grid">
        <Stat label="Gap between means" value={fixed(gap, 4)} note="bits per byte" />
        <Stat label="Half spreads added" value={fixed(x.half + y.half, 4)} />
      </div>
      <div class={`verdict tint hue-${clear && a !== b ? "green" : "orange"}`}>
        {a === b ? "Pick two different variants." : clear ? "The ranges do not overlap. The gap is bigger than the seed noise." : "The ranges overlap. Three seeds cannot tell these two apart."}
      </div>
    </div>
  );
}
