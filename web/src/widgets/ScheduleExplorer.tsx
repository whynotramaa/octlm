import { useState } from "preact/hooks";
import LineChart from "../charts/LineChart.tsx";
import { type Schedule, learningRate } from "../lib/schedule.ts";
import { Slider, Stat } from "./ui.tsx";

type Preset = { name: string; schedule: Schedule };

export default function ScheduleExplorer({ presets }: { presets: Preset[] }) {
  const [s, setS] = useState<Schedule>(presets[0].schedule);
  const count = 120;
  const points: [number, number][] = Array.from({ length: count + 1 }, (_, i) => {
    const step = Math.round((i / count) * s.steps);
    return [step, learningRate(step, s)];
  });
  const half = Math.round((s.steps + s.warmup_steps) / 2);
  return (
    <div class="widget">
      <div class="widget-controls">
        <div class="segmented" role="group" aria-label="Preset">
          {presets.map((p) => <button type="button" class={p.schedule === s ? "on" : ""} onClick={() => setS(p.schedule)}>{p.name}</button>)}
        </div>
      </div>
      <div class="grid-controls widget-controls">
        <Slider label="Warmup steps" value={s.warmup_steps} min={0} max={Math.round(s.steps / 2)} onInput={(v) => setS({ ...s, warmup_steps: v })} />
        <Slider label="Peak rate" value={s.learning_rate} min={0.0001} max={0.002} step={0.0001} shown={s.learning_rate.toExponential(1)} onInput={(v) => setS({ ...s, learning_rate: v })} />
        <Slider label="Floor, share of peak" value={s.min_learning_rate / s.learning_rate} min={0} max={1} step={0.05} shown={`${Math.round((s.min_learning_rate / s.learning_rate) * 100)}%`} onInput={(v) => setS({ ...s, min_learning_rate: v * s.learning_rate })} />
      </div>
      <LineChart series={[{ name: "learning rate", points }]} x={{ label: "step" }} y={{ label: "learning rate", format: "rate", domain: [0, s.learning_rate * 1.05] }} height={280} />
      <div class="readout-grid">
        <Stat label="Step 0" value={learningRate(0, s).toExponential(2)} note="peak × 1 / warmup" />
        <Stat label={`Step ${s.warmup_steps}`} value={learningRate(s.warmup_steps, s).toExponential(2)} note="peak" />
        <Stat label={`Step ${half}`} value={learningRate(half, s).toExponential(2)} note="halfway down the cosine" />
        <Stat label={`Step ${s.steps}`} value={learningRate(s.steps, s).toExponential(2)} note="floor" />
      </div>
    </div>
  );
}
