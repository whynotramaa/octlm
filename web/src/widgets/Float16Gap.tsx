import { useState } from "preact/hooks";
import { toFloat16 } from "../lib/floats.ts";
import { Slider, Stat } from "./ui.tsx";

const spacing = (value: number) => 2 ** (Math.floor(Math.log2(Math.abs(value))) - 10);

type Measured = { name: string; value: number };

export default function Float16Gap({ tolerance, measured }: { tolerance: number; measured: Measured[] }) {
  const [logit, setLogit] = useState(38);
  const step = spacing(logit);
  const marks = [1, 2, 4, 8, 16, 32, 64];
  const top = Math.max(...measured.map((m) => m.value), tolerance, spacing(64)) * 1.1;
  const sample = logit + step * 0.37;
  return (
    <div class="widget">
      <Slider label="Logit size" value={logit} min={1} max={64} step={0.5} shown={String(logit)} onInput={setLogit} />
      <div class="bars">
        {marks.map((m) => (
          <div class={`bar-row${spacing(m) > tolerance ? " leak" : ""}`} style={{ gridTemplateColumns: "96px 1fr 72px" }}>
            <span class="bar-name">{m} to {m * 2}</span>
            <span class="bar-track"><span class="bar-fill" style={{ width: `${(100 * spacing(m)) / top}%` }} /></span>
            <span class="bar-value">{spacing(m).toFixed(4)}</span>
          </div>
        ))}
        {measured.map((m) => (
          <div class="bar-row after" style={{ gridTemplateColumns: "96px 1fr 72px" }}>
            <span class="bar-name">{m.name}</span>
            <span class="bar-track"><span class="bar-fill" style={{ width: `${(100 * m.value) / top}%` }} /></span>
            <span class="bar-value">{m.value.toFixed(4)}</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="float16 step at this size" value={step.toFixed(4)} note={`gap between neighbouring values near ${logit}`} />
        <Stat label="Tolerance set before the run" value={tolerance.toFixed(2)} note={step > tolerance ? "smaller than one rounding step" : "larger than one rounding step"} />
        <Stat label="Example" value={`${sample.toFixed(5)} → ${toFloat16(sample).toFixed(5)}`} note="a float32 value stored as float16" />
      </div>
      <p class="widget-note">Each row is one power-of-two range of logit sizes and its float16 step, the distance between adjacent float16 values. Red rows have steps above the 0.05 tolerance. Blue rows are the measured merge differences on the T4. Qwen's largest logit on the gate prompt is about 38, in the 32 to 64 range, where one step is already 0.031.</p>
    </div>
  );
}
