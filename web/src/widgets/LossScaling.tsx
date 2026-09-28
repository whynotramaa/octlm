import { useMemo, useState } from "preact/hooks";
import { toFloat16 } from "../lib/floats.ts";
import { mulberry32 } from "../lib/sampling.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

const COUNT = 4000;

function gradients(center: number) {
  const random = mulberry32(5);
  return Array.from({ length: COUNT }, () => {
    const normal = Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
    return 10 ** (center + 1.3 * normal);
  });
}

function histogram(values: number[]) {
  const bins = Array.from({ length: 24 }, (_, i) => ({ low: -14 + i, count: 0 }));
  for (const v of values) {
    const index = Math.floor(Math.log10(v)) + 14;
    if (index >= 0 && index < bins.length) bins[index].count++;
  }
  return bins;
}

export default function LossScaling() {
  const [center, setCenter] = useState(-6);
  const [power, setPower] = useState(0);
  const grads = useMemo(() => gradients(center), [center]);
  const scale = 2 ** power;
  const scaled = grads.map((g) => g * scale);
  const stored = scaled.map(toFloat16);
  const zero = stored.filter((v) => v === 0).length / COUNT;
  const overflow = stored.filter((v) => !Number.isFinite(v)).length / COUNT;
  const bins = histogram(scaled);
  const peak = Math.max(...bins.map((b) => b.count));
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Typical gradient size" value={center} min={-9} max={-2} step={0.5} shown={`1e${center}`} onInput={setCenter} />
        <Slider label="Loss scale" value={power} min={0} max={24} shown={`2^${power} = ${scale.toLocaleString("en-US")}`} onInput={setPower} />
      </div>
      <div class="histogram" role="img" aria-label="Histogram of scaled gradient magnitudes">
        {bins.map((b) => {
          const zone = b.low <= -8 ? "under" : b.low >= 5 ? "over" : "ok";
          return <div class={`hist-bar ${zone}`} style={{ height: `${(b.count / peak) * 100}%` }} title={`1e${b.low}: ${b.count}`} />;
        })}
      </div>
      <div class="hist-axis"><span>1e−14</span><span>scaled gradient magnitude, one bar per power of ten</span><span>1e9</span></div>
      <div class="legend">
        <span class="legend-item"><i class="swatch" style={{ background: "color-mix(in oklch, var(--ink-3) 60%, var(--surface))" }} />below 1e−7: float16 rounds these to 0 or keeps one or two significant bits</span>
        <span class="legend-item"><i class="swatch" style={{ background: "var(--series-1)" }} />representable</span>
        <span class="legend-item"><i class="swatch" style={{ background: "var(--series-2)" }} />above 1e5: past float16's 65,504 maximum</span>
      </div>
      <div class="readout-grid">
        <Stat label="Lost to zero in float16" value={`${fixed(zero * 100, 1)}%`} />
        <Stat label="Overflow to inf" value={`${fixed(overflow * 100, 1)}%`} note={overflow > 0 ? "GradScaler skips this step and halves the scale" : "step is applied"} />
      </div>
      <p class="widget-note">Multiply the loss by the scale, and every gradient grows by the same factor, because the backward pass is linear in the loss. The scaler divides the gradients back down in float32 before the optimizer step. Slide the scale until the zero share falls without anything overflowing. GradScaler searches for that scale on its own: it starts at 2^16, halves on any inf, and doubles after 2,000 clean steps.</p>
    </div>
  );
}
