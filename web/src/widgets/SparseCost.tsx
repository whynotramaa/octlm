import { useState } from "preact/hooks";
import LineChart from "../charts/LineChart.tsx";
import { Slider, Stat, fixed, int } from "./ui.tsx";

const LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192];

function scores(length: number, window: number, stride: number) {
  let total = 0;
  for (let i = 0; i < length; i++) {
    total += Math.min(i + 1, window);
    if (stride && i >= window) total += Math.floor((i - window) / stride) + 1;
  }
  return total;
}

export default function SparseCost() {
  const [window, setWindow] = useState(256);
  const [stride, setStride] = useState(0);
  const full = (t: number) => (t * (t + 1)) / 2;
  const at = 4096;
  return (
    <div class="widget">
      <LineChart
        series={[
          { name: "full causal", points: LENGTHS.map((t) => [t, full(t)]), slot: 2 },
          { name: stride ? "window + stride" : "window", points: LENGTHS.map((t) => [t, scores(t, window, stride)]), slot: 1 },
        ]}
        x={{ label: "sequence length", log: true }}
        y={{ label: "scores per head", log: true, format: "exp" }}
        markers
        height={280}
      />
      <div class="grid-controls widget-controls">
        <Slider label="Window" value={window} min={32} max={1024} step={32} onInput={setWindow} />
        <Slider label="Stride (0 = none)" value={stride} min={0} max={512} step={32} onInput={setStride} />
      </div>
      <div class="readout-grid">
        <Stat label={`Scores at ${int(at)}`} value={scores(at, window, stride).toExponential(2)} note={`full: ${full(at).toExponential(2)}`} />
        <Stat label="Density" value={`${fixed((scores(at, window, stride) / full(at)) * 100, 1)}%`} />
      </div>
      <p class="widget-note">On log axes, full causal attention climbs with slope 2. A window climbs with slope 1 once the sequence is longer than the window: each query reads at most w keys. Global stride columns add back a quadratic term, T²/(2s), so a small stride gives up most of the saving at long lengths. Below the window length the two lines are the same, so a window only pays at lengths well past it.</p>
    </div>
  );
}
