import { useState } from "preact/hooks";
import LineChart from "../charts/LineChart.tsx";
import { crossEntropy, perplexity } from "../lib/metrics.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

const ps = Array.from({ length: 100 }, (_, i) => 0.01 + i * 0.01);

export default function ProbabilityCurve() {
  const [p, setP] = useState(0.25);
  const loss = crossEntropy(p);
  return (
    <div class="widget">
      <LineChart
        series={[
          { name: "loss, nats (−ln p)", points: ps.map((x) => [x, crossEntropy(x)]), slot: 1 },
          { name: "bits (−log₂ p)", points: ps.map((x) => [x, crossEntropy(x) / Math.LN2]), slot: 2 },
          { name: "your p", points: [[p, 0], [p, 6.7]], slot: 4, dashed: true },
        ]}
        x={{ label: "probability of the correct token", format: "fixed1", domain: [0, 1] }}
        y={{ label: "penalty", format: "fixed1", domain: [0, 6.7] }}
        height={280}
      />
      <Slider label="p(correct)" value={p} min={0.01} max={1} step={0.01} shown={fixed(p, 2)} onInput={setP} />
      <div class="readout-grid">
        <Stat label="Loss" value={`${fixed(loss, 3)} nats`} />
        <Stat label="Bits" value={fixed(loss / Math.LN2, 3)} />
        <Stat label="Perplexity" value={fixed(perplexity(loss), 2)} note="1 / p" />
        <Stat label="Halving p costs" value={`+${fixed(Math.LN2, 3)} nats`} note="exactly one bit, at any p" />
      </div>
      <p class="widget-note">The curve is steep near 0 and flat near 1. A model that gives the right token 1 percent pays 4.6 nats. Raising that to 2 percent saves as much loss as raising 50 percent to 100 percent. The loss punishes confident mistakes far more than it rewards extra confidence.</p>
    </div>
  );
}
