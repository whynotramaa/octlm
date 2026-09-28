import { useState } from "preact/hooks";
import LineChart from "../charts/LineChart.tsx";
import { hiddenSize } from "../lib/model.ts";
import { Segmented, Slider, Stat, fixed, int } from "./ui.tsx";

const silu = (x: number) => x / (1 + Math.exp(-x));
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const range = Array.from({ length: 81 }, (_, i) => -4 + i * 0.1);

export default function SwiGluGate() {
  const [gate, setGate] = useState(1.5);
  const [up, setUp] = useState(-0.8);
  const [width, setWidth] = useState(256);
  const geluHidden = hiddenSize({ d_model: width, n_layers: 1, n_heads: 1, feed_forward: "gelu" });
  const swigluHidden = hiddenSize({ d_model: width, n_layers: 1, n_heads: 1, feed_forward: "swiglu" });
  return (
    <div class="widget">
      <LineChart
        series={[
          { name: "ReLU", points: range.map((x) => [x, Math.max(0, x)]), slot: 3 },
          { name: "GELU", points: range.map((x) => [x, gelu(x)]), slot: 1 },
          { name: "SiLU (Swish)", points: range.map((x) => [x, silu(x)]), slot: 2, dashed: true },
        ]}
        x={{ label: "input", format: "fixed1" }}
        y={{ label: "output", format: "fixed1" }}
        height={260}
      />
      <div class="grid-controls widget-controls">
        <Slider label="gate(x) for one hidden unit" value={gate} min={-4} max={4} step={0.1} shown={fixed(gate, 1)} onInput={setGate} />
        <Slider label="up(x) for the same unit" value={up} min={-3} max={3} step={0.1} shown={fixed(up, 1)} onInput={setUp} />
      </div>
      <div class="readout-grid">
        <Stat label="SiLU(gate)" value={fixed(silu(gate), 3)} note="how open the gate is" />
        <Stat label="SiLU(gate) × up" value={fixed(silu(gate) * up, 3)} note="what reaches the down projection" />
        <Stat label="GELU(up), no gate" value={fixed(gelu(up), 3)} note="the Day 1 unit" />
      </div>
      <Segmented label="Model width d" value={width} options={[128, 256, 512].map((v) => ({ value: v, text: String(v) }))} onChange={setWidth} />
      <div class="readout-grid">
        <Stat label="GELU hidden" value={int(geluHidden)} note={`2 × ${width} × ${geluHidden} = ${int(2 * width * geluHidden)}`} />
        <Stat label="SwiGLU hidden" value={int(swigluHidden)} note={`3 × ${width} × ${swigluHidden} = ${int(3 * width * swigluHidden)}`} />
        <Stat label="Difference" value={`${fixed(((3 * swigluHidden) / (2 * geluHidden) - 1) * 100, 2)}%`} note="rounding to a multiple of 8" />
      </div>
    </div>
  );
}
