import { useState } from "preact/hooks";
import { Segmented, Slider, Stat, fixed } from "./ui.tsx";

const EPSILON = 1e-6;
const mean = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length;

export function layerNorm(x: number[]) {
  const m = mean(x);
  const variance = mean(x.map((v) => (v - m) ** 2));
  return x.map((v) => (v - m) / Math.sqrt(variance + 1e-5));
}

export function rmsNorm(x: number[]) {
  const rms = Math.sqrt(mean(x.map((v) => v * v)) + EPSILON);
  return x.map((v) => v / rms);
}

function Bars({ title, values }: { title: string; values: number[] }) {
  const peak = 3;
  return (
    <div>
      <div class="chart-title" style={{ marginBottom: "8px" }}>{title}</div>
      <div class="diverging">
        {values.map((v) => (
          <div class="div-row">
            <span class="div-track"><span class={v < 0 ? "div-fill neg" : "div-fill"} style={{ width: `${Math.min(50, (Math.abs(v) / peak) * 50)}%`, [v < 0 ? "right" : "left"]: "50%" }} /></span>
            <span class="bar-value">{fixed(v)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function NormPlay({ initial = "both" }: { initial?: "layernorm" | "rmsnorm" | "both" }) {
  const [x, setX] = useState([1.2, -0.4, 2.1, 0.3, -1.0, 0.8]);
  const [view, setView] = useState(initial);
  const m = mean(x);
  const rms = Math.sqrt(mean(x.map((v) => v * v)));
  const std = Math.sqrt(mean(x.map((v) => (v - m) ** 2)));
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        {x.map((v, i) => <Slider label={`x[${i}]`} value={v} min={-3} max={3} step={0.1} shown={fixed(v, 1)} onInput={(n) => setX(x.map((w, j) => (j === i ? n : w)))} />)}
      </div>
      <div class="widget-controls">
        <button type="button" class="button" onClick={() => setX(x.map((v) => Math.min(3, v + 1)))}>Add 1 to every entry</button>
        <button type="button" class="button" onClick={() => setX(x.map((v) => Math.max(-3, Math.min(3, v * 2))))}>Double every entry</button>
        <Segmented label="Show" value={view} options={[{ value: "both", text: "Both" }, { value: "layernorm", text: "LayerNorm" }, { value: "rmsnorm", text: "RMSNorm" }]} onChange={setView} />
      </div>
      <div class={view === "both" ? "maps maps-3" : "maps maps-2"}>
        <Bars title="input x" values={x} />
        {view !== "rmsnorm" && <Bars title="LayerNorm(x), gain 1, bias 0" values={layerNorm(x)} />}
        {view !== "layernorm" && <Bars title="RMSNorm(x), gain 1" values={rmsNorm(x)} />}
      </div>
      <div class="readout-grid">
        <Stat label="mean(x)" value={fixed(m)} note="LayerNorm subtracts it" />
        <Stat label="std(x)" value={fixed(std)} note="LayerNorm divides by it" />
        <Stat label="rms(x)" value={fixed(rms)} note="RMSNorm divides by it" />
        <Stat label="mean of RMSNorm(x)" value={fixed(mean(rmsNorm(x)))} note="not forced to 0" />
      </div>
    </div>
  );
}
