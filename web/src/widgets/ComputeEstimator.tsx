import { useState } from "preact/hooks";
import { Segmented, Slider, Stat, fixed, int } from "./ui.tsx";

type Preset = { name: string; parameters: number; tokens: number };

const hours = (flops: number, tflops: number) => flops / (tflops * 1e12) / 3600;
const human = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${fixed(h, 1)} h`);

export default function ComputeEstimator({ presets, measured, assumed }: { presets: Preset[]; measured: number; assumed: number }) {
  const [p, setP] = useState(presets[0]);
  const [rate, setRate] = useState(measured);
  const flops = 6 * p.parameters * p.tokens;
  const ratio = p.tokens / p.parameters;
  return (
    <div class="widget">
      <div class="widget-controls">
        <div class="segmented" role="group" aria-label="Run">
          {presets.map((preset) => <button type="button" class={preset === p ? "on" : ""} onClick={() => setP(preset)}>{preset.name}</button>)}
        </div>
      </div>
      <div class="grid-controls widget-controls">
        <Slider label="Parameters" value={Math.log10(p.parameters)} min={5} max={9} step={0.01} shown={`${fixed(p.parameters / 1e6, 1)}M`} onInput={(v) => setP({ ...p, name: "custom", parameters: 10 ** v })} />
        <Slider label="Training tokens" value={Math.log10(p.tokens)} min={5} max={10} step={0.01} shown={`${fixed(p.tokens / 1e6, 1)}M`} onInput={(v) => setP({ ...p, name: "custom", tokens: 10 ** v })} />
        <Segmented label="T4 throughput" value={rate} options={[{ value: measured, text: `${measured} TFLOPs measured` }, { value: assumed, text: `${assumed} assumed` }]} onChange={setRate} />
      </div>
      <div class="readout-grid">
        <Stat label="Training compute" value={`${flops.toExponential(1)} FLOPs`} note="6 × N × D" />
        <Stat label="Tokens per parameter" value={fixed(ratio, ratio < 1 ? 2 : 1)} note="compute-optimal is about 20" />
        <Stat label="Share of the optimal budget" value={`${fixed(Math.min(ratio / 20, 99) * 100, 0)}%`} />
        <Stat label="T4 time" value={human(hours(flops, rate))} note={`at ${rate} TFLOPs, before overhead`} />
        <Stat label="Optimal tokens" value={`${int((20 * p.parameters) / 1e6)}M`} note={`${human(hours(6 * p.parameters * 20 * p.parameters, rate))} on one T4`} />
      </div>
    </div>
  );
}
