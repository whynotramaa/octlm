import { useState } from "preact/hooks";
import Heatmap from "../charts/Heatmap.tsx";
import { attentionMask, maskDensity } from "../lib/masks.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

export default function MaskBuilder() {
  const [length, setLength] = useState(16);
  const [window, setWindow] = useState(4);
  const [stride, setStride] = useState(0);
  const mask = attentionMask(length, window, window ? stride : 0);
  const matrix = mask.map((row) => row.map((allowed) => (allowed ? 1 : -Infinity)));
  const density = maskDensity(length, window, window ? stride : 0);
  const farthest = mask[length - 1].indexOf(true);
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Sequence length" value={length} min={8} max={24} onInput={(v) => { setLength(v); setWindow(Math.min(window, v)); }} />
        <Slider label="Window (0 = full causal)" value={window} min={0} max={length} onInput={setWindow} />
        <Slider label="Stride (global columns)" value={stride} min={0} max={8} onInput={setStride} />
      </div>
      <div class="maps maps-1">
        <Heatmap title="attention_mask(T, window, stride)" matrix={matrix} mode="weight" rowLabel={(i) => `q${i}`} columnLabel={(j) => `${j}`} />
      </div>
      <div class="readout-grid">
        <Stat label="Density" value={`${fixed(density * 100, 1)}%`} note="kept share of the causal triangle" />
        <Stat label="Scores per row, last position" value={mask[length - 1].filter(Boolean).length} note={`full causal: ${length}`} />
        <Stat label="Oldest key the last row reads" value={farthest} />
      </div>
      <p class="widget-note">A window keeps the {window || "all"} most recent keys per query. With a stride, every column divisible by it stays visible to all later rows, a cheap long-range path. Any mask other than plain causal makes PyTorch's SDPA leave the flash kernel, which Day 3 would have measured.</p>
    </div>
  );
}
