import { useState } from "preact/hooks";
import { Slider, Stat, fixed, int } from "./ui.tsx";

const D = 64;

export default function MemoryTraffic() {
  const [log, setLog] = useState(11);
  const [block, setBlock] = useState(64);
  const n = 2 ** log;
  const naive = 4 * n * D + 4 * n * n;
  const flash = 2 * n * D + 2 * n * D * Math.ceil(n / block);
  const top = Math.max(naive, flash);
  const rows = [
    { name: "standard", value: naive, cls: "leak", note: "Q, K, V, O plus S and P written and read" },
    { name: "tiled", value: flash, cls: "after", note: "K and V reread once per query block" },
  ];
  return (
    <div class="widget">
      <div class="diagram">
        <svg viewBox="0 0 640 150" role="img" aria-label="GPU memory hierarchy">
          <rect class="box muted" x="10" y="20" width="250" height="110" rx="12" />
          <text class="t" x="135" y="50" text-anchor="middle">HBM, large and slow</text>
          <text class="t small" x="135" y="72" text-anchor="middle">16 GB on a T4</text>
          <text class="t small" x="135" y="92" text-anchor="middle">Q, K, V, output{naive > flash ? ", and S, P if standard" : ""}</text>
          <rect class="box on" x="420" y="35" width="210" height="80" rx="12" />
          <text class="t" x="525" y="65" text-anchor="middle">on-chip SRAM, small, fast</text>
          <text class="t small" x="525" y="87" text-anchor="middle">one {block}-row tile at a time</text>
          <path class="link" d="M 262 62 H 416" marker-end="url(#mt-arrow)" />
          <path class="link" d="M 418 92 H 264" marker-end="url(#mt-arrow)" />
          <text class="t small" x="340" y="54" text-anchor="middle">read</text>
          <text class="t small" x="340" y="110" text-anchor="middle">write</text>
          <defs><marker id="mt-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="arrowhead" d="M0 0 L10 5 L0 10 z" /></marker></defs>
        </svg>
      </div>
      <div class="grid-controls widget-controls">
        <Slider label="Sequence length" value={log} min={8} max={15} shown={int(n)} onInput={setLog} />
        <Slider label="Query block rows" value={block} min={16} max={256} step={16} onInput={setBlock} />
      </div>
      <div class="bars">
        {rows.map((r) => (
          <div class={`bar-row ${r.cls}`} style={{ gridTemplateColumns: "76px 1fr 96px" }}>
            <span class="bar-name">{r.name}</span>
            <span class="bar-track"><span class="bar-fill" style={{ width: `${(r.value / top) * 100}%` }} /></span>
            <span class="bar-value">{(r.value / 1e6).toFixed(1)}M</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        {rows.map((r) => <Stat label={`${r.name}, elements moved`} value={`${fixed(r.value / 1e6, 1)}M`} note={r.note} />)}
        <Stat label="Ratio" value={`${fixed(naive / flash, 1)}×`} note="standard ÷ tiled" />
      </div>
      <p class="widget-note">Element counts for one head of width {D}, simplified from the FlashAttention paper's analysis. Standard attention writes the T × T score matrix to HBM, reads it back for the softmax, writes the probabilities and reads them again. The tiled version never stores the square. It pays by rereading K and V once per query block, which is why larger tiles, if SRAM fits them, move less.</p>
    </div>
  );
}
