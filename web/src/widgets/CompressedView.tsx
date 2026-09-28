import { useState } from "preact/hooks";
import Heatmap from "../charts/Heatmap.tsx";
import { compressedMask } from "../lib/masks.ts";
import { kvCacheBytes } from "../lib/model.ts";
import { Segmented, Slider, Stat } from "./ui.tsx";

const SHAPE = { n_layers: 4, d_model: 256, n_heads: 8, kv_heads: 2 };

export default function CompressedView() {
  const [window, setWindow] = useState(4);
  const [block, setBlock] = useState(4);
  const length = 16;
  const mask = compressedMask(length, window, block);
  const blocks = Math.floor(length / block);
  const matrix = mask.map((row) => row.map((allowed) => (allowed ? 1 : -Infinity)));
  const cached = (L: number) => kvCacheBytes({ ...SHAPE, attention_window: 256, kv_compress_block: block }, L);
  const plain = (L: number) => kvCacheBytes(SHAPE, L);
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Exact window" value={window} min={1} max={12} onInput={setWindow} />
        <Segmented label="Block size" value={block} options={[2, 4, 8].map((v) => ({ value: v, text: String(v) }))} onChange={setBlock} />
      </div>
      <div class="maps maps-1">
        <Heatmap title={`columns 0–${blocks - 1}: pooled blocks, then 16 exact keys`} matrix={matrix} mode="weight" rowLabel={(i) => `q${i}`} columnLabel={(j) => (j < blocks ? `b${j}` : `${j - blocks}`)} />
      </div>
      <div class="readout-grid">
        <Stat label="Pooled entries" value={blocks} note={`mean of ${block} keys each`} />
        <Stat label="Cache at 4K, GQA-2" value={`${(plain(4096) / 2 ** 20).toFixed(2)} MiB`} note="every position cached" />
        <Stat label={`Cache at 4K, window 256, block ${block}`} value={`${(cached(4096) / 2 ** 20).toFixed(2)} MiB`} note="256 exact + pooled blocks" />
      </div>
      <p class="widget-note">A query reads its most recent {window} keys exactly. Older keys reach it only as block averages, and only once the whole block has left the window. The cache figures use the Day 2 shape through kv_cache_bytes.</p>
    </div>
  );
}
