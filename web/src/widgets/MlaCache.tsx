import { useState } from "preact/hooks";
import LineChart from "../charts/LineChart.tsx";
import { cacheDims } from "../lib/model.ts";
import { Segmented, Slider, Stat } from "./ui.tsx";

export default function MlaCache() {
  const [rank, setRank] = useState(64);
  const [ropeDim, setRopeDim] = useState(16);
  const [width, setWidth] = useState(256);
  const heads = 8;
  const mla = (r: number) => cacheDims({ n_layers: 1, d_model: width, n_heads: heads, attention: "mla", mla_rank: r, mla_rope_dim: ropeDim });
  const gqa = (kv: number) => cacheDims({ n_layers: 1, d_model: width, n_heads: heads, kv_heads: kv });
  const ranks = [8, 16, 32, 48, 64, 96, 128, 192, 256];
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Latent rank" value={rank} min={8} max={256} step={8} onInput={setRank} />
        <Segmented label="RoPE key width" value={ropeDim} options={[8, 16, 32].map((v) => ({ value: v, text: String(v) }))} onChange={setRopeDim} />
        <Segmented label="Model width" value={width} options={[256, 1024, 4096].map((v) => ({ value: v, text: String(v) }))} onChange={setWidth} />
      </div>
      <LineChart
        series={[
          { name: "MLA", points: ranks.map((r) => [r, mla(r)]), slot: 1 },
          { name: "MHA", points: ranks.map((r) => [r, gqa(heads)]), slot: 2, dashed: true },
          { name: "GQA-2", points: ranks.map((r) => [r, gqa(2)]), slot: 3, dashed: true },
        ]}
        x={{ label: "latent rank" }}
        y={{ label: "cached numbers per token per layer" }}
        markers
        height={280}
      />
      <div class="readout-grid">
        <Stat label="MLA at this rank" value={mla(rank)} note={`${rank} latent + ${ropeDim} RoPE key`} />
        <Stat label="GQA-2" value={gqa(2)} note={`2 × 2 × ${width / heads}`} />
        <Stat label="MHA" value={gqa(heads)} note={`2 × ${heads} × ${width / heads}`} />
        <Stat label="MLA against GQA-2" value={`${(gqa(2) / mla(rank)).toFixed(2)}×`} note={mla(rank) < gqa(2) ? "smaller" : "not smaller"} />
      </div>
    </div>
  );
}
