import { useState } from "preact/hooks";
import { cacheDims } from "../lib/model.ts";
import { Segmented, Stat } from "./ui.tsx";

type Mode = "mha" | "gqa" | "mla";
const BASE = { d_model: 256, n_heads: 8, n_layers: 1 };
const SHAPES: Record<Mode, object> = { mha: {}, gqa: { kv_heads: 2 }, mla: { attention: "mla", mla_rank: 64, mla_rope_dim: 16 } };

function Heads({ x, y, count, cached, label }: { x: number; y: number; count: number; cached: boolean; label: string }) {
  const w = Math.min(34, 300 / count);
  return (
    <g>
      {Array.from({ length: count }, (_, i) => <rect class={`box ${cached ? "good" : "on"}`} x={x + i * w} y={y} width={w - 5} height="28" rx="5" />)}
      <text class="t small" x={x + (count * w) / 2} y={y + 44} text-anchor="middle">{label}</text>
    </g>
  );
}

export default function MlaPaths() {
  const [mode, setMode] = useState<Mode>("mla");
  const dims = cacheDims({ ...BASE, ...SHAPES[mode] });
  const kv = mode === "gqa" ? 2 : 8;
  return (
    <div class="widget">
      <Segmented label="Attention" value={mode} options={[{ value: "mha", text: "MHA" }, { value: "gqa", text: "GQA-2" }, { value: "mla", text: "MLA" }]} onChange={setMode} />
      <div class="diagram" style={{ overflowX: "auto" }}>
        <svg viewBox="0 0 600 240" style={{ minWidth: "440px" }} role="img" aria-label={`${mode} projection path`}>
          <rect class="box muted" x="10" y="100" width="70" height="34" rx="8" />
          <text class="t" x="45" y="122" text-anchor="middle">hₜ</text>
          {mode === "mla" ? (
            <g>
              <path class="link" d="M 80 117 H 130" />
              <rect class="box good" x="130" y="100" width="100" height="34" rx="8" />
              <text class="t" x="180" y="122" text-anchor="middle">latent cₜ, 64</text>
              <path class="link" d="M 80 117 V 190 H 130" />
              <rect class="box good" x="130" y="173" width="100" height="34" rx="8" />
              <text class="t" x="180" y="195" text-anchor="middle">kᵣₒₚₑ, 16</text>
              <path class="link" d="M 230 117 H 270 V 50 H 290 M 270 117 V 150 H 290" />
              <Heads x={290} y={36} count={kv} cached={false} label="K, 8 heads, rebuilt from cₜ" />
              <Heads x={290} y={136} count={kv} cached={false} label="V, 8 heads, rebuilt from cₜ" />
            </g>
          ) : (
            <g>
              <path class="link" d="M 80 117 H 110 V 50 H 150 M 110 117 V 150 H 150" />
              <Heads x={150} y={36} count={kv} cached label={`K, ${kv} heads × 32`} />
              <Heads x={150} y={136} count={kv} cached label={`V, ${kv} heads × 32`} />
            </g>
          )}
        </svg>
      </div>
      <div class="readout-grid">
        <Stat label="Cached per token per layer" value={dims} note="green boxes" />
        <Stat label="Against MHA" value={`${(256 * 2 / dims).toFixed(1)}× smaller`} />
        <Stat label="Against GQA-2" value={`${(128 / dims).toFixed(1)}×`} />
      </div>
      <p class="widget-note">Width 256, 8 heads of 32, the Day 2 shape. Green is what the cache stores. MHA and GQA store keys and values directly. MLA stores a 64-wide latent and a small RoPE key, and rebuilds the per-head keys and values with an up-projection at every step, trading memory for extra matrix work.</p>
    </div>
  );
}
