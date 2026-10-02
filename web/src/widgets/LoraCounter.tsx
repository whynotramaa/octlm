import { useState } from "preact/hooks";
import { type QwenShape, loraParameters, qwenProjections } from "../lib/lora.ts";
import { Segmented, Stat, int } from "./ui.tsx";

const RANKS = [1, 4, 8, 16, 32, 64, 128];
const GB = 1e9;

export default function LoraCounter({ config, total }: { config: QwenShape; total: number }) {
  const projections = qwenProjections(config);
  const [rank, setRank] = useState(16);
  const [on, setOn] = useState(projections.map((p) => p.name));
  const shapes = projections.filter((p) => on.includes(p.name)).map((p) => p.shape);
  const trainable = loraParameters(shapes, rank, config.n_layers);
  const full = total * 16;
  const lora = total * 2 + trainable * 16;
  const toggle = (name: string) => setOn(on.includes(name) ? on.filter((n) => n !== name) : [...on, name]);
  return (
    <div class="widget">
      <div class="widget-controls">
        <Segmented label="Rank r" value={rank} options={RANKS.map((r) => ({ value: r, text: String(r) }))} onChange={setRank} />
        <div class="control">
          <span class="control-label">Adapted projections</span>
          <div class="segmented" role="group" aria-label="Adapted projections" style={{ flexWrap: "wrap" }}>
            {projections.map((p) => (
              <button type="button" class={on.includes(p.name) ? "on" : ""} aria-pressed={on.includes(p.name)} onClick={() => toggle(p.name)}>
                {p.name} <small class="muted">{p.shape[0]}→{p.shape[1]}</small>
              </button>
            ))}
          </div>
        </div>
      </div>
      <div class="bars">
        <div class="bar-row leak" style={{ gridTemplateColumns: "120px 1fr 72px" }}>
          <span class="bar-name">Full fine-tune</span>
          <span class="bar-track"><span class="bar-fill" style={{ width: "100%" }} /></span>
          <span class="bar-value">{(full / GB).toFixed(1)} GB</span>
        </div>
        <div class="bar-row after" style={{ gridTemplateColumns: "120px 1fr 72px" }}>
          <span class="bar-name">LoRA</span>
          <span class="bar-track"><span class="bar-fill" style={{ width: `${(100 * lora) / full}%` }} /></span>
          <span class="bar-value">{(lora / GB).toFixed(1)} GB</span>
        </div>
      </div>
      <div class="readout-grid">
        <Stat label="Trainable parameters" value={int(trainable)} note={`${((100 * trainable) / total).toFixed(2)}% of ${int(total)}`} />
        <Stat label="Per layer" value={int(trainable / config.n_layers)} note={`r × (in + out), summed over ${shapes.length} projections`} />
        <Stat label="Adapter file" value={`${((trainable * 4) / 2 ** 20).toFixed(1)} MiB`} note="float32 A and B" />
      </div>
      <p class="widget-note">Memory counts weights and optimizer state only, no activations. A full fine-tune keeps float32 master weights, gradients and two AdamW moments: 16 bytes per parameter. LoRA keeps the frozen model in float16, 2 bytes per parameter, and pays the 16 bytes only for A and B. Rank 16 on all seven projections is the setting octlm trained.</p>
    </div>
  );
}
