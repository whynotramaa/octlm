import { useState } from "preact/hooks";
import { type Config, hiddenSize, parameterBreakdown, parameterCount } from "../lib/model.ts";
import { Stat, int } from "./ui.tsx";

type Preset = { name: string; config: Config };

const PARTS = [
  ["embedding", "Token embedding (shared with the output head)"],
  ["positions", "Learned position table"],
  ["attention", "Attention: query, key, value, output"],
  ["feedForward", "Feed-forward"],
  ["norms", "Norm gains and biases"],
] as const;

const FIELDS: [keyof Config, string][] = [
  ["vocab_size", "Vocabulary"],
  ["d_model", "Width"],
  ["n_layers", "Layers"],
  ["n_heads", "Query heads"],
  ["kv_heads", "KV heads (0 = same)"],
  ["context_length", "Context"],
];

export default function ParameterCounter({ presets, start = 0 }: { presets: Preset[]; start?: number }) {
  const [config, setConfig] = useState<Config>(presets[start].config);
  const parts = parameterBreakdown(config);
  const total = parameterCount(config);
  const set = (key: keyof Config, value: string | number) => setConfig({ ...config, [key]: value });
  return (
    <div class="widget">
      <div class="widget-controls">
        <div class="segmented" role="group" aria-label="Preset">
          {presets.map((p) => <button type="button" class={p.config === config ? "on" : ""} onClick={() => setConfig(p.config)}>{p.name}</button>)}
        </div>
      </div>
      <div class="grid-controls widget-controls">
        {FIELDS.map(([key, label]) => (
          <label class="control">
            <span class="control-label">{label}</span>
            <input type="number" min="0" value={config[key] as number} onInput={(e) => set(key, Number(e.currentTarget.value) || 0)} />
          </label>
        ))}
        <label class="control">
          <span class="control-label">Feed-forward</span>
          <select value={config.feed_forward ?? "gelu"} onChange={(e) => set("feed_forward", e.currentTarget.value)}>
            <option value="gelu">GELU, 2 matrices</option>
            <option value="swiglu">SwiGLU, 3 matrices</option>
          </select>
        </label>
        <label class="control">
          <span class="control-label">Positions</span>
          <select value={config.position ?? "learned"} onChange={(e) => set("position", e.currentTarget.value)}>
            <option value="learned">Learned table</option>
            <option value="sinusoidal">Sinusoidal</option>
            <option value="rope">RoPE</option>
          </select>
        </label>
        <label class="control">
          <span class="control-label">Norm</span>
          <select value={config.norm ?? "layernorm"} onChange={(e) => set("norm", e.currentTarget.value)}>
            <option value="layernorm">LayerNorm</option>
            <option value="rmsnorm">RMSNorm</option>
          </select>
        </label>
      </div>
      <div class="bars" role="list">
        {PARTS.map(([key, label]) => (
          <div class="bar-row" role="listitem" style={{ gridTemplateColumns: "minmax(120px, 260px) 1fr 96px", fontFamily: "var(--font-sans)" }}>
            <span class="bar-name">{label}</span>
            <span class="bar-track"><span class="bar-fill" style={{ width: `${(parts[key] / total) * 100}%`, background: "var(--series-1)" }} /></span>
            <span class="bar-value">{int(parts[key])}</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="Parameters" value={int(total)} />
        <Stat label="Feed-forward hidden width" value={int(hiddenSize(config))} note={config.feed_forward === "swiglu" ? "⅔ × 4d, rounded to 8" : "4d"} />
        <Stat label="float32 weights" value={`${(total * 4 / 2 ** 20).toFixed(1)} MiB`} />
      </div>
    </div>
  );
}
