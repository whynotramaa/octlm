import { useState } from "preact/hooks";
import { Segmented, Slider, Stat, int } from "./ui.tsx";

const MAX = 12;
const C = 36;

export default function KvDecode() {
  const [step, setStep] = useState(5);
  const [cache, setCache] = useState(true);
  const projected = cache ? step : (step * (step + 1)) / 2;
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Tokens generated" value={step} min={1} max={MAX} onInput={setStep} />
        <Segmented label="Decoding" value={cache ? "on" : "off"} options={[{ value: "on", text: "with KV cache" }, { value: "off", text: "recompute everything" }]} onChange={(v) => setCache(v === "on")} />
      </div>
      <div class="diagram" style={{ overflowX: "auto" }}>
        <svg viewBox={`0 0 ${40 + MAX * C} ${4 * C + 20}`} style={{ minWidth: "440px" }} role="img" aria-label="Key and value cache during decoding">
          {["K", "V"].map((name, row) => (
            <g>
              <text class="t" x="14" y={row * C + C / 2 + 5} text-anchor="middle">{name}</text>
              {Array.from({ length: MAX }, (_, i) => {
                const fresh = i === step - 1 || (!cache && i < step);
                const cls = i >= step ? "muted" : fresh ? "warn" : "on";
                return <rect class={`box ${cls}`} x={32 + i * C} y={row * C + 2} width={C - 6} height={C - 6} rx="5" style={i >= step ? { strokeDasharray: "3 3" } : undefined} />;
              })}
            </g>
          ))}
          <text class="t" x="14" y={2 * C + C / 2 + 12} text-anchor="middle">q</text>
          <rect class="box good" x={32 + (step - 1) * C} y={2 * C + 10} width={C - 6} height={C - 6} rx="5" />
          {Array.from({ length: step }, (_, i) => (
            <line class="link" x1={32 + (step - 1) * C + (C - 6) / 2} y1={2 * C + 10} x2={32 + i * C + (C - 6) / 2} y2={2 * C - 4} style={{ opacity: 0.5 }} />
          ))}
          {Array.from({ length: MAX }, (_, i) => <text class="t mono" x={32 + i * C + (C - 6) / 2} y={4 * C + 10} text-anchor="middle">{i}</text>)}
        </svg>
      </div>
      <div class="readout-grid">
        <Stat label="K and V computed this step" value={cache ? 1 : step} note="per layer, per head" />
        <Stat label="Computed so far" value={int(projected)} note={cache ? "one per token" : "1 + 2 + … + t"} />
        <Stat label="Cached positions" value={cache ? step : 0} />
      </div>
      <p class="widget-note">Orange cells are keys and values computed at this step, blue cells are reused from the cache, green is the new query. The query must read every earlier key, but those keys never change once written, so the cache keeps them. Without it, generating t tokens projects about t²/2 keys and values. With it, t. The price is the memory the cells take, which is what GQA shrinks.</p>
    </div>
  );
}
