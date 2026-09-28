import { useState } from "preact/hooks";
import { Segmented, Slider, Stat } from "./ui.tsx";

function formula(layers: number, pre: boolean) {
  if (pre) return `out = x₀ + ${Array.from({ length: layers }, (_, i) => `f${i + 1}`).join(" + ")}`;
  let text = "x₀";
  for (let i = 1; i <= layers; i++) text = `N(${text} + f${i})`;
  return `out = ${text}`;
}

export default function ResidualStream() {
  const [layers, setLayers] = useState(4);
  const [mode, setMode] = useState<"pre" | "post">("pre");
  const pre = mode === "pre";
  const H = 70;
  const height = 60 + layers * H;
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Sublayers" value={layers} min={1} max={8} onInput={setLayers} />
        <Segmented label="Placement" value={mode} options={[{ value: "pre", text: "Pre-norm" }, { value: "post", text: "Post-norm" }]} onChange={setMode} />
      </div>
      <div class="two-col">
        <div class="diagram">
          <svg viewBox={`0 0 280 ${height}`} role="img" aria-label={`${mode}-norm residual stream, ${layers} sublayers`}>
            <text class="t small" x="70" y="18" text-anchor="middle">x₀</text>
            <line class="link on" x1="70" y1="26" x2="70" y2={height - 20} />
            <text class="t small" x="70" y={height - 4} text-anchor="middle">out</text>
            {Array.from({ length: layers }, (_, i) => {
              const y = 40 + i * H;
              return (
                <g>
                  <path class="link" d={`M 70 ${y} H 150 V ${y + 8}`} />
                  <rect class="box on" x="110" y={y + 8} width="130" height="28" rx="8" />
                  <text class="t" x="175" y={y + 27} text-anchor="middle">{pre ? `f${i + 1}(N(x))` : `f${i + 1}(x)`}</text>
                  <path class="link" d={`M 175 ${y + 36} V ${y + 50} H 84`} />
                  <circle class="box muted" cx="70" cy={y + 50} r="11" />
                  <text class="t" x="70" y={y + 55} text-anchor="middle">+</text>
                  {!pre && <><rect class="box warn" x="46" y={y + 60} width="48" height="18" rx="5" /><text class="t small" x="70" y={y + 73} text-anchor="middle">Norm</text></>}
                </g>
              );
            })}
          </svg>
        </div>
        <div>
          <code class="shape" style={{ display: "block", whiteSpace: "normal", overflowWrap: "anywhere" }}>{formula(layers, pre)}</code>
          <div class="readout-grid" style={{ marginTop: "16px" }}>
            <Stat label="Norms on the main path" value={pre ? 0 : layers} />
            <Stat label="x₀ reaches the output" value={pre ? "unchanged" : `through ${layers} norms`} />
          </div>
          <p class="widget-note" style={{ marginTop: "16px" }}>{pre ? "In pre-norm the output is a plain sum. The gradient of the output with respect to x₀ always includes an identity term, however deep the stack." : "In post-norm every addition is followed by a norm, so the input is renormalized once per sublayer. The gradient to x₀ passes through every norm, and nothing carries it straight through."}</p>
        </div>
      </div>
    </div>
  );
}
