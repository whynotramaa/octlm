import { useState } from "preact/hooks";
import { FORMATS } from "../lib/floats.ts";
import { Slider, Stat } from "./ui.tsx";

const LO = -150;
const HI = 130;
const W = 640;
const px = (e: number) => 90 + ((e - LO) / (HI - LO)) * (W - 100);

function limits(exponent: number, mantissa: number) {
  const bias = 2 ** (exponent - 1) - 1;
  return { tiny: 1 - bias - mantissa, normal: 1 - bias, max: bias + Math.log2(2 - 2 ** -mantissa) };
}

export default function FormatRanges() {
  const [k, setK] = useState(-20);
  const value = 1.3 * 2 ** k;
  return (
    <div class="widget">
      <div class="diagram" style={{ overflowX: "auto" }}>
        <svg viewBox={`0 0 ${W} 170`} style={{ minWidth: "480px" }} role="img" aria-label="Representable magnitudes of each format on a log scale">
          {FORMATS.map((f, i) => {
            const l = limits(f.exponent, f.mantissa);
            const y = 20 + i * 40;
            return (
              <g>
                <text class="t" x="0" y={y + 13}>{f.name}</text>
                <rect class="fill-1" x={px(l.tiny)} y={y} width={px(l.normal) - px(l.tiny)} height="18" rx="4" opacity="0.3" />
                <rect class="fill-1" x={px(l.normal)} y={y} width={px(l.max) - px(l.normal)} height="18" rx="4" />
              </g>
            );
          })}
          {[-140, -100, -60, -20, 0, 20, 60, 100].map((e) => (
            <g>
              <line class="dial-axis" x1={px(e)} x2={px(e)} y1="136" y2="142" />
              <text class="t mono" x={px(e)} y="156" text-anchor="middle">2^{e}</text>
            </g>
          ))}
          <line class="rule" x1={px(Math.log2(value))} x2={px(Math.log2(value))} y1="10" y2="136" />
        </svg>
      </div>
      <Slider label="Magnitude, 1.3 × 2^k" value={k} min={-145} max={128} shown={`k = ${k}, ${value.toExponential(2)}`} onInput={setK} />
      <div class="readout-grid">
        {FORMATS.map((f) => {
          const r = f.round(value);
          const text = !Number.isFinite(r) ? "overflow, ∞" : r === 0 ? "underflow, 0" : `${Math.abs((r - value) / value * 100).toPrecision(2)}% error`;
          return <Stat label={f.name} value={text} note={Number.isFinite(r) ? r.toExponential(3) : "inf"} />;
        })}
      </div>
      <p class="widget-note">Solid bars are the normal range, faded bars the subnormals, where precision thins out toward zero. float16 covers a narrow window around 1, from about 2^−24 to 65,504. Gradients often sit near 2^−20 to 2^−30, below it, which is why float16 training scales the loss up first. bfloat16's bar is as long as float32's.</p>
    </div>
  );
}
