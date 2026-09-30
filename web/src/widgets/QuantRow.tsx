import { useState } from "preact/hooks";
import { quantizeRow } from "../lib/day5.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

export default function QuantRow() {
  const [outlier, setOutlier] = useState(1);
  const weights = [-0.8, -0.35, 0, 0.32, outlier];
  const { scale, codes, restored } = quantizeRow(weights);
  const error = Math.max(...weights.map((weight, index) => Math.abs(weight - restored[index])));
  return (
    <div class="widget">
      <Slider label="Largest weight in this output row" value={outlier} min={1} max={20} step={0.5} shown={fixed(outlier, 1)} onInput={setOutlier} />
      <div class="diagram" style={{ overflowX: "auto" }}>
        <svg viewBox="0 0 640 126" style={{ minWidth: "500px" }} role="img" aria-label="A float weight becomes an int8 code and is then restored approximately">
          <rect class="box on" x="10" y="25" width="168" height="54" rx="9" />
          <rect class="box alt" x="236" y="25" width="168" height="54" rx="9" />
          <rect class="box good" x="462" y="25" width="168" height="54" rx="9" />
          <text class="t" x="94" y="47" text-anchor="middle">float weight</text>
          <text class="t mono" x="94" y="67" text-anchor="middle">−0.350</text>
          <text class="t" x="320" y="47" text-anchor="middle">int8 code</text>
          <text class="t mono" x="320" y="67" text-anchor="middle">{codes[1]}</text>
          <text class="t" x="546" y="47" text-anchor="middle">restored weight</text>
          <text class="t mono" x="546" y="67" text-anchor="middle">{fixed(restored[1], 3)}</text>
          <path class="link" d="M 181 52 H 231 M 407 52 H 457" />
          <text class="t small" x="207" y="104" text-anchor="middle">÷ scale, round</text>
          <text class="t small" x="433" y="104" text-anchor="middle">× scale</text>
        </svg>
      </div>
      <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Column</th>{weights.map((_, index) => <th>{index + 1}</th>)}</tr></thead>
          <tbody>
            <tr><th>Float weight</th>{weights.map((value) => <td>{fixed(value, 3)}</td>)}</tr>
            <tr><th>Signed int8</th>{codes.map((value) => <td>{value}</td>)}</tr>
            <tr><th>Restored</th>{restored.map((value) => <td>{fixed(value, 3)}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <div class="readout-grid">
        <Stat label="Scale for this row" value={fixed(scale, 5)} note="largest absolute weight ÷ 127" />
        <Stat label="Largest absolute error" value={fixed(error, 4)} />
        <Stat label="Storage for this row" value="9 bytes" note="5 int8 weights + one float32 scale" />
      </div>
      <p class="widget-note">This is the same symmetric, per-output-row rule as octlm's Int8Linear. Raise the outlier: the row needs a coarser scale, so smaller weights lose precision. The restored numbers are what the current PyTorch path multiplies in float16 or float32 before its linear operation. The 9-byte count excludes file metadata and alignment.</p>
    </div>
  );
}
