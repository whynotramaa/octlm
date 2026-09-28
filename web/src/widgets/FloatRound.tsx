import { useState } from "preact/hooks";
import { FORMATS, bitFields } from "../lib/floats.ts";

const PRESETS = [0.1, 3.14159265, 1000.7, 65504, 70000, 1e-5, 3e-8, 1e-9];

function show(value: number) {
  if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
  if (value === 0) return "0";
  return Math.abs(value) >= 1e5 || Math.abs(value) < 1e-3 ? value.toExponential(6) : value.toPrecision(9).replace(/\.?0+$/, "");
}

export default function FloatRound() {
  const [text, setText] = useState("0.1");
  const value = Number(text);
  return (
    <div class="widget">
      <div class="widget-controls">
        <label class="control" style={{ flex: "1 1 200px" }}>
          <span class="control-label">A number</span>
          <input type="text" inputMode="decimal" value={text} onInput={(e) => setText(e.currentTarget.value)} />
        </label>
        <div class="segmented" role="group" aria-label="Presets" style={{ flexWrap: "wrap" }}>
          {PRESETS.map((p) => <button type="button" class={Number(text) === p ? "on" : ""} onClick={() => setText(String(p))}>{show(p)}</button>)}
        </div>
      </div>
      <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Format</th><th>Sign · exponent · mantissa bits</th><th class="num">Stored value</th><th class="num">Relative error</th></tr></thead>
          <tbody>
            {FORMATS.map((f) => {
              const stored = f.round(value);
              const [sign, exponent, mantissa] = Number.isNaN(value) ? ["", "", ""] : bitFields(f, value);
              const error = value === 0 ? 0 : Math.abs(stored - value) / Math.abs(value);
              return (
                <tr>
                  <td>{f.name}<br /><small class="muted">{f.exponent} exp, {f.mantissa} mantissa</small></td>
                  <td><code class="bits"><span class="b-sign">{sign}</span> <span class="b-exp">{exponent}</span> <span class="b-man">{mantissa}</span></code></td>
                  <td class="num">{Number.isNaN(value) ? "–" : show(stored)}</td>
                  <td class="num">{Number.isFinite(stored) ? error.toExponential(1) : "overflow"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p class="widget-note">float16 spends 5 bits on the exponent, so it tops out at 65,504 and rounds values below about 3e-8 to zero. bfloat16 keeps float32's 8 exponent bits and so its range, but has only 7 mantissa bits, about 2 to 3 decimal digits. Try 70000, 3e-8 and 1e-9.</p>
    </div>
  );
}
