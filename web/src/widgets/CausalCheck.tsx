import { useMemo, useState } from "preact/hooks";
import fixture from "../../fixtures/attention.json";
import { causalSelfAttention } from "../lib/attention.ts";
import { mulberry32 } from "../lib/sampling.ts";

const weights = { query: fixture.query, key: fixture.key, value: fixture.value, output: fixture.output_weight };
const random = mulberry32(7);
const direction = fixture.x[0].map(() => random() * 2 - 1);

function rowChanges(changed: number, amount: number, causal: boolean): number[] {
  const edited = fixture.x.map((row, i) => (i === changed ? row.map((v, c) => v + amount * direction[c]) : row));
  const before = causalSelfAttention(fixture.x, weights, fixture.n_heads, causal).output;
  const after = causalSelfAttention(edited, weights, fixture.n_heads, causal).output;
  return before.map((row, i) => Math.max(...row.map((v, c) => Math.abs(v - after[i][c]))));
}

export default function CausalCheck() {
  const [changed, setChanged] = useState(7);
  const [amount, setAmount] = useState(1.5);
  const [causal, setCausal] = useState(true);
  const changes = useMemo(() => rowChanges(changed, amount, causal), [changed, amount, causal]);
  const peak = Math.max(1e-9, ...changes);
  const earlier = changes.slice(0, changed);
  const leaked = earlier.some((d) => d > 0);
  const verdict = changed === 0
    ? { hue: "blue", text: "Position t0 has no earlier rows to check. Pick a later position." }
    : leaked
      ? { hue: "red", text: `Rows t0 to t${changed - 1} changed. Earlier positions read the future.` }
      : { hue: "green", text: `Rows t0 to t${changed - 1} changed by exactly 0. No position read the future.` };
  return (
    <div class="widget">
      <div class="widget-controls grid-controls">
        <label class="control">
          <span class="control-label">Change position</span>
          <select value={changed} onChange={(e) => setChanged(Number(e.currentTarget.value))}>
            {fixture.x.map((_, i) => <option value={i}>t{i}</option>)}
          </select>
        </label>
        <label class="control">
          <span class="control-label">Size of change <output>{amount.toFixed(1)}</output></span>
          <input type="range" min="0.1" max="4" step="0.1" value={amount} onInput={(e) => setAmount(Number(e.currentTarget.value))} />
        </label>
        <div class="control">
          <span class="control-label">Mask</span>
          <div class="segmented" role="group" aria-label="Mask">
            <button type="button" class={causal ? "on" : ""} aria-pressed={causal} onClick={() => setCausal(true)}>Causal</button>
            <button type="button" class={causal ? "" : "on"} aria-pressed={!causal} onClick={() => setCausal(false)}>None</button>
          </div>
        </div>
      </div>
      <div class="bars" role="list">
        {changes.map((d, i) => {
          const state = i === changed ? "edited" : d === 0 ? "same" : i < changed ? "leak" : "after";
          return (
            <div class={`bar-row ${state}`} role="listitem">
              <span class="bar-name">t{i}</span>
              <span class="bar-track"><span class="bar-fill" style={{ width: `${(d / peak) * 100}%` }} /></span>
              <span class="bar-value">{d === 0 ? "0" : d.toExponential(1)}</span>
            </div>
          );
        })}
      </div>
      <div class={`verdict tint hue-${verdict.hue}`}>{verdict.text}</div>
      <p class="widget-note">Each bar is the largest change in that row of the attention output after adding a random vector to x[t{changed}].</p>
    </div>
  );
}
