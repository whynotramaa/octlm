import { useMemo, useState } from "preact/hooks";
import fixture from "../../fixtures/tiled.json";
import { causalMask, attendHead } from "../lib/attention.ts";
import { tiledAttention } from "../lib/online-softmax.ts";
import { Segmented, Slider, Stat, fixed } from "./ui.tsx";

const full = attendHead(fixture.query, fixture.key, fixture.value, causalMask(fixture.query.length));

export default function OnlineSoftmax() {
  const [block, setBlock] = useState(4);
  const [row, setRow] = useState(11);
  const [step, setStep] = useState(0);
  const tiled = useMemo(() => tiledAttention(fixture.query, fixture.key, fixture.value, block), [block]);
  const current = tiled.steps[Math.min(step, tiled.steps.length - 1)];
  const scores = full.scores[row];
  const done = step >= tiled.steps.length - 1;
  const partial = current.output[row].map((o) => o / current.runningSum[row]);
  const error = Math.max(...tiled.output[row].map((v, c) => Math.abs(v - full.out[row][c])));
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Segmented label="Block size" value={block} options={[2, 3, 4, 6].map((v) => ({ value: v, text: String(v) }))} onChange={(v) => { setBlock(v); setStep(0); }} />
        <Slider label="Query row" value={row} min={0} max={11} onInput={setRow} />
      </div>
      <div class="score-strip">
        {scores.map((s, j) => {
          const state = j > row ? "masked" : j < current.start ? "seen" : j < current.stop ? "now" : "later";
          return <div class={`score ${state}`}><small>k{j}</small>{j > row ? "−∞" : fixed(s)}</div>;
        })}
      </div>
      <div class="readout-grid">
        <Stat label="Keys read so far" value={`0 to ${current.stop - 1}`} note={`block ${step + 1} of ${tiled.steps.length}`} />
        <Stat label="Running max m" value={fixed(current.runningMax[row], 3)} note="largest score seen" />
        <Stat label="Running sum ℓ" value={fixed(current.runningSum[row], 3)} note="Σ exp(score − m)" />
        <Stat label="Output so far, entry 0" value={fixed(partial[0], 4)} note={`full softmax: ${fixed(full.out[row][0], 4)}`} />
      </div>
      <div class="widget-nav">
        <button type="button" class="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button>
        <span class="muted">{done ? `Done. Largest difference from full attention on row ${row}: ${error.toExponential(1)}` : "Each block rescales the old sum and output by exp(m_old − m_new)."}</span>
        <button type="button" class="button primary" disabled={done} onClick={() => setStep(step + 1)}>Next block</button>
      </div>
    </div>
  );
}
