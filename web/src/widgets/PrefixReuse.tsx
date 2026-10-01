import { useState } from "preact/hooks";
import { Segmented, Stat, int } from "./ui.tsx";

type Turn = { text: string; reused_chars: number; rendered: number; keep: number };
type Trace = { task: string; seed: number; turns: Turn[] };

export default function PrefixReuse({ traces }: { traces: Trace[] }) {
  const [index, setIndex] = useState(0);
  const [at, setAt] = useState(0);
  const trace = traces[index];
  const turn = trace.turns[Math.min(at, trace.turns.length - 1)];
  const seen = trace.turns.slice(0, Math.min(at, trace.turns.length - 1) + 1);
  const rendered = seen.reduce((sum, t) => sum + t.rendered, 0);
  const prefilled = seen.reduce((sum, t) => sum + t.rendered - t.keep, 0);
  const pick = (value: number) => { setIndex(value); setAt(0); };
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Segmented label="Run" value={index} options={traces.map((t, value) => ({ value, text: `${t.task}, seed ${t.seed}, ${t.turns.length} turns` }))} onChange={pick} />
      </div>
      <pre class="template-text prefix-text"><span class="reused">{turn.text.slice(0, turn.reused_chars)}</span><mark>{turn.text.slice(turn.reused_chars)}</mark></pre>
      <div class="readout-grid">
        <Stat label="This turn's prompt" value={`${int(turn.rendered)} tokens`} note="rendered by the chat template" />
        <Stat label="From the cache" value={`${int(turn.keep)} tokens`} note="gray text" />
        <Stat label="Prefilled" value={`${int(turn.rendered - turn.keep)} tokens`} note="highlighted text" />
        <Stat label="Run so far" value={`${Math.round((100 * prefilled) / rendered)}%`} note={`prefilled, ${int(prefilled)} of ${int(rendered)}`} />
      </div>
      <div class="widget-nav">
        <button type="button" class="button" disabled={at === 0} onClick={() => setAt(at - 1)}>Back</button>
        <span class="muted">Turn {Math.min(at, trace.turns.length - 1) + 1} of {trace.turns.length}</span>
        <button type="button" class="button primary" disabled={at >= trace.turns.length - 1} onClick={() => setAt(at + 1)}>Next</button>
      </div>
      <p class="widget-note">Each turn the harness renders the whole conversation again, finds where the new token IDs first differ from what the cache holds, cuts the cache there and prefills only the rest. From turn 2 on, the highlight starts at the previous assistant turn: the template drops the empty think block from it, so its tokens change.</p>
    </div>
  );
}
