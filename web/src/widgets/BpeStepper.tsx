import { useMemo, useState } from "preact/hooks";
import { BASE_SIZE, type Merge, applyMerge, byteTable, encode, initialSequences, pairCounts, tokenText, train } from "../lib/bpe.ts";
import { Stat, int } from "./ui.tsx";

const show = (text: string) => text.replace(/\n/g, "\\n").replace(/\t/g, "\\t").replace(/ /g, "␠");

function stateAfter(text: string, merges: Merge[]) {
  let sequences = initialSequences([text]);
  for (const [left, right, id] of merges) sequences = applyMerge(sequences, [left, right], id);
  const counts = [...pairCounts(sequences)]
    .map(([key, count]) => ({ pair: key.split(",").map(Number) as [number, number], count }))
    .sort((a, b) => b.count - a.count || a.pair[0] - b.pair[0] || a.pair[1] - b.pair[1]);
  return { sequences, counts };
}

export default function BpeStepper({ text: initial }: { text: string }) {
  const [text, setText] = useState(initial);
  const [step, setStep] = useState(0);
  const all = useMemo(() => train([text], BASE_SIZE + 60), [text]);
  const merges = all.slice(0, step);
  const table = byteTable(all);
  const { counts } = useMemo(() => stateAfter(text, merges), [text, step, all]);
  const encoded = encode(text, merges);
  const bytes = new TextEncoder().encode(text).length;
  const top = counts.slice(0, 8);
  const tied = top.length > 1 && top[0].count === top[1].count;
  const name = (id: number) => show(tokenText(id, table));
  const last = merges[merges.length - 1];
  return (
    <div class="widget">
      <textarea rows={5} value={text} onInput={(e) => { setText(e.currentTarget.value); setStep(0); }} aria-label="Training text" />
      <div class="widget-nav">
        <button type="button" class="button" disabled={step === 0} onClick={() => setStep(0)}>Reset</button>
        <span class="muted">Merge {step} of {all.length} learnable</span>
        <div style={{ display: "flex", gap: "8px" }}>
          <button type="button" class="button" disabled={step >= all.length} onClick={() => setStep(Math.min(all.length, step + 10))}>+10</button>
          <button type="button" class="button primary" disabled={step >= all.length} onClick={() => setStep(step + 1)}>Next merge</button>
        </div>
      </div>
      <div class="two-col">
        <div>
          <div class="label" style={{ marginBottom: "8px" }}>Most frequent adjacent pairs now</div>
          <table class="data-table">
            <thead><tr><th>Pair</th><th>IDs</th><th class="num">Count</th></tr></thead>
            <tbody>
              {top.map((row, i) => (
                <tr class={i === 0 ? "on" : ""}>
                  <td><code>{name(row.pair[0])}</code> + <code>{name(row.pair[1])}</code></td>
                  <td>{row.pair[0]} + {row.pair[1]}</td>
                  <td class="num">{row.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p class="widget-note">
            {top.length === 0 ? "No pair left inside any chunk." : top[0].count < 2 ? "The best pair occurs once. Training stops here, because min_frequency is 2." : tied ? `Tie at ${top[0].count}. The lower ID pair wins, so ${top[0].pair.join(" + ")} becomes ID ${BASE_SIZE + step}.` : `Next merge: ${top[0].pair.join(" + ")} becomes ID ${BASE_SIZE + step}.`}
          </p>
        </div>
        <div>
          <div class="label" style={{ marginBottom: "8px" }}>Merges learned</div>
          <div class="table-scroll" style={{ maxHeight: "300px", overflowY: "auto" }}>
            <table class="data-table">
              <thead><tr><th class="num">#</th><th>Rule</th><th>New token</th></tr></thead>
              <tbody>
                {merges.map(([left, right, id], i) => (
                  <tr class={i === step - 1 ? "on" : ""}>
                    <td class="num">{i + 1}</td>
                    <td>{left} + {right} → {id}</td>
                    <td><code>{name(id)}</code></td>
                  </tr>
                )).reverse()}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="tokens">
        {encoded.slice(0, 120).map((id) => <span class={last && id === last[2] ? "token new" : "token"}>{name(id)}</span>)}
        {encoded.length > 120 && <span class="muted">…</span>}
      </div>
      <div class="readout-grid">
        <Stat label="UTF-8 bytes" value={int(bytes)} />
        <Stat label="Tokens now" value={int(encoded.length)} />
        <Stat label="Bytes per token" value={(bytes / Math.max(1, encoded.length)).toFixed(2)} />
        <Stat label="Vocabulary" value={int(BASE_SIZE + step)} note="256 bytes + 5 specials + merges" />
      </div>
    </div>
  );
}
