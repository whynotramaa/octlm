import { useState } from "preact/hooks";
import { pretokenize } from "../lib/pretokenize.ts";
import { Stat, int } from "./ui.tsx";

const SAMPLE = "def area(radius):\n    return 3.14159 * radius ** 2\n\nprint(area(2))  # café ☕";

export default function SequenceLength() {
  const [text, setText] = useState(SAMPLE);
  const rows = [
    { name: "bytes", count: new TextEncoder().encode(text).length },
    { name: "characters", count: [...text].length },
    { name: "chunks", count: pretokenize(text).length },
  ];
  const top = Math.max(1, ...rows.map((r) => r.count ** 2));
  return (
    <div class="widget">
      <textarea rows={4} value={text} onInput={(e) => setText(e.currentTarget.value)} aria-label="Text to measure" />
      <div class="bars">
        {rows.map((r) => (
          <div class="bar-row" style={{ gridTemplateColumns: "92px 1fr 96px" }}>
            <span class="bar-name">{r.name}</span>
            <span class="bar-track"><span class="bar-fill" style={{ width: `${(r.count ** 2 / top) * 100}%` }} /></span>
            <span class="bar-value">{int(r.count ** 2)}</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        {rows.map((r) => <Stat label={`Sequence length in ${r.name}`} value={int(r.count)} />)}
      </div>
      <p class="widget-note">The bars are attention scores per head, the sequence length squared. A character tokenizer makes one position per character, so the same text costs several times the attention of a tokenizer that groups characters. Chunks are what pre-tokenization makes, a rough floor for what BPE can reach. Indentation is the worst case: four spaces are four positions.</p>
    </div>
  );
}
