import { useState } from "preact/hooks";
import { toBytes } from "../lib/bpe.ts";
import { KIND_NAMES, kind, pretokenize } from "../lib/pretokenize.ts";

const PRESETS: Record<string, string> = {
  Code: "def f(x_1):\n\treturn x_1 + 30.5\r\n",
  Prose: "Naïve café, 3 cups.",
  Unicode: "日本語 Привет 🙂👍🏽 e\u0301",
  Special: "a <|eos|> b",
};

const show = (text: string) => text.replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t").replace(/ /g, "␠");
const hex = (bytes: number[]) => bytes.map((b) => b.toString(16).padStart(2, "0")).join(" ");

export default function PretokenView() {
  const [text, setText] = useState(PRESETS.Code);
  const [chosen, setChosen] = useState(0);
  const chunks = pretokenize(text);
  const current = chunks[Math.min(chosen, chunks.length - 1)] ?? "";
  const bytes = toBytes(current);
  return (
    <div class="widget">
      <div class="widget-controls">
        <div class="segmented" role="group" aria-label="Preset">
          {Object.entries(PRESETS).map(([name, value]) => (
            <button type="button" class={value === text ? "on" : ""} onClick={() => { setText(value); setChosen(0); }}>{name}</button>
          ))}
        </div>
      </div>
      <textarea rows={2} value={text} onInput={(e) => setText(e.currentTarget.value)} aria-label="Text to pre-tokenize" />
      <div class="tokens">
        {chunks.map((chunk, i) => (
          <button type="button" class={`token k${kind(Array.from(chunk)[0])}${i === chosen ? " new" : ""}`} onClick={() => setChosen(i)} onMouseEnter={() => setChosen(i)}>
            {show(chunk)}<small>{KIND_NAMES[kind(Array.from(chunk)[0])]}</small>
          </button>
        ))}
      </div>
      <div class="legend">
        {KIND_NAMES.map((name, i) => <span class="legend-item"><span class={`token k${i}`}>{name}</span></span>)}
      </div>
      <p class="widget-note">
        Chunk <code>{show(current)}</code> is {Array.from(current).length} characters and {bytes.length} UTF-8 bytes: <code>{hex(bytes)}</code>. BPE starts from these bytes and never merges across a chunk edge.
      </p>
    </div>
  );
}
