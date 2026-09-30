import { useState } from "preact/hooks";
import { Segmented, Stat, fixed, int } from "./ui.tsx";

type Generation = { case: number; prompt: string; pieces: string[]; ids: number[] };
type Row = { case: number; tokens_per_second: number; reference_tokens_per_second: number; cache_bytes: number; greedy_equal: boolean };
type Props = { generations: Generation[]; rows: Row[]; imEnd: number };

const show = (piece: string) => piece.replace(/\n/g, "↵");

export default function GenerationTrace({ generations, rows, imEnd }: Props) {
  const [index, setIndex] = useState(0);
  const [stop, setStop] = useState(false);
  const generation = generations[index];
  const row = rows.find((r) => r.case === generation.case)!;
  const end = generation.ids.indexOf(imEnd);
  const kept = stop && end >= 0 ? end + 1 : generation.ids.length;
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Segmented label="Prompt" value={index} options={generations.map((g, value) => ({ value, text: g.case === 0 ? "Robot sentence" : "2 + 2" }))} onChange={setIndex} />
        <Segmented label="Stop at <|im_end|>" value={stop ? "yes" : "no"} options={[{ value: "no", text: "No (benchmark)" }, { value: "yes", text: "Yes (harness)" }]} onChange={(v) => setStop(v === "yes")} />
      </div>
      <div class="tokens" aria-label="Generated tokens">
        {generation.pieces.map((piece, i) => (
          <span class={`token${generation.ids[i] === imEnd ? " special" : ""}${i >= kept ? " dim" : ""}`}>
            {show(piece)}
            <small>{i + 1}</small>
          </span>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="octlm, T4 float16" value={`${fixed(row.tokens_per_second)} tok/s`} note="median of 3 runs" />
        <Stat label="Reference, same GPU" value={`${fixed(row.reference_tokens_per_second)} tok/s`} note="Transformers with SDPA" />
        <Stat label="Greedy tokens" value={row.greedy_equal ? "all equal" : "diverged"} note="every repeat" />
        <Stat label="Answer ends at token" value={end >= 0 ? end + 1 : "never"} note={`${generation.ids.length - (end + 1)} of ${generation.ids.length} tokens after it`} />
        <Stat label="KV cache at the end" value={`${int(row.cache_bytes)} B`} note="matched the formula" />
      </div>
      <p class="widget-note">The benchmark generates exactly 32 tokens so both implementations do the same work. The model finishes its answer and emits &lt;|im_end|&gt;, then keeps going and invents a new "Human:" turn, because nothing tells the loop to stop. Switch to the harness view to see what Day 7 has to keep.</p>
    </div>
  );
}
