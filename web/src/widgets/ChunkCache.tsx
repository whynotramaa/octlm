import { useState } from "preact/hooks";
import { pretokenize } from "../lib/pretokenize.ts";
import { Stat, int } from "./ui.tsx";

const STORY = "Once upon a time, there was a little girl named Lily. Lily liked to play in the park. One day, Lily saw a big dog in the park. The dog was happy and wanted to play. Lily and the dog played all day. They were very happy.";

const show = (text: string) => text.replace(/ /g, "␠").replace(/\n/g, "\\n");

export default function ChunkCache() {
  const [text, setText] = useState(STORY);
  const seen = new Set<string>();
  const chunks = pretokenize(text).map((chunk) => {
    const hit = seen.has(chunk);
    seen.add(chunk);
    return { chunk, hit };
  });
  const hits = chunks.filter((c) => c.hit).length;
  return (
    <div class="widget">
      <textarea rows={4} value={text} onInput={(e) => setText(e.currentTarget.value)} aria-label="Story text" />
      <div class="tokens">
        {chunks.map(({ chunk, hit }) => <span class={hit ? "token hit" : "token miss"}>{show(chunk)}</span>)}
      </div>
      <div class="legend">
        <span class="legend-item"><span class="token miss">miss</span> run the merge loop, store the IDs</span>
        <span class="legend-item"><span class="token hit">hit</span> copy the stored IDs</span>
      </div>
      <div class="readout-grid">
        <Stat label="Chunks" value={int(chunks.length)} />
        <Stat label="Distinct chunks" value={int(seen.size)} note="merge loops run" />
        <Stat label="Hit rate" value={`${chunks.length ? ((hits / chunks.length) * 100).toFixed(1) : "0.0"}%`} />
      </div>
      <p class="widget-note">Pre-tokenization splits text into chunks and BPE never merges across a chunk edge, so a chunk always encodes to the same IDs. On one paragraph the hit rate is modest. Over 2.7 million stories drawn from a small vocabulary, almost every chunk is a word the encoder has already seen.</p>
    </div>
  );
}
