import { useState } from "preact/hooks";
import { mulberry32 } from "../lib/sampling.ts";
import { Segmented, Stat, int } from "./ui.tsx";

type Split = { name: string; code: number; prose: number };
const SOURCES = [
  { name: "stdlib", chunks: 20 },
  ...[1, 2, 3, 4, 5, 6].map((i) => ({ name: `book ${i}`, chunks: 5 + ((i * 3) % 5) })),
];
const random = mulberry32(5);
const CHUNKS = SOURCES.flatMap((s, source) => Array.from({ length: s.chunks }, () => ({ source, draw: random() })));
const ranked = [...CHUNKS].sort((a, b) => a.draw - b.draw);
const cut = ranked[Math.round(CHUNKS.length * 0.1)].draw;

export default function CorpusSplit({ splits }: { splits: Split[] }) {
  const [mode, setMode] = useState<"chunk" | "book">("chunk");
  const isVal = (c: { source: number; draw: number }) => (mode === "chunk" ? c.draw < cut : c.source === 3);
  const present = new Set(CHUNKS.filter(isVal).map((c) => c.source)).size;
  const total = Math.max(...splits.map((s) => s.code + s.prose));
  return (
    <div class="widget">
      <div class="bars">
        {splits.map((s) => (
          <div class="bar-row" style={{ gridTemplateColumns: "92px 1fr 110px" }}>
            <span class="bar-name">{s.name}</span>
            <span class="bar-track" style={{ display: "flex" }}>
              <span class="bar-fill" style={{ width: `${(s.code / total) * 100}%`, background: "var(--series-1)", borderRadius: "6px 0 0 6px" }} title={`code ${int(s.code)} bytes`} />
              <span class="bar-fill" style={{ width: `${(s.prose / total) * 100}%`, background: "var(--series-2)", borderRadius: "0 6px 6px 0" }} title={`prose ${int(s.prose)} bytes`} />
            </span>
            <span class="bar-value">{int((s.code + s.prose) / 1000)} KB</span>
          </div>
        ))}
      </div>
      <div class="legend">
        <span class="legend-item"><i class="swatch s1" />code</span>
        <span class="legend-item"><i class="swatch s2" />prose</span>
      </div>
      <Segmented label="Hold out 10 percent" value={mode} options={[{ value: "chunk", text: "by chunk (octlm)" }, { value: "book", text: "by whole book" }]} onChange={setMode} />
      <div class="bars">
        {SOURCES.map((s, source) => (
          <div class="bar-row" style={{ gridTemplateColumns: "64px 1fr" }}>
            <span class="bar-name">{s.name}</span>
            <span class="tokens" style={{ gap: "3px" }}>
              {CHUNKS.filter((c) => c.source === source).map((c) => <span class={`token ${isVal(c) ? "miss" : ""}`} style={{ width: "14px", height: "14px", padding: 0 }} />)}
            </span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="Sources in validation" value={`${present} of ${SOURCES.length}`} />
        <Stat label="Validation chunks" value={CHUNKS.filter(isVal).length} note={`of ${CHUNKS.length}`} />
      </div>
      <p class="widget-note">Top: the real Colab build, from notes/day2.md. Bottom: an illustration with made-up chunk counts, orange squares held out. Holding out whole books would test one author the model never read. Holding out chunks keeps every source in both splits, and since chunks share no text, no validation sentence is ever trained on.</p>
    </div>
  );
}
