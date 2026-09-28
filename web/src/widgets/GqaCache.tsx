import { useState } from "preact/hooks";
import { kvCacheBytes } from "../lib/model.ts";
import { Segmented, Slider, Stat, int } from "./ui.tsx";

const QUERIES = 8;
const W = 640;

function Groups({ kv }: { kv: number }) {
  const qx = (i: number) => 40 + i * ((W - 80) / (QUERIES - 1));
  const kx = (j: number) => (kv === 1 ? W / 2 : 40 + j * ((W - 80) / (kv - 1)));
  const group = QUERIES / kv;
  return (
    <div class="diagram">
      <svg viewBox={`0 0 ${W} 170`} role="img" aria-label={`${QUERIES} query heads sharing ${kv} key and value heads`}>
        {Array.from({ length: QUERIES }, (_, i) => (
          <path class="link" d={`M ${qx(i)} 44 C ${qx(i)} 90, ${kx(Math.floor(i / group))} 90, ${kx(Math.floor(i / group))} 122`} />
        ))}
        {Array.from({ length: QUERIES }, (_, i) => (
          <g>
            <rect class="box on" x={qx(i) - 26} y="12" width="52" height="32" rx="8" />
            <text class="t" x={qx(i)} y="33" text-anchor="middle">Q{i}</text>
          </g>
        ))}
        {Array.from({ length: kv }, (_, j) => (
          <g>
            <rect class="box alt" x={kx(j) - 34} y="122" width="68" height="34" rx="8" />
            <text class="t" x={kx(j)} y="144" text-anchor="middle">K{j} V{j}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

const PRESETS = {
  day2: { name: "Day 2 grid", layers: 4, width: 256, length: 4096 },
  paper: { name: "Paper exercise", layers: 12, width: 512, length: 4096 },
  day4: { name: "Day 4 model", layers: 8, width: 512, length: 512 },
};

export default function GqaCache() {
  const [kv, setKv] = useState(2);
  const [preset, setPreset] = useState<keyof typeof PRESETS>("day2");
  const [length, setLength] = useState(PRESETS.day2.length);
  const [bytes, setBytes] = useState(2);
  const p = PRESETS[preset];
  const shape = { n_layers: p.layers, d_model: p.width, n_heads: QUERIES, kv_heads: kv };
  const cache = kvCacheBytes(shape, length, bytes);
  const full = kvCacheBytes({ ...shape, kv_heads: QUERIES }, length, bytes);
  const perToken = 2 * kv * (p.width / QUERIES) * bytes;
  return (
    <div class="widget">
      <Segmented label="KV heads" value={kv} options={[{ value: 8, text: "8 (MHA)" }, { value: 4, text: "4 (GQA)" }, { value: 2, text: "2 (GQA)" }, { value: 1, text: "1 (MQA)" }]} onChange={setKv} />
      <Groups kv={kv} />
      <div class="grid-controls widget-controls">
        <Segmented label="Shape" value={preset} options={Object.entries(PRESETS).map(([value, v]) => ({ value: value as keyof typeof PRESETS, text: v.name }))} onChange={(v) => { setPreset(v); setLength(PRESETS[v].length); }} />
        <Slider label="Positions cached" value={Math.log2(length)} min={7} max={15} shown={int(length)} onInput={(v) => setLength(2 ** v)} />
        <Segmented label="Element" value={bytes} options={[{ value: 4, text: "float32" }, { value: 2, text: "bf16 / fp16" }, { value: 1, text: "int8" }]} onChange={setBytes} />
      </div>
      <div class="readout-grid">
        <Stat label="Per token per layer" value={`${int(perToken)} B`} note={`2 × ${kv} × ${p.width / QUERIES} × ${bytes}`} />
        <Stat label="Cache for one sequence" value={`${(cache / 2 ** 20).toLocaleString("en-US", { maximumFractionDigits: 2 })} MiB`} note={`× ${p.layers} layers × ${int(length)} positions`} />
        <Stat label="Against MHA" value={`${QUERIES / kv}× smaller`} note={`MHA needs ${(full / 2 ** 20).toLocaleString("en-US", { maximumFractionDigits: 2 })} MiB`} />
      </div>
      <p class="widget-note">Each group of {QUERIES / kv} query head{kv === QUERIES ? "" : "s"} reads the same key and value head. Query heads still compute their own scores, so they can look for different things. They just search one shared set of keys.</p>
    </div>
  );
}
