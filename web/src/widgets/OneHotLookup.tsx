import { useState } from "preact/hooks";
import Heatmap from "../charts/Heatmap.tsx";
import { matmul, type Matrix } from "../lib/matrix.ts";
import { mulberry32 } from "../lib/sampling.ts";

const WORDS = ["the", "cat", "sat", "on", "mat", "dog"];
const random = mulberry32(7);
const table: Matrix = WORDS.map(() => Array.from({ length: 5 }, () => Math.round((random() * 2 - 1) * 100) / 100));

export default function OneHotLookup() {
  const [id, setId] = useState(1);
  const oneHot = [WORDS.map((_, i) => (i === id ? 1 : 0))];
  return (
    <div class="widget">
      <div class="tokens" role="group" aria-label="Token">
        {WORDS.map((w, i) => (
          <button type="button" class={`token${i === id ? " new" : ""}`} aria-pressed={i === id} onClick={() => setId(i)}>
            {w}<small>ID {i}</small>
          </button>
        ))}
      </div>
      <div class="maps maps-3">
        <Heatmap title="one-hot, 1 × V" matrix={oneHot} showValues columnLabel={(c) => String(c)} rowLabel={() => ""} maxAbs={1} />
        <Heatmap title="table E, V × C" matrix={table} showValues highlightRow={id} rowLabel={(r) => String(r)} maxAbs={1} />
        <Heatmap title="one-hot × E, 1 × C" matrix={matmul(oneHot, table)} showValues rowLabel={() => ""} maxAbs={1} />
      </div>
      <p class="widget-note">Multiplying a one-hot vector by the table picks out one row, because every other row is multiplied by 0. That is why an embedding is written as a matrix product in papers and computed as an index in code: <code>E[{id}]</code> gives the same numbers with no multiplications. The gradient reaches only row {id}.</p>
    </div>
  );
}
