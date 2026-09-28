import { useMemo, useState } from "preact/hooks";
import Heatmap from "../charts/Heatmap.tsx";
import { sinusoidal } from "../lib/positions.ts";
import { Slider } from "./ui.tsx";

function similarity(table: number[][]) {
  const norms = table.map((row) => Math.hypot(...row));
  return table.map((a, i) => table.map((b, j) => a.reduce((s, v, c) => s + v * b[c], 0) / (norms[i] * norms[j])));
}

export default function SinusoidalMap() {
  const [width, setWidth] = useState(16);
  const [length, setLength] = useState(24);
  const table = useMemo(() => sinusoidal(length, width), [length, width]);
  const sim = useMemo(() => similarity(table), [table]);
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Width (columns)" value={width} min={4} max={32} step={2} onInput={setWidth} />
        <Slider label="Positions" value={length} min={8} max={32} onInput={setLength} />
      </div>
      <div class="maps maps-2">
        <Heatmap title="sinusoidal_positions(T, C)" matrix={table} maxAbs={1} rowLabel={(i) => `${i}`} />
        <Heatmap title="cosine similarity between positions" matrix={sim} maxAbs={1} />
      </div>
      <p class="widget-note">Left: each row is one position. Columns come in sin and cos pairs, and the pair frequency falls from left to right, so the first columns flicker and the last barely move. Right: row i, column j compares position i with position j. The bright diagonal band says nearby positions look alike, and the band has the same width everywhere, because similarity depends on the distance i − j.</p>
    </div>
  );
}
