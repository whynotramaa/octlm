import { useState } from "preact/hooks";
import { toFloat16 } from "../lib/floats.ts";
import { loraForward, mergeLora } from "../lib/lora.ts";
import { type Matrix, linear, maxAbsDifference } from "../lib/matrix.ts";
import { Segmented, Stat } from "./ui.tsx";

const IN = 6, OUT = 4, ALPHA = 4;

function random(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 2 ** 32;
    return s / 2 ** 32 - 0.5;
  };
}

const fill = (rows: number, cols: number, next: () => number, size: number): Matrix =>
  Array.from({ length: rows }, () => Array.from({ length: cols }, () => next() * size));

const round16 = (m: Matrix) => m.map((row) => row.map(toFloat16));

export default function LoraMerge() {
  const [rank, setRank] = useState(2);
  const [trained, setTrained] = useState<"zero" | "trained">("zero");
  const next = random(7);
  const base = fill(OUT, IN, next, 8);
  const x = fill(3, IN, next, 4);
  const a = fill(rank, IN, next, 1);
  const b = trained === "zero" ? fill(OUT, rank, () => 0, 0) : fill(OUT, rank, next, 1);
  const scale = ALPHA / rank;
  const plain = linear(x, base);
  const adapted = loraForward(x, base, a, b, scale);
  const merged = mergeLora(base, a, b, scale);
  const shown = (m: Matrix) => m.map((row) => row.map((v) => v.toFixed(3)).join("  ")).join("\n");
  return (
    <div class="widget">
      <div class="widget-controls">
        <Segmented label="Rank r" value={rank} options={[1, 2, 3].map((r) => ({ value: r, text: String(r) }))} onChange={setRank} />
        <Segmented label="B" value={trained} options={[{ value: "zero", text: "step 0, B = 0" }, { value: "trained", text: "after training" }]} onChange={setTrained} />
      </div>
      <div class="grid-controls">
        <div><span class="control-label">Merged W = W₀ + (α/r)·B·A, {OUT}×{IN}</span><pre class="template-text">{shown(merged)}</pre></div>
      </div>
      <div class="readout-grid">
        <Stat label="Adapted vs base output" value={maxAbsDifference(adapted, plain).toExponential(1)} note={trained === "zero" ? "exactly zero: B·A is zero" : "the adapter changed the layer"} />
        <Stat label="Merged vs unmerged, float64" value={maxAbsDifference(linear(x, merged), adapted).toExponential(1)} note="the merge is exact up to rounding" />
        <Stat label="Merged weights stored as float16" value={maxAbsDifference(linear(x, round16(merged)), adapted).toExponential(1)} note="the rounding cost of storing W" />
      </div>
      <p class="widget-note">A toy layer with {IN} inputs and {OUT} outputs, α = {ALPHA}. The adapter adds x·Aᵀ·Bᵀ·α/r to the base output. Merging folds B·A·α/r into W once, so inference runs one matmul instead of three. The merge itself is exact. The error comes from storing the merged W in float16, and it grows with the size of the weights and the length of the sums.</p>
    </div>
  );
}
