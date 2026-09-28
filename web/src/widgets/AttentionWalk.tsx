import { useMemo, useState } from "preact/hooks";
import fixture from "../../fixtures/attention.json";
import Heatmap from "../charts/Heatmap.tsx";
import { type Head, causalSelfAttention } from "../lib/attention.ts";
import { type Matrix, linear } from "../lib/matrix.ts";

const weights = { query: fixture.query, key: fixture.key, value: fixture.value, output: fixture.output_weight };
const position = (i: number) => `t${i}`;

type Stage = {
  name: string;
  shape: string;
  text: string;
  maps: (view: View) => { title: string; matrix: Matrix; mode?: "signed" | "weight"; values?: boolean; rows?: boolean }[];
};

type View = Head & { x: Matrix; fullQ: Matrix; fullK: Matrix; fullV: Matrix; joined: Matrix; output: Matrix };

const full = { fullQ: linear(fixture.x, weights.query), fullK: linear(fixture.x, weights.key), fullV: linear(fixture.x, weights.value) };

const STAGES: Stage[] = [
  {
    name: "Input",
    shape: "x: [1, 8, 16]",
    text: "Eight positions, each a vector of 16 numbers. In a trained model these come from the embedding table. Here they are random draws from torch.randn with seed 0.",
    maps: (h) => [{ title: "x", matrix: h.x }],
  },
  {
    name: "Project",
    shape: "q, k, v: [1, 8, 16]",
    text: "Three learned matrices turn each position into a query (what it looks for), a key (what it offers) and a value (what it passes on). Each is x times a 16 × 16 weight.",
    maps: (h) => [
      { title: "q = x Wqᵀ", matrix: h.fullQ },
      { title: "k = x Wkᵀ", matrix: h.fullK },
      { title: "v = x Wvᵀ", matrix: h.fullV },
    ],
  },
  {
    name: "Split heads",
    shape: "q, k, v: [1, 2, 8, 8]",
    text: "The 16 columns split into 2 heads of width 8. Each head runs its own attention on its own slice, so it can learn a different pattern.",
    maps: (h) => [
      { title: "q (this head)", matrix: h.q },
      { title: "k (this head)", matrix: h.k },
      { title: "v (this head)", matrix: h.v },
    ],
  },
  {
    name: "Scores",
    shape: "q kᵀ / √8: [1, 2, 8, 8]",
    text: "Row i, column j is the dot product of query i with key j, divided by √8. A large score means position i finds position j relevant.",
    maps: (h) => [{ title: "scores", matrix: h.scores, values: true, rows: true }],
  },
  {
    name: "Mask",
    shape: "masked scores: [1, 2, 8, 8]",
    text: "Every column to the right of the diagonal is a future position. The mask sets those scores to −∞, so no position can read a token that comes after it.",
    maps: (h) => [{ title: "masked scores", matrix: h.masked, values: true, rows: true }],
  },
  {
    name: "Softmax",
    shape: "weights: [1, 2, 8, 8]",
    text: "Softmax along each row turns scores into weights that sum to 1. The exponent of −∞ is 0, so masked positions get exactly zero weight.",
    maps: (h) => [{ title: "weights", matrix: h.weights, mode: "weight", values: true, rows: true }],
  },
  {
    name: "Mix values",
    shape: "weights · v: [1, 2, 8, 8]",
    text: "Each output row is a weighted average of the value rows it may see. Row 0 can only copy v[0]. Row 7 blends all eight.",
    maps: (h) => [
      { title: "weights", matrix: h.weights, mode: "weight", rows: true },
      { title: "out = weights · v", matrix: h.out, rows: true },
    ],
  },
  {
    name: "Join",
    shape: "output: [1, 8, 16]",
    text: "The heads are placed side by side again, back to 16 columns, and one more 16 × 16 matrix mixes them. This matches octlm's CausalSelfAttention to within 1e-5.",
    maps: (h) => [
      { title: "joined heads", matrix: h.joined },
      { title: "output = joined Woᵀ", matrix: h.output },
    ],
  },
];

export default function AttentionWalk() {
  const [stage, setStage] = useState(0);
  const [headIndex, setHeadIndex] = useState(0);
  const [row, setRow] = useState<number | null>(null);
  const result = useMemo(() => causalSelfAttention(fixture.x, weights, fixture.n_heads), []);
  const view: View = { ...result.heads[headIndex], ...full, x: fixture.x, joined: result.joined, output: result.output };
  const current = STAGES[stage];
  const maps = current.maps(view);
  const perHead = stage >= 2 && stage <= 6;
  return (
    <div class="widget">
      <div class="widget-controls">
        <div class="stepper" role="tablist" aria-label="Attention stage">
          {STAGES.map((s, i) => (
            <button type="button" role="tab" aria-selected={i === stage} class={i === stage ? "step on" : "step"} onClick={() => setStage(i)}>
              <span class="step-index">{i + 1}</span>
              <span class="step-name">{s.name}</span>
            </button>
          ))}
        </div>
      </div>
      <div class="widget-status">
        <code class="shape">{current.shape}</code>
        {perHead && (
          <div class="segmented" role="group" aria-label="Head">
            {[0, 1].map((h) => (
              <button type="button" class={h === headIndex ? "on" : ""} aria-pressed={h === headIndex} onClick={() => setHeadIndex(h)}>
                Head {h + 1}
              </button>
            ))}
          </div>
        )}
      </div>
      <p class="widget-note">{current.text}</p>
      <div class={`maps maps-${maps.length}`}>
        {maps.map((m) => (
          <Heatmap
            title={m.title}
            matrix={m.matrix}
            mode={m.mode}
            showValues={m.values}
            rowLabel={position}
            columnLabel={m.matrix[0].length === 8 && m.rows ? position : undefined}
            highlightRow={m.rows ? row : null}
            onRow={m.rows ? setRow : undefined}
          />
        ))}
      </div>
      <div class="widget-nav">
        <button type="button" class="button" disabled={stage === 0} onClick={() => setStage(stage - 1)}>Back</button>
        <span class="muted">Step {stage + 1} of {STAGES.length}</span>
        <button type="button" class="button primary" disabled={stage === STAGES.length - 1} onClick={() => setStage(stage + 1)}>Next</button>
      </div>
    </div>
  );
}
