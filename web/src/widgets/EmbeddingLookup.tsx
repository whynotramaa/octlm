import { useState } from "preact/hooks";
import Heatmap from "../charts/Heatmap.tsx";
import type { Matrix } from "../lib/matrix.ts";
import { sinusoidal } from "../lib/positions.ts";
import { mulberry32 } from "../lib/sampling.ts";
import { Segmented } from "./ui.tsx";

const WORDS = ["the", "cat", "sat", "on", "mat", "dog", "ran", "a"];
const WIDTH = 8;
const LENGTH = 6;
const random = mulberry32(3);
const normal = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
const tokenTable: Matrix = WORDS.map(() => Array.from({ length: WIDTH }, () => normal() * 0.5));
const learnedTable: Matrix = Array.from({ length: LENGTH }, () => Array.from({ length: WIDTH }, () => normal() * 0.5));

export default function EmbeddingLookup() {
  const [sentence, setSentence] = useState([0, 1, 2, 3, 0, 4]);
  const [position, setPosition] = useState<"learned" | "sinusoidal">("learned");
  const table = position === "learned" ? learnedTable : sinusoidal(LENGTH, WIDTH);
  const tokens = sentence.map((id) => tokenTable[id]);
  const sum = tokens.map((row, i) => row.map((v, c) => v + table[i][c]));
  const label = (i: number) => WORDS[sentence[i]];
  return (
    <div class="widget">
      <div class="widget-controls">
        {sentence.map((id, i) => (
          <label class="control" style={{ minWidth: "80px" }}>
            <span class="control-label">pos {i}</span>
            <select value={id} onChange={(e) => setSentence(sentence.map((v, j) => (j === i ? Number(e.currentTarget.value) : v)))}>
              {WORDS.map((word, w) => <option value={w}>{word} (ID {w})</option>)}
            </select>
          </label>
        ))}
        <Segmented label="Positions" value={position} options={[{ value: "learned", text: "Learned" }, { value: "sinusoidal", text: "Sinusoidal" }]} onChange={setPosition} />
      </div>
      <div class="maps maps-3">
        <Heatmap title="token_embedding[ids]" matrix={tokens} rowLabel={label} />
        <Heatmap title="position rows 0..5" matrix={table} rowLabel={(i) => `p${i}`} />
        <Heatmap title="sum = model input" matrix={sum} rowLabel={label} />
      </div>
      <p class="widget-note">Rows with the same word share one token row. Set pos 0 and pos 4 to "the": their token rows match, and the sum differs only by the position row. Without that row, attention could not tell the two apart.</p>
    </div>
  );
}
