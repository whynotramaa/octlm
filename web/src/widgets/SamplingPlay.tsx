import { useState } from "preact/hooks";
import { draw, mulberry32, samplingDistribution } from "../lib/sampling.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

export const WORDS = [" happy", " sad", " big", " little", " red", " very", " the", " ran"];
export const LOGITS = [3.1, 1.2, 2.2, 2.6, 0.4, 1.9, -0.5, -1.2];
const DRAWS = 200;

export default function SamplingPlay() {
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(0);
  const [seed, setSeed] = useState(0);
  const greedy = temperature === 0;
  const probabilities = greedy ? LOGITS.map((_, i) => (i === 0 ? 1 : 0)) : samplingDistribution(LOGITS, temperature, topK);
  const random = mulberry32(seed);
  const counts = new Array(WORDS.length).fill(0);
  for (let i = 0; i < DRAWS; i++) counts[greedy ? 0 : draw(probabilities, random())]++;
  const entropy = -probabilities.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0);
  return (
    <div class="widget">
      <p class="widget-note">Context: <code>The dog was very</code>. These are made-up scores for eight candidate tokens.</p>
      <div class="grid-controls widget-controls">
        <Slider label="Temperature (0 = greedy)" value={temperature} min={0} max={2} step={0.05} shown={fixed(temperature)} onInput={setTemperature} />
        <Slider label="Top-k (0 = off)" value={topK} min={0} max={8} onInput={setTopK} />
      </div>
      <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Token</th><th class="num">Logit</th><th>Probability</th><th class="num">p</th><th class="num">Drawn in {DRAWS}</th></tr></thead>
          <tbody>
            {WORDS.map((word, i) => (
              <tr>
                <td><code>"{word}"</code></td>
                <td class="num">{fixed(LOGITS[i], 1)}</td>
                <td style={{ width: "40%" }}><span class="bar-track" style={{ display: "block" }}><span class="bar-fill" style={{ width: `${probabilities[i] * 100}%`, background: "var(--series-1)" }} /></span></td>
                <td class="num">{fixed(probabilities[i], 3)}</td>
                <td class="num">{counts[i]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div class="widget-controls">
        <Stat label="Entropy" value={`${fixed(entropy)} bits`} note={`uniform over 8 is 3.00`} />
        <button type="button" class="button" onClick={() => setSeed(seed + 1)}>Draw again (seed {seed + 1})</button>
      </div>
    </div>
  );
}
