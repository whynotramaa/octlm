import { useMemo, useState } from "preact/hooks";
import { mulberry32 } from "../lib/sampling.ts";
import { Segmented, Stat, fixed } from "./ui.tsx";

const VOCAB = 1024;
const WIDTH = 128;
const DRAWS = 24;

function initialLoss(std: number, seed: number) {
  const random = mulberry32(seed);
  const normal = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
  let total = 0;
  let peak = 0;
  for (let draw = 0; draw < DRAWS; draw++) {
    const logits = Array.from({ length: VOCAB }, () => normal() * std * Math.sqrt(WIDTH));
    const top = Math.max(...logits);
    const lse = top + Math.log(logits.reduce((s, z) => s + Math.exp(z - top), 0));
    total += lse - logits[0];
    peak += Math.exp(top - lse);
  }
  return { loss: total / DRAWS, peak: peak / DRAWS };
}

export default function InitScale() {
  const [std, setStd] = useState(1);
  const result = useMemo(() => initialLoss(std, 11), [std]);
  return (
    <div class="widget">
      <Segmented label="Standard deviation of embedding and linear weights" value={std} options={[
        { value: 1, text: "1.0 (PyTorch Embedding default)" },
        { value: 0.1, text: "0.1" },
        { value: 0.02, text: "0.02 (GPT-2, octlm)" },
      ]} onChange={setStd} />
      <div class="readout-grid">
        <Stat label="Logit standard deviation" value={fixed(std * Math.sqrt(WIDTH))} note={`σ × √${WIDTH}`} />
        <Stat label="Top token probability" value={fixed(result.peak, 3)} note={`uniform is ${fixed(1 / VOCAB, 4)}`} />
        <Stat label="Expected loss at step 0" value={`${fixed(result.loss)} nats`} note={`ln ${VOCAB} = ${fixed(Math.log(VOCAB))}`} />
      </div>
      <p class="widget-note">The tied head scores each token by a dot product between the final hidden vector, about unit size per entry after the last norm, and that token's embedding row. With rows drawn at σ = 1, the scores spread by σ√{WIDTH} ≈ 11, one random token gets most of the probability, and the loss starts far above the ln V a blank model should show. Average of {DRAWS} random draws, V = {VOCAB}, width {WIDTH}, the Day 1 shape.</p>
    </div>
  );
}
