import { useState } from "preact/hooks";
import { softmax } from "../lib/matrix.ts";
import { crossEntropy, perplexity } from "../lib/metrics.ts";
import { Slider, Stat, fixed } from "./ui.tsx";

const WORDS = [" mat", " floor", " sofa", " roof", " dog", " moon"];
const OTHERS = [1.6, 1.1, 0.4, -0.6, -1.5];

export default function LossMeter() {
  const [score, setScore] = useState(1.2);
  const [bytesPerToken, setBytesPerToken] = useState(2.3);
  const probabilities = softmax([score, ...OTHERS]);
  const p = probabilities[0];
  const loss = crossEntropy(p);
  const bits = loss / Math.LN2;
  return (
    <div class="widget">
      <p class="widget-note">Context: <code>The cat sat on the</code>. The true next token is <code>" mat"</code>. Move the model's score for it and watch every metric follow.</p>
      <div class="grid-controls widget-controls">
        <Slider label={'Score (logit) for " mat"'} value={score} min={-4} max={8} step={0.1} shown={fixed(score, 1)} onInput={setScore} />
        <Slider label="Bytes per token" value={bytesPerToken} min={1} max={5} step={0.1} shown={fixed(bytesPerToken, 1)} onInput={setBytesPerToken} />
      </div>
      <div class="bars" role="list">
        {WORDS.map((word, i) => (
          <div class={`bar-row ${i === 0 ? "edited" : ""}`} role="listitem" style={{ gridTemplateColumns: "64px 1fr 64px" }}>
            <span class="bar-name">"{word}"</span>
            <span class="bar-track"><span class="bar-fill" style={{ width: `${probabilities[i] * 100}%` }} /></span>
            <span class="bar-value">{fixed(probabilities[i], 3)}</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="p(correct)" value={fixed(p, 3)} note="softmax of the scores" />
        <Stat label="Cross entropy" value={`${fixed(loss, 3)} nats`} note="−ln p" />
        <Stat label="Bits" value={fixed(bits, 3)} note="−log₂ p" />
        <Stat label="Perplexity" value={fixed(perplexity(loss), 2)} note="e^loss = 1/p" />
        <Stat label="Bits per byte" value={fixed(bits / bytesPerToken, 3)} note="bits ÷ bytes per token" />
      </div>
      <p class="widget-note">A uniform guess over these 6 words gives p = 0.167, loss {fixed(Math.log(6), 3)} nats and perplexity 6. Perplexity reads as "the model is as unsure as a fair choice among this many tokens".</p>
    </div>
  );
}
