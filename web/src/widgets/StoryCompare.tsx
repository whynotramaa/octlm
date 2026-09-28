import { useState } from "preact/hooks";
import { repeatedGrams } from "../lib/repeats.ts";
import { Stat } from "./ui.tsx";

type Sample = { prompt: string; temperature: number; top_k: number; seed: number; text: string };

function Story({ title, sample }: { title: string; sample: Sample }) {
  const { pieces, repeated, total } = repeatedGrams(sample.text.slice(sample.prompt.length));
  return (
    <div class="story-col">
      <div class="chart-title">{title}</div>
      <p class="story">
        <span class="story-prompt">{sample.prompt}</span>
        {pieces.map((piece) => (piece.repeat ? <mark class="repeat">{piece.text}</mark> : piece.text))}
        <span class="story-cut">…</span>
      </p>
      <Stat label="Repeated 4-grams" value={`${repeated} of ${total}`} note={`${((repeated / Math.max(total, 1)) * 100).toFixed(1)}% of the generated text`} />
    </div>
  );
}

export default function StoryCompare({ samples }: { samples: Sample[] }) {
  const prompts = [...new Set(samples.map((s) => s.prompt))];
  const [prompt, setPrompt] = useState(prompts[0]);
  const greedy = samples.find((s) => s.prompt === prompt && s.temperature === 0)!;
  const sampled = samples.find((s) => s.prompt === prompt && s.temperature > 0)!;
  return (
    <div class="widget">
      <label class="control">
        <span class="control-label">Prompt</span>
        <select value={prompt} onChange={(e) => setPrompt(e.currentTarget.value)}>
          {prompts.map((p, i) => <option value={p}>{i + 1}. {p}</option>)}
        </select>
      </label>
      <div class="two-col">
        <Story title="Greedy" sample={greedy} />
        <Story title={`Temperature ${sampled.temperature}, top-k ${sampled.top_k}, seed ${sampled.seed}`} sample={sampled} />
      </div>
      <div class="legend">
        <span class="legend-item"><span class="story-prompt">Prompt</span> given to the model</span>
        <span class="legend-item"><mark class="repeat">marked words</mark> belong to a 4-word run that already appeared earlier in the same story</span>
      </div>
    </div>
  );
}
