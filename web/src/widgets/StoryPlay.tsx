import { useEffect, useState } from "preact/hooks";
import { repeatedGrams } from "../lib/repeats.ts";

type Sample = { prompt: string; temperature: number; top_k: number; seed: number; text: string };

const SCRAMBLE_TAIL = 4;
const scramble = (text: string) => text.replace(/[A-Za-z]/g, () => String.fromCharCode(97 + Math.floor(Math.random() * 26)));

export default function StoryPlay({ samples }: { samples: Sample[] }) {
  const prompts = [...new Set(samples.map((s) => s.prompt))];
  const [prompt, setPrompt] = useState(prompts[0]);
  const [greedy, setGreedy] = useState(false);
  const sample = samples.find((s) => s.prompt === prompt && (s.temperature === 0) === greedy)!;
  const { pieces, repeated, total } = repeatedGrams(sample.text.slice(sample.prompt.length));
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return setCount(pieces.length);
    setCount(0);
    const tick = setInterval(() => setCount((c) => (c >= pieces.length ? (clearInterval(tick), c) : c + 1)), 26);
    return () => clearInterval(tick);
  }, [prompt, greedy]);

  const done = count >= pieces.length;
  const settled = done ? count : Math.max(count - SCRAMBLE_TAIL, 0);
  return (
    <div class="play">
      <div class="play-prompts" role="radiogroup" aria-label="Prompt">
        {prompts.map((p, i) => (
          <button type="button" role="radio" aria-checked={p === prompt} class="play-prompt" onClick={() => setPrompt(p)}>
            <span class="play-n">{String(i + 1).padStart(2, "0")}</span>{p}
          </button>
        ))}
      </div>
      <div class="play-page">
        <div class="play-head">
          <div class="play-switch" role="radiogroup" aria-label="Decoding">
            <button type="button" role="radio" aria-checked={!greedy} onClick={() => setGreedy(false)}>Temperature 0.8 · top-k 40</button>
            <button type="button" role="radio" aria-checked={greedy} onClick={() => setGreedy(true)}>Greedy</button>
          </div>
          <span class="play-meta">seed {sample.seed} · 26.3M params · 200 tokens</span>
        </div>
        <p class="play-story">
          <strong>{sample.prompt}</strong>
          <span class={`play-gen${done ? " in" : ""}`}>
            {pieces.slice(0, count).map((piece, i) =>
              i >= settled ? <span class="play-scramble">{scramble(piece.text)}</span> : piece.repeat && done ? <mark>{piece.text}</mark> : piece.text,
            )}
          </span>
          {!done && <span class="play-caret" />}
          {done && <span class="play-cut">…</span>}
        </p>
        <div class={`play-foot${done ? " in" : ""}`}>
          <span><b>{repeated}</b> of {total} four-word runs repeat</span>
          <span>{greedy ? "Greedy decoding loops. That is why Day 4 samples." : "Sampling keeps the story moving."}</span>
        </div>
      </div>
    </div>
  );
}
