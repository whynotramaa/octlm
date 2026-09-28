import { useState } from "preact/hooks";

export type Stage = { name: string; shape?: string; detail: string };

export default function Flow({ stages, loop = false }: { stages: Stage[]; loop?: boolean }) {
  const [at, setAt] = useState(0);
  return (
    <div class="widget">
      <div class="flow" role="list">
        {stages.map((s, i) => (
          <>
            {i > 0 && <span class="flow-arrow" aria-hidden="true">→</span>}
            <button type="button" role="listitem" class={`flow-stage${i === at ? " on" : ""}`} aria-pressed={i === at} onClick={() => setAt(i)}>
              {s.name}
              {s.shape && <small>{s.shape}</small>}
            </button>
          </>
        ))}
        {loop && <span class="flow-arrow" aria-label="back to the first stage">↺</span>}
      </div>
      <p class="widget-note">{stages[at].detail}</p>
      <div class="widget-nav">
        <button type="button" class="button" disabled={at === 0} onClick={() => setAt(at - 1)}>Back</button>
        <span class="muted">Stage {at + 1} of {stages.length}</span>
        <button type="button" class="button primary" disabled={at === stages.length - 1} onClick={() => setAt(at + 1)}>Next</button>
      </div>
    </div>
  );
}
