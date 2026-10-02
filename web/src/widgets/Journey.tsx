import { useState } from "preact/hooks";

type Stop = { day: number; title: string; act: string; built: string; metric: string; unit: string; status: string; href: string; posts: number; exps: string };

const pad = (n: number) => String(n).padStart(2, "0");

export default function Journey({ stops }: { stops: Stop[] }) {
  const [pick, setPick] = useState(stops.length - 2);
  const stop = stops[pick];
  return (
    <div class="journey">
      <div class="journey-rail" role="tablist" aria-label="Roadmap day">
        {stops.map((s, i) => (
          <button type="button" role="tab" aria-selected={i === pick} class={`journey-stop${i < pick ? " past" : ""}${s.status === "Running" ? " running" : ""}`} onClick={() => setPick(i)} onMouseEnter={() => setPick(i)}>
            <span class="journey-num">{pad(s.day)}</span>
            <span class="journey-node" aria-hidden="true" />
            <span class="journey-act">{s.act}</span>
          </button>
        ))}
      </div>
      <div class="journey-card" role="tabpanel" key={stop.day}>
        <div class="journey-copy">
          <span class="journey-day">Day {pad(stop.day)} · {stop.exps}</span>
          <h3>{stop.title}</h3>
          <p>{stop.built}</p>
          <a class="journey-link" href={stop.href}>
            Read {stop.posts} {stop.posts === 1 ? "post" : "posts"}
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h11m-4-4.5L15.5 10 11 14.5" /></svg>
          </a>
        </div>
        <div class="journey-figure">
          <span class={`journey-status${stop.status === "Running" ? " running" : ""}`}>{stop.status}</span>
          <span class="journey-metric">{stop.metric}</span>
          <span class="journey-unit">{stop.unit}</span>
        </div>
      </div>
    </div>
  );
}
