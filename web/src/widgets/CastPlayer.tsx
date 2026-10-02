import { useEffect, useRef, useState } from "preact/hooks";

type Span = { text: string; cls: string };
type Event = [number, string];

const SGR = /\x1b\[([0-9;]*)m/;

function load(text: string, idle: number): Event[] {
  const rows = text.trim().split("\n");
  const limit = JSON.parse(rows[0]).idle_time_limit ?? idle;
  let clock = 0;
  let last = 0;
  return rows.slice(1).map((row) => {
    const [time, , data] = JSON.parse(row) as [number, string, string];
    clock += Math.min(time - last, limit);
    last = time;
    return [clock, data];
  });
}

function write(lines: Span[][], style: string[], data: string) {
  const parts = data.split(SGR);
  parts.forEach((part, i) => {
    if (i % 2) {
      const codes = part.split(";").filter(Boolean);
      if (!codes.length || codes.includes("0")) style.length = 0;
      style.push(...codes.filter((c) => c !== "0").map((c) => `t${c}`));
      return;
    }
    part.replace(/\r\n/g, "\n").replace(/\r/g, "").split("\n").forEach((text, j) => {
      if (j) lines.push([]);
      if (text) lines[lines.length - 1].push({ text, cls: style.join(" ") });
    });
  });
}

export default function CastPlayer({ src, title, height = 22, status }: { src: string; title: string; height?: number; status?: "pass" | "fail" }) {
  const [events, setEvents] = useState<Event[]>([]);
  const [at, setAt] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const root = useRef<HTMLDivElement>(null);
  const screen = useRef<HTMLPreElement>(null);

  useEffect(() => {
    fetch(src).then((r) => r.text()).then((text) => setEvents(load(text, 1.5)));
  }, [src]);

  useEffect(() => {
    if (!root.current) return;
    const seen = new IntersectionObserver(([entry]) => setPlaying(entry.isIntersecting && !matchMedia("(prefers-reduced-motion: reduce)").matches), { threshold: 0.35 });
    seen.observe(root.current);
    return () => seen.disconnect();
  }, []);

  useEffect(() => {
    if (!playing || !events.length) return;
    if (at >= events.length) {
      const again = setTimeout(() => setAt(0), 4000);
      return () => clearTimeout(again);
    }
    const wait = at === 0 ? 600 : ((events[at][0] - events[at - 1][0]) * 1000) / speed;
    const next = setTimeout(() => setAt(at + 1), Math.max(wait, 0));
    return () => clearTimeout(next);
  }, [playing, at, events, speed]);

  useEffect(() => {
    if (screen.current) screen.current.scrollTop = screen.current.scrollHeight;
  }, [at]);

  const lines: Span[][] = [[]];
  const style: string[] = [];
  const shown = playing || at ? at : events.length;
  events.slice(0, shown).forEach(([, data]) => write(lines, style, data));
  const done = shown >= events.length && events.length > 0;
  const progress = events.length ? shown / events.length : 0;

  return (
    <div class="term" ref={root}>
      <div class="term-bar">
        <span class="term-dots" aria-hidden="true"><i /><i /><i /></span>
        <span class="term-title">{title}</span>
        {status && done && <span class={`term-verdict ${status}`}>{status === "pass" ? "PASS" : "FAIL"}</span>}
        <span class="term-tools">
          <button type="button" class="term-btn" onClick={() => setSpeed(speed === 1 ? 2 : speed === 2 ? 4 : 1)} aria-label="Playback speed">{speed}×</button>
          <button type="button" class="term-btn" onClick={() => { setAt(0); setPlaying(true); }} aria-label="Restart">
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10a6 6 0 1 0 2-4.5M4 4v3.5h3.5" /></svg>
          </button>
          <button type="button" class="term-btn" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play"}>
            {playing ? <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 5v10M13 5v10" /></svg> : <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 5l8 5-8 5z" /></svg>}
          </button>
        </span>
      </div>
      <pre class="term-screen" ref={screen} style={{ height: `${height * 1.55}em` }} aria-label={`Recording: ${title}`}>
        {lines.map((line, i) => (
          <div class="term-line">
            {line.map((span) => <span class={span.cls}>{span.text}</span>)}
            {i === lines.length - 1 && !done && <span class="term-caret" />}
          </div>
        ))}
      </pre>
      <div class="term-progress" style={{ transform: `scaleX(${progress})` }} />
    </div>
  );
}
