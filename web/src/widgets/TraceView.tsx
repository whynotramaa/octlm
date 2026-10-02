import { useState } from "preact/hooks";
import { Stat, int } from "./ui.tsx";

type Message = { role: string; content: string };
type Trace = { task: string; kind: string; turns: [number, number][] };

export default function TraceView({ traces, messages }: { traces: Trace[]; messages: Record<string, Message[]> }) {
  const [task, setTask] = useState(traces.find((t) => t.kind === "tests")!.task);
  const trace = traces.find((t) => t.task === task)!;
  const shown = messages[task];
  const longest = Math.max(...trace.turns.map(([p, t]) => p + t));
  const target = trace.turns.reduce((sum, [, t]) => sum + t, 0);
  const all = trace.turns.reduce((sum, [p, t]) => sum + p + t, 0);
  let turn = -1;
  return (
    <div class="widget">
      <label class="control">
        <span class="control-label">Training task</span>
        <select value={task} onChange={(e) => setTask(e.currentTarget.value)}>
          {["answer", "file", "hidden", "tests"].map((kind) => (
            <optgroup label={kind}>{traces.filter((t) => t.kind === kind).map((t) => <option value={t.task}>{t.task}</option>)}</optgroup>
          ))}
        </select>
      </label>
      <div class="trace-messages">
        {shown.map((m, i) => {
          if (m.role === "assistant") turn += 1;
          const text = m.role === "system" ? `${m.content.split("\n")[0]} … plus the tool list` : m.content;
          return (
            <div class={`trace-message ${m.role}`} key={i}>
              <span class="trace-role">{m.role === "assistant" ? `assistant, example ${turn + 1}, trained on` : `${m.role}, context only`}</span>
              <pre>{text.length > 600 ? `${text.slice(0, 600)}…` : text}</pre>
            </div>
          );
        })}
      </div>
      <div class="bars">
        {trace.turns.map(([prompt, tokens], i) => (
          <div class="bar-row" style={{ gridTemplateColumns: "88px 1fr 120px" }}>
            <span class="bar-name">example {i + 1}</span>
            <span class="bar-track stacked" style={{ width: `${(100 * (prompt + tokens)) / longest}%` }}>
              <span class="bar-fill" style={{ width: `${(100 * prompt) / (prompt + tokens)}%` }} />
              <span class="bar-fill target" style={{ width: `${(100 * tokens) / (prompt + tokens)}%` }} />
            </span>
            <span class="bar-value">{int(prompt)} + {int(tokens)}</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="SFT examples" value={trace.turns.length} note="one per assistant turn" />
        <Stat label="Target tokens" value={int(target)} note={`${((100 * target) / all).toFixed(1)}% of ${int(all)}`} />
      </div>
      <p class="widget-note">Each assistant turn becomes one example: the rendered conversation up to that turn is the prompt (gray), and the reply plus <code>&lt;|im_end|&gt;</code> is the target (blue). The loss only counts blue tokens. Every example repeats the system prompt and tool list, so the prompts dwarf the targets.</p>
    </div>
  );
}
