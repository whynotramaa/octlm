import { useState } from "preact/hooks";
import { Segmented, Stat, fixed } from "./ui.tsx";

type Run = { task: string; kind: string; seed: number; success: boolean; calls: number; valid_calls: number; turns: number; answer: string | null; stop: string };
type Task = { id: string; prompt: string; check: { kind: string } };
type Props = { tasks: Task[]; v1: Run[]; v2: Run[] };

export default function EvalGrid({ tasks, v1, v2 }: Props) {
  const [scoring, setScoring] = useState<"v1" | "v2">("v2");
  const [picked, setPicked] = useState<string>(`${v2.find((r) => r.success)?.task}:0`);
  const runs = scoring === "v1" ? v1 : v2;
  const find = (rows: Run[], task: string, seed: number) => rows.find((r) => r.task === task && r.seed === seed)!;
  const seeds = [...new Set(runs.map((r) => r.seed))].sort();
  const [task, seed] = picked.split(":");
  const run = find(runs, task, Number(seed));
  const info = tasks.find((t) => t.id === task)!;
  const passes = runs.filter((r) => r.success).length;
  const tried = runs.filter((r) => r.calls > 0).length;
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Segmented label="Checks" value={scoring} options={[{ value: "v1", text: "v1 (run 1)" }, { value: "v2", text: "v2 (baseline)" }]} onChange={setScoring} />
      </div>
      <div class="chart-scroll">
        <div class="run-grid" style={{ gridTemplateColumns: `52px repeat(${tasks.length}, 14px)` }}>
          <span />
          {tasks.map((t, i) => <span class={`run-kind${i === 0 || tasks[i - 1].check.kind !== t.check.kind ? " start" : ""}`}>{i === 0 || tasks[i - 1].check.kind !== t.check.kind ? t.check.kind : ""}</span>)}
          {seeds.map((s) => (
            <>
              <span class="run-seed">seed {s}</span>
              {tasks.map((t) => {
                const r = find(runs, t.id, s);
                const flipped = find(v1, t.id, s).success !== find(v2, t.id, s).success;
                const state = r.success ? "pass" : r.calls ? "tried" : "none";
                return <button type="button" class={`run-cell ${state}${flipped ? " flipped" : ""}${picked === `${t.id}:${s}` ? " on" : ""}`} aria-label={`${t.id}, seed ${s}: ${state}`} onClick={() => setPicked(`${t.id}:${s}`)} />;
              })}
            </>
          ))}
        </div>
      </div>
      <div class="legend">
        <span class="legend-item"><span class="swatch run-cell pass" />passed</span>
        <span class="legend-item"><span class="swatch run-cell tried" />called a tool, failed</span>
        <span class="legend-item"><span class="swatch run-cell none" />no tool call</span>
        <span class="legend-item"><span class="swatch run-cell flipped" />scored differently by v1 and v2</span>
      </div>
      <div class="readout-grid">
        <Stat label="Runs passed" value={`${passes} of ${runs.length}`} note={`pass^1 ${fixed(passes / runs.length, 3)}`} />
        <Stat label="Runs with a tool call" value={`${tried} of ${runs.length}`} note="same in both runs" />
        <Stat label="Picked run" value={run.success ? "passed" : "failed"} note={`${run.turns} turns, ${run.valid_calls} of ${run.calls} calls valid`} />
      </div>
      <p class="widget-note"><strong>{info.id}</strong> ({info.check.kind}): {info.prompt}<br />Final answer: {run.answer === null ? `none, the run hit the ${run.stop} limit` : `"${run.answer.length > 220 ? `${run.answer.slice(0, 220)}…` : run.answer}"`}</p>
    </div>
  );
}
