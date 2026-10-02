import { useState } from "preact/hooks";
import LineChart, { type Series } from "../charts/LineChart.tsx";
import { Segmented } from "./ui.tsx";

type Step = { type: string; step: number; loss: number };
type Run = { name: string; rows: Step[]; dashed?: boolean };

export default function LossLines({ runs, note }: { runs: Run[]; note: string }) {
  const [log, setLog] = useState<"log" | "linear">("log");
  const series: Series[] = runs.map((run, i) => ({
    name: run.name,
    points: run.rows.filter((r) => r.type === "sft_step").map((r) => [r.step, r.loss] as [number, number]),
    slot: (i + 1) as 1 | 2 | 3,
    dashed: run.dashed,
  }));
  return (
    <div class="widget">
      <Segmented label="Loss axis" value={log} options={[{ value: "log", text: "log" }, { value: "linear", text: "linear" }]} onChange={setLog} />
      <LineChart series={series} x={{ label: "optimizer step" }} y={{ label: "loss on target tokens", format: log === "log" ? "exp" : "fixed2", log: log === "log" }} height={300} />
      <p class="widget-note">{note}</p>
    </div>
  );
}
