import { useState } from "preact/hooks";
import LineChart, { type Series } from "../charts/LineChart.tsx";
import type { FormatKey } from "../charts/format.ts";
import { Segmented, Slider, Stat, fixed, int } from "./ui.tsx";

type Record = { step: number; train_loss: number; validation_loss: number; validation_bits_per_byte: number; validation_perplexity: number; learning_rate: number; gradient_norm: number };
type Metric = "loss" | "bpb" | "perplexity" | "gain" | "lr" | "norm";

const METRICS: { value: Metric; text: string; label: string; format: FormatKey; log?: boolean }[] = [
  { value: "loss", text: "Loss", label: "nats per token", format: "fixed2" },
  { value: "bpb", text: "Bits per byte", label: "bits per byte", format: "fixed3" },
  { value: "perplexity", text: "Perplexity", label: "perplexity", format: "fixed2" },
  { value: "gain", text: "Gain per 1,000", label: "validation loss drop", format: "fixed4", log: true },
  { value: "lr", text: "Learning rate", label: "learning rate", format: "rate" },
  { value: "norm", text: "Gradient norm", label: "gradient norm", format: "fixed2" },
];

function series(records: Record[], metric: Metric): Series[] {
  const at = (key: keyof Record) => records.map((r) => [r.step, r[key]] as [number, number]);
  if (metric === "loss") return [{ name: "validation", points: at("validation_loss"), slot: 1 }, { name: "train batch", points: at("train_loss"), slot: 2, dashed: true }];
  if (metric === "bpb") return [{ name: "validation", points: at("validation_bits_per_byte"), slot: 1 }];
  if (metric === "perplexity") return [{ name: "validation", points: at("validation_perplexity"), slot: 1 }];
  if (metric === "lr") return [{ name: "learning rate", points: at("learning_rate"), slot: 3 }];
  if (metric === "norm") return [{ name: "gradient norm", points: at("gradient_norm"), slot: 4 }];
  const gains = records.slice(1).map((r, i) => [r.step, records[i].validation_loss - r.validation_loss] as [number, number]);
  return [{ name: "drop", points: gains.filter((p) => p[1] > 0), slot: 1 }];
}

export default function RunCurve({ records }: { records: Record[] }) {
  const [metric, setMetric] = useState<Metric>("loss");
  const [from, setFrom] = useState(records[0].step);
  const shown = records.filter((r) => r.step >= from);
  const spec = METRICS.find((m) => m.value === metric)!;
  const first = shown[0];
  const last = shown[shown.length - 1];
  const perThousand = shown.length > 1 ? ((first.validation_loss - last.validation_loss) / (last.step - first.step)) * 1000 : 0;
  return (
    <div class="widget">
      <Segmented label="Metric" value={metric} options={METRICS} onChange={setMetric} />
      <Slider label="Start the x axis at step" value={from} min={records[0].step} max={records[records.length - 3].step} step={1000} shown={int(from)} onInput={setFrom} />
      <LineChart series={series(metric === "gain" ? records.filter((r) => r.step >= from - 1000) : shown, metric)} x={{ label: "step" }} y={{ label: spec.label, format: spec.format, log: spec.log }} markers height={320} />
      <div class="readout-grid">
        <Stat label={`Validation loss, step ${int(first.step)}`} value={fixed(first.validation_loss, 4)} />
        <Stat label={`Validation loss, step ${int(last.step)}`} value={fixed(last.validation_loss, 4)} />
        <Stat label="Average drop per 1,000 steps" value={fixed(perThousand, 4)} note={`over steps ${int(first.step)} to ${int(last.step)}`} />
      </div>
      <p class="widget-note">Train loss is one batch per record, so it jitters. Validation loss is the same 131,072 held-out tokens every time, so it is smooth. Slide the start to step 15,000 and the curve that looked flat from step 1,000 still slopes down. Gain per 1,000 uses a log scale, because the drops shrink by a factor of 100.</p>
    </div>
  );
}
