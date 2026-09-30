import { useState } from "preact/hooks";
import { Segmented, Stat, fixed } from "./ui.tsx";

type Result = {
  baseline_model_bytes: number;
  int8_model_bytes: number;
  baseline_tokens_per_second: number;
  int8_tokens_per_second: number;
  baseline_bits_per_byte: number;
  int8_bits_per_byte: number;
};

export default function QuantCompare({ result }: { result: Result }) {
  const [metric, setMetric] = useState<"size" | "speed">("size");
  const size = metric === "size";
  const rows = size
    ? [result.baseline_model_bytes / 2 ** 20, result.int8_model_bytes / 2 ** 20]
    : [result.baseline_tokens_per_second, result.int8_tokens_per_second];
  const saved = 100 * (1 - result.int8_model_bytes / result.baseline_model_bytes);
  const slower = 100 * (1 - result.int8_tokens_per_second / result.baseline_tokens_per_second);
  return (
    <div class="widget">
      <Segmented label="Measured comparison" value={metric} options={[{ value: "size", text: "Saved file" }, { value: "speed", text: "Decode speed" }]} onChange={setMetric} />
      <div class="bars" role="list" aria-label={size ? "Inference-only state file sizes" : "Cached generation speeds"}>
        {rows.map((value, index) => (
          <div class="bar-row" role="listitem" style={{ gridTemplateColumns: "64px 1fr 110px", fontFamily: "var(--font-sans)" }}>
            <span class="bar-name">{index ? "int8" : "float16"}</span>
            <span class="bar-track"><span class="bar-fill" style={{ width: `${(value / Math.max(...rows)) * 100}%`, background: index ? "var(--series-2)" : "var(--series-1)" }} /></span>
            <span class="bar-value">{fixed(value, 1)} {size ? "MiB" : "tok/s"}</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="File size saved" value={`${fixed(saved, 1)}%`} note="inference state dictionaries" />
        <Stat label="Int8 decode slowdown" value={`${fixed(slower, 1)}%`} note="256 tokens, including prefill" />
        <Stat label="Validation bits per byte" value={`${fixed(result.baseline_bits_per_byte, 6)} → ${fixed(result.int8_bits_per_byte, 6)}`} note="lower is better" />
      </div>
      <p class="widget-note">One timed T4 run per variant. The file shrank and held-out quality barely moved, but int8 decoded more slowly. This PyTorch path reconstructs floating-point weights on every call; the timing does not isolate that cost. The bars compare observed values.</p>
    </div>
  );
}
