import { useState } from "preact/hooks";
import { Segmented, Stat } from "./ui.tsx";

type Row = { type: string; case: number; path: string; max_absolute_error: number; mean_absolute_error: number; argmax_agreement: number; tolerance: number };
type Props = { cpu: Row[]; t4: Row[]; labels: string[] };

const sci = (value: number) => value.toExponential(2).replace("-", "−");

export default function ParityErrors({ cpu, t4, labels }: Props) {
  const [path, setPath] = useState<"full" | "cached">("full");
  const pick = (rows: Row[]) => rows.filter((r) => r.type === "logit_parity" && r.path === path);
  const devices = [
    { name: "laptop CPU", rows: pick(cpu), slot: 1 },
    { name: "Kaggle T4", rows: pick(t4), slot: 2 },
  ];
  const tolerance = devices[0].rows[0].tolerance;
  const worst = (rows: Row[]) => Math.max(...rows.map((r) => r.max_absolute_error));
  return (
    <div class="widget">
      <Segmented label="Forward pass" value={path} options={[{ value: "full", text: "One full pass" }, { value: "cached", text: "Two cached chunks" }]} onChange={setPath} />
      <div class="legend">
        {devices.map((d) => <span class="legend-item"><i class={`swatch s${d.slot}`} />{d.name}</span>)}
        <span class="legend-item">bar end = frozen tolerance {sci(tolerance)}</span>
      </div>
      <div class="bars" role="list" aria-label="Largest logit error per prompt as a share of the tolerance">
        {labels.map((label, index) => (
          <div class="bar-row" role="listitem" style={{ gridTemplateColumns: "150px 1fr 150px", fontFamily: "var(--font-sans)" }}>
            <span class="bar-name">{label}</span>
            <span style={{ display: "grid", gap: "3px" }}>
              {devices.map((d) => (
                <span class="bar-track"><span class="bar-fill" style={{ width: `${(d.rows[index].max_absolute_error / tolerance) * 100}%`, background: `var(--series-${d.slot})` }} /></span>
              ))}
            </span>
            <span class="bar-value">{devices.map((d) => sci(d.rows[index].max_absolute_error)).join(" · ")}</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        {devices.map((d) => <Stat label={`Worst error, ${d.name}`} value={sci(worst(d.rows))} note={`${((worst(d.rows) / tolerance) * 100).toFixed(1)}% of the tolerance`} />)}
        <Stat label="Top-token agreement" value={devices.every((d) => d.rows.every((r) => r.argmax_agreement > 0.9999)) ? "every position" : "mismatch"} note="both devices, all prompts" />
      </div>
      <p class="widget-note">Each bar is the largest gap between one of octlm's logits and the reference logit for that position and vocabulary entry, over every position in the prompt. A full bar would hit the limit written down before the run. The T4 bars run longer than the CPU bars because the GPU adds numbers in a different order, not because the math differs.</p>
    </div>
  );
}
