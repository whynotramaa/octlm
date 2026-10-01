import { useState } from "preact/hooks";
import { passAt, passHat } from "../lib/harness.ts";
import { Segmented, Stat, fixed } from "./ui.tsx";

export default function PassK({ successes, trials }: { successes: number[]; trials: number }) {
  const [k, setK] = useState(trials);
  const counts = Array.from({ length: trials + 1 }, (_, c) => successes.filter((value) => value === c).length);
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Segmented label="k" value={k} options={Array.from({ length: trials }, (_, i) => ({ value: i + 1, text: String(i + 1) }))} onChange={setK} />
      </div>
      <div class="bars">
        {counts.map((count, c) => (
          <div class="bar-row"><span class="bar-name">{c}/{trials}</span><span class="bar-track"><span class="bar-fill" style={{ width: `${(100 * count) / successes.length}%` }} /></span><span class="bar-value">{count} tasks</span></div>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label={`pass^${k}`} value={fixed(passHat(successes, trials, k), 3)} note={`all ${k} sampled tries succeed`} />
        <Stat label={`pass@${k}`} value={fixed(passAt(successes, trials, k), 3)} note={`at least one of ${k} succeeds`} />
      </div>
      <p class="widget-note">Bars count the 40 tasks by how many of their {trials} seeds passed in the baseline. At k = 1 both metrics equal the mean success rate. As k grows, pass^k only counts tasks the model solves every time, and pass@k counts any task it solved once.</p>
    </div>
  );
}
