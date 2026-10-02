import { useState } from "preact/hooks";
import { advantages } from "../lib/grpo.ts";
import { Segmented, Stat, fixed } from "./ui.tsx";

export default function GroupAdvantage({ size }: { size: number }) {
  const [rewards, setRewards] = useState<number[]>(Array.from({ length: size }, (_, i) => (i < 2 ? 1 : 0)));
  const [rule, setRule] = useState<"dr" | "std">("dr");
  const values = advantages(rewards, rule === "std");
  const uniform = new Set(rewards).size === 1;
  const flip = (i: number) => setRewards(rewards.map((r, j) => (j === i ? 1 - r : r)));
  const peak = Math.max(1, ...values.map(Math.abs));
  return (
    <div class="widget grpo">
      <Segmented label="Advantage" value={rule} options={[{ value: "dr", text: "r − mean (Dr. GRPO)" }, { value: "std", text: "(r − mean) / std (GRPO)" }]} onChange={setRule} />
      <div class="bars">
        {rewards.map((r, i) => (
          <div class={`bar-row${values[i] < 0 ? " leak" : " after"}`} style={{ gridTemplateColumns: "128px 1fr 56px" }}>
            <button type="button" class={`button small${r ? " primary" : ""}`} onClick={() => flip(i)}>rollout {i + 1}: {r ? "pass" : "fail"}</button>
            <span class="bar-track"><span class="bar-fill" style={{ width: `${(50 * Math.abs(values[i])) / peak}%` }} /></span>
            <span class="bar-value">{fixed(values[i], 2)}</span>
          </div>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="Passes in the group" value={`${rewards.filter(Boolean).length} of ${size}`} />
        <Stat label="Group" value={uniform ? "dropped" : "kept"} note={uniform ? "every advantage is zero, so it carries no gradient" : "mixed rewards give a signal"} />
      </div>
      <p class="widget-note">Click a rollout to flip its reward. A pass above the group mean gets a positive advantage, and every token it generated is pushed up. A fail below the mean is pushed down. With one pass in {size}, dividing by the standard deviation makes that pass's advantage {fixed(advantages([1, ...Array(size - 1).fill(0)], true)[0], 2)} instead of {fixed(advantages([1, ...Array(size - 1).fill(0)], false)[0], 2)}, so rare successes on hard tasks weigh more. That is the difficulty bias Dr. GRPO removes.</p>
    </div>
  );
}
