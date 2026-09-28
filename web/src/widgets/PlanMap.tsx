import { useState } from "preact/hooks";
import { Segmented } from "./ui.tsx";

type Row = { first: string; now: string };
type Kind = "all" | "moved" | "shelved" | "dropped";
const kindOf = (r: Row): Exclude<Kind, "all"> => (r.now.startsWith("Dropped") ? "dropped" : r.now.includes("Not scheduled") ? "shelved" : "moved");
const HUE = { moved: "green", shelved: "orange", dropped: "gray" } as const;

export default function PlanMap({ rows }: { rows: Row[] }) {
  const [kind, setKind] = useState<Kind>("all");
  const count = (k: Exclude<Kind, "all">) => rows.filter((r) => kindOf(r) === k).length;
  const shown = rows.filter((r) => kind === "all" || kindOf(r) === kind);
  return (
    <div class="widget">
      <div style={{ display: "flex", height: "14px", borderRadius: "7px", overflow: "hidden", gap: "2px" }}>
        {(["moved", "shelved", "dropped"] as const).map((k) => (
          <span style={{ flex: count(k), background: k === "moved" ? "var(--series-3)" : k === "shelved" ? "var(--series-4)" : "var(--ink-3)" }} title={`${k}: ${count(k)}`} />
        ))}
      </div>
      <Segmented label="Show" value={kind} options={[
        { value: "all", text: `All ${rows.length}` },
        { value: "moved", text: `Renumbered or folded ${count("moved")}` },
        { value: "shelved", text: `Code kept, not scheduled ${count("shelved")}` },
        { value: "dropped", text: `Dropped ${count("dropped")}` },
      ]} onChange={setKind} />
      <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>First plan</th><th /><th>Now</th></tr></thead>
          <tbody>
            {shown.map((r) => (
              <tr>
                <td>{r.first}</td>
                <td aria-hidden="true">→</td>
                <td><span class={`chip-inline tint hue-${HUE[kindOf(r)]}`}>{r.now}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
