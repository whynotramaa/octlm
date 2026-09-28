import { useState } from "preact/hooks";
import RangeChart from "../charts/RangeChart.tsx";
import { Segmented } from "./ui.tsx";

export type GridRow = { variant: string; code: number; codeSpread: number; prose: number; proseSpread: number; note: string };

export default function GridChart({ rows, highlight = [] }: { rows: GridRow[]; highlight?: string[] }) {
  const [split, setSplit] = useState<"code" | "prose">("code");
  const data = rows.map((r) => ({
    name: r.variant,
    value: split === "code" ? r.code : r.prose,
    spread: split === "code" ? r.codeSpread : r.proseSpread,
    note: r.note,
  }));
  return (
    <div class="widget">
      <Segmented label="Held-out split" value={split} options={[{ value: "code", text: "Code" }, { value: "prose", text: "Prose" }]} onChange={setSplit} />
      <RangeChart rows={data} label={`${split} bits per byte, lower is better`} highlight={highlight} baseline="baseline" />
    </div>
  );
}
