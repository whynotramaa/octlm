import { useState } from "preact/hooks";
import { Segmented } from "./ui.tsx";

type Box = { label: string; y: number; kind: string };

const SKIPS = { pre: [46, 186], post: [46, 196] };

const LAYOUT: Record<string, { boxes: Box[]; formula: string; note: string }> = {
  pre: {
    boxes: [
      { label: "Norm", y: 70, kind: "alt" },
      { label: "Attention", y: 120, kind: "on" },
      { label: "+", y: 170, kind: "muted" },
      { label: "Norm", y: 230, kind: "alt" },
      { label: "Feed-forward", y: 280, kind: "on" },
      { label: "+", y: 330, kind: "muted" },
    ],
    formula: "x = x + Attention(Norm(x));  x = x + FFN(Norm(x))",
    note: "The straight line on the left never passes through a norm. Every block adds to it, and the gradient reaches the embedding through additions only. octlm uses this placement.",
  },
  post: {
    boxes: [
      { label: "Attention", y: 70, kind: "on" },
      { label: "+", y: 120, kind: "muted" },
      { label: "Norm", y: 170, kind: "alt" },
      { label: "Feed-forward", y: 230, kind: "on" },
      { label: "+", y: 280, kind: "muted" },
      { label: "Norm", y: 330, kind: "alt" },
    ],
    formula: "x = Norm(x + Attention(x));  x = Norm(x + FFN(x))",
    note: "The residual line runs into a norm after every block, so there is no clean path from the output back to the input. This is the 2017 layout. Xiong et al. show its gradients near the output are large at initialization, which is why it needs warmup.",
  },
};

export default function ResidualPlacement() {
  const [mode, setMode] = useState<"pre" | "post">("pre");
  const { boxes, formula, note } = LAYOUT[mode];
  const adds = boxes.filter((b) => b.label === "+");
  return (
    <div class="widget">
      <Segmented label="Placement" value={mode} options={[{ value: "pre", text: "Pre-norm" }, { value: "post", text: "Post-norm" }]} onChange={setMode} />
      <div class="two-col">
        <div class="diagram">
          <svg viewBox="0 0 300 390" role="img" aria-label={`${mode}-norm block`}>
            <text class="t small" x="150" y="24" text-anchor="middle">x in</text>
            <line class="link" x1="150" y1="32" x2="150" y2="360" />
            <text class="t small" x="150" y="382" text-anchor="middle">x out</text>
            {adds.map((a, i) => {
              const top = SKIPS[mode][i];
              return <path class="link on" d={`M 150 ${top} H 40 V ${a.y} H 134`} />;
            })}
            {boxes.map((b) => (
              b.label === "+"
                ? <g><circle class="box muted" cx="150" cy={b.y} r="15" /><text class="t" x="150" y={b.y + 5} text-anchor="middle">+</text></g>
                : <g><rect class={`box ${b.kind}`} x="90" y={b.y - 17} width="120" height="34" rx="9" /><text class="t" x="150" y={b.y + 5} text-anchor="middle">{b.label}</text></g>
            ))}
          </svg>
        </div>
        <div>
          <code class="shape" style={{ display: "block", whiteSpace: "normal" }}>{formula}</code>
          <p class="widget-note" style={{ marginTop: "16px" }}>{note}</p>
        </div>
      </div>
    </div>
  );
}
