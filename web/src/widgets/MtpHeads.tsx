import { useState } from "preact/hooks";
import { Slider, Stat, int } from "./ui.tsx";

const TOKENS = ["def", " total", "(", "values", "):", "\n    ", "return", " sum", "(", "values", ")"];

export default function MtpHeads({ width = 256 }: { width?: number }) {
  const [depth, setDepth] = useState(2);
  const [position, setPosition] = useState(2);
  const inputs = TOKENS.slice(0, -1);
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Slider label="Depth (futures per position)" value={depth} min={1} max={3} onInput={setDepth} />
        <Slider label="Position" value={position} min={0} max={inputs.length - 1} onInput={setPosition} />
      </div>
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr><th>Head</th>{inputs.map((t, i) => <th class={i === position ? "num" : ""}>{i}</th>)}</tr>
          </thead>
          <tbody>
            <tr><td>input</td>{inputs.map((t) => <td><span class="token">{JSON.stringify(t).slice(1, -1)}</span></td>)}</tr>
            {Array.from({ length: depth }, (_, k) => (
              <tr>
                <td>t+{k + 1}</td>
                {inputs.map((_, i) => {
                  const target = TOKENS[i + k + 1];
                  return <td>{target === undefined ? <span class="muted">–</span> : <span class={i === position ? "token new" : "token"}>{JSON.stringify(target).slice(1, -1)}</span>}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div class="readout-grid">
        <Stat label="Targets at this position" value={depth} note={`tokens ${position + 1} to ${position + depth}`} />
        <Stat label="Positions lost at the end" value={depth - 1} note="depth k has no target for the last k − 1" />
        <Stat label="Extra parameters" value={int((depth - 1) * width * width)} note={`${depth - 1} adapter${depth === 2 ? "" : "s"} of ${width} × ${width}`} />
      </div>
      <p class="widget-note">Every head reads the same final hidden state. Depth 1 goes straight into the tied output head. Depth k first passes through its own linear adapter, so it can learn a different map to the token two or three steps ahead.</p>
    </div>
  );
}
