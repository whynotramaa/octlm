import { useState } from "preact/hooks";
import { pretokenize } from "../lib/pretokenize.ts";

export default function ShiftedTargets() {
  const [text, setText] = useState("Once upon a time, there was a little cat.");
  const [row, setRow] = useState(3);
  const tokens = ["<|bos|>", ...pretokenize(text).slice(0, 23), "<|eos|>"];
  const inputs = tokens.slice(0, -1);
  const targets = tokens.slice(1);
  const active = Math.min(row, inputs.length - 1);
  return (
    <div class="widget">
      <label class="control">
        <span class="control-label">Text (split at character-type boundaries so the example stays readable)</span>
        <input type="text" value={text} onInput={(e) => setText(e.currentTarget.value)} />
      </label>
      <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Position</th><th>Model sees</th><th>Must predict</th></tr></thead>
          <tbody>
            {inputs.map((_, i) => (
              <tr class={i === active ? "on" : ""} onMouseEnter={() => setRow(i)}>
                <td class="num">{i}</td>
                <td><div class="tokens">{inputs.slice(0, i + 1).map((t) => <span class="token">{t}</span>)}</div></td>
                <td><div class="tokens"><span class="token new">{targets[i]}</span></div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p class="widget-note">{inputs.length} tokens in, {targets.length} predictions out, all from one forward pass. The target row is the input row shifted left by one. The causal mask is what stops position {active} from reading "{targets[active]}" in its own input.</p>
    </div>
  );
}
