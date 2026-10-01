import { useState } from "preact/hooks";
import { type Schema, parseCalls } from "../lib/harness.ts";
import { Stat } from "./ui.tsx";

type Props = { schemas: Record<string, Schema>; presets: { name: string; text: string }[] };

const describe = (call: { name: string; arguments: Record<string, string> }) =>
  `${call.name}(${Object.entries(call.arguments).map(([key, value]) => `${key}=${JSON.stringify(value.length > 40 ? `${value.slice(0, 40)}…` : value)}`).join(", ")})`;

export default function ToolCallLab({ schemas, presets }: Props) {
  const [text, setText] = useState(presets[0].text);
  const calls = parseCalls(text, schemas);
  const valid = calls.filter((call) => typeof call !== "string").length;
  return (
    <div class="widget">
      <div class="widget-controls">
        <div class="segmented" role="group" aria-label="Model output">
          {presets.map((preset) => (
            <button type="button" class={preset.text === text ? "on" : ""} aria-pressed={preset.text === text} onClick={() => setText(preset.text)}>{preset.name}</button>
          ))}
        </div>
      </div>
      <textarea rows={5} value={text} onInput={(e) => setText(e.currentTarget.value)} aria-label="Model output to parse" />
      <div class="readout-grid">
        <Stat label="Tool-call blocks" value={calls.length} note={calls.length ? "each parsed on its own" : "no block, so this is the final answer"} />
        <Stat label="Valid" value={valid} note="these run" />
        <Stat label="Errors" value={calls.length - valid} note="sent back as tool results" />
      </div>
      {calls.length > 0 && (
        <div class="table-scroll">
          <table class="data-table">
            <thead><tr><th>Block</th><th>What the harness does</th></tr></thead>
            <tbody>
              {calls.map((call, index) => (
                <tr><td>{index + 1}</td><td><code>{typeof call === "string" ? call : `run ${describe(call)}`}</code></td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p class="widget-note">Edit the text. The parser finds every &lt;tool_call&gt; block, including an unclosed last one, and checks each one against the five tool schemas: valid JSON, exactly "name" and "arguments", a known tool, every required argument, no unknown argument, and string values. A broken block becomes an error message for the model. The loop keeps going.</p>
    </div>
  );
}
