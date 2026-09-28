import { useState } from "preact/hooks";
import { bitsPerByte, perplexity } from "../lib/metrics.ts";
import { Slider, Stat, fixed, int } from "./ui.tsx";

type Run = { name: string; loss: number; tokens: number; bpb: number };

function Side({ run, loss, setLoss, bytes }: { run: Run; loss: number; setLoss: (v: number) => void; bytes: number }) {
  return (
    <div class="widget" style={{ gap: "12px" }}>
      <div class="chart-title">{run.name}</div>
      <Slider label="Loss per token (nats)" value={loss} min={0.5} max={8} step={0.01} shown={fixed(loss)} onInput={setLoss} />
      <div class="readout-grid">
        <Stat label="Perplexity" value={fixed(perplexity(loss), 1)} />
        <Stat label="Tokens" value={int(run.tokens)} note={`${fixed(bytes / run.tokens)} bytes each`} />
        <Stat label="Bits per byte" value={fixed(bitsPerByte(loss, run.tokens, bytes), 3)} />
      </div>
    </div>
  );
}

export default function BpbCompare({ runs }: { runs: [Run, Run] }) {
  const [a, b] = runs;
  const bytes = (run: Run) => (run.loss * run.tokens) / (Math.LN2 * run.bpb);
  const [lossA, setLossA] = useState(a.loss);
  const [lossB, setLossB] = useState(b.loss);
  const bpbA = bitsPerByte(lossA, a.tokens, bytes(a));
  const bpbB = bitsPerByte(lossB, b.tokens, bytes(b));
  const better = bpbA < bpbB ? a.name : b.name;
  return (
    <div class="widget">
      <div class="two-col">
        <Side run={a} loss={lossA} setLoss={setLossA} bytes={bytes(a)} />
        <Side run={b} loss={lossB} setLoss={setLossB} bytes={bytes(b)} />
      </div>
      <div class="verdict tint hue-blue">
        By perplexity, {perplexity(lossA) < perplexity(lossB) ? a.name : b.name} looks better. By bits per byte, which both tokenizers share, {better} is better by {fixed(Math.abs(bpbA - bpbB), 3)} bits.
      </div>
      <button type="button" class="button" style={{ alignSelf: "start" }} onClick={() => { setLossA(a.loss); setLossB(b.loss); }}>Reset to the measured losses</button>
    </div>
  );
}
