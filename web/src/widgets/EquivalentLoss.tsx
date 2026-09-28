import { useState } from "preact/hooks";
import LineChart from "../charts/LineChart.tsx";
import { Slider, Stat, fixed } from "./ui.tsx";

const xs = Array.from({ length: 41 }, (_, i) => 1 + i * 0.1);
const lossFor = (bpb: number, bytesPerToken: number) => bpb * bytesPerToken * Math.LN2;

export default function EquivalentLoss() {
  const [bpb, setBpb] = useState(2);
  const [bytes, setBytes] = useState(2.5);
  const loss = lossFor(bpb, bytes);
  return (
    <div class="widget">
      <LineChart
        series={[
          { name: `${fixed(bpb, 1)} bits per byte`, points: xs.map((x) => [x, lossFor(bpb, x)]), slot: 1 },
          { name: "this tokenizer", points: [[bytes, 0], [bytes, lossFor(bpb, 5)]], slot: 4, dashed: true },
        ]}
        x={{ label: "bytes per token", format: "fixed1", domain: [1, 5] }}
        y={{ label: "loss per token, nats", format: "fixed1" }}
        height={260}
      />
      <div class="grid-controls widget-controls">
        <Slider label="Model quality, bits per byte" value={bpb} min={0.5} max={4} step={0.1} shown={fixed(bpb, 1)} onInput={setBpb} />
        <Slider label="Tokenizer, bytes per token" value={bytes} min={1} max={5} step={0.1} shown={fixed(bytes, 1)} onInput={setBytes} />
      </div>
      <div class="readout-grid">
        <Stat label="Loss per token" value={`${fixed(loss, 3)} nats`} note="bpb × bytes per token × ln 2" />
        <Stat label="Perplexity" value={fixed(Math.exp(loss), 1)} />
        <Stat label="Same model, 1 byte per token" value={`${fixed(lossFor(bpb, 1), 3)} nats`} note={`perplexity ${fixed(Math.exp(lossFor(bpb, 1)), 1)}`} />
      </div>
      <p class="widget-note">Every point on the line predicts text equally well. Only the tokenizer changes. A tokenizer that packs more bytes into each token must pay more loss per token to reach the same bits per byte, and perplexity grows exponentially with it. That is why the two metrics disagreed.</p>
    </div>
  );
}
