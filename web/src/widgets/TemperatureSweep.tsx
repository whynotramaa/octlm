import LineChart from "../charts/LineChart.tsx";
import { samplingDistribution } from "../lib/sampling.ts";
import { LOGITS, WORDS } from "./SamplingPlay.tsx";

const TEMPS = Array.from({ length: 39 }, (_, i) => 0.1 + i * 0.05);
const SHOWN = [0, 3, 2, 1];

export default function TemperatureSweep() {
  const curves = TEMPS.map((t) => samplingDistribution(LOGITS, t, 0));
  return (
    <div class="widget">
      <LineChart
        series={SHOWN.map((w, i) => ({ name: `"${WORDS[w]}"`, points: TEMPS.map((t, j) => [t, curves[j][w]] as [number, number]), slot: (i + 1) as 1 | 2 | 3 | 4 }))}
        x={{ label: "temperature", format: "fixed1", domain: [0.1, 2] }}
        y={{ label: "probability", format: "fixed1", domain: [0, 1] }}
        height={300}
      />
      <p class="widget-note">The same eight scores as the playground above, four of them drawn. At low temperature the top token takes almost everything, and sampling turns into greedy decoding. As the temperature rises the curves pull together toward 1/8, a uniform guess. Hover to read the probabilities at one temperature.</p>
    </div>
  );
}
