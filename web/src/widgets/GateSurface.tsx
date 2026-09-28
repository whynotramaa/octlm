import Heatmap from "../charts/Heatmap.tsx";

const silu = (x: number) => x / (1 + Math.exp(-x));
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const GATE = Array.from({ length: 9 }, (_, i) => 4 - i);
const UP = Array.from({ length: 9 }, (_, i) => -3 + i * 0.75);

export default function GateSurface() {
  const swiglu = GATE.map((g) => UP.map((u) => silu(g) * u));
  const plain = GATE.map(() => UP.map(gelu));
  const label = (v: number) => String(v).replace("-", "−");
  return (
    <div class="widget">
      <div class="maps maps-2">
        <Heatmap title="SiLU(gate) × up" matrix={swiglu} showValues maxAbs={3} rowLabel={(r) => label(GATE[r])} columnLabel={(c) => label(UP[c])} />
        <Heatmap title="GELU(up), no gate" matrix={plain} showValues maxAbs={3} rowLabel={(r) => label(GATE[r])} columnLabel={(c) => label(UP[c])} />
      </div>
      <p class="widget-note">Rows are the gate value, columns the up value. Hover a cell to read it. On the left, a negative gate zeroes the whole row, and a positive gate lets both signs of up through, scaled. On the right the gate does not exist, so every row is the same: the unit's output depends on one number and is almost never negative. The gated unit responds to a pair of inputs.</p>
    </div>
  );
}
