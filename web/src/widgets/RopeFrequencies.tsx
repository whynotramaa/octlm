import { useState } from "preact/hooks";
import { ROPE_BASE } from "../lib/positions.ts";
import { Segmented, Slider, Stat, int } from "./ui.tsx";

const R = 36;

export default function RopeFrequencies() {
  const [head, setHead] = useState(32);
  const [position, setPosition] = useState(7);
  const thetas = Array.from({ length: head / 2 }, (_, i) => 1 / Math.pow(ROPE_BASE, (2 * i) / head));
  const period = (t: number) => (2 * Math.PI) / t;
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Segmented label="Head size" value={head} options={[16, 32, 64].map((v) => ({ value: v, text: String(v) }))} onChange={setHead} />
        <Slider label="Position" value={position} min={0} max={1024} onInput={setPosition} />
      </div>
      <div class="dial-grid">
        {thetas.map((t, i) => {
          const a = position * t;
          return (
            <div class="s1" style={{ textAlign: "center" }}>
              <svg viewBox="-46 -46 92 92" class="dial" role="img" aria-label={`pair ${i}, angle ${a.toFixed(2)} radians`}>
                <circle r={R} class="dial-ring" />
                <line x1="0" y1="0" x2={Math.cos(a) * R} y2={-Math.sin(a) * R} class="dial-vec" />
                <circle cx={Math.cos(a) * R} cy={-Math.sin(a) * R} r="3" class="dial-dot" />
              </svg>
              <div class="stat-note">pair {i}<br />turns every {int(period(t))}</div>
            </div>
          );
        })}
      </div>
      <div class="readout-grid">
        <Stat label="Fastest pair" value={`${period(thetas[0]).toFixed(1)} tokens`} note="per full turn" />
        <Stat label="Slowest pair" value={`${int(period(thetas[thetas.length - 1]))} tokens`} note="per full turn" />
      </div>
      <p class="widget-note">Each dial is one channel pair of a query at this position. Pair 0 spins once every 6.3 tokens and tells nearby positions apart. The last pairs barely move over a thousand tokens and carry coarse, long-range position. Drag the position slowly and watch the left dials spin while the right ones creep. The spread comes from θᵢ = {int(ROPE_BASE)}^(−2i/d).</p>
    </div>
  );
}
