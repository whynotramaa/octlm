const OPS = [
  { name: "mean μ = Σx / d", layer: true, rms: false },
  { name: "center x − μ", layer: true, rms: false },
  { name: "mean of squares", layer: true, rms: true },
  { name: "√(· + ε), divide", layer: true, rms: true },
  { name: "× gain g", layer: true, rms: true },
  { name: "+ bias b", layer: true, rms: false },
];

function Column({ x, title, kind }: { x: number; title: string; kind: "layer" | "rms" }) {
  return (
    <g>
      <text class="t" x={x + 100} y="20" text-anchor="middle">{title}</text>
      {OPS.map((op, i) => {
        const y = 36 + i * 46;
        const kept = op[kind];
        return (
          <g>
            {i > 0 && <line class="link" x1={x + 100} x2={x + 100} y1={y - 12} y2={y} />}
            <rect class={`box ${kept ? (kind === "layer" && !op.rms ? "warn" : "on") : "muted"}`} x={x} y={y} width="200" height="32" rx="8" style={kept ? undefined : { strokeDasharray: "4 4" }} />
            <text class={`t${kept ? "" : " small"}`} x={x + 100} y={y + 21} text-anchor="middle">{kept ? op.name : "skipped"}</text>
          </g>
        );
      })}
    </g>
  );
}

export default function NormOps() {
  return (
    <div class="widget">
      <div class="diagram">
        <svg viewBox="0 0 480 320" role="img" aria-label="Operations in LayerNorm and RMSNorm">
          <Column x={10} title="LayerNorm" kind="layer" />
          <Column x={270} title="RMSNorm" kind="rms" />
        </svg>
      </div>
      <p class="widget-note">Orange steps are the ones RMSNorm removes: the mean, the centering, and the bias. Without centering, the mean of squares is the RMS instead of the variance. Each norm is one pass over the vector either way, so the saving is a reduction and a subtraction per vector, small next to the matrix multiplies around it.</p>
    </div>
  );
}
