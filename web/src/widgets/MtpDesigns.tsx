import { useState } from "preact/hooks";
import { Segmented } from "./ui.tsx";

type Mode = "parallel" | "sequential" | "octlm" | "tied";
const DEPTH = 3;
const LABEL: Record<Mode, string> = { parallel: "Transformer layer", sequential: "module + emb", octlm: "linear adapter", tied: "" };
const NOTE: Record<Mode, string> = {
  parallel: "Gloeckle et al.: each depth has its own Transformer layer on the shared trunk output, then a shared unembedding. All heads run at once.",
  sequential: "DeepSeek-V3: depth k takes depth k−1's output and the embedding of the token k steps ahead, so each prediction is conditioned on the one before. The chain runs in order.",
  octlm: "octlm: Gloeckle's parallel form with a single linear adapter per depth instead of a Transformer layer. At 3.3M parameters a full block per depth would compare capacity, not the objective.",
  tied: "The first Day 3 proposal: every head is the tied unembedding on the same hᵗ. The inputs are identical, so the logits are identical, and depth 2 predicts token t+1 again. Each depth needs its own transformation.",
};

export default function MtpDesigns() {
  const [mode, setMode] = useState<Mode>("octlm");
  const x = (k: number) => 30 + k * 180;
  return (
    <div class="widget">
      <Segmented label="Design" value={mode} options={[
        { value: "parallel", text: "Parallel" },
        { value: "sequential", text: "Sequential" },
        { value: "octlm", text: "octlm" },
        { value: "tied", text: "All tied" },
      ]} onChange={setMode} />
      <div class="diagram" style={{ overflowX: "auto" }}>
        <svg viewBox="0 0 580 300" style={{ minWidth: "440px" }} role="img" aria-label={`${mode} multi-token prediction`}>
          <defs><marker id="mtp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="arrowhead" d="M0 0 L10 5 L0 10 z" /></marker></defs>
          <rect class="box muted" x="30" y="10" width="520" height="40" rx="10" />
          <text class="t" x="290" y="35" text-anchor="middle">shared trunk: embedding + blocks → hₜ</text>
          {Array.from({ length: DEPTH }, (_, k) => {
            const cx = x(k) + 70;
            const from = mode === "sequential" && k > 0 ? `M ${x(k - 1) + 140} 125 H ${x(k)}` : `M ${cx} 50 V 104`;
            return (
              <g>
                <path class="link" d={from} marker-end="url(#mtp-arrow)" />
                {mode === "sequential" && k > 0 && <path class="link" d={`M ${cx} 50 V 104`} style={{ opacity: 0.25 }} />}
                {mode !== "tied" && <>
                  <rect class="box on" x={x(k)} y="106" width="140" height="38" rx="9" />
                  <text class="t" x={cx} y="130" text-anchor="middle">{k === 0 && mode !== "sequential" ? "identity" : LABEL[mode]}</text>
                </>}
                {mode === "tied" && <text class="t small" x={cx} y="130" text-anchor="middle">same hₜ</text>}
                <path class="link" d={`M ${cx} 146 V 190`} marker-end="url(#mtp-arrow)" />
                <rect class="box alt" x={x(k)} y="192" width="140" height="34" rx="9" />
                <text class="t" x={cx} y="214" text-anchor="middle">tied unembedding</text>
                <rect class={`box ${mode === "tied" && k > 0 ? "warn" : "good"}`} x={x(k) + 10} y="246" width="120" height="34" rx="9" />
                <text class="t" x={cx} y="268" text-anchor="middle">{mode === "tied" ? "predicts xₜ₊₁" : `predicts xₜ₊${"₁₂₃"[k]}`}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <p class="widget-note">{NOTE[mode]}</p>
    </div>
  );
}
