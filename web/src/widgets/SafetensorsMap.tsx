import { useState } from "preact/hooks";
import { Segmented, Slider, Stat, fixed, int } from "./ui.tsx";

type Tensor = { name: string; octlm: string; shape: number[]; start: number; end: number };
type Props = { fileBytes: number; headerBytes: number; tensors: Tensor[]; layers: number };

const MIB = 2 ** 20;

export default function SafetensorsMap({ fileBytes, headerBytes, tensors, layers }: Props) {
  const [group, setGroup] = useState<"layer" | "global">("layer");
  const [layer, setLayer] = useState(0);
  const prefix = `model.layers.${layer}.`;
  const shown = tensors.filter((t) => (group === "layer" ? t.name.startsWith(prefix) : !t.name.startsWith("model.layers.")));
  const [picked, setPicked] = useState<string | null>(null);
  const tensor = shown.find((t) => t.name === picked) ?? shown[0];
  const payload = fileBytes - headerBytes;
  const left = ((headerBytes + tensor.start) / fileBytes) * 100;
  const width = ((tensor.end - tensor.start) / fileBytes) * 100;
  const layerBytes = shown.reduce((sum, t) => sum + t.end - t.start, 0);
  return (
    <div class="widget">
      <div class="grid-controls widget-controls">
        <Segmented label="Tensors" value={group} options={[{ value: "layer", text: "One layer" }, { value: "global", text: "Outside the layers" }]} onChange={(v) => { setGroup(v); setPicked(null); }} />
        {group === "layer" && <Slider label="Layer" value={layer} min={0} max={layers - 1} onInput={(v) => { setLayer(v); setPicked(null); }} />}
      </div>
      <div class="byte-strip" role="img" aria-label={`${tensor.name} occupies bytes ${int(headerBytes + tensor.start)} to ${int(headerBytes + tensor.end)} of the file`}>
        <span style={{ left: 0, width: "3px", background: "var(--series-4)" }} />
        <span style={{ left: `${left}%`, width: `max(3px, ${width}%)`, background: "var(--series-1)" }} />
      </div>
      <div class="legend">
        <span class="legend-item"><i class="swatch s4" />8-byte length + JSON header, {int(headerBytes)} bytes</span>
        <span class="legend-item"><i class="swatch s1" />{tensor.name}</span>
      </div>
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr><th>Checkpoint name</th><th>octlm name</th><th>Shape</th><th class="num">MiB</th></tr>
          </thead>
          <tbody>
            {shown.map((t) => (
              <tr class={`pick${t === tensor ? " on" : ""}`} onClick={() => setPicked(t.name)}>
                <td><code>{t.name.replace(prefix, "…")}</code></td>
                <td><code>{t.octlm.replace(`blocks.${layer}.`, "…")}</code></td>
                <td>{t.shape.join(" × ")}</td>
                <td class="num">{fixed((t.end - t.start) / MIB, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div class="readout-grid">
        <Stat label="Selected tensor starts at" value={`byte ${int(headerBytes + tensor.start)}`} note="8 + header length + its start offset" />
        <Stat label={group === "layer" ? "This layer" : "These tensors"} value={`${fixed(layerBytes / MIB, 1)} MiB`} note={`${shown.length} tensors`} />
        <Stat label="Whole payload" value={`${fixed(payload / MIB, 1)} MiB`} note={`${tensors.length} tensors, BF16`} />
      </div>
      <p class="widget-note">Click a row to find it in the file. Every layer holds the same eleven tensors, so the loader maps names with one table and a loop over 28 layers. The header is so small next to the weights that it shows as a sliver at the left edge.</p>
    </div>
  );
}
