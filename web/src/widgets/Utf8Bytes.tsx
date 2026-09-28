import { useState } from "preact/hooks";

const PREFIX = [1, 3, 4, 5];

function split(byte: number, lead: boolean, size: number) {
  const bits = byte.toString(2).padStart(8, "0");
  const cut = lead ? (size === 1 ? 1 : PREFIX[size - 1]) : 2;
  return [bits.slice(0, cut), bits.slice(cut)];
}

export default function Utf8Bytes() {
  const [text, setText] = useState("aé€😀");
  const chars = [...text].slice(0, 12).map((c) => ({ c, bytes: [...new TextEncoder().encode(c)] }));
  return (
    <div class="widget">
      <input type="text" value={text} onInput={(e) => setText(e.currentTarget.value)} aria-label="Characters to encode" />
      <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Character</th><th>Code point</th><th class="num">Bytes</th><th>Byte IDs</th><th>Bits</th></tr></thead>
          <tbody>
            {chars.map(({ c, bytes }) => (
              <tr>
                <td>{c === " " ? "␠" : c}</td>
                <td><code>U+{c.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")}</code></td>
                <td class="num">{bytes.length}</td>
                <td><code>{bytes.join(" ")}</code></td>
                <td class="bits">{bytes.map((b, i) => {
                  const [head, tail] = split(b, i === 0, bytes.length);
                  return <span><span class="b-prefix">{head}</span><span class="b-man">{tail}</span> </span>;
                })}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p class="widget-note">The orange bits are UTF-8's markers. A lead byte starting 0 is a whole ASCII character. A lead byte starting 110, 1110 or 11110 says 2, 3 or 4 bytes follow in total, and every continuation byte starts 10. The gray bits carry the code point. Byte-level BPE starts from these 256 byte values, so any character, even one never seen in training, has an encoding.</p>
    </div>
  );
}
