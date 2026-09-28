import { useState } from "preact/hooks";
import { Stat, int } from "./ui.tsx";

const SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"];
const PRESETS: Record<string, string> = {
  English: "The model reads text one character at a time.",
  Python: "def total(values):\n    return sum(values)",
  Japanese: "日本語のテキストを読む",
  Emoji: "ship it 🚀 then test 🙂",
};

const show = (character: string) => (character === "\n" ? "\\n" : character === " " ? "␠" : character);

export default function CharTokenizer({ training }: { training: string }) {
  const vocabulary = [...SPECIALS, ...[...new Set(Array.from(training))].sort()];
  const lookup = new Map(vocabulary.map((token, id) => [token, id]));
  const [text, setText] = useState(PRESETS.English);
  const characters = Array.from(text);
  const ids = characters.map((character) => lookup.get(character) ?? 3);
  const unknown = ids.filter((id) => id === 3).length;
  return (
    <div class="widget">
      <div class="widget-controls">
        <div class="segmented" role="group" aria-label="Preset">
          {Object.entries(PRESETS).map(([name, value]) => (
            <button type="button" class={value === text ? "on" : ""} onClick={() => setText(value)}>{name}</button>
          ))}
        </div>
      </div>
      <textarea rows={2} value={text} onInput={(e) => setText(e.currentTarget.value)} aria-label="Text to tokenize" />
      <div class="tokens">
        {characters.map((character, i) => (
          <span class={ids[i] === 3 ? "token unk" : "token"}>{show(character)}<small>{ids[i]}</small></span>
        ))}
      </div>
      <div class="readout-grid">
        <Stat label="Vocabulary" value={int(vocabulary.length)} note={`4 specials + ${vocabulary.length - 4} characters`} />
        <Stat label="Tokens" value={int(ids.length)} note="one per character" />
        <Stat label="UTF-8 bytes" value={int(new TextEncoder().encode(text).length)} />
        <Stat label="Unknown" value={`${ids.length ? ((unknown / ids.length) * 100).toFixed(1) : "0.0"}%`} note="ID 3, <|unk|>" />
      </div>
      <p class="widget-note">The vocabulary is every distinct character in the training text, sorted. Anything the training text never contained becomes ID 3, and the original character is gone. Try the Japanese or emoji preset.</p>
    </div>
  );
}
