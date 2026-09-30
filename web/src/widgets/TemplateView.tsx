import { useState } from "preact/hooks";
import { Segmented, Stat, int } from "./ui.tsx";

type Template = { chat: number; thinking: boolean; text: string; ids: number[] };
type Encoded = { text: string; ids: number[]; native_ids: number[] };
type Props = { templates: Template[]; encoded: Encoded[]; chats: string[] };

const MARKUP = /(<\|[a-z_]+\|>|<\/?(?:think|tool_call|tool_response|tools)>)/g;
const visible = (text: string) => text.replace(/\t/g, "→\t").replace(/\r/g, "␍").replace(/\x00/g, "␀");

function Marked({ text }: { text: string }) {
  return <pre class="template-text">{visible(text).split(MARKUP).map((part, i) => (i % 2 ? <mark>{part}</mark> : part))}</pre>;
}

function ChatMode({ templates, chats }: { templates: Template[]; chats: string[] }) {
  const [chat, setChat] = useState(0);
  const [thinking, setThinking] = useState(false);
  const row = templates.find((t) => t.chat === chat && t.thinking === thinking)!;
  const markers = row.text.match(MARKUP) ?? [];
  return (
    <>
      <div class="grid-controls widget-controls">
        <Segmented label="Conversation" value={chat} options={chats.map((text, value) => ({ value, text }))} onChange={setChat} />
        <Segmented label="Thinking mode" value={thinking ? "on" : "off"} options={[{ value: "off", text: "Off" }, { value: "on", text: "On" }]} onChange={(v) => setThinking(v === "on")} />
      </div>
      <Marked text={row.text} />
      <div class="readout-grid">
        <Stat label="Rendered characters" value={int(row.text.length)} note="identical to the reference" />
        <Stat label="Token IDs" value={int(row.ids.length)} note="identical to the reference" />
        <Stat label="Markup pieces" value={markers.length} note="highlighted above" />
      </div>
    </>
  );
}

function TokenMode({ encoded }: { encoded: Encoded[] }) {
  const [index, setIndex] = useState(2);
  const row = encoded[index];
  const differ = row.ids.length !== row.native_ids.length || row.ids.some((id, i) => id !== row.native_ids[i]);
  return (
    <>
      <Segmented label="Test string" value={index} options={encoded.map((_, value) => ({ value, text: `Case ${value + 1}` }))} onChange={setIndex} />
      <Marked text={row.text} />
      <div class="two-col">
        {[{ title: "Ordinary text (literal)", ids: row.ids }, { title: "Trusted template (native)", ids: row.native_ids }].map(({ title, ids }) => (
          <div>
            <div class="chart-title">{title}, {ids.length} IDs</div>
            <div class="tokens">{ids.map((id) => <span class={`token${id >= 151643 ? " special" : ""}`}>{id}</span>)}</div>
          </div>
        ))}
      </div>
      <p class="verdict">{differ ? "The two modes give different IDs for this string. Both lists match their own reference." : "Both modes give the same IDs for this string."}</p>
    </>
  );
}

export default function TemplateView({ templates, encoded, chats }: Props) {
  const [mode, setMode] = useState<"chat" | "tokens">("chat");
  return (
    <div class="widget">
      <Segmented label="Show" value={mode} options={[{ value: "chat", text: "Chat template" }, { value: "tokens", text: "Two tokenizer modes" }]} onChange={setMode} />
      {mode === "chat" ? <ChatMode templates={templates} chats={chats} /> : <TokenMode encoded={encoded} />}
      <p class="widget-note">{mode === "chat" ? "The exact text octlm renders from Qwen's own Jinja template, taken from the T4 run. Highlighted pieces are the markup the model learned during post-training. Thinking off adds an empty think block, which tells the model to answer directly." : "Arrows mark tabs, ␍ a carriage return and ␀ a NUL byte. Orange IDs are Qwen's added control tokens, numbered from 151,643 up. Case 3 has a combining accent that the native mode merges into one character, and case 4 has literal control strings that only the native mode turns into special tokens."}</p>
    </div>
  );
}
