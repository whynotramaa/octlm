import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(process.cwd(), "..");

export type Row = Record<string, string>;

const cells = (line: string) => line.trim().slice(1, -1).split("|").map((cell) => cell.trim().replaceAll("`", ""));

export function noteTable(day: number, heading: string, nth = 0): Row[] {
  const lines = readFileSync(resolve(ROOT, `notes/day${day}.md`), "utf-8").split("\n");
  const start = lines.findIndex((line) => /^#+ /.test(line) && line.replace(/^#+ /, "").startsWith(heading));
  if (start < 0) throw new Error(`notes/day${day}.md has no heading "${heading}"`);
  const tables = lines.map((line, index) => index).filter((index) => index > start && lines[index].startsWith("|") && !lines[index - 1].startsWith("|"));
  const first = tables[nth] ?? -1;
  if (first < 0) throw new Error(`no table under "${heading}" in notes/day${day}.md`);
  const header = cells(lines[first]);
  const rows: Row[] = [];
  for (let index = first + 2; lines[index]?.startsWith("|"); index++) {
    const values = cells(lines[index]);
    rows.push(Object.fromEntries(header.map((name, column) => [name, values[column]])));
  }
  return rows;
}

export function num(text: string): number {
  return Number(text.replace(/,/g, "").match(/-?[\d.]+(e-?\d+)?/)?.[0] ?? NaN);
}

export function configSection(name: string, section: string): Record<string, string | number> {
  const text = readFileSync(resolve(ROOT, `configs/${name}.toml`), "utf-8");
  const body = text.split(/^\[(\w+)\]$/m);
  const index = body.indexOf(section);
  if (index < 0) throw new Error(`configs/${name}.toml has no [${section}]`);
  const entries = [...body[index + 1].matchAll(/^(\w+) = (.+)$/gm)].map(([, key, value]) => {
    const text = value.trim();
    return [key, text.startsWith('"') ? text.slice(1, -1) : Number(text)] as const;
  });
  return Object.fromEntries(entries);
}

export function modelConfig(name: string, overrides: Record<string, string | number> = {}) {
  const model = configSection(name, "model");
  const vocab = configSection(name, "tokenizer").vocab_size;
  return { vocab_size: vocab, ...model, ...overrides } as any;
}

export function repoText(path: string): string {
  return readFileSync(resolve(ROOT, path), "utf-8");
}

export function variantGrid() {
  return noteTable(2, "EXP-010 to EXP-013 and EXP-015, the variant grid").map((r) => ({
    variant: r["Variant"],
    parameters: num(r["Parameters"]),
    kvHeads: num(r["KV heads"]),
    code: num(r["Code bpb"]),
    codeSpread: num(r["Code spread"]),
    prose: num(r["Prose bpb"]),
    proseSpread: num(r["Prose spread"]),
    secondsPerStep: num(r["Seconds per step"]),
    note: `${r["Parameters"]} parameters, ${r["Seconds per step"]} s/step`,
  }));
}
