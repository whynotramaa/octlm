import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(process.cwd(), "..");

const indent = (line: string) => line.length - line.trimStart().length;

function block(lines: string[], name: string, from: number, to: number): [number, number] {
  const pattern = new RegExp(`^\\s*(?:async\\s+)?(?:def|class)\\s+${name}\\b`);
  const start = lines.slice(from, to).findIndex((line) => pattern.test(line));
  if (start < 0) throw new Error(`symbol ${name} not found`);
  const first = from + start;
  const level = indent(lines[first]);
  let head = first;
  while (head > from && lines[head - 1].trimStart().startsWith("@")) head--;
  let end = first;
  while (end < to && !lines[end].trimEnd().endsWith(":")) end++;
  end++;
  while (end < to && (lines[end].trim() === "" || indent(lines[end]) > level)) end++;
  while (end > first && lines[end - 1].trim() === "") end--;
  return [head, end];
}

export function readSymbol(file: string, symbol: string): { code: string; line: number } {
  const lines = readFileSync(resolve(ROOT, file), "utf-8").split("\n");
  let range: [number, number] = [0, lines.length];
  for (const name of symbol.split(".")) range = block(lines, name, range[0], range[1]);
  const body = lines.slice(range[0], range[1]);
  const level = indent(body[0]);
  return { code: body.map((line) => line.slice(level)).join("\n"), line: range[0] + 1 };
}
