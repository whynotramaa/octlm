import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(process.cwd(), "..");

export type Status = "done" | "switched" | "built" | "pending" | "planned";
export type Experiment = { id: string; title: string; status: Status };
export type Day = { day: number; title: string; experiments: Experiment[] };
export type Retired = { first: string; now: string };

const read = (path: string) => readFileSync(resolve(ROOT, path), "utf-8");

function noteFor(day: number): string | null {
  const path = resolve(ROOT, `notes/day${day}.md`);
  return existsSync(path) ? readFileSync(path, "utf-8") : null;
}

export function resultStatus(note: string, id: string): Status {
  if (note.includes(`- [x] ${id}`)) return "done";
  if (note.includes(`- [ ] ${id}`)) return "built";
  const start = note.search(new RegExp(`^## ${id}\\b`, "m"));
  if (start < 0) return "planned";
  const section = note.slice(start).split(/^## (?!#)/m)[1] ?? "";
  const result = section.split(/^### Result\s*$/m)[1];
  if (result === undefined) return "planned";
  return result.trim().startsWith("Not yet run") ? "pending" : "done";
}

function noteDay(day: number, note: string, status: Status): Day {
  const title = note.match(/^# Day \w+: (.+)$/m)?.[1] ?? `Day ${day}`;
  const heading = /^#{2,3} (EXP-\d{3}(?:(?: to | and )EXP-\d{3})*)[:,]? ([^.\n]+)/gm;
  const experiments = [...note.matchAll(heading)]
    .map(([, id, title]) => ({ id, title: title.trim(), status }))
    .sort((a, b) => a.id.localeCompare(b.id));
  return { day, title, experiments };
}

function planDays(): Day[] {
  const plan = read("PLAN.md");
  const sections = plan.split(/^### Day (\d+)\. (.+)$/m).slice(1);
  const days: Day[] = [];
  for (let index = 0; index < sections.length; index += 3) {
    const day = Number(sections[index]);
    const note = noteFor(day);
    const experiments = [...sections[index + 2].matchAll(/^- (EXP-\d{3})\. ([^.,]+)/gm)].map(([, id, title]) => ({
      id,
      title,
      status: note ? resultStatus(note, id) : ("planned" as Status),
    }));
    days.push({ day, title: sections[index + 1], experiments });
  }
  return days;
}

export function roadmap(): Day[] {
  const early = [1, 2, 3].map((day) => noteDay(day, noteFor(day) ?? "", day === 3 ? "switched" : "done"));
  return [...early, ...planDays()];
}

export function retired(): Retired[] {
  const note = read("notes/day3.md");
  const table = note.split("### Where each first-plan experiment went")[1] ?? "";
  return [...table.matchAll(/^\| (?!First plan|---)(.+?) \| (.+?) \|$/gm)].map(([, first, now]) => ({ first, now }));
}
