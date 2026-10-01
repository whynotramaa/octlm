export type Call = { name: string; arguments: Record<string, string> };
export type Schema = { properties: Record<string, unknown>; required: string[] };

const CALL = /<tool_call>([\s\S]*?)<\/tool_call>|<tool_call>([\s\S]*)/g;
const quote = (items: string[]) => `[${[...items].sort().map((item) => `'${item}'`).join(", ")}]`;
const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

export function validateCall(block: string, schemas: Record<string, Schema>): Call | string {
  let call: unknown;
  try {
    call = JSON.parse(block);
  } catch (error) {
    return `error: tool call is not valid JSON: ${(error as Error).message}`;
  }
  const keys = isObject(call) ? Object.keys(call).sort().join() : "";
  if (!isObject(call) || keys !== "arguments,name") return 'error: a tool call must be a JSON object with exactly "name" and "arguments"';
  const schema = typeof call.name === "string" && Object.hasOwn(schemas, call.name) ? schemas[call.name] : null;
  if (!schema) return `error: unknown tool ${typeof call.name === "string" ? `'${call.name}'` : JSON.stringify(call.name)}`;
  if (!isObject(call.arguments)) return "error: arguments must be a JSON object";
  const names = Object.keys(call.arguments);
  const missing = schema.required.filter((name) => !names.includes(name));
  const unknown = names.filter((name) => !Object.hasOwn(schema.properties, name));
  if (missing.length || unknown.length) return `error: missing arguments ${quote(missing)}, unknown arguments ${quote(unknown)}`;
  if (Object.values(call.arguments).some((value) => typeof value !== "string")) return "error: every argument must be a string";
  return call as Call;
}

export function parseCalls(text: string, schemas: Record<string, Schema>): (Call | string)[] {
  return [...text.matchAll(CALL)].map((match) => validateCall(match[1] ?? match[2], schemas));
}

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");

export function answerMatches(expected: string, answer: string): boolean {
  const word = "[\\p{L}\\p{N}_]";
  const pattern = new RegExp(`(?<!${word}|[.-])${escape(expected.toLowerCase())}(?!${word}|-|\\.\\p{N})`, "u");
  return pattern.test(answer.toLowerCase());
}

function choose(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 1; i <= k; i++) result = (result * (n - k + i)) / i;
  return result;
}

export function passHat(successes: number[], trials: number, k: number): number {
  return successes.reduce((sum, c) => sum + choose(c, k) / choose(trials, k), 0) / successes.length;
}

export function passAt(successes: number[], trials: number, k: number): number {
  return successes.reduce((sum, c) => sum + 1 - choose(trials - c, k) / choose(trials, k), 0) / successes.length;
}

export function schemasOf(tools: { function: { name: string; parameters: Schema } }[]): Record<string, Schema> {
  return Object.fromEntries(tools.map((tool) => [tool.function.name, tool.function.parameters]));
}
