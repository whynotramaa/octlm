import { pretokenize } from "./pretokenize.ts";

export const SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|tool_call|>", "<|tool_result|>"];
export const BASE_SIZE = 256 + SPECIALS.length;

export type Pair = [number, number];
export type Merge = [number, number, number];
export type Sequences = Map<string, { ids: number[]; frequency: number }>;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export function toBytes(text: string): number[] {
  return Array.from(encoder.encode(text));
}

export function initialSequences(texts: string[]): Sequences {
  const sequences: Sequences = new Map();
  for (const text of texts) {
    for (const chunk of pretokenize(text)) {
      const ids = toBytes(chunk);
      const key = ids.join(",");
      const entry = sequences.get(key);
      if (entry) entry.frequency += 1;
      else sequences.set(key, { ids, frequency: 1 });
    }
  }
  return sequences;
}

export function pairCounts(sequences: Sequences): Map<string, number> {
  const counts = new Map<string, number>();
  for (const { ids, frequency } of sequences.values()) {
    for (let index = 0; index + 1 < ids.length; index++) {
      const key = `${ids[index]},${ids[index + 1]}`;
      counts.set(key, (counts.get(key) ?? 0) + frequency);
    }
  }
  return counts;
}

function parsePair(key: string): Pair {
  const [left, right] = key.split(",").map(Number);
  return [left, right];
}

function before(a: Pair, b: Pair): boolean {
  return a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
}

export function bestPair(counts: Map<string, number>): { pair: Pair; count: number } | null {
  let best: { pair: Pair; count: number } | null = null;
  for (const [key, count] of counts) {
    const pair = parsePair(key);
    if (!best || count > best.count || (count === best.count && before(pair, best.pair))) {
      best = { pair, count };
    }
  }
  return best;
}

export function mergeIds(ids: number[], pair: Pair, tokenId: number): number[] {
  const output: number[] = [];
  let index = 0;
  while (index < ids.length) {
    if (index + 1 < ids.length && ids[index] === pair[0] && ids[index + 1] === pair[1]) {
      output.push(tokenId);
      index += 2;
    } else {
      output.push(ids[index]);
      index += 1;
    }
  }
  return output;
}

export function applyMerge(sequences: Sequences, pair: Pair, tokenId: number): Sequences {
  const merged: Sequences = new Map();
  for (const { ids, frequency } of sequences.values()) {
    const next = mergeIds(ids, pair, tokenId);
    const key = next.join(",");
    const entry = merged.get(key);
    if (entry) entry.frequency += frequency;
    else merged.set(key, { ids: next, frequency });
  }
  return merged;
}

export function train(texts: string[], vocabSize: number, minFrequency = 2): Merge[] {
  let sequences = initialSequences(texts);
  const merges: Merge[] = [];
  while (BASE_SIZE + merges.length < vocabSize) {
    const best = bestPair(pairCounts(sequences));
    if (!best || best.count < minFrequency) break;
    const tokenId = BASE_SIZE + merges.length;
    sequences = applyMerge(sequences, best.pair, tokenId);
    merges.push([best.pair[0], best.pair[1], tokenId]);
  }
  return merges;
}

export function encodeChunk(ids: number[], merges: Merge[]): number[] {
  const ranks = new Map(merges.map(([left, right, id], rank) => [`${left},${right}`, { rank, id }]));
  let sequence = ids;
  while (sequence.length > 1) {
    let chosen: { pair: Pair; rank: number; id: number } | null = null;
    for (let index = 0; index + 1 < sequence.length; index++) {
      const found = ranks.get(`${sequence[index]},${sequence[index + 1]}`);
      if (found && (!chosen || found.rank < chosen.rank)) {
        chosen = { pair: [sequence[index], sequence[index + 1]], ...found };
      }
    }
    if (!chosen) break;
    sequence = mergeIds(sequence, chosen.pair, chosen.id);
  }
  return sequence;
}

export function encode(text: string, merges: Merge[]): number[] {
  return pretokenize(text).flatMap((chunk) => encodeChunk(toBytes(chunk), merges));
}

export function byteTable(merges: Merge[]): Map<number, number[]> {
  const table = new Map<number, number[]>();
  for (let index = 0; index < 256; index++) table.set(index, [index]);
  for (const [left, right, id] of merges) table.set(id, [...table.get(left)!, ...table.get(right)!]);
  return table;
}

export function tokenText(id: number, table: Map<number, number[]>): string {
  if (id >= 256 && id < BASE_SIZE) return SPECIALS[id - 256];
  return decoder.decode(new Uint8Array(table.get(id) ?? []));
}
