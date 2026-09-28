const SPACE = /^(?:[\t\n\v\f\r\x1c-\x1f\x85]|\p{Zs}|\p{Zl}|\p{Zp})$/u;
const WORD = /^[\p{L}\p{M}_]$/u;
const NUMBER = /^\p{N}$/u;

export type Kind = 0 | 1 | 2 | 3;

export const KIND_NAMES = ["space", "word", "number", "symbol"] as const;

export function kind(character: string): Kind {
  if (SPACE.test(character)) return 0;
  if (WORD.test(character)) return 1;
  if (NUMBER.test(character)) return 2;
  return 3;
}

export function pretokenize(text: string): string[] {
  const characters = Array.from(text);
  const chunks: string[] = [];
  let current = "";
  let previous: Kind | null = null;
  for (const character of characters) {
    const next = kind(character);
    if (previous !== null && next !== previous) {
      chunks.push(current);
      current = "";
    }
    current += character;
    previous = next;
  }
  if (current) chunks.push(current);
  return chunks;
}
