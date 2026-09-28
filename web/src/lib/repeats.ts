export type Piece = { text: string; word: boolean; repeat: boolean };

export function repeatedGrams(text: string, n = 4) {
  const pieces: Piece[] = (text.match(/[A-Za-z']+|[^A-Za-z']+/g) ?? []).map((part) => ({ text: part, word: /[A-Za-z]/.test(part), repeat: false }));
  const words = pieces.filter((piece) => piece.word);
  const seen = new Set<string>();
  let repeated = 0;
  for (let i = 0; i + n <= words.length; i++) {
    const gram = words.slice(i, i + n).map((w) => w.text.toLowerCase()).join(" ");
    if (seen.has(gram)) {
      repeated++;
      words.slice(i, i + n).forEach((w) => (w.repeat = true));
    }
    seen.add(gram);
  }
  return { pieces, repeated, total: Math.max(words.length - n + 1, 0) };
}
