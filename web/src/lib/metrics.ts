export function crossEntropy(probability: number): number {
  return -Math.log(probability);
}

export function perplexity(loss: number): number {
  return Math.exp(loss);
}

export function bitsPerByte(loss: number, tokens: number, bytes: number): number {
  return (loss * tokens) / (Math.LN2 * bytes);
}
