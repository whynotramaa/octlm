const buffer = new DataView(new ArrayBuffer(4));

export function float32Bits(value: number): number {
  buffer.setFloat32(0, value);
  return buffer.getUint32(0);
}

function fromBits(bits: number): number {
  buffer.setUint32(0, bits >>> 0);
  return buffer.getFloat32(0);
}

export function toBfloat16(value: number): number {
  if (Number.isNaN(value)) return NaN;
  const bits = float32Bits(value);
  const rounded = bits + 0x7fff + ((bits >>> 16) & 1);
  return fromBits(rounded & 0xffff0000);
}

export const FLOAT16_MAX = 65504;

export function toFloat16(value: number): number {
  const x = Math.fround(value);
  if (!Number.isFinite(x)) return x;
  const magnitude = Math.abs(x);
  if (magnitude >= 65520) return Math.sign(x) * Infinity;
  const exponent = Math.max(Math.floor(Math.log2(magnitude || 1)), -14);
  const step = 2 ** (exponent - 10);
  const scaled = magnitude / step;
  const floor = Math.floor(scaled);
  const remainder = scaled - floor;
  const even = remainder > 0.5 || (remainder === 0.5 && floor % 2 === 1) ? floor + 1 : floor;
  return Math.sign(x) * even * step;
}

export type Format = { name: string; exponent: number; mantissa: number; round: (value: number) => number };

export const FORMATS: Format[] = [
  { name: "float32", exponent: 8, mantissa: 23, round: Math.fround },
  { name: "bfloat16", exponent: 8, mantissa: 7, round: toBfloat16 },
  { name: "float16", exponent: 5, mantissa: 10, round: toFloat16 },
];

export function bitFields(format: Format, value: number): [string, string, string] {
  const rounded = format.round(value);
  if (format.name === "float16") {
    const sign = rounded < 0 || Object.is(rounded, -0) ? "1" : "0";
    const magnitude = Math.abs(rounded);
    if (!Number.isFinite(magnitude)) return [sign, "11111", "0".repeat(10)];
    if (magnitude < 2 ** -14) return [sign, "00000", Math.round(magnitude / 2 ** -24).toString(2).padStart(10, "0")];
    const exponent = Math.floor(Math.log2(magnitude));
    const fraction = Math.round((magnitude / 2 ** exponent - 1) * 1024);
    return [sign, (exponent + 15).toString(2).padStart(5, "0"), fraction.toString(2).padStart(10, "0")];
  }
  const bits = float32Bits(rounded).toString(2).padStart(32, "0");
  return [bits[0], bits.slice(1, 9), bits.slice(9, 9 + format.mantissa)];
}
