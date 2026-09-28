export type FormatKey = "int" | "fixed1" | "fixed2" | "fixed3" | "fixed4" | "mib" | "exp" | "percent" | "rate";

export function format(value: number, key: FormatKey = "fixed2"): string {
  if (!Number.isFinite(value)) return value > 0 ? "∞" : value < 0 ? "−∞" : "–";
  const text = {
    int: () => Math.round(value).toLocaleString("en-US"),
    fixed1: () => value.toFixed(1),
    fixed2: () => value.toFixed(2),
    fixed3: () => value.toFixed(3),
    fixed4: () => value.toFixed(4),
    mib: () => `${(value / 2 ** 20).toLocaleString("en-US", { maximumFractionDigits: value < 2 ** 24 ? 1 : 0 })} MiB`,
    exp: () => value.toExponential(1),
    percent: () => `${(value * 100).toFixed(1)}%`,
    rate: () => value.toExponential(2),
  }[key]();
  return text.replace("-", "−");
}
