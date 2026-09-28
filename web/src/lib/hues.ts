export type Hue = "blue" | "green" | "orange" | "red" | "purple" | "cyan" | "gray";

const DAY_HUES: Hue[] = ["cyan", "blue", "green", "purple", "orange", "red"];

export const dayHue = (day: number): Hue => DAY_HUES[day % DAY_HUES.length];
