import type { Status } from "./roadmap.ts";

export const STATUS: Record<Status, { hue: "green" | "orange" | "cyan" | "gray"; label: string }> = {
  done: { hue: "green", label: "Done" },
  built: { hue: "orange", label: "Built, not trained" },
  pending: { hue: "cyan", label: "Next" },
  planned: { hue: "gray", label: "Planned" },
};
