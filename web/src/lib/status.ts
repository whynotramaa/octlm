import type { Status } from "./roadmap.ts";

export const STATUS: Record<Status, { hue: "green" | "purple" | "orange" | "cyan" | "gray"; label: string }> = {
  done: { hue: "green", label: "Done" },
  switched: { hue: "purple", label: "Plan switched" },
  built: { hue: "orange", label: "Built, result pending" },
  pending: { hue: "cyan", label: "Next" },
  planned: { hue: "gray", label: "Planned" },
};
