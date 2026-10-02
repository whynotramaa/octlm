import type { Status } from "./roadmap.ts";

export const STATUS: Record<Status, { label: string }> = {
  done: { label: "Done" },
  switched: { label: "Plan switched" },
  built: { label: "Built, result pending" },
  pending: { label: "Next" },
  planned: { label: "Planned" },
};
