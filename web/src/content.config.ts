import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const posts = defineCollection({
  loader: glob({ pattern: "*.mdx", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    day: z.number().int().min(0).max(9),
    order: z.number().int(),
    exps: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    measured: z.boolean().default(true),
  }),
});

export const collections = { posts };
