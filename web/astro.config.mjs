import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import preact from "@astrojs/preact";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  integrations: [mdx(), preact()],
  redirects: { "/posts/[slug]": "/blogs/[slug]" },
  markdown: {
    processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex], smartypants: false }),
    shikiConfig: { themes: { light: "github-light", dark: "github-dark" } },
  },
});
