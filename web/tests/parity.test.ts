import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { causalSelfAttention } from "../src/lib/attention.ts";
import { encode, train } from "../src/lib/bpe.ts";
import { attentionMask, compressedMask, maskDensity } from "../src/lib/masks.ts";
import { type Matrix, maxAbsDifference } from "../src/lib/matrix.ts";
import { toBfloat16, toFloat16 } from "../src/lib/floats.ts";
import { hiddenSize, kvCacheBytes, parameterCount } from "../src/lib/model.ts";
import { tiledAttention } from "../src/lib/online-softmax.ts";
import { applyRope, ropeAngles, ropeTables, sinusoidal } from "../src/lib/positions.ts";
import { pretokenize } from "../src/lib/pretokenize.ts";
import { samplingDistribution } from "../src/lib/sampling.ts";
import { learningRate } from "../src/lib/schedule.ts";

const TOLERANCE = 1e-5;

const fixture = (name: string) => JSON.parse(readFileSync(new URL(`../fixtures/${name}.json`, import.meta.url), "utf-8"));
const close = (a: Matrix, b: Matrix) => assert.ok(maxAbsDifference(a, b) < TOLERANCE, `max diff ${maxAbsDifference(a, b)}`);
const ints = (mask: boolean[][]) => mask.map((row) => row.map(Number));

test("pretokenize matches octlm.tokenizer.pretokenize", () => {
  for (const { text, chunks } of fixture("tokenizer").pretokenize) assert.deepEqual(pretokenize(text), chunks);
});

test("BPE merges and encoding match ByteBPETokenizer", () => {
  const { text, vocab_size, merges, encoded } = fixture("tokenizer").bpe;
  const learned = train([text], vocab_size);
  assert.deepEqual(learned, merges);
  assert.deepEqual(encode(text, learned), encoded);
});

test("causal self-attention matches CausalSelfAttention", () => {
  const f = fixture("attention");
  const weights = { query: f.query, key: f.key, value: f.value, output: f.output_weight };
  close(causalSelfAttention(f.x, weights, f.n_heads).output, f.output);
});

test("tiled attention matches octlm.day2.tiled_attention", () => {
  const f = fixture("tiled");
  close(tiledAttention(f.query, f.key, f.value, f.block).output, f.output);
});

test("position tables match sinusoidal_positions, rope_tables and apply_rope", () => {
  const f = fixture("positions");
  close(sinusoidal(16, 16), f.sinusoidal);
  const tables = ropeTables(f.rope.length, f.rope.head_size);
  close(tables.cos, f.rope.cos);
  close(tables.sin, f.rope.sin);
  const scaled = ropeTables(f.rope.length, f.rope.head_size, f.rope_scaled.scale);
  close(scaled.cos, f.rope_scaled.cos);
  close(applyRope(f.apply.x, ropeAngles(16, 8)), f.apply.rotated);
});

test("masks match attention_mask, compressed_mask and mask_density", () => {
  const f = fixture("masks");
  for (const m of f.masks) {
    assert.deepEqual(ints(attentionMask(m.length, m.window, m.stride)), m.mask);
    assert.ok(Math.abs(maskDensity(m.length, m.window, m.stride) - m.density) < TOLERANCE);
  }
  for (const m of f.compressed) assert.deepEqual(ints(compressedMask(m.length, m.window, m.block)), m.mask);
});

test("cache bytes match kv_cache_bytes", () => {
  for (const row of fixture("cache")) assert.equal(kvCacheBytes(row.config, row.length), row.bytes);
});

test("schedule and hidden size match learning_rate and DecoderConfig.hidden_size", () => {
  const f = fixture("training");
  for (const schedule of Object.values(f.schedules) as any[]) {
    for (const [step, rate] of schedule.points) assert.ok(Math.abs(learningRate(step, schedule) - rate) < 1e-12);
  }
  for (const row of f.hidden_size) assert.equal(hiddenSize(row), row.hidden_size);
});

test("sampling distribution matches sampling_distribution", () => {
  for (const row of fixture("sampling")) {
    close([samplingDistribution(row.logits, row.temperature, row.top_k)], [row.probabilities]);
  }
});

test("parameter count matches parameter_count(Decoder)", () => {
  for (const row of fixture("parameters")) assert.equal(parameterCount(row.config), row.parameters);
});

test("float16 and bfloat16 rounding match torch casts", () => {
  for (const row of fixture("floats")) {
    assert.equal(toFloat16(row.value), row.float16, `float16 ${row.value}`);
    assert.equal(toBfloat16(row.value), row.bfloat16, `bfloat16 ${row.value}`);
  }
});
