import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { causalSelfAttention } from "../src/lib/attention.ts";
import { encode, train } from "../src/lib/bpe.ts";
import { attentionMask, compressedMask, maskDensity } from "../src/lib/masks.ts";
import { type Matrix, maxAbsDifference } from "../src/lib/matrix.ts";
import { toBfloat16, toFloat16 } from "../src/lib/floats.ts";
import { quantizeRow, seedSpread } from "../src/lib/day5.ts";
import { answerMatches, parseCalls, passHat, schemasOf } from "../src/lib/harness.ts";
import { loraForward, loraParameters, mergeLora, qwenProjections } from "../src/lib/lora.ts";
import { hiddenSize, kvCacheBytes, parameterCount } from "../src/lib/model.ts";
import { tiledAttention } from "../src/lib/online-softmax.ts";
import { applyRope, applyRopeSplit, ropeAngles, ropeTables, sinusoidal } from "../src/lib/positions.ts";
import { pretokenize } from "../src/lib/pretokenize.ts";
import { samplingDistribution } from "../src/lib/sampling.ts";
import { learningRate } from "../src/lib/schedule.ts";
import { resultStatus, roadmap } from "../src/lib/roadmap.ts";

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

test("Day 5 seed spread and row quantization match Python", () => {
  const f = fixture("day5");
  const range = seedSpread([0.48, 0.5, 0.52]);
  assert.ok(Math.abs(range.mean - f.spread.mean) < 1e-12);
  assert.ok(Math.abs(range.spread - f.spread.spread) < 1e-12);
  const row = quantizeRow(f.weights);
  assert.deepEqual(row.codes, f.codes);
  assert.ok(Math.abs(row.scale - f.scale) < 1e-7);
  assert.equal(quantizeRow([-2.5 / 127, 1]).codes[0], -2);
});

test("roadmap waits for an experiment's exit check", () => {
  const note = "## EXP-071: cache\n### Result\nLocal parity passed. GPU timing pending.\n- [ ] EXP-071 has trained speed.\n";
  assert.equal(resultStatus(note, "EXP-071"), "built");
  assert.equal(resultStatus(note.replace("[ ]", "[x]"), "EXP-071"), "done");
});

test("Day 3 is marked as a switched plan", () => {
  assert.ok(roadmap().find((day) => day.day === 3)?.experiments.every((exp) => exp.status === "switched"));
});

test("published Day 5 inference records preserve the measurement invariants", () => {
  const cache = JSON.parse(readFileSync(new URL("../src/data/runs/day5-cache.json", import.meta.url), "utf-8"));
  assert.deepEqual(cache.map((row: any) => row.length), [64, 256, 512]);
  for (const row of cache) {
    assert.equal(row.greedy_equal, true);
    assert.equal(row.cache_bytes, row.length * 4096);
  }
  const [quant] = JSON.parse(readFileSync(new URL("../src/data/runs/day5-quant.json", import.meta.url), "utf-8"));
  assert.ok(Math.abs(quant.int8_bits_per_byte - quant.baseline_bits_per_byte - quant.bits_per_byte_delta) < 1e-12);
  assert.ok(quant.int8_model_bytes < quant.baseline_model_bytes);
});

test("completed Day 5 MTP uses all matched seeds and the frozen threshold", () => {
  const mtp = JSON.parse(readFileSync(new URL("../src/data/runs/mtp.json", import.meta.url), "utf-8"));
  const controls = JSON.parse(readFileSync(new URL("../src/data/runs/seed-spread.json", import.meta.url), "utf-8"));
  assert.deepEqual(mtp.map((row: any) => row.seed), [1337, 1338, 1339]);
  for (const row of mtp) {
    assert.equal(row.tokens, 98304000);
    assert.equal(row.data_hash, controls[0].data_hash);
    assert.equal(row.tokenizer_hash, controls[0].tokenizer_hash);
  }
  const modern = seedSpread(controls.filter((row: any) => row.variant === "modern").map((row: any) => row.bits_per_byte));
  const baseline = seedSpread(controls.filter((row: any) => row.variant === "baseline").map((row: any) => row.bits_per_byte));
  assert.ok(seedSpread(mtp.map((row: any) => row.bits_per_byte)).mean - modern.mean > Math.max(modern.spread, baseline.spread));
  assert.ok(roadmap().find((day) => day.day === 5)?.experiments.every((exp) => exp.status === "done"));
});

test("split-half RoPE with base 1e6 matches apply_rope(split=True)", () => {
  const f = fixture("day6");
  close(applyRopeSplit(f.x, ropeAngles(16, 8, 1, f.base)), f.rotated);
});

test("Day 7 parser, answer check and pass^k match octlm.harness", () => {
  const f = fixture("day7");
  const data = JSON.parse(readFileSync(new URL("../src/data/day7.json", import.meta.url), "utf-8"));
  const schemas = schemasOf(data.tools);
  const json = "error: tool call is not valid JSON";
  const shape = (call: unknown) => (typeof call === "string" && call.startsWith(json) ? json : call);
  for (const { text, calls } of f.parse) assert.deepEqual(parseCalls(text, schemas).map(shape), calls.map(shape), text);
  for (const { expected, answer, match } of f.answers) assert.equal(answerMatches(expected, answer), match, answer);
  for (const { successes, trials, k, value } of f.pass_hat) assert.ok(Math.abs(passHat(successes, trials, k) - value) < 1e-12);
});

test("published Day 7 baseline matches its own runs", () => {
  const rows = JSON.parse(readFileSync(new URL("../src/data/runs/day7-eval.json", import.meta.url), "utf-8"));
  const runs = rows.filter((row: any) => row.type === "day7_run");
  const summary = rows.at(-1);
  assert.equal(runs.length, 120);
  const tasks = [...new Set(runs.map((row: any) => row.task))];
  const successes = tasks.map((task) => runs.filter((row: any) => row.task === task && row.success).length);
  assert.ok(Math.abs(passHat(successes, 3, 1) - summary.pass_1_mean) < 1e-12);
  assert.equal(passHat(successes, 3, 3), summary.pass_hat_3);
  assert.ok(roadmap().find((day) => day.day === 7)?.experiments.every((exp) => exp.status === "done"));
});

test("LoRA parameter count, merge and forward match octlm.day8", () => {
  const f = fixture("day8");
  assert.equal(loraParameters(f.shapes, f.rank, f.layers), f.trainable);
  close(mergeLora(f.base, f.a, f.b, f.scale), f.merged);
  close(loraForward(f.x, f.base, f.a, f.b, f.scale), f.output);
  const qwen = JSON.parse(readFileSync(new URL("../src/data/qwen.json", import.meta.url), "utf-8")).config;
  const train = JSON.parse(readFileSync(new URL("../src/data/runs/day8-train-0.json", import.meta.url), "utf-8"));
  const shapes = qwenProjections(qwen).map((p) => p.shape);
  assert.equal(loraParameters(shapes, 16, qwen.n_layers), train[0].trainable_parameters);
});

test("published Day 8 runs match their summaries", () => {
  for (const name of ["day8-eval-0", "day8-eval-1", "day8-eval-0-int8"]) {
    const rows = JSON.parse(readFileSync(new URL(`../src/data/runs/${name}.json`, import.meta.url), "utf-8"));
    const runs = rows.filter((row: any) => row.type === "day7_run");
    const tasks = [...new Set(runs.map((row: any) => row.task))];
    const successes = tasks.map((task) => runs.filter((row: any) => row.task === task && row.success).length);
    assert.ok(Math.abs(passHat(successes, 3, 1) - rows.at(-1).pass_1_mean) < 1e-12, name);
  }
  assert.ok(roadmap().find((day) => day.day === 8)?.experiments.every((exp) => exp.status === "done"));
});
