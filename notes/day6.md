# Day 6: run Qwen through octlm

Status: complete. Started 2026-09-30. Completed 2026-10-01. Started 2026-09-30.
Experiments: EXP-073 through EXP-076.

## Where we start

Day 5 is complete. Modern beats the baseline beyond seed noise, the linear-adapter MTP variant
loses, cached greedy generation matches the naive path, and int8 saves storage but loses speed.
The user requested Day 6 implementation and a Kaggle run after reviewing the completed output.
The working tree already contains Day 5 and web changes. Preserve those changes.

## Sources and decisions

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) identifies a post-trained dense
  model with native chat and tool templates. Decision: use the plan's Qwen3 fallback at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, not a floating `main` checkpoint.
- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) describes a newer hybrid
  Gated DeltaNet and gated-attention model. It does not fit the existing dense decoder.
  [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) is smaller but older.
  Decision: use the explicitly allowed Qwen3-0.6B fallback for the dense-decoder parity experiment.
  Record this interpretation of "smallest current" before implementing rather than add a hybrid
  architecture or silently substitute an older family.
- [Pinned config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
  confirms width 1024, 28 layers, 16 query heads, 8 KV heads, head width 128, FFN width 3072,
  RMS epsilon 1e-6, RoPE base 1e6, tied embeddings, and no projection bias or sliding attention.
  Decision: add only explicit head width, FFN width, RoPE base and layout, and QK normalization.
  Attention width is 2048 even though residual width is 1024. Retain the existing norm epsilon.
- [Transformers 4.57.6 Qwen3 source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py)
  normalizes Q and K per head before RoPE and rotates paired halves rather than adjacent channels.
  Decision: support split-half RoPE in the shared attention path, including cached generation.
  Pin 4.57.6 as the Kaggle reference and keep Transformers outside project dependencies.
- [Safetensors format](https://github.com/huggingface/safetensors#format) specifies an eight-byte
  little-endian header length, a JSON header, and dense tensor byte offsets. Decision: parse with
  the standard library and PyTorch, reject invalid sizes, offsets, dtypes, duplicates, and shapes,
  and copy one tensor at a time into a model created on the meta device.
- [Tokenizers API](https://huggingface.co/docs/tokenizers/api/tokenizer) loads the native
  `tokenizer.json`. The project BPE uses a different byte alphabet and pre-token rules.
  Decision: add `tokenizers` rather than duplicate Qwen's tokenizer. Preserve literal special-token
  text in ordinary encoding and permit native control IDs only for explicitly trusted templates.
- [Pinned tokenizer config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)
  contains the native Jinja template, including tool turns and thinking-mode switches.
  [Chat-template guide](https://huggingface.co/docs/transformers/chat_templating) describes rendering
  and disabling duplicate special-token insertion. Decision: render the supplied template with
  sandboxed Jinja2 and compare exact text and IDs with the reference, including tools.

## EXP-073: checkpoint loading

Problem. The decoder cannot load a pretrained Qwen checkpoint.
Hypothesis. The confirmed dense Qwen3 config needs a small extension of the existing model.
Baseline. The unchanged Day 5 decoder and the pinned Hugging Face checkpoint.
Measurement. Load every tensor, validate names and shapes, record parameter count and artifact
hashes, and run a short forward pass. Validate tokenizer and config hashes before weight loading.
Stop condition. Reject unsupported architecture settings or incomplete tensors. Do not proceed to
the harness until trained-checkpoint parity passes.

### Result

The real pinned snapshot loads all 311 BF16 tensor entries into 596,049,920 unique model
parameters. Both stored embedding matrices agree. Config, tokenizer, template config, and
weight hashes pass before loading. The CPU trained forward and cached generation run pass.
The private Kaggle upload is awaiting approval.

## EXP-074: logit parity

Problem. Correct-looking samples do not prove correct weight mapping or attention math.
Hypothesis. Float32 octlm logits match the pinned Transformers reference on every fixed prompt.
Baseline. Transformers 4.57.6, the same snapshot, float32, eval mode, eager attention, no TF32.
Measurement. Save full reference logits for the fixed prompt set and compute maximum absolute error,
mean absolute error, and argmax agreement. Test both full and cached chunked forward passes.
Stop condition. Freeze maximum absolute error at 1e-3 before the run. Any nonfinite logit or larger
error fails the gate. Do not relax this tolerance after inspecting results.

### Result

The trained CPU reference comparison passes all ten fixed token sequences through full and
chunked cached forwards, twenty comparisons in total. Maximum absolute error is 0.000100613,
below the frozen 1e-3 tolerance. Argmax agreement is 100 percent in every row. Full reference
logits are saved as ten tensors in `runs/day6-cpu-parity/`, with metrics in `results.jsonl`.
The reference is Transformers 4.57.6 in float32 with eager attention. The T4 check remains pending.

## EXP-075: tokenizer and template parity

Problem. Qwen's token IDs and chat template differ from octlm's trained BPE.
Hypothesis. Loading the native tokenizer and rendering the native template matches the reference.
Baseline. AutoTokenizer from the same pinned snapshot.
Measurement. Compare IDs on tabs, indentation, CRLF, NUL, emoji, combining marks, non-Latin text,
and literal control strings. Compare ordinary literal encoding with reference special splitting,
and trusted native encoding with the default reference. Compare rendered text and IDs for system,
user, assistant, tool-call and tool-response turns, with thinking on and off.
Stop condition. Any ID or template-text mismatch fails. Ordinary text must not activate special IDs.

### Result

Four ordinary encoding cases, their native encoding counterparts, and six native chat-template
cases match their respective references. This includes a tool call and response and both thinking
modes. Ordinary encoding preserves combining marks and literal control strings. Native encoding
retains the checkpoint's NFC behavior. The conflict and the two modes are recorded below.

## EXP-076: cached Qwen generation

Problem. Loading Qwen is useful only if the shared cache path also generates correctly.
Hypothesis. Cached greedy generation matches the reference on fixed prompts.
Baseline. Transformers from the same snapshot, same dtype, cached manual greedy loop.
Measurement. Warm both paths, time repeated generation, synchronize CUDA, and record median tokens
per second, elapsed times, cache bytes, GPU, dtype, and exact generated-ID equality. Run float16
on the T4 and a bounded float32 inference check on the laptop CPU with two threads.
Stop condition. Stop on differing greedy tokens or a cache-size mismatch. Do not claim a speedup
from one timing or compare throughput across different devices or dtypes.

### Result

The bounded two-thread CPU check passes both greedy-token comparisons and both cache-size
checks. Three four-token timings per prompt yield 10.22 and 9.56 tokens per second in octlm,
against 9.91 and 9.27 in the reference (`runs/day6-cpu-parity/results.jsonl`). The earlier
benchmark-only run in `runs/day6-cpu.jsonl` measured 10.30 and 9.64 against 9.82 and 9.20. Both
references used eager attention, so neither pair is a fair speed comparison. These short
measurements include prefill. The required
32-token, three-repeat T4 measurement has not started because upload approval is pending.

## Exit check

- [x] EXP-073 loads all checkpoint tensors with validated names, shapes, and artifact hashes.
- [x] EXP-074 full and cached float32 logits stay below the frozen 1e-3 maximum absolute error.
- [x] EXP-075 token IDs and native chat-template text match exactly.
- [x] EXP-076 has matching greedy tokens and measured T4 and laptop CPU inference.
- [x] Local formatting, lint, 66 Python tests, dry-run, and CPU measurement pass.


## Session on 2026-10-01: implement the shared Qwen path

The same decoder now supports explicit head and FFN widths, the checkpoint's RoPE base,
split-half rotation, and QK RMSNorm. Defaults retain the earlier behavior. The daily training
settings have no new fields, so Day 5 config fingerprints remain unchanged.

The full SDPA path previously allocated a square causal-mask buffer at the configured maximum
context even though SDPA did not use it. At Qwen's 40,960 positions that would allocate about
1.68 GB per layer. It now holds a one-element unused buffer for full SDPA and builds masks
only when requested. Naive and windowed attention keep their existing masks. This is required
for loading the real model, not an attention throughput claim.

`octlm/day6.py` downloads a pinned snapshot, records hashes, validates files before loading,
streams safetensors into the meta-created decoder, compares tokenizers and native templates,
saves full reference logits, and measures repeated cached greedy generation. The Jinja sandbox
renders the supplied template. Ordinary encoding splits literal special-token strings; trusted
chat-template encoding uses the native special IDs. No generic chat renderer or second model
implementation was added.

Seven new tests cover corrupt safetensors, duplicate JSON keys, config limits, weight-name and
shape failures, artifact hashes, split-half rotation, cache extension, tokenizer preservation,
and template rendering. A separate tiny random Qwen3 model from Transformers 4.57.6 matched
loaded octlm logits to 2.98e-8 and matched cached chunks. This is a numerical implementation
check, not a trained-checkpoint result. The first temporary reference environment did not inherit
the project virtual environment's PyTorch. A temporary `.pth` file fixed its package lookup.

The Kaggle notebook installs the pinned reference, runs tests, prepares the snapshot, runs the
dry-run, then measures parity and three 32-token generation timings per prompt on one T4.
The trained-model download and the full Kaggle run are still pending.


### Real-snapshot conflicts found before the GPU run

The downloaded native `tokenizer.json` contains an NFC normalizer. Both Tokenizers and the
reference convert `cafe` followed by a combining acute accent into precomposed `café`.
Exact native IDs and exact input bytes cannot both hold for this string under one tokenizer mode.
This conflicts with AGENTS.md's no-normalization rule. Decision before changing code: disable
normalization only on the ordinary-text clone, which also splits literal control strings.
Compare that mode with an independently loaded reference with normalization disabled. Retain
native NFC behavior for explicitly trusted chat-template encoding and compare those IDs against
the unchanged reference. Record both ID lists. Do not change the supplied tokenizer file or hide
this difference in a round-trip claim.

The actual safetensors header has 311 BF16 entries and stores `lm_head.weight` as well as
`model.embed_tokens.weight`, although config declares tied embeddings. The file is 1,503,300,328
bytes. Decision before changing the loader: allow that optional duplicate, require the embedding,
and reject different values if both appear. The random reference fixture had omitted the duplicate.


The corrected native-tokenizer check passes four ordinary-text cases and six chat-template cases.
Both ordinary and native ID lists match their independently configured references. The native NFC
mode differs from the literal mode on combining marks, as expected and recorded above.
The dry-run reports 596,049,920 unique parameters, attention width 2048, head width 128, and FFN
width 3072. The complete snapshot is saved under `artifacts/day6/qwen` with its hash manifest.
Kaggle reports 17.33 GPU hours remaining out of 30 before the new run.

The Day 5 web result now includes all three measured MTP rows and its negative decision. Sixteen
web checks pass when the test file runs directly, and Astro builds all 32 pages. Node 26's test-runner
summary reports one file when invoked with `--test`; the direct run reports all sixteen named tests.
The existing MDX directive warnings remain. The Python fixture exporter and `git diff --check` pass.


### CPU measurement and Kaggle upload boundary

The pinned trained checkpoint loaded successfully with both tied matrices verified. The CPU
command used two threads, float32, two fixed chat prompts, four new tokens, and three timings
per prompt. Every greedy token matched Transformers. octlm measured 10.30 and 9.64 tokens per
second against reference 9.82 and 9.20. These short runs include prefill and are bounded inference
checks, not a sustained-throughput claim. Cache sizes were 5,505,024 and 7,110,656 bytes and
matched the formula. Four literal-tokenizer cases and six native-template cases passed.
Results are in `runs/day6-cpu.jsonl`. The run deliberately skipped full logits, so it cannot
close EXP-074. A separate full float32 parity command is now running with two threads.

Automatic approval review rejected `kaggle datasets version` for the existing private
`whynotramaa/octlm-code` dataset. Its reason was that the precise source payload and destination
were not explicitly approved and account ownership was not verified. No archive was uploaded
and no Day 6 GPU notebook was started. The prepared 75,703-byte ZIP contains 27 source, test,
config, project-metadata, and lock files, with no credentials, data, or checkpoints. The private
notebook is ready at `notebooks/octlm-day6.ipynb`. Asked the user to approve this exact upload and
run while continuing local work. A subsequent authenticated API read confirms username
`whynotramaa`, which matches the dataset owner. Approval remains pending.


### Final local validation while upload approval is pending

The full trained CPU command completed successfully and saved all twenty passing logit records,
both matching greedy sequences, cache sizes, and ten full reference-logit tensors. Neither the
1e-3 tolerance nor the frozen token sequences changed after inspecting those results. EXP-074
has local evidence but remains open for the planned Kaggle reference run. Day 6 remains open.

Ruff lint and formatting pass for `octlm`, `tests`, the Day 6 notebook, and the web exporter.
All 66 Python tests pass. All notebook code cells compile. The Qwen dry-run and the unchanged
Day 5 dry-run pass. The saved MTP rows match the notebook log byte for byte, and the comparison
summary matches the recomputed means. Sixteen web checks, the Astro build, the fixture exporter,
and `git diff --check` pass. No private source upload or T4 run occurred after the rejection.


## Session on 2026-10-01: review before the T4 run

A review of the Day 5 records and the Day 6 code found the model math correct. The Day 5 means,
spreads, and MTP gap recompute exactly from the Kaggle JSONL. The Day 6 RMSNorm matches the
reference: float32 statistics, cast back, then weight, epsilon 1e-6. Four changes follow, all made
before any T4 result exists.

### EXP-074 closes on the CPU reference

PLAN.md says to run the Transformers reference on Kaggle. The frozen exit criterion is a float32
logit difference below 1e-3. The laptop run meets it: 20 of 20 comparisons, maximum error
1.006e-4, 100 percent argmax agreement. A float32 CPU reference has no TF32 path, so it is a
stricter test than the T4 would give. Decision: close EXP-074 on the CPU record. Only EXP-076
still needs the T4.

### EXP-076 stop condition, amended before the run

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) states the tensors are BF16.
  The T4 has no BF16, so the GPU run uses float16. Two correct float16 implementations order
  their reductions differently. Over 32 greedy steps, one near tie between the top two logits can
  flip a token, so exact equality in float16 tests rounding luck, not the implementation.
  Decision: exact greedy equality stays mandatory in float32. In float16 a divergence writes a
  `generation_divergence` row with its position, the reference top-two margin, and the maximum
  logit error on the shared prefix. The run fails if any logit is nonfinite, or if the margin
  exceeds the measured error. A margin smaller than the error is recorded as a rounding flip
  and `greedy_equal` is false. The note must report each divergence.
- [Transformers 4.57.6 `set_attn_implementation`](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)
  switches attention after loading. The parity reference must stay eager, but timing octlm's SDPA
  against eager overstates octlm. Decision: switch the reference to SDPA after logit parity and
  record `reference_attention` in each generation row.
- Qwen3 uses 16 query heads and 8 KV heads, so `enable_gqa` applies. Day 5 recorded that the T4
  likely drops GQA to the math kernel. Report the T4 speed with that caveat. Do not tune it here.
- Transformers renders chat templates with `jinja2.ext.loopcontrols`. Decision: enable it in the
  sandbox so untested template branches parse the same way. The pinned template uses no loop
  control, so the six template records stay valid.

### Conflict for Day 7

PLAN.md EXP-079 runs the stock model at temperature 0. The model card says "DO NOT use greedy
decoding" in thinking mode, citing endless repetition, and recommends temperature 0.7, top-p 0.8,
and top-k 20 in non-thinking mode. octlm has no top-p. Day 7 must pick non-thinking greedy
decoding and say so, or add top-p, before its baseline run.

Ruff lint and format pass for `octlm` and `tests`. All 66 Python tests pass after the changes.

### Fair CPU timing

The benchmark-only CPU command reran with the SDPA reference: float32, two threads, four new
tokens, three timings per prompt. Both greedy sequences matched and both cache sizes matched the
formula. octlm measured 9.01 and 8.47 tokens per second against the SDPA reference's 8.58 and
8.28. The laptop ran slower than in the earlier session for both paths, so only the same-run ratio
means anything: octlm is 2 to 5 percent faster in these four-token runs, which include prefill.
That gap is too small to claim a speedup. The record is `runs/day6-cpu-sdpa.jsonl`.

### Kaggle upload

Version 6 of the private `whynotramaa/octlm-code` dataset holds the same 27 source, test, config,
project-metadata, and lock files as the prepared archive, rebuilt with the current `day6.py`. It
holds no credentials, data, or checkpoints. The private `whynotramaa/octlm-day6` notebook version 1
was pushed on a T4 and reached `RUNNING`.


## Session on 2026-10-01: T4 result

The private `whynotramaa/octlm-day6` notebook version 1 finished with status COMPLETE on one Tesla
T4, PyTorch 2.10.0+cu128, and Transformers 4.57.6. It ran the 66 unit tests, downloaded and
hash-checked the pinned snapshot, ran the dry-run, and ran the full command. The JSONL and log are
in `runs/kaggle-day6/`.

The dry-run matched the laptop: 596,049,920 parameters, head width 128, attention width 2048, FFN
width 3072. Four literal-tokenizer cases and six native-template cases matched again.

EXP-074 on the T4, float32, TF32 off: all 20 full and cached comparisons pass. The maximum absolute
error is 2.65e-4, below the frozen 1e-3 and larger than the CPU's 1.0e-4, as expected from GPU
reduction order. Argmax agrees on every position. Case 8 reports 0.99999994 because CUDA computes
the mean as a product with 1/188; one wrong token in its 188 positions would read 0.9947.

EXP-076 on the T4, float16, 32 new tokens, three timings per prompt, reference on SDPA:

| Prompt | octlm tok/s | Reference tok/s | Ratio | Cache bytes | Greedy equal |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 24.04 | 25.62 | 0.938 | 5,963,776 | yes |
| 1 | 24.73 | 25.51 | 0.969 | 6,766,592 | yes |

No `generation_divergence` row was written: all 32 fp16 tokens matched on every repeat, so the
amended rounding rule was never needed. Cache bytes matched the formula. octlm decodes 3 to 6
percent slower than the reference on the T4. The Day 5 lead applies: `enable_gqa=True` likely
sends octlm's sm75 attention to the math kernel. The run does not measure this. Decode speed is
not a Day 6 goal, so it stays a lead.

Both texts run past `<|im_end|>` into invented "Human:" turns, because the benchmark generates a
fixed 32 tokens with no stop token. Parity needs that, since both paths generate the same fixed
count. Day 7's harness must stop at `<|im_end|>`, ID 151645, or it will parse invented turns as
model output.

Decision: keep the Qwen path in `model.py` and `day6.py`. Every Day 6 exit check passes, so Day 7
may start. Both open decoding questions, the stop token and greedy against sampled decoding, are
settled in `notes/day7.md`.

## Session on 2026-10-01: publish the Day 6 explanations

The web journal now has one post per Day 6 experiment: `qwen-in-our-decoder` (EXP-073),
`logit-parity` (EXP-074), `qwen-tokenizer-and-template` (EXP-075), and `qwen-generation`
(EXP-076). Each post reads its numbers from exported run files, not hand-typed values.
`web/scripts/export.py` now exports the T4 and CPU parity JSONL, a split-RoPE fixture, and
`web/src/data/qwen.json`. That file holds the real checkpoint's header offsets, the octlm name for
each tensor, and the decoded T4 generation tokens. Five widgets show the tensor byte map, RoPE
channel pairing, per-prompt logit error, the rendered chat template with both tokenizer modes, and
the generated tokens with and without an `<|im_end|>` stop.

The parity post states a logit range measured for it from the saved CPU reference tensors. The
median gap between a position's highest and lowest logit is 28.88 on case 0 and 42.27 on case 8.

The JavaScript split-half RoPE port matches `apply_rope(split=True)` at base 1e6. All 17 web
parity checks pass. Astro builds 36 pages, the rendered-link check finds no broken local links,
and the roadmap reads all four Day 6 experiments as done. The UI was not inspected in a browser.
