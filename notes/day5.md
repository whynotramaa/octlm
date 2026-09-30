# Day 5: comparisons and inference

Status: complete. Started 2026-09-29. Completed 2026-09-30.

## EXP-069: seed spread at the Day 4 scale

Problem. The Day 2 comparison used too few tokens to separate most variant gaps from seed noise.

Hypothesis. At about 100M tokens per run, the validation bits per byte of three seeds will give a
useful spread for deciding whether the modern stack beats the baseline at this scale.

Baseline. Day 4 trained the 26,255,872-parameter modern model for 393M tokens on a Kaggle T4. It
reached 0.4839 validation bits per byte and took 0.614 seconds per step. This is a throughput and
quality reference, not a matched control for the shorter runs.

Measurement. Train three seeds each of the Day 2 `baseline` and `modern` stacks on the Day 4 train
tokens, with the same 8,192-token vocabulary, 512-token context, 32 sequences per step, and 6,000
steps. That is 98.3M tokens per run. Evaluate the same fixed held-out blocks. Report each final
loss and bits per byte, each variant's mean and max-minus-min spread, and the gap between means.
Use the larger spread as the minimum detectable effect for Day 5. Record parameters, seconds per
step, the GPU, and the PyTorch version. The baseline retains full multi-head attention, GELU,
LayerNorm, and learned positions. Both variants use SDPA so the attention implementation does not
become a separate variable.

Stop condition. If the first run is unstable, or a run cannot finish within the Kaggle session,
stop and revise the budget from the measured rate. If the two variants' mean gap is no larger than
the seed spread, record the result as unresolved.

### Sources and decisions

- [PyTorch reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html) says
  exact repeatability depends on the software, device, and release. Decision: keep each comparison
  on one Kaggle GPU type and record the runtime version; the spread is local to that setup.
- [GQA paper](https://arxiv.org/abs/2305.13245) defines the difference between full multi-head
  attention and grouped-query attention. Decision: use all eight KV heads in the baseline and two
  in `modern`, as in Day 2.
- [Day 4 result](day4.md) supplies the measured T4 time and the fixed validation blocks. Decision:
  reuse its token files and evaluation function.

### Result

The runs have not started. The sandboxed Kaggle CLI could not resolve `api.kaggle.com`, but an
approved network command reached the API. Version 2 of the private `octlm-code` dataset now contains
`octlm/day5.py` and `configs/day5.toml`. The local one-step CPU smoke run completed both variants,
verified the output records, and exercised the resume guard. Its 32-token runs are a command check,
not evidence for the architecture comparison. The Day 5 unit tests and Ruff check pass. No seed
spread or architecture claim exists yet.

The baseline has 29,639,680 parameters; `modern` has 26,255,872. The baseline has more attention
weights because it keeps eight KV heads. That difference is part of the Day 2 stack comparison.
The full local suite passed 56 tests. Ruff lint and format passed for `octlm/` and `tests/`.
`octlm.train --config configs/day5.toml --dry-run` reported 26,255,872 parameters and logits
shape `[1, 512, 8192]`. The private Kaggle notebook `octlm-day5-spread` version 1 was queued after
upload; it reads the completed Day 4 notebook output.

#### Kaggle result, 2026-09-30

The private `octlm-day5-spread` notebook completed on one Tesla T4 with PyTorch 2.10.0+cu128 in
float16. All six rows share data hash `6144d010…`, tokenizer hash `5fae2d2c…`, and 98,304,000
training tokens. Every validation curve fell at each 1,000-step check; no run diverged. The rows
are in `runs/kaggle-day5-spread/day5/seed-spread.jsonl`, and the notebook log is beside them.

| Variant | Seed 1337 | Seed 1338 | Seed 1339 | Mean bpb | Spread | s/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5884 | 0.5864 | 0.5884 | 0.5877 | 0.0020 | 0.283 |
| modern | 0.5569 | 0.5519 | 0.5529 | 0.5539 | 0.0050 | 0.625 |

The larger spread is 0.0050 bits per byte, from `modern`. The gap between means is 0.0338, 6.8
times that threshold. The worst `modern` seed beats the best baseline seed by 0.0295. `modern`
reaches 5.8% lower bits per byte with 11% fewer parameters.

`modern` took 2.2 times as long per step: 3,752 s per run against 1,698 s for the baseline. The
comparison is matched on tokens, not on wall-clock time. A baseline given the same seconds would
see about 2.2 times the tokens, and this experiment does not measure where it would land. The
[PyTorch 2.10 SDPA documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
says GQA "works only for Flash_attention and math kernel on CUDA". PyTorch's flash kernel needs an
sm80 or newer GPU, and the T4 is sm75. So `enable_gqa=True` likely drops `modern` to the math
kernel, while the eight-KV-head baseline keeps the memory-efficient kernel. This is a lead, not a
measurement; SwiGLU and RoPE also add cost. Day 4's 0.614 s per step would carry the same cost.

Decision: keep `modern` as the default stack. At 98.3M tokens it beats the baseline by far more than
the seed noise. The minimum detectable effect for the rest of Day 5 is 0.0050 bits per byte. Open
question: time the T4 step with `repeat_interleave` keys in place of `enable_gqa` before any further
T4 training.

## EXP-070: depth-2 multi-token prediction

Problem. The Day 3a depth-2 head exists, but it has no measured quality result at the Day 4 model
size or on TinyStories.

Hypothesis. Depth-2 prediction will not improve held-out bits per byte by more than the EXP-069
modern seed spread. This is an inference from the scale trend in Gloeckle et al., not a result for
our 26M model.

Baseline. The three modern runs from EXP-069 are the matched control. They use the same tokenizer,
corpus, model trunk, seeds, step count, batch size, and validation blocks.

Measurement. Train three depth-2 runs with the EXP-069 settings. Record next-token validation bits
per byte, per-step time, added parameters, and depth-2 top-1 accuracy on the held-out blocks. Compare
the mean bits per byte with the modern control mean. Use EXP-069's larger variant spread as the
decision threshold. Do not compare raw token perplexity between tokenizers.

Stop condition. Do not run MTP until EXP-069 has all six seed results. If the MTP mean improves by no
more than the spread, record it as unresolved or negative and leave the depth-2 head off by default.

### Sources and decisions

- [Gloeckle et al., Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
  trains independent future-token heads on one trunk and reports stronger gains at larger sizes.
  Decision: use the already implemented Day 3a depth-2 adapter and evaluate the next-token head
  against the same modern control.
- [Day 3a experiment note](day3.md) records that the local implementation uses a linear adapter
  before the tied output matrix, while the paper uses a transformer layer per future head.
  Decision: keep that cheaper design and report its added parameter count.
- [PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
  defines `ignore_index` for excluded targets. Decision: keep the existing PAD exclusion in
  `mtp_loss` and evaluate ordinary next-token loss for both variants.

### Result

No GPU MTP run has started. `octlm.day5 mtp` now reuses the EXP-069 modern control, checks that all
requested control seeds finished, and records the depth-2 head's held-out top-1 accuracy. One local
32-token CPU smoke run completed the command and its output checks. Its bits per byte and accuracy
are not an MTP result. EXP-069 remains queued, so no threshold or comparison is available.

On 2026-09-30, EXP-069 finished all six seeds. The frozen threshold is 0.0050 bits per byte and
the control mean is 0.5539. The private `octlm-day5-mtp` notebook version 1 was pushed the same day
with the `octlm-code` dataset and the `octlm-day4` and `octlm-day5-spread` outputs, on a T4, and
reached `RUNNING`. It uses dataset version 2, whose `day5.py` predates the local `train_variant`
refactor; the MTP row fields and training call are the same. No MTP result exists yet.

## EXP-071: KV-cached generation

Problem. `Decoder.generate` recomputes every previous token at every decoding step.

Hypothesis. Caching the projected keys and values will increase generated tokens per second as the
sequence grows, while greedy output stays identical on the fixed prompt.

Baseline. Day 4's naive `Decoder.generate` with the trained 26M modern checkpoint, temperature 0,
batch 1, and one BOS token as the prompt. The 512-token context lets the 512-token measurement stay
inside the trained window.

Measurement. On the same device and dtype, warm both paths, then time 64, 256, and 512 generated
tokens. Synchronize CUDA around each timer. Record tokens per second, speed ratio, actual cache
bytes, and greedy token equality. Run a local small-model check for multi-token cache extension and
for crossing the context boundary.

Stop condition. If cached generation changes any greedy token, stop speed claims and fix parity.
If it is slower, keep only if the saved computation matters at a longer measured length; otherwise
record the result and revert the cache path.

### Sources and decisions

- [Hugging Face cache guide](https://huggingface.co/docs/transformers/kv_cache) describes storing
  prior key and value states for autoregressive inference. Decision: cache one key and one value
  tensor per layer and keep the training forward path unchanged.
- [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
  says `is_causal=True` on a non-square query-key matrix uses upper-left alignment. Decision: use
  an explicit offset-aware causal mask when a cached call adds more than one new token. A single
  decode token may attend to every cached key.
- [RoFormer](https://arxiv.org/abs/2104.09864) applies position rotations to queries and keys.
  Decision: rotate each new key once at its absolute position before placing it in the cache.

### Result

The small-model CPU check matched every greedy token through 12 new tokens with an eight-token
context. It also matched full logits when a cached call added a three-token chunk. The cache byte
count matched the tensor sizes. The local four- and eight-token timing command ran on a random
model; those tiny timings are command checks, not a speed claim. No trained-checkpoint timing has
run yet. The private `octlm-day5-inference` Kaggle notebook version 1 now runs this measurement
before the int8 measurement. EXP-069 is running on Kaggle.

#### Trained-checkpoint result, 2026-09-29

The completed private Kaggle `octlm-day5-inference` notebook used one Tesla T4, PyTorch
2.10.0+cu128, float16, batch 1, a BOS-only prompt, and the saved Day 4 checkpoint. The warmup and
CUDA-synchronized timers followed the measurement plan. One timed generation per length was saved;
the speeds have no repeated-run spread. The downloaded JSONL and notebook log are in
`runs/kaggle-day5-inference/`.

| New tokens | Naive tok/s | Cached tok/s | Speed ratio | Actual cache | Greedy equal |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 64 | 90.74 | 105.05 | 1.158× | 256 KiB | yes |
| 256 | 90.06 | 101.37 | 1.126× | 1 MiB | yes |
| 512 | 94.06 | 104.00 | 1.106× | 2 MiB | yes |

All three actual cache byte counts matched `kv_cache_bytes`. The trained model's greedy output
matched the naive path exactly at every length. The speed advantage is 10.6 to 15.8 percent in
these single timings, smaller than the projection-work ratio because decoding still reads cached
keys, projects logits, and pays PyTorch call overhead. Decision: keep `generate_cached` for full
SDPA attention. The generic `generate` remains available for configurations the cache path rejects.

## EXP-072: int8 weight-only quantization

Problem. The float16 Day 4 checkpoint spends two bytes per projection weight during inference.

Hypothesis. Per-output-channel int8 weights will reduce saved model bytes. A PyTorch-only decode
path that dequantizes weights for each forward may run slower; the experiment must measure that cost.
Quantization may increase held-out bits per byte.

Baseline. The trained Day 4 checkpoint in float16 on the Kaggle T4. Both variants use the same
KV-cached generation command and the same fixed validation blocks.

Measurement. Quantize attention and feed-forward linear weights per output channel. Keep the tied
embedding and output weight in float16. Save each model's inference state dictionary and compare
file bytes. Measure held-out bits per byte and cached generated tokens per second at 256 tokens,
including prefill, for both variants. Record the bits-per-byte change and speed ratio.

Stop condition. If int8 raises bits per byte by more than 0.01 or is slower without a useful file
size reduction, do not make it the default inference path. Keep the measured implementation for the
Day 8 merge experiment only if it has a useful size result.

### Sources and decisions

- [TorchAO int8 weight-only configuration](https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html)
  uses per-channel weight quantization. Decision: use one symmetric scale per output row with
  signed int8 weights, and store the scale in float32.
- [TorchAO inference guide](https://docs.pytorch.org/ao/stable/workflows/inference.html) separates
  weight-only storage from activation quantization. Decision: leave activations in the baseline
  dtype and report speed even if repeated dequantization loses time.
- [PyTorch `torch.save` guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
  supports saving a model state dictionary. Decision: compare inference-only state files so the Day
  4 optimizer and random states do not distort model bytes.

### Result

`Int8Linear` now stores signed int8 rows and float32 row scales. `quantize_int8` replaces projection
layers inside decoder blocks and leaves the tied embedding and output weight alone. The local
one-step CPU command wrote both inference state files, evaluated the same blocks, and generated
eight tokens. Its 0.00003 bits-per-byte change and tiny speed numbers describe a random model, so
they do not support a quality or speed decision. No trained-checkpoint quantization has run yet.
The private `octlm-code` Kaggle dataset now includes the cache and int8 code. The
`octlm-day5-inference` notebook will save float16 and int8 inference state files with its metrics.

#### Trained-checkpoint result, 2026-09-29

The same T4 notebook evaluated the same held-out Day 4 blocks and generated 256 tokens with both
cached paths. Inference-only state files measured 52,542,613 bytes for float16 and 30,652,347 bytes
for int8: 21,890,266 fewer bytes, or 41.66 percent smaller. Validation bits per byte changed from
0.48392883 to 0.48395385, a +0.00002503 increase, well below the +0.01 stop threshold. Cached
generation fell from 104.31 to 83.30 tokens per second, a 20.14 percent slowdown. These are single
timings, not a speed distribution; dequantizing weights on every linear call is a likely cause,
but the run does not isolate that cost from other operations. The JSONL is
`runs/kaggle-day5-inference/day5/day5-quant.jsonl`.

Decision: keep the int8 implementation as an opt-in smaller checkpoint for storage-constrained use
and the Day 8 merge experiment. Keep float16 as the default inference path because it decodes faster
and its bits per byte is marginally lower. Do not describe int8 as a speed optimization.

## Commands and exit check

On 2026-09-29, local commands passed:

- `python -m unittest discover -s tests`: 59 tests.
- `ruff check octlm tests` and `ruff format --check octlm tests`: passed.
- Ruff initially flagged import order and formatting in the three new notebooks. After fixing the
  notebooks, lint and format passed for all three. Their code cells compile.
- `python -m octlm.train --config configs/day5.toml --dry-run`: 26,255,872 parameters and logits
  shape `[1, 512, 8192]`.
- `python -m octlm.day5 spread`, `mtp`, `cache`, and `quant` each completed on a synthetic small CPU
  model. The control gate, resume check, token parity, and quantized state output ran end to end.

The private `octlm-day5-spread` notebook is running. `octlm-day5-inference` is queued. The MTP
notebook is ready but must wait for the six seed results before upload.

## Session on 2026-09-29: publish the Day 5 explanations

Problem. The web journal ends at Day 4. Its Day 2 and Day 4 posts still say KV-cached generation
has not been built, and the roadmap treats any Day 5 `Result` heading as a completed experiment.

Hypothesis. One detailed post per Day 5 experiment, with an interactive explanation and an explicit
measurement status, will let readers distinguish implemented code from trained-checkpoint results.

Baseline. The site has 26 posts through Day 4 and no Day 5 post. The Day 5 local suite has 59
passing tests, but none of its four exit checks has passed. The spread and inference Kaggle notebooks
were both running when checked for this session; MTP still waits for the spread results.

Measurement. Check that every Day 5 experiment has a linked post, that older pages no longer claim
the cache is absent, that roadmap status follows exit checks, and that the web build and tests pass.

Stop condition. Do not publish a GPU speed, quality, or architecture conclusion from the random
CPU smoke runs. Add trained results only after the notebook output and note are checked.

### Sources and decisions

- [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
  defines causal masking for query and key lengths. Decision: explain why cached multi-token calls
  need an offset-aware mask.
- [Hugging Face cache guide](https://huggingface.co/docs/transformers/kv_cache) explains reuse of
  past keys and values, and the memory and speed tradeoff of dynamic caches. Decision: show both
  saved projection work and cache growth in the cache post.
- [Gloeckle et al.](https://arxiv.org/abs/2404.19737) report independent future heads and stronger
  gains at larger scale. Decision: explain the Day 5 matched control and keep the 26M result open.
- [TorchAO quantized inference guide](https://docs.pytorch.org/ao/stable/workflows/inference.html)
  separates weight-only quantization from activation quantization and lists per-channel int8.
  Decision: show rowwise scales, reconstruction error, and the unmeasured decode cost.

### Result

The web journal now has one full post for each Day 5 experiment: seed spread, depth-2 MTP,
KV-cached generation, and int8 weight-only quantization. The Day 3 MTP page links to its new-scale
experiment. The Day 1, Day 2, and Day 4 pages that said generation still recomputed the context
now name the Day 5 cache and keep its speed result open. The roadmap reads the unchecked Day 5 exit
items as `Built, result pending`; it will read a checked item as `Done`. Post badges say `Result
pending`, and the shared warning no longer claims every unmeasured experiment is a default-off
Day 3 flag. The audit found no other implemented experiment without a matching post through Day 4,
and no post is marked as a draft. A later web pass labels the withdrawn Day 3 runs `Plan switched`
and explains that choice in their shared callout.

The seed-spread playground shows illustrative scores and says so. The quantization playground uses
the same per-row rule as `Int8Linear`; a Python-exported fixture checks its codes and scale. The
new seed-range calculation also matches Python. Existing MTP, KV decode, cache-size, and Day 4
training-curve playgrounds serve the other explanations. The web test suite passed 13 checks,
Ruff lint and format passed for the fixture exporter, the Astro build produced 32 pages, and a
rendered-HTML link check found no broken local links. The Impeccable detector returned no findings.

Two optional checks could not complete. The in-app browser had no available backend, so visual
interaction was not inspected. Standalone `tsc --noEmit` still fails because this site does not
install Node type declarations; its errors are in existing Node imports and tests, while Astro
build and the Node tests pass. The build emitted the existing MDX `astro:head-inject` bundler
warnings and completed. No new dependency was added solely for that optional type check.

The spread and inference Kaggle notebooks were still running at the final status check. None of the
four Day 5 trained-model exit results is available, so every Day 5 post remains marked pending.
Decision: keep the posts and status fix; add measured tables and conclusions only after checking
the notebook outputs and updating this note.

The later web pass also moved each Day 5 playground before the experiment mechanics. The seed chart
now labels its axis and means; the MTP target table uses illustrative story tokens and counts valid
targets at the block edge; the cache post starts at one visible decode step; and the int8 playground
draws float weight → integer code → restored weight. The post descriptions now introduce the idea
before the run plan. These are presentation changes, not new experiment results.

The inference notebook finished after that site check. A first attempt to download all output
failed partway through a large state file; downloading only the JSONL files succeeded. Their rows
match the completed notebook log. The web journal now needs the measured cache and int8 results,
while the seed-spread and MTP posts must stay pending.

The cache and int8 posts now import JSON exports of those run files. Their badges and the roadmap
show `Done`; the Day 5 spread and MTP posts remain `Result pending`. The cache post shows the three
measured speeds as a hoverable chart and a small table. The int8 post lets readers switch between
measured file-size and decode-speed bars while reading the exact bits-per-byte change. The Day 1,
Day 2, and Day 4 cache references now point to the trained result. The web exporter copied both
Kaggle JSONL files; their rows were checked against the notebook log, and the web test checks the
published cache and quality invariants. All 15 web tests passed, the Astro
build produced 32 pages, Ruff and `git diff --check` passed, and the rendered status and link check
found no mismatches or broken local links. Decision: keep both measured posts and their status
updates; leave the unmeasured comparisons open.

- [x] EXP-069 has six measured runs, a seed spread, and an architecture decision.
- [x] EXP-070 has three measured MTP runs and a decision against the seed spread.
- [x] EXP-071 has trained-checkpoint speed and cache bytes at 64, 256, and 512 tokens, with exact
  greedy token parity.
- [x] EXP-072 has trained-checkpoint model bytes, speed, bits-per-byte change, and a keep or revert
  decision.


## Session on 2026-09-30: close Day 5 from the MTP output

The private `whynotramaa/octlm-day5-mtp` notebook version 1 finished with status COMPLETE.
Downloaded `day5/mtp.jsonl` and the notebook log to `runs/kaggle-day5-mtp/`.
`read_runs` accepted all three config hashes, data hashes, and tokenizer hashes against the
six EXP-069 control records. Each run used 98,304,000 tokens, one T4, float16, and PyTorch
2.10.0+cu128. The log's final rows and comparison match the downloaded JSONL.

| Seed | MTP bits per byte | Depth-2 top-1 accuracy | Seconds per step |
| ---: | ---: | ---: | ---: |
| 1337 | 0.572361 | 0.730354 | 0.692906 |
| 1338 | 0.571251 | 0.731378 | 0.694173 |
| 1339 | 0.574450 | 0.730813 | 0.694187 |

The MTP mean is 0.572688 bits per byte against the modern control's 0.553899.
The regression is +0.018788, 3.77 times the frozen 0.004989 seed threshold.
MTP's own spread is 0.003199. All three MTP seeds are worse than all three control seeds.
Training costs 10.95 percent more per step and adds 262,144 parameters. Mean depth-2 top-1
accuracy is 73.08 percent. That accuracy is against the true second future token, not agreement
with a speculative verifier, and proves no generation speedup.

Decision: keep the measured implementation behind `mtp_depth = 2`, but revert its use in the
default model. `mtp_depth = 1` remains the keeper. This is a negative result for octlm's linear
adapter at 26M parameters and this token budget, not a reproduction of the paper's transformer
heads or a rejection of MTP at larger scales.

Both architecture comparisons now have decisions against the measured spread. The cache matches
every greedy token at all three lengths. Int8 remains opt-in for smaller storage and is slower.
Every Day 5 exit check passes, so Day 6 may start.

Validation: 59 local unit tests, Ruff lint and formatting for `octlm`, `tests`, and the Day 5
notebooks, and the Day 5 training dry-run pass. The first test command ran from `web` without
the project on its import path and failed collection. Running from the project root fixed it.
The measurement check recomputed the means, spreads, training cost, and input-hash validation
from the saved Kaggle records without retraining on the laptop.
