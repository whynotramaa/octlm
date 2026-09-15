# Day 2: modern decoder core

Status: complete. EXP-009 through EXP-015 are measured. The training runs moved to a Colab T4, and
the corpus moved with them.
Started: 2026-09-14
Experiments: EXP-009 through EXP-015

## What we need to finish

Day 2 ends when RoPE, RMSNorm, SwiGLU, and grouped-query attention each have a measured result
against the Day 1 baseline. The KV-head sweep needs a defended keeper choice. The SDPA benchmark
needs memory and tokens per second at every context length this machine reaches. Each comparison
changes one component. Token budget, seed set, and data stay fixed inside a comparison.

## Where we start

Day 1 is complete and its exit checks pass on a rerun. The baseline is a 541,952-parameter decoder
with learned absolute positions, pre-norm blocks, handwritten causal multi-head attention, GELU
feed-forward layers, and a tied head. The machine is still CPU only. PyTorch is 2.14.0+cpu.

## What we read

- [RoFormer](https://arxiv.org/abs/2104.09864) defines RoPE. It rotates each adjacent pair of
  channels by an angle proportional to the position, with frequencies
  `theta_i = 10000^(-2(i-1)/d)` for `i` in `1..d/2`. A rotation by `m` on the query and a rotation by
  `n` on the key compose into a rotation by `m - n` inside the dot product, so attention becomes a
  function of relative distance. The paper also gives an elementwise form. It multiplies `x` by
  `cos(m*theta)`, swaps each pair and flips the sign of the first element, and multiplies that by
  `sin(m*theta)`. We implement the elementwise form, apply it to Q and K only, never to V, and build
  the cosine and sine tables per forward pass from the sequence length.
- [Extending Context Window via Position Interpolation](https://arxiv.org/abs/2306.15595) reports
  that RoPE stops working past the length it was trained on. Perplexity can rise above 10^3, which is
  what an untrained model scores. The fix divides position indices by `L'/L` before the rotation, and
  1000 fine-tuning steps recover the quality. EXP-009 therefore measures where our model breaks
  instead of assuming it reaches 8K, and interpolation becomes the second half of that experiment.
  `rope_scale` is the knob, and `rope_tables` divides positions by it.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) normalizes by
  `RMS(a) = sqrt(mean(a_i^2))` and multiplies by a learned gain. It drops the mean subtraction, so it
  keeps re-scaling invariance and gives up re-centering invariance. Reported speedups run from 7 to
  64 percent across models, and from 6.9 to 9.3 percent on a Transformer. We implement the gain-only
  form that LLaMA uses rather than the paper's `a_bar + b`, and we compute the statistic in float32
  even when the activations are bfloat16.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) defines
  `FFN_SwiGLU(x) = (Swish(xW) * xV) W2`, where `*` is elementwise multiplication. Three matrices
  replace two, so the paper cuts the hidden width to 2/3 to hold parameters and compute constant. Its
  perplexity table puts SwiGLU at 1.944 against GELU at 1.983 and ReLU at 1.997 after 65,536 steps.
  EXP-011 compares at matched parameter count rather than matched hidden width. Matching the width
  instead would only measure that we added weights.
- [GQA](https://arxiv.org/abs/2305.13245) groups query heads so that each group shares one key head
  and one value head. One group is MQA and one group per head is MHA. The T5-XXL table shows MHA at
  1.51 s and 47.2 average ROUGE, MQA at 0.24 s and 46.6, and GQA-8 at 0.28 s and 47.1. The authors
  pick 8 groups because the slowdown from MQA stays small at first and grows as the group count
  approaches MHA. Converting an MHA checkpoint means mean-pooling the key and value projections
  within each group, then training for 5 percent of the original steps. EXP-013 reports cache bytes,
  throughput, and loss together, because any two of the three mislead on their own.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) shows
  that post-norm gradients near the output stay large at initialization whatever the depth, which is
  the reason post-norm needs warmup. Pre-norm gradients scale with `1/sqrt(L)` and train without it.
  We keep the Day 1 pre-norm placement as the control in EXP-015. Removing warmup is a separate
  change and never travels with a residual change.
- [FlashAttention](https://arxiv.org/abs/2205.14135) argues that attention is bound by memory, not by
  arithmetic. On an A100, HBM runs at 1.5 to 2.0 TB/s across 40 to 80 GB, while on-chip SRAM runs
  near 19 TB/s across 192 KB per multiprocessor. Standard attention writes the `N x N` score and
  probability matrices to HBM at a cost of `Theta(Nd + N^2)` accesses. Tiling with online softmax
  rescaling cuts that to `Theta(N^2 d^2 / M)` and `O(N)` extra memory, paid for by recomputing the
  scores in the backward pass. We write the tiled forward sketch to learn the blocking and the
  running maximum and sum, then benchmark the PyTorch kernels. We do not write a CUDA kernel.
- [LLaMA](https://arxiv.org/abs/2302.13971) uses the same set of parts we are about to assemble.
  Pre-norm with RMSNorm, SwiGLU at `2/3 * 4d`, RoPE at every layer, AdamW with betas 0.9 and 0.95,
  weight decay 0.1, gradient clipping 1.0, and a cosine schedule that ends at 10 percent of the peak
  rate. The Day 1 optimizer already matches every one of those except beta2, so the optimizer stays
  untouched through Day 2 and beta2 becomes a Day 4 question.
- PyTorch's `scaled_dot_product_attention` signature, read from the installed 2.14.0+cpu build, is
  `(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
  enable_gqa=False)`. We pass `is_causal=True` rather than build a mask tensor, and we never pass
  both, because `is_causal` and an explicit mask do not compose.

## What the paper exercises showed

### KV cache bytes for MHA against GQA

Twelve layers, model width 512, 8 query heads, head width 64, 4096 positions, bfloat16, batch 1.
Each layer caches both K and V, so the per-token cost is `2 * kv_heads * head_width * 2 bytes`.

| Variant | KV heads | Bytes per token per layer | Bytes per token | Cache at 4K |
| --- | ---: | ---: | ---: | ---: |
| MHA | 8 | 2048 | 24,576 | 96 MiB |
| GQA | 4 | 1024 | 12,288 | 48 MiB |
| GQA | 2 | 512 | 6,144 | 24 MiB |
| MQA | 1 | 256 | 3,072 | 12 MiB |

The cache scales with the KV head count and nothing else moves, so the saving is exact and
predictable. Two KV heads against eight is a 4x cut, which returns 72 MiB at this shape.

### RoPE rotation for one 2D pair

Take the first channel pair, where `theta_1 = 10000^0 = 1` radian per position, and the unit vector
`x = (1, 0)`.

| Position `m` | `cos(m)` | `sin(m)` | Rotated pair |
| ---: | ---: | ---: | --- |
| 0 | 1.0000 | 0.0000 | (1.0000, 0.0000) |
| 1 | 0.5403 | 0.8415 | (0.5403, 0.8415) |
| 2 | -0.4161 | 0.9093 | (-0.4161, 0.9093) |

The pair turns around the unit circle at a fixed rate. Its length never changes, and that is why RoPE
does not rescale the query or the key. Later pairs use smaller `theta_i`, so they turn slowly and
carry long-range information. Early pairs turn fast and carry local information.
`test_rotation_preserves_length` and `test_scores_depend_only_on_relative_distance` hold the
implementation to both properties.

### Why MQA is fastest and still loses quality

Decoding one token reads the whole KV cache and does little arithmetic, so decoding is bound by
memory bandwidth. Cutting to one KV head divides that read by the query-head count. The 1.51 s to
0.24 s gap in the GQA table comes from exactly that division. The cost is that every query head then
retrieves from one shared key and value subspace, so heads cannot specialize what they look up. GQA
keeps most of the bandwidth saving and still leaves several distinct subspaces, which is the reason
its ROUGE reaches 47.1 against 47.2 for MHA and 46.6 for MQA.

### SDPA call signature and mask behavior

A boolean `attn_mask` marks the positions to keep, and the masked positions become negative infinity
before the softmax. A float mask is added to the scores instead. `is_causal=True` builds the
lower-triangular mask internally, and you must not combine it with an explicit mask. `enable_gqa=True`
lets the key and value tensors carry fewer heads than the query tensor and broadcasts each KV head
across its group, so GQA needs no manual `repeat_interleave`.

We probed the installed build rather than trust the documentation.

- `enable_gqa=True` works on CPU with 8 query heads and 2 KV heads.
- `FLASH_ATTENTION` and `MATH` both run on CPU in float32 and bfloat16.
- `EFFICIENT_ATTENTION` has no CPU kernel and raises "No viable backend".

EXP-014 therefore compares `MATH` against `FLASH_ATTENTION` on CPU. The FlashAttention numbers
describe GPU HBM traffic. This machine has no HBM, so our measurement says nothing about the GPU
speedup. We do not quote one.

## The corpus

Day 1 trained on the project's own planning documents, roughly 111 KB. That is too small to measure
anything at long context. It is also too narrow to decide the KV-head count, and the KV-head count is
the keeper choice for the main model. So we built a code and prose corpus before starting the
experiments.

Command: `uv run python -m octlm.corpus`

| Split | Documents | Bytes | Code | Prose |
| --- | ---: | ---: | ---: | ---: |
| Train | 207 | 7,039,878 | 4,241,184 | 2,798,694 |
| Validation | 23 | 896,905 | 487,054 | 409,851 |

The mix lands at 60 percent code and 40 percent prose by bytes. `PLAN.md` asks for 50 percent code,
30 percent prose, and 20 percent instruction data. We have no instruction data yet, so compare the
two categories we do have. The corpus holds them at 3 to 2 where the plan implies 5 to 3. The code
side ran out before it reached its budget, and that gap is the whole difference.

Code comes from the top-level standard-library modules of the Python 3.14 install at
`/usr/lib/python3.14`, sorted, excluding `test_` files and dunder files. Python Software Foundation
License.

Prose comes from six Project Gutenberg books, all public domain in the United States. Four are
narrative and two are expository, so the prose side is not one voice.

| Gutenberg ID | Title | Kind |
| ---: | --- | --- |
| 84 | Frankenstein; or, The Modern Prometheus | Narrative |
| 1228 | On the Origin of Species | Expository |
| 1342 | Pride and Prejudice | Narrative |
| 1497 | The Republic | Expository |
| 2701 | Moby Dick; or, The Whale | Narrative |
| 2814 | Dubliners | Narrative |

`octlm/corpus.py` splits each source into 64 KB chunks on line boundaries, then assigns chunks to
train or validation by a seeded shuffle at a 0.1 validation fraction. Splitting by chunk rather than
by whole book keeps every source present in both splits. Two adjacent chunks of one book share a
style, but they share no text. Documents stay separate in JSONL, so the BPE trainer never merges
across a document boundary.

`data/manifest.json` records the fingerprints. The train document hash is
`b420e6d9fb6eef5d3fbaeb44a9ecba68f326a39f22805a839b53f5df8f9d6078` and the validation document hash
is `a3e6999f8bf99460801919a0cd09725858a77c389a709d1c49fadfd75e5757a0`.

Known limitations:

- The code is Python only, so the TypeScript and JavaScript targets in `PLAN.md` have no
  representation here.
- The prose is nineteenth-century literature, not the modern editing and rewriting text that the plan
  eventually wants.
- The stdlib file set depends on this machine's Python install. The manifest hashes pin what we used.
  They do not promise that a rebuild on another machine matches.

Day 2 needs a corpus large enough to separate architectures. It does not need the final data mix, and
the final mix stays a later experiment.

### The corpus moved to Colab

The development laptop overheats under a grid that saturates every core for an hour, so all training
now runs on a Colab T4. `octlm/corpus.py` reads the standard library of the machine it runs on, and
Colab ships Python 3.13.15 against the laptop's 3.14. The corpus therefore changed, and the Colab
build is now the one every result below uses.

| Split | Documents | Bytes | Code | Prose |
| --- | ---: | ---: | ---: | ---: |
| Train | 205 | 6,918,277 | 4,103,183 | 2,815,094 |
| Validation | 23 | 860,062 | 466,611 | 393,451 |

The train document fingerprint is `c6572704984dfd02662b080f8273bf0080be207f034c6edb8663cdfea89e01bc`
and the validation fingerprint is
`4a7bccfbfa637d50c94ad89611f491e087b049fffbed5a69980b88d5daa79c9d`. Check both in
`data/manifest.json` before you compare a new run against the tables below. Colab upgrades its Python
image without warning, and a different image is a different corpus.

The Colab corpus is 1.7 percent smaller than the laptop corpus and holds the same 60 to 40 code and
prose mix. The size is not the reason to prefer it. The reason is that the corpus should be built
where the training runs, and every run from here is a Colab run.

The 2048-entry tokenizer retrained on this corpus stops at the same 1787 merges and reaches 0.431
tokens per byte, against 0.407 on the laptop corpus. Its fingerprint is
`a2a18fe1e011f8c0cf774161f7fab5761f2e1aa0095ecae86ce0392d260af358`. Blocks at a 256-token context:
11,508 training blocks and 1,528 validation blocks, of which 782 are code and 745 are prose. That is
2.96M training tokens and 393K validation tokens.

EXP-014 is unaffected. It measures attention kernels on random tensors, so it depends on no corpus
and its CPU numbers stand.

### The Day 2 tokenizer

Day 1 flagged that the BPE trainer recounts every pair after every merge. On 7 MB that cost blocks
work, so we measured it before choosing. Training on 142 KB with a 512 vocabulary took 2.7 seconds.
Training on 702 KB with a 2048 vocabulary took 57 seconds. The cost tracks the number of distinct
pre-tokens, not the raw byte count, and 702 KB of this corpus holds 11,665 distinct pre-tokens.

So we train the tokenizer on every tenth training document, 21 documents and 702 KB, and then encode
all 7.9 MB with it. The 2048 request stopped at 1787 merges. Encoding measured 0.407 tokens per byte
against Day 1's 0.525, because this corpus is large enough to earn its merges. The artifact is
`artifacts/day2/bpe-2048.json`, fingerprint
`d757537c914a962f5ab662fc547a208517f79f42fc562f60544dd2dadf5e6d7f`.

Blocks at a 256-token context: 11,725 training blocks, 1,519 validation blocks, of which 756 are code
and 763 are prose. That is 3.0M training tokens and 389K validation tokens.

## What we built

Day 2's components live in `octlm/model.py` behind config switches, and every switch defaults to the
Day 1 behavior. One code path serves both generations, so there is no second model file to keep in
step. `configs/day2.toml` sets the Day 2 shape and `configs/day2-long.toml` is the same shape at a
512-token context for the length sweep.

| Switch | Values | Default |
| --- | --- | --- |
| `position` | `learned`, `sinusoidal`, `rope` | `learned` |
| `norm` | `layernorm`, `rmsnorm` | `layernorm` |
| `feed_forward` | `gelu`, `swiglu` | `gelu` |
| `attention` | `naive`, `sdpa` | `naive` |
| `residual` | `pre`, `post` | `pre` |
| `kv_heads` | 0 for MHA, else a divisor of `n_heads` | 0 |
| `rope_scale` | positive float, the position-interpolation divisor | 1.0 |

`octlm/day2.py` runs the experiments as stages: `tiled`, `equivalence`, `sdpa`, `variants`, `length`,
`cache`, and `report`. Each stage appends JSONL to `runs/`.

Two implementation notes worth keeping:

- Attention now projects Q, K, and V separately instead of through one fused matrix, because K and V
  carry fewer heads than Q under GQA. The parameter count is unchanged, but the initialization draws
  in a different order, so the exact loss values Day 1 recorded belong to commit `532c7f4`. Day 1's
  exit checks are behavioral and still pass, and all 15 Day 1 tests pass against the new model.
- `ModelSettings` gained seven fields, so the Day 1 config fingerprint changed from
  `f77b2a99c8d4d42e1d62093ff6c71d010ea75268fc8d2325d7f276bf6c7c3ee8` to
  `12cc0f1de67889e04411e8d5d78e62ae284318e001f7403f3c8068fb613eae5e`. Day 1 checkpoints written
  before this change will refuse to resume, which is the hash guard doing its job.
- A model with learned positions still refuses a sequence longer than its table. RoPE and sinusoidal
  positions build their tables from the sequence length, so they run past the trained length. That
  asymmetry is the reason EXP-009 uses sinusoidal positions as its control.

## Results

### EXP-014, SDPA and the tiled sketch. Keep.

Problem: the Day 1 attention path materializes the full score matrix, and every later experiment pays
for it. Hypothesis: the PyTorch flash kernel matches the handwritten path and costs less memory.

The tiled forward sketch in `tiled_attention` walks one key block at a time and rescales by a running
maximum and sum, which is the FlashAttention idea without the CUDA. It matches
`scaled_dot_product_attention` to 4.8e-7 at lengths 128, 512, and 1024 and at block sizes 64, 128,
and 256. Block size changes nothing, which is the point of the algorithm.

The full model through SDPA matches the handwritten path to about 1e-6 at 8, 4, 2, and 1 KV heads.
That equivalence is what licenses `attention = "sdpa"` in every later Day 2 run.

Benchmark at batch 1, 8 heads, head width 32, one forward pass per row. Each measurement runs in its
own process, because `ru_maxrss` reports a process high-water mark and the first version of this
benchmark let the `math` peak leak into the `flash` row.

| Length | dtype | Backend | Seconds | Tokens/s | RSS growth |
| ---: | --- | --- | ---: | ---: | ---: |
| 1024 | float32 | math | 0.0220 | 46,491 | 80 MB |
| 1024 | float32 | flash | 0.0038 | 268,554 | 9 MB |
| 4096 | float32 | math | 0.3177 | 12,891 | 1.29 GB |
| 4096 | float32 | flash | 0.0428 | 95,673 | 33 MB |
| 8192 | float32 | math | 1.2644 | 6,479 | 5.17 GB |
| 8192 | float32 | flash | 0.1200 | 68,250 | 62 MB |
| 8192 | bfloat16 | math | 1.4368 | 5,702 | 5.18 GB |
| 8192 | bfloat16 | flash | 0.1374 | 59,629 | 50 MB |

At 8192 tokens the flash kernel is 10.5x faster on 84x less memory growth. The memory gap is the
result to keep. The math backend allocates the `N x N` matrix, so its footprint grows quadratically,
from 80 MB at 1K to 5.17 GB at 8K. The flash kernel never materializes it, so its footprint tracks
the inputs. This machine has no HBM, so the speedup here comes from cache traffic rather than the
HBM traffic the paper measures, and the direction of the argument is the part that transfers.

Bfloat16 is slower than float32 on this CPU at every length. That is the opposite of the GPU case and
worth remembering before Day 4 plans mixed precision.

### EXP-013, KV cache bytes at the Day 2 shape. Measured.

Four layers, model width 256, 8 query heads, head width 32, bfloat16, batch 1, from
`kv_cache_bytes`.

| KV heads | Cache at 1K | Cache at 4K |
| ---: | ---: | ---: |
| 8 | 4 MiB | 16 MiB |
| 4 | 2 MiB | 8 MiB |
| 2 | 1 MiB | 4 MiB |
| 1 | 0.5 MiB | 2 MiB |

The quality side of this experiment is the sweep, and the sweep now sits in the variant grid below.
Keeper choice: two KV heads. It cuts the 4K cache from 16 MiB to 4 MiB and its quality difference
against the baseline is 0.011 bits per byte on code, well inside the 0.043 seed spread the baseline
itself shows. One KV head cuts the cache to 2 MiB at the same measured quality, and the GQA paper's
ROUGE table is the reason we do not take it: the MQA penalty appears as the model grows, and a
400-step run at width 256 cannot resolve it.

### EXP-010 to EXP-013 and EXP-015, the variant grid. Measured.

Eight architectures, three seeds each, 400 steps at a 256-token context on one Colab T4. Bits per
byte on the held-out code and prose splits, averaged over seeds, with the spread across seeds in the
next column. Lower is better.

| Variant | Parameters | KV heads | Cache at 4K | Code bpb | Code spread | Prose bpb | Prose spread | Seconds per step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3,740,160 | 8 | 16 MiB | 2.9339 | 0.0434 | 2.6998 | 0.0038 | 0.0196 |
| rmsnorm | 3,737,856 | 8 | 16 MiB | 2.9352 | 0.0390 | 2.7004 | 0.0031 | 0.0213 |
| swiglu | 3,731,968 | 8 | 16 MiB | 2.8521 | 0.0085 | 2.6802 | 0.0074 | 0.0218 |
| post-norm | 3,740,160 | 8 | 16 MiB | 3.9310 | 0.0047 | 3.3995 | 0.0096 | 0.0209 |
| gqa-4 | 3,478,016 | 4 | 8 MiB | 2.8997 | 0.0262 | 2.6915 | 0.0077 | 0.0244 |
| gqa-2 | 3,346,944 | 2 | 4 MiB | 2.9449 | 0.0235 | 2.7022 | 0.0119 | 0.0241 |
| mqa-1 | 3,281,408 | 1 | 2 MiB | 2.9288 | 0.0261 | 2.6992 | 0.0205 | 0.0236 |
| modern | 3,270,912 | 2 | 4 MiB | 2.8045 | 0.1194 | 2.5906 | 0.0429 | 0.0275 |

The 24 runs took 249 seconds of training and evaluation on the T4. The same grid was measured at
0.235 to 0.270 seconds per step on the laptop, so the GPU is about 11x faster per step and turns a
50-minute grid into a 4-minute one.

Read the spread column before any gap. The baseline moves 0.0434 bits per byte on code across three
seeds with nothing else changed, so a gap smaller than that is noise at this budget.

**EXP-010, RMSNorm. Keep inside `modern`, not on its own.** The quality difference against the
baseline is 0.0013 on code and 0.0006 on prose, which is two orders of magnitude inside the seed
spread. RMSNorm also costs 8.9 percent more step time here, the opposite of the 6.9 to 9.3 percent
speedup the paper reports on a Transformer. The reason is that PyTorch fuses its LayerNorm kernel
and our RMSNorm is plain tensor operations, and at width 256 the norm is a large share of the block.
Re-measure at width 512 and above before treating the speed result as general.

**EXP-011, SwiGLU. Keep.** It improves code by 0.082 and prose by 0.020 bits per byte at matched
parameter count, and the code gain is twice the baseline seed spread. It costs 11 percent more step
time. This is the only single-component swap in the grid that moves quality outside the noise.

**EXP-012 and EXP-013, grouped-query attention. Keep at two KV heads.** All three head counts land
inside the seed spread on quality: gqa-4 is 0.034 better on code, gqa-2 is 0.011 worse, mqa-1 is
0.005 better. None of that is a signal. All three are 20 to 24 percent slower per training step than
MHA, because `enable_gqa` broadcasts the key and value heads across their group and training never
reads a cache. The payoff is decode memory, and it is exact: 4 MiB against 16 MiB at 4K positions.

**EXP-015, post-norm. Revert.** It is 1.00 bits per byte worse on code and 0.70 worse on prose, at
identical parameter count and no speed benefit. Four layers with warmup is the regime where the
pre-norm paper says post-norm still trains, and it trains, badly. The Day 1 pre-norm placement stays.

**The `modern` stack. Best measured, not yet separated.** RoPE, RMSNorm, SwiGLU, and two KV heads
together give the best code and prose numbers in the grid on a quarter of the baseline cache. Its
0.047 margin over `swiglu` on code is smaller than its own 0.119 seed spread, so this grid ranks the
two and does not separate them. Separating them needs steps, not seeds. At 3.27M parameters and 2.96M
training tokens the model sees roughly one token per parameter, against the 20 that Chinchilla-style
scaling suggests, so every architecture here is measured deep in the undertrained regime.

### EXP-009, the length sweep. Keep RoPE. Interpolate only past 4x.

Two models trained at a 512-token context, one with sinusoidal positions and one with RoPE, then
evaluated on held-out text at five lengths. Bits per byte, lower is better. The interpolated column
divides position indices by the evaluation length over 512.

| Evaluation length | Sinusoidal | RoPE | RoPE interpolated | RoPE scale |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 3.3687 | 2.8083 | 2.8083 | 1.0 |
| 1024 | 3.2752 | 2.5340 | 2.5651 | 2.0 |
| 2048 | 3.0565 | 2.2924 | 2.3330 | 4.0 |
| 4096 | 3.5159 | 2.4929 | 2.5141 | 8.0 |
| 8192 | 4.1394 | 2.9023 | 2.8701 | 16.0 |

RoPE beats sinusoidal at every length, by 0.76 bits per byte at 2048. Both curves improve from 512 to
2048, because a longer window gives each prediction more context and the padding share falls, and
both turn back up after 4096.

The Position Interpolation paper predicts that raw extrapolation fails past the trained length.
Ours degrades rather than collapses: 2.29 at 2048 against 2.90 at 8192, which is 4x and 16x the
trained length. Nothing here approaches the 10^3 perplexity the paper reports, and the reason is
that our trained length is 512, not 2048, so the absolute distances stay small.

Interpolation earns its place only at 8192, where it gives back 0.032 bits per byte. At 1024, 2048,
and 4096 it costs 0.02 to 0.04. Dividing positions by the length ratio compresses the rotation
angles, and compressing them below what the model saw in training is a loss when extrapolation was
still working. The rule this gives us: apply interpolation past 4x the trained length, not before,
and re-check the crossover after any change to the trained context.

The paper's other half, 1000 fine-tuning steps at the longer length, is not tested here and belongs
to Phase 4.

## Conflicts with the plans

`AGENTS.md` says to record where a current source disagrees with a plan.

1. `day-wise.md` frames RoPE as "extrapolation to 2K, 4K, 8K". The Position Interpolation paper shows
   that direct extrapolation past the trained length fails, and fails hard. EXP-009 keeps the length
   sweep, states the opposite hypothesis, and adds interpolation as the fix.
2. `day-wise.md` carries the GQA paper's 8 KV heads into our setting. Eight groups was a choice for a
   64-head XXL model. With 8 query heads, 8 KV heads is plain MHA. The ratio transfers and the shape
   of the curve transfers. The number does not, so EXP-013 sweeps 1, 2, 4, and 8.
3. The RMSNorm paper keeps a bias term. RMSNorm as LLaMA uses it has a gain and no bias. We follow
   LLaMA and record here that we did not test the bias variant.
4. The pre-norm paper says warmup can be dropped. Day 1 uses warmup. We keep warmup through Day 2 so
   that the comparisons stay valid, and we revisit it on Day 4.
5. Day 1 recorded that bfloat16 would be a Day 4 concern. EXP-014 shows bfloat16 is slower than
   float32 for attention on this CPU, so Day 4 should treat mixed precision as a GPU-only win until
   measured otherwise.
6. The RMSNorm paper reports a 6.9 to 9.3 percent speedup on a Transformer. EXP-010 measures 8.9
   percent slower at width 256 on a T4, because PyTorch fuses its LayerNorm kernel and our RMSNorm
   is plain tensor operations. The paper's claim is about arithmetic saved. Ours is about which
   kernel the framework has. Both can be true, and the one that decides our config is the kernel.
7. `PLAN.md` assumes a CPU-only lab. Day 2 ends that: the laptop overheats under an hour-long grid,
   so training moved to a Colab T4 and the corpus moved with it.

## Operating constraint

Training does not run on the laptop. A grid saturates all 12 logical CPUs for the better part of an
hour and the machine overheats, so every training run goes to a Colab T4 through
`notebooks/octlm-colab.ipynb` or the `colab-mcp` bridge. The laptop keeps the short checks: the test
suite, the linters, a dry run, and the corpus-independent stages `tiled`, `equivalence`, `sdpa`,
`cache`, and `report`.

`sdpa` stays on the laptop on purpose. It reports process resident memory, which does not describe
GPU allocation, so running it on the T4 would produce a number that looks like a measurement and is
not one.

## Commands

Every entry point takes `--device`, which defaults to `auto` and takes the GPU when the machine has
one. The costs below are measured, laptop rows on 12 CPUs and Colab rows on one T4 with Python
3.13.15 and PyTorch 2.11.0+cu128.

On the laptop:

```sh
uv run python -m octlm.day2 tiled             # 5 s
uv run python -m octlm.day2 equivalence       # 5 s
uv run python -m octlm.day2 sdpa              # 2 min, one process per measurement
uv run python -m octlm.day2 cache             # instant
uv run python -m octlm.day2 report            # instant, reads runs/day2-variants.jsonl
uv run python -m unittest                     # 33 tests, under a second
uv run ruff check . && uv run ruff format --check .
```

On Colab, after cloning the repository into the VM:

```sh
python -m octlm.corpus                        # 40 s, downloads six books once
python -m octlm.day2 variants --device cuda   # 24 runs, 4 min of training
python -m octlm.day2 length --config configs/day2-long.toml --device cuda   # 2 trainings, 2 min
```

The first Colab stage of a session pays a one-time cost the timings above exclude. Training the
tokenizer on every tenth document and encoding the corpus into blocks takes about 5 minutes of
single-core Python, and the results cache under `artifacts/day2/`. The VM is deleted between
sessions, so that cost returns every session unless you copy `data/` and `artifacts/day2/` out to
Drive and back.

Run a long stage with `subprocess.Popen` into a log file rather than as a foreground cell. A cell
driven over the Colab MCP bridge has a 30-second tool timeout, and a 4-minute cell will hit it.

## Exit check

- [x] The tiled attention sketch matches the naive path.
- [x] SDPA memory and throughput are recorded per backend, per dtype, and per context length.
- [x] SDPA matches the Day 1 handwritten attention at every KV-head count.
- [x] You can derive KV cache bytes for any head count without notes.
- [x] Every Day 2 component has a test that fails if its defining property breaks.
- [x] RoPE, RMSNorm, SwiGLU, and GQA each have a measured result against the baseline.
- [x] The KV-head sweep has a defended keeper choice with cache, loss, and throughput numbers.
- [x] Every experiment records a keep or revert decision, including the failures.

Day 2 is complete. Keep RoPE, SwiGLU, and two KV heads. Revert post-norm. RMSNorm rides with the
`modern` stack and has not earned a standalone place at this width.

What the day did not settle: whether `modern` beats `swiglu` alone. Its margin is 0.047 bits per byte
on code and its seed spread is 0.119. The M2 milestone asks for one `modern-50M` config that beats
the Generation 0 baseline on perplexity and decode speed at equal parameter count, and a 400-step run
at 3.3M parameters cannot answer that. Day 3 should raise the step budget before it adds another
component, because every gap this grid measured except SwiGLU's sits inside the noise.
