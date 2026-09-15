# Day 3a: multi-token prediction and the attention variants

Status: in progress.
Started: 2026-09-15
Experiments: EXP-016 through EXP-019

## What we need to finish

Day 3a ends when multi-token prediction, sliding-window attention, block-compressed KV, and MLA each
have a measured result against a control run at the same step budget, and each carries a keep or
revert decision. Before any of that, the day needs a readable noise floor, because Day 2 could not
separate five of its six variants from seed noise.

EXP-020 through EXP-025 (MoE v0, MoE v1, the combined run, asymmetric compute, Muon, mHC) belong to
Day 3b. `day-wise.md` sanctions the split: "Split Day 3 into two days, one for attention variants
and one for MoE plus MTP plus optimizers." We keep MTP here instead, because EXP-016 is the lowest
unfinished experiment and EXP-022 cannot run until it exists.

## Where we start

Day 2 is complete and its exit checks pass. The keepers are RoPE, SwiGLU, and two KV heads.
Post-norm is reverted. RMSNorm rides inside the `modern` stack without a standalone result.

The measured problem Day 2 handed us, in its own words: the baseline moved 0.043 bits per byte
across three seeds with nothing else changed, and every single-component gap except SwiGLU's 0.082
sat inside that band. The grid ran 400 steps at 3.3M parameters on a T4.

Training does not run on the laptop. Every training stage goes to a Colab T4, free tier.

## Stage 0: the noise floor

Problem. A 0.043 bits-per-byte seed spread swallows every effect Day 3 wants to measure.

Hypothesis. Five times the step budget shrinks the spread below 0.03, which is small enough to read
a real component change. The budget is the only thing that moves: `configs/day3.toml` is
`configs/day2.toml` with `steps` 400 to 2000, `warmup_steps` 40 to 200, `eval_interval` 100 to 250.
Width, depth, data, and seeds stay fixed, so the Day 2 rows remain the comparison.

Baseline. The Day 2 `baseline` variant at 400 steps, spread 0.0434 on held-out code.

Stop condition. If the spread at 2000 steps is still above 0.03, raise to 4000 and rerun Stage 0
before any experiment starts. Do not build four experiments on an unreadable floor.

Second question, free with the same runs. Day 2 could not separate `modern` from `swiglu`: 0.047
bits per byte apart against a 0.119 spread. Stage 0 runs `baseline`, `swiglu`, and `modern`, so the
M2 question gets its cheapest shot.

### Result

Not yet run.

## EXP-016: multi-token prediction

Problem. Next-token prediction gives one gradient signal per position. Predicting several futures
densifies that signal and may hand Phase 5 a free draft head for speculative decoding.

### What we read

- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
  (Gloeckle et al. 2024) predicts n futures from one shared trunk with n independent output heads,
  each head a transformer layer feeding a shared unembedding matrix. Its finding that decides our
  hypothesis is the scale coupling: improvements grow with capacity, and for smaller models gains
  are muted or harmful on some benchmarks. At 7B on 200B tokens the 2-future model matched the
  baseline and the 4-future model regressed. The gains that survive are on code and at 3B and above.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) rejects Gloeckle's parallel heads for D sequential
  modules, each a full transformer block plus a projection that mixes the trunk hidden state with
  the embedding of the next ground-truth token, keeping the complete causal chain at every depth. It
  reports an MTP1 acceptance rate above 80 percent, worth about 1.8x on decode throughput.

Decision. Build Gloeckle's parallel form, not DeepSeek's. A sequential transformer block per depth
adds roughly a full model's worth of parameters at 3.3M, which turns an objective comparison into a
capacity comparison. Record the deviation: we use one linear adapter per depth where Gloeckle uses a
transformer layer, which is the cheapest thing that keeps the heads distinct.

Correction to the Day 3 plan. The plan proposed tying every MTP head to `token_embedding.weight` to
keep the added parameter count at zero. That is wrong. Heads tied to the same matrix and fed the
same hidden state compute identical logits, so depth 2 would predict token t+1, not t+2. Each depth
needs its own transformation before the shared unembedding. One `Linear(d_model, d_model)` per extra
depth costs 65,536 parameters, about 2 percent of the model per depth. That overhead is reported in
the result table rather than hidden.

Hypothesis. **MTP does not improve bits per byte at 3.3M parameters.** Gloeckle's scale coupling
points the other way at our size. What we are actually measuring is the cost and the depth-2
agreement rate, which is the number that tells Phase 5 whether a self-draft head is worth building.

Baseline. `mtp_depth = 1` at the Stage 0 budget, which must be numerically identical to the
pre-Day-3 model.

Stop condition. Depths 1, 2, and 3 at three seeds. Keep only if depth 2 improves code bits per byte
by more than the Stage 0 floor, or if the depth-2 agreement rate is high enough to justify a draft
head in Phase 5. Otherwise revert and record Gloeckle's scale finding as the explanation.

Still to read: [Babies Learn to Look Ahead](https://aclanthology.org/2025.babylm-main.41.pdf), the
closest published small-scale MTP result. Record whether it agrees with our measurement.

### Result

Not yet run.

## EXP-017: sparse attention

Problem. Full causal attention costs O(T^2) in both time and score memory. Most of those scores are
near zero. The question is what a windowed model gives up on code, which refers backwards further
than prose does.

### What we read

Longformer and Mistral's sliding window are the pattern. PyTorch's `flex_attention` is the native
way to express a block-sparse mask and is present in the local 2.14 build.

Decision. The primary path is a boolean mask into `scaled_dot_product_attention`. It is correct on
every backend and every device. It also costs the flash kernel: SDPA drops to the memory-efficient
or math backend as soon as an arbitrary mask replaces `is_causal=True`. That cost is a measurement,
not a defect, and the result table records which backend actually ran. `flex_attention` needs
`torch.compile` to beat that, the Colab T4 is sm_75, and chasing it is not what EXP-017 is for.

Hypothesis. At a 512-token trained context, a 128-token window loses measurable bits per byte on
code and loses less on prose, and the windowed run is no faster than full causal, because losing the
flash kernel costs more than the skipped scores save at this length. The speed win belongs to
lengths this lab has not trained at yet.

Baseline. Full causal attention at the same budget and the same context.

Stop condition. Windows 128 and 256, strides 0 and 64, control full causal, three seeds. Report code
and prose bits per byte, seconds per step, the SDPA backend, and the computed mask density.

Still to read: the Longformer and Mistral sections on how window size is chosen.

### Result

Not yet run.

## EXP-018: compressed attention and the needle probe

Problem. A window throws distant tokens away entirely. Compression keeps a lossy summary of them for
a fraction of the cache.

### What we read

[Native Sparse Attention](https://arxiv.org/abs/2502.11089) (DeepSeek, 2025) runs three branches:
compressed, which mean-pools consecutive KV blocks into one vector; selected, which picks top-k
blocks by their compressed attention scores; and sliding window, which keeps recent tokens exact.
A learned gate combines the three.

Decision. Build the compressed branch and the sliding window. Skip the selection branch and the
gate. Selection is where NSA's hardware-aligned kernel work lives, and a top-k over blocks at 3.3M
parameters would measure our softmax, not the idea.

Decision on rotation. Mean-pooling keys that have already been rotated by RoPE averages vectors
pointing in different directions, which shrinks their norm in proportion to how spread out the block
is, and biases the scores by block position. We pool the pre-rotation keys and then rotate each
pooled block at its midpoint position. NSA solves the same problem with an intra-block position
encoding. Ours is the cheaper version of that idea and the note records it as a deviation.

**The needle design, and it is a judgment call.** `PLAN.md` asks for "retrieval accuracy on a needle
style synthetic test". A 3.3M base model cannot answer a question about a needle, so the usual
prompt-and-check form would measure nothing but the model's inability to follow an instruction. The
form that works at this scale is a copy probe: place a random token sequence at depth p, fill the
rest with corpus text, repeat the same sequence at the end, and compare the mean negative log
likelihood on the repeated copy against the same tokens with no earlier occurrence. A model that
carried the reference forward scores far lower on the repeat. This is induction, it is present in
tiny models, and it disappears when the mechanism carrying the reference is removed. It is not the
long-context needle test the literature runs, and the note says so wherever the number appears.

Hypothesis. Block compression cuts cache bytes roughly by the block size outside the window and
costs less code bits per byte than a plain window of the same cache budget. On the copy probe, full
causal recovers the needle at every depth, the window recovers it only inside the window, and
compression recovers it partially at every depth.

Baseline. Full causal attention, and the EXP-017 window at a matched cache budget.

Stop condition. Blocks 4 and 8 against an uncompressed control, three seeds, plus the copy probe
swept across needle depth for all three attention configurations.

### Result

Not yet run.

## EXP-019: MLA

Problem. The KV cache dominates decode memory. Day 2 cut it 4x with two KV heads. MLA claims more.

### What we read

[DeepSeek-V2](https://arxiv.org/abs/2405.04434) compresses keys and values into one low-rank latent
`c_KV` per token, shared across heads, and up-projects at use. Position rides in a separate decoupled
RoPE key, also shared across heads, because a rotation cannot survive the low-rank projection. Only
the latent and the RoPE key are cached. Reported cache reductions run past 90 percent.

Decision. Build the compression and the decoupled RoPE key. Skip absorbing the up-projection into
the query and output matrices. That absorption is a decode-time identity with no training-time
effect, it belongs to Phase 5, and it is the part of MLA most likely to hide a silent bug.

**Conflict with the plans, recorded per `AGENTS.md`.** `day-wise.md` and `PLAN.md` both carry MLA in
as a large cache win. That figure is measured against multi-head attention with many heads.
Day 2 already took the cache win with GQA-2. At `d_model` 256, `head_size` 32, and `kv_heads` 2, the
present cache holds 2 x 2 x 32 = 128 dimensions per token per layer. MLA at rank 64 with a 16
dimension RoPE key holds 80. That is 1.6x, not 10x.

Hypothesis. **MLA loses to GQA-2 at this width.** The useful number is the rank at which MLA's cache
first falls below GQA-2's 128 dimensions, and whether quality survives that rank. MLA's advantage is
a property of wide models with many heads, and this lab is neither.

Baseline. GQA-2 at the same budget, which is the Day 2 keeper.

Stop condition. Ranks 32, 64, and 128 against the GQA-2 control, three seeds. Report cache
dimensions per token per layer, code and prose bits per byte, and seconds per step. Mark
experimental either way.

### Result

Not yet run.

## Conflicts with the plans

1. `PLAN.md` and `day-wise.md` treat MLA as a cache win. Against GQA-2 at 8 query heads and head
   width 32 it is a 1.6x win at best. See EXP-019.
2. `day-wise.md` lists multi-token prediction under "possible gains for code". Gloeckle's own scale
   analysis predicts no gain, or a regression, at 3.3M parameters. EXP-016 states the negative
   hypothesis before running.
3. The Day 3 plan proposed tying MTP heads to the embedding at zero parameter cost. Tied heads fed
   one hidden state are identical, so that design cannot predict distinct futures. Corrected to one
   linear adapter per depth, at about 2 percent of the model per depth.
4. `PLAN.md` asks for a needle-style retrieval test. At 3.3M parameters the standard form measures
   instruction following, not retrieval. Replaced with a copy probe and flagged wherever it appears.

## Housekeeping

`runs/day2-variants.jsonl` holds two records. The full 24-run Day 2 grid ran on Colab and its numbers
reached the Day 2 note, but the JSONL never came back from the VM. A local `octlm.day2 report`
therefore summarizes two rows and must not be read as the Day 2 result. The note is the record.

## Commands

Costs below are estimates from the Day 2 rate of roughly 10 seconds per 400-step run on a T4, scaled
to 2000 steps. Nothing here has been timed on a GPU yet.

On the laptop, before any GPU time:

```sh
uv run ruff check . && uv run ruff format --check .
uv run python -m unittest                                          # 54 tests, under a second
uv run python -m octlm.train --config configs/day3.toml --dry-run  # 3,740,160 parameters
uv run python -m octlm.day3 cache --config configs/day3-long.toml  # arithmetic, instant
```

On Colab, in order. Stage 0 gates the rest:

```sh
python -m octlm.day2 variants --config configs/day3.toml \
    --variants baseline swiglu modern --out runs/day3-floor.jsonl --device cuda   # 9 runs, ~8 min
python -m octlm.day2 report --out runs/day3-floor.jsonl
python -m octlm.day3 mtp --config configs/day3.toml --device cuda                 # 9 runs, ~10 min
python -m octlm.day3 sparse --config configs/day3-long.toml --device cuda         # 12 runs, ~12 min
python -m octlm.day3 compressed --config configs/day3-long.toml --device cuda     # 9 runs, ~12 min
python -m octlm.day3 mla --config configs/day3-long.toml --device cuda            # 12 runs, ~12 min
python -m octlm.day3 report
```

Roughly 55 minutes of T4 in total. Restore `data/` and `artifacts/day2/` from Drive first, or the
first stage pays another 5 minutes for the tokenizer and the block build. The 512-token blocks are
not in the cache yet either way, so the first long stage pays for building those.

Two decisions that shrank the plan while building it:

- The copy probe has no stage of its own. It rides on the first seed of `sparse` and `compressed`,
  reusing the models those stages already trained. A separate stage would have retrained three
  models to probe them, for about 12 minutes of T4 and no extra information.
- Stage 0 needed no new code. `octlm.day2 variants` already takes `--config` and `--variants`, so a
  new config file and an `--out` flag were the whole change.

`flash_accepts_mask` is probed on the device the model trained on. PyTorch's CPU flash path takes an
explicit mask and the CUDA kernel does not, so probing on CPU for a CUDA run would report the wrong
backend. This is the same trap `notes/day2.md` recorded for process RSS on a GPU.

## Exit check

- [ ] The seed spread at the Day 3 budget is measured and frozen as the minimum detectable effect.
- [ ] `modern` against `swiglu` is resolved, or recorded as unresolved with the numbers that failed.
- [ ] `mtp_depth = 1` produces logits identical to the pre-Day-3 model.
- [ ] MTP has a measured cost, a depth-2 agreement rate, and a keep or revert decision.
- [ ] Sliding window and strided attention each have quality, speed, mask density, and the SDPA
      backend that actually ran.
- [ ] The copy probe produces a depth curve for full, windowed, and compressed attention.
- [ ] MLA has cache dimensions per rank against the GQA-2 control, and the crossover rank is named.
- [ ] Every component that lost is still in the note with its numbers and its reason.
