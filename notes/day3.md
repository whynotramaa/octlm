# Day 3a: multi-token prediction and the attention variants

Status: stopped on 2026-09-23 before any GPU run. The plan changed. See the last section.
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

## Session on 2026-09-23: the plan changes

### What happened

No Day 3a stage ran. Before the Stage 0 runs, the user asked whether the project was on the right
track, and then whether to continue it or fine-tune a Qwen model instead. The user's goal, in their
words, is to learn ML "in depth" for placement season and to end with "a harness that can control
my own model". The user approved a rewrite of `PLAN.md` around that goal.

### The problem, from our own numbers

The Day 2 grid trained models of about 3.3M parameters for 400 steps at batch 8 and 256-token
context. That is 819,200 tokens per run. `configs/day3.toml` raises it to 2000 steps, which is
4,096,000 tokens for a 3,740,160-parameter model. `configs/day3-long.toml` reaches 8,192,000 tokens.
The Day 2 corpus holds only 2.96M training tokens, so longer runs repeat data.

Hoffmann et al. put the compute-optimal budget near 20 tokens per parameter, about 75M tokens for
the Day 3 model. Day 2 ran at about 0.25 tokens per parameter and Day 3 would have run at about 1.1.
The 0.043 bits-per-byte seed spread that blocked Day 2 is what an undertrained model looks like.
Stage 0 would have measured it again at a budget that is still about 18x too small.

The second problem is the goal. No model we can pretrain on free Colab will follow instructions or
emit reliable tool calls, and the harness needs both.

### Decision

Keep the from-scratch track and train it once at a readable budget, about 20M parameters on
TinyStories. Then load a small Qwen instruct model into `octlm/model.py`, which already has the same
RoPE, RMSNorm, SwiGLU, and grouped-query attention blocks. Write LoRA in octlm, fine-tune Qwen for
tool calls, and build the harness and its eval around it. `PLAN.md` now holds Days 4 to 9.

We considered two alternatives and rejected both.

- Continue the first plan. It spends the remaining time on variants the harness never uses, and it
  ends with a model too weak to drive a harness.
- Fine-tune Qwen with an off-the-shelf trainer and drop octlm. It gives the harness a working model
  but leaves nothing to defend when an interviewer asks how attention, the KV cache, or LoRA work.

### Sources

These come from the planning conversation, not from pages fetched in this session. Each one is a
lead until the day that depends on it fetches and confirms it.

- [Hoffmann et al., Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
  puts the compute-optimal budget near 20 tokens per parameter. This sets the EXP-067 budget.
- [Eldan and Li, TinyStories](https://arxiv.org/abs/2305.07759) reports coherent English from models
  under about 30M parameters trained on a synthetic dataset of simple stories. This picks the Day 4
  data.
- [Hu et al., LoRA](https://arxiv.org/abs/2106.09685) adds a trainable low-rank update with one
  factor initialized to zero. This gives EXP-080 its step-0 parity test.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) describes dense models from 0.6B
  parameters with grouped-query attention, SwiGLU, RoPE, RMSNorm, and QK-norm. This makes Day 6
  possible without a second model file.
- The NVIDIA T4 is a Turing card with float16 tensor cores and no bfloat16 support. This sets
  float16 with `GradScaler` in EXP-066, and it conflicts with the bfloat16 plan in `day-wise.md`
  Day 4.
- Kaggle's free GPU quota, about 30 hours a week with background execution, is a lead for runs
  longer than a Colab session. EXP-066 checks it.

### Where each first-plan experiment went

| First plan | Now |
| --- | --- |
| Stage 0 noise floor | Replaced by EXP-069 at the new scale |
| EXP-016 multi-token prediction | Code kept. Runs as EXP-070 |
| EXP-017 sparse attention | Code kept behind default-off flags. Not scheduled |
| EXP-018 compressed attention and copy probe | Code kept behind default-off flags. Not scheduled |
| EXP-019 MLA | Code kept behind default-off flags. Not scheduled |
| EXP-020 to EXP-025, MoE, combined run, asymmetric compute, Muon, mHC | Dropped |
| EXP-026 mixed precision | EXP-066, float16 only |
| EXP-027 to EXP-029, batch size, schedule, checkpointing | Folded into EXP-066 and EXP-067 where the run needs them |
| EXP-030 scaling check | Dropped |
| EXP-031 to EXP-034, DDP, FSDP, tensor and pipeline parallelism | Dropped. One GPU |
| EXP-035 and EXP-036, naive generation and KV cache | EXP-071, then EXP-076 on Qwen |
| EXP-037 and EXP-038, prefill split and `torch.compile` | Folded into EXP-071 and EXP-076 if time allows |
| EXP-039 quantization | EXP-072, int8 only |
| EXP-040 to EXP-042, batching, prefix cache, speculative decoding | Dropped. Cache reuse across turns lives in EXP-078 |
| EXP-043 SFT | EXP-082 |
| EXP-044 LoRA | EXP-080 |
| EXP-045 and EXP-046, DPO and reward model | Dropped |
| EXP-047 and EXP-048, GRPO and RLVR | Day 9 option |
| EXP-049 to EXP-053, evals and regression gate | Replaced by the EXP-079 harness eval and bits per byte |
| EXP-054 task classifier | Dropped |
| EXP-055 and EXP-056, hybrid retrieval | Dropped. The harness has a `grep` tool |
| EXP-057 tool loop | EXP-078 |
| EXP-058 tool-call fine-tune | EXP-081 and EXP-082 |
| EXP-059 writing path, EXP-060 cache stack | Dropped |
| EXP-061 router and cost log | Day 9 option |
| EXP-062 to EXP-064, long context, stability run, model card | Dropped |

New numbers start at EXP-065, so no retired number means two things.

### Conflicts with the rules

1. `AGENTS.md` forbids starting Day 4 before every Day 3 exit check passes. None of them pass. The
   user's request outranks the plan, so the Day 3a checks above are withdrawn, not failed. Day 4
   starts from Day 2's passed checks.
2. `AGENTS.md` says to learn BPE merges from our training data only. Qwen's tokenizer ships with its
   weights. The rule keeps applying to every tokenizer octlm trains. For Qwen, EXP-075 applies the
   hash check instead, on `tokenizer.json`.
3. `day-wise.md` is a reading schedule for the first plan. Days 1 and 2 match the work done. Days 3
   to 8 cover topics the new plan dropped. The file now says so at the top, and each new day lists
   its reading in its own note.

### What changed in the repository

- `PLAN.md` was rewritten around Days 4 to 9.
- `README.md` now describes the new goal, status, and roadmap.
- `day-wise.md` gained a note at the top on which days still apply.
- No code changed. The Day 3a code, configs, and tests stay as they were.
- `octlm.train` and `octlm.tokenizer` default to `PLAN.md` as training text and `day-wise.md` as
  held-out text. Both files changed, so a Day 1 command run now trains on different text. The Day 1
  numbers belong to the versions at commit `ab0d49c`. No test reads either file.

### Exit check for this session

- [x] The reason for the change is recorded with our own numbers.
- [x] Every first-plan experiment has a new number or a reason it was dropped.
- [x] Every conflict with `AGENTS.md` and `day-wise.md` is recorded.
