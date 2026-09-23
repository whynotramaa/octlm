# Day 4: train the from-scratch model properly

Status: code built, nothing run.
Started: 2026-09-23
Experiments: EXP-065 through EXP-068

## What we need to finish

Day 4 ends when one model of about 26M parameters has trained on TinyStories to a flat validation
loss, and ten fixed prompts produce stories a reader judges coherent. `PLAN.md` explains why this
replaces the rest of the first plan.

## Where we start

Nothing trained carries over. The user moved to a new Colab account with a Pro subscription on
2026-09-23, and the old account's Drive files are gone. Nothing needs retraining for that reason.
The Day 2 corpus and grid belong to the first plan, Day 3a never ran, and Day 5 remeasures the
comparisons at the new scale. The Day 2 numbers stay in `notes/day2.md` as the record.

The new Colab Pro account offers two accelerators: a T4 GPU and a TPU v5e-1. No L4 or A100. We
train on the T4, which has no bfloat16, so the trainer uses float16 with `GradScaler`. Every
training record carries the `dtype` that ran.

We rejected the TPU for Day 4. PyTorch reaches a TPU through `torch_xla`, and octlm's device
selection, autocast, `GradScaler`, SDPA kernel choice, and checkpoint saving are all written for
CUDA or CPU. The port would cost more time than it saves on a run estimated at 70 to 180 minutes,
and no later day needs a TPU. On the T4, SDPA uses the memory-efficient kernel, because
FlashAttention needs an sm80 or newer GPU.

## What we read

- [TinyStories on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories), fetched
  2026-09-23. The V2 files are `TinyStoriesV2-GPT4-train.txt` at 2,227,753,162 bytes and
  `TinyStoriesV2-GPT4-valid.txt` at 22.5 MB. The license is CDLA-Sharing-1.0. A line holding only
  `<|endoftext|>` separates stories, and blank lines sometimes follow it. The validation file
  begins in the middle of a story. Decision: split on the delimiter line, strip surrounding
  newlines from each story, drop empty stories, and keep the truncated first story, because
  dropping it would be a special case for one story out of thousands.
- [Eldan and Li, TinyStories](https://arxiv.org/abs/2305.07759), not yet reread for this day. It
  remains the lead from `notes/day3.md` that models under about 30M parameters write coherent
  stories on this data. EXP-067 tests it directly.

## Engineering decisions

**Vocabulary 8,192, without the 4,096 comparison.** `PLAN.md` asked to pick between the two by bits
per byte. Bits per byte is a property of a trained model, so the comparison costs two full runs.
The TinyStories authors used a 10K vocabulary. We take 8,192 and record the compression it reaches.
If Day 5 has spare GPU time, a 4,096 run at the EXP-069 budget answers the question.

**The tokenizer trains on the first 10M characters of the train split.** `ByteBPETokenizer.train`
recounts every pair after every merge, so its cost grows with the sample and the merge count. The
TinyStories vocabulary is small, so 10M characters should cover it. The prepare log records the
training seconds.

**The encoder caches chunk IDs.** BPE encodes each pre-token on its own, so a pre-token always maps
to the same IDs. `ByteBPETokenizer.encode` now keeps a dictionary from chunk bytes to IDs. The
output does not change. EXP-065 measures the speed with and without the cache.

**Token files are flat `uint16`.** Each story is encoded with BOS and EOS and appended to one
stream. Training reads random 513-token windows from a `numpy.memmap`, so a window can span a
story boundary with EOS and BOS between the two stories. BPE merges still never cross a story.

**Validation is the first 256 non-overlapping 512-token blocks of the validation split.** That is
131,328 tokens. `train.evaluate` computes bits per byte from a per-token byte table.

**The trainer writes a checkpoint at every eval.** A Colab disconnect loses at most 1,000 steps.
The checkpoint stores the `GradScaler` state. `octlm.day4 train` resumes when the checkpoint file
exists.

**Mixed precision is opt-in inside `train_model`.** Days 1 to 3 keep float32 on every device, so
their numbers stay comparable. Day 4 turns it on.

## EXP-065: corpus and tokenizer

Problem. The Day 2 corpus had 2.96M training tokens. The Day 4 model needs about 400M.

Hypothesis. The pure-Python encoder is too slow to encode 2.2 GB in one session. The chunk cache
fixes the merge cost. The per-character pre-tokenizer stays the bottleneck.

Baseline. `encode_chunk` on every chunk of the first 2M characters of the validation split, no
cache.

Measurement. The `encode_benchmark` record in `data/tinystories/prepare.jsonl` gives bytes per
second with and without the cache. The `encode` records give tokens, bytes per token, and seconds
per split. `artifacts/day4/bpe.jsonl` gives the tokenizer training time.

Stop condition. If the train split takes longer than two hours to encode, stop and speed up the
pre-tokenizer before training anything.

### Result

Not yet run.

## EXP-066: mixed precision

Problem. Float32 wastes most of the GPU's matrix throughput.

Hypothesis. Mixed precision at least doubles tokens per second, and its validation loss at step
1,000 stays within 0.02 of float32.

Baseline. The same 1,000 steps with `--full-precision`.

Measurement. `train_seconds` and `validation_loss` at step 1,000 in `day4-precision.jsonl`, with
the `dtype` and `device` of each run.

Stop condition. If the gap is larger than 0.02, train the main run in float32 and record why.

### Result

First attempt, 2026-09-23, invalid. `autocast_dtype` chose bfloat16 because
`torch.cuda.is_bf16_supported()` returns `True` on the T4, but the T4 (compute capability 7.5) has
no bfloat16 tensor cores and emulates it. The "mixed" run took 1,529 s against 1,173 s for float32,
with validation loss 1.259 against 1.250. Fix: choose bfloat16 only at compute capability 8.0 or
higher, otherwise float16 with `GradScaler`. The float16 rerun appends to `day4-precision.jsonl`.

## EXP-067: the main run

Problem. Every model so far was too undertrained to read an architecture effect or produce text.

Hypothesis. About 26M parameters trained on about 393M tokens writes coherent short stories.

Setup. `configs/day4.toml` is the Day 2 `modern` stack: RoPE, RMSNorm, SwiGLU, and two KV heads,
through SDPA. Width 512, 8 layers, 8 query heads, 512-token context, vocabulary 8,192. The dry run
reports 26,255,872 parameters. 24,000 steps of 32 x 512 tokens is 393,216,000 tokens, about 15
tokens per parameter. The learning rate schedule is the Day 3 one with a longer warmup: peak 6e-4,
floor 6e-5, 500 warmup steps, cosine decay.

Estimate. 6 x 26.3M x 393M is about 6.2e16 FLOPs. At an assumed 15 TFLOPs on a T4 that is about 70
minutes, before attention and data-loading overhead. EXP-066 gives the real rate.

Baseline. None at this scale. The exit check is absolute: a flat validation curve and readable
samples.

Stop condition. If validation loss stops improving for 3,000 steps, stop the run and keep the best
checkpoint. If the loss diverges, halve the learning rate and restart.

### Result

Not yet run.

## EXP-068: sampling

Problem. Greedy decoding on story data repeats phrases, which would make the exit check judge the
decoding instead of the model.

Change. `Decoder.generate` takes `temperature`, `top_k`, and a `torch.Generator`. Temperature 0
keeps greedy decoding, so earlier days' output does not change.

Measurement. `octlm.day4 samples` writes the ten prompts in `day4.PROMPTS`, once greedy and once at
temperature 0.8 with top-k 40, seed 0. Paste both sets into this note.

### Result

Not yet run.

## What changed in the code

- `octlm/day4.py` has the `prepare`, `train`, and `samples` stages.
- `octlm/tokenizer.py` gained `encode_chunk` and `chunk_cache`. `encode` uses both.
- `octlm/model.py` gained `sample_token` and the sampling arguments on `generate`.
- `octlm/train.py` gained `autocast_dtype`, the `mixed_precision` argument, a checkpoint at every
  eval, and the scaler state in the checkpoint.
- `configs/day4.toml` is new. `notebooks/octlm-colab.ipynb` now runs Day 4 only.

No tests were added and the suite was not run, per the user's standing instruction. `AGENTS.md`
asks for both. That conflict is open.

## Commands

On Colab, in notebook order:

```sh
python -m octlm.day4 prepare
python -m octlm.day4 train --stop-after 1000 --checkpoint /content/mixed.pt ...
python -m octlm.day4 train --stop-after 1000 --full-precision --checkpoint /content/full.pt ...
python -m octlm.day4 train --checkpoint $DRIVE/run.pt --metrics $DRIVE/day4.jsonl ...
python -m octlm.day4 samples --checkpoint $DRIVE/run.pt --temperature 0
python -m octlm.day4 samples --checkpoint $DRIVE/run.pt
```

## Exit check

- [ ] Encode throughput with and without the chunk cache is recorded.
- [ ] Mixed precision has a speed and loss comparison against float32.
- [ ] Validation loss has flattened, and the curve is in this note.
- [ ] Bits per byte on the validation blocks is recorded.
- [ ] Ten greedy and ten sampled stories are in this note, with a coherence judgment.
