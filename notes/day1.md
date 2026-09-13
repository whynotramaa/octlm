# Day 1: foundations and the Generation 0 Transformer

Status: complete  
Started: 2026-09-13  
Completed: 2026-09-13  
Experiments: EXP-001 through EXP-008

## What we need to finish

Day 1 ends when both tokenizers work, the baseline decoder overfits one block, validation perplexity
is logged, and a resumed CPU run matches an uninterrupted run. The repository must also pass its
tests, lint check, format check, dry-run, and dummy benchmark.

This is an outcome-based roadmap day. The reading schedule budgets six to eight hours for study, but
the implementation and measurements can take longer. We will not shorten the correctness work to
fit that estimate.

## Where we started

The repository contained only `PLAN.md` and `day-wise.md`. It was not a Git repository. The machine
has an Intel Core i5-12450H with 12 logical CPUs and 15 GiB of RAM. Python 3.14.7 and `uv` 0.12.3
were installed. PyTorch was not installed. `nvidia-smi` could not contact an NVIDIA driver, so Day 1
uses the CPU.

## Session on 2026-09-13

We read both project plans before choosing files or APIs. `day-wise.md` describes Day 1 as a reading
day and maps it to EXP-003 through EXP-008. Because the repository was empty, EXP-001 and EXP-002
also belong to the first implementation milestone.

We chose one daily note instead of eight experiment files. The single note keeps the explanation,
research, failures, and results together. `AGENTS.md` holds only rules that future work must follow.

## What we read

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) defines scaled dot-product attention,
  multiple attention heads, causal masking, positional information, residual paths, and the original
  Transformer block. We will write the causal attention calculation with PyTorch tensor operations.
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
  explains how repeated pair merges build a fixed subword vocabulary. We will learn ordered merges
  from the training corpus instead of using a pretrained tokenizer.
- The [GPT-2 technical report](https://cdn.openai.com/better-language-models/language-models.pdf) and
  [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) motivate a
  reversible byte alphabet and boundaries that stop poor merges across character categories. We
  will use all 256 bytes and preserve whitespace and Unicode exactly.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) normalizes a sample across its feature
  dimension. We will use pre-LayerNorm blocks and never normalize across the batch or sequence.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) separates weight decay
  from Adam's gradient update. We will use PyTorch AdamW and exclude biases and normalization weights
  from decay.
- PyTorch's [reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html) says
  that repeatability is limited to a fixed release, platform, and device. Our exact-resume claim has
  the same boundary.
- PyTorch's [checkpoint guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
  says that training resume needs both model and optimizer state. We will also save random-generator
  state and artifact hashes.

## What the paper exercises showed

For an input shaped `[B, T, C] = [1, 8, 16]` with two heads, each head has width 8. After projection,
`Q`, `K`, and `V` each have shape `[1, 2, 8, 8]`. `QK^T` has shape `[1, 2, 8, 8]`. We divide those
scores by `sqrt(8)`, mask future columns, apply softmax along the last dimension, multiply by `V`,
and join the heads back into `[1, 8, 16]`.

The allowed positions for a sequence of length four are:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

A character tokenizer makes common code fragments long, spends many attention positions on repeated
syntax, and makes the model learn multi-character names one character at a time. BPE can compress
frequent fragments while byte fallback still represents unseen text.

Raw token perplexity cannot compare two tokenizers. A tokenizer that combines more bytes into each
token changes the prediction unit. We will use bits per UTF-8 byte for cross-tokenizer comparisons.

We repeated one Python function and traced the first 20 available merges. The token IDs start at 261
because IDs 0 through 255 represent bytes and IDs 256 through 260 are reserved special tokens.

| Merge | Input IDs | New ID | Bytes after the merge |
| ---: | --- | ---: | --- |
| 1 | 32 + 32 | 261 | two spaces |
| 2 | 97 + 108 | 262 | `al` |
| 3 | 95 + 110 | 263 | `_n` |
| 4 | 98 + 101 | 264 | `be` |
| 5 | 109 + 264 | 265 | `mbe` |
| 6 | 117 + 265 | 266 | `umbe` |
| 7 | 263 + 266 | 267 | `_numbe` |
| 8 | 267 + 114 | 268 | `_number` |
| 9 | 105 + 110 | 269 | `in` |
| 10 | 111 + 116 | 270 | `ot` |
| 11 | 116 + 270 | 271 | `tot` |
| 12 | 269 + 116 | 272 | `int` |
| 13 | 271 + 262 | 273 | `total` |
| 14 | 10 + 261 | 274 | newline and two spaces |
| 15 | 95 + 118 | 275 | `_v` |
| 16 | 98 + 273 | 276 | `btotal` |
| 17 | 99 + 111 | 277 | `co` |
| 18 | 100 + 268 | 278 | `d_number` |
| 19 | 101 + 277 | 279 | `eco` |
| 20 | 102 + 105 | 280 | `fi` |

The exact trace comes from deterministic frequency counts and token-ID tie-breaking. The merge table
can contain partial words such as `btotal`; later merges can combine those pieces into an identifier.

## Engineering decisions

The baseline follows the small GPT-2 decoder shape rather than reproducing the translation model in
the 2017 paper. It uses learned absolute positions, pre-LayerNorm residual blocks, causal multi-head
attention, GELU, and a tied language-model head. Later experiments can replace one component at a
time without changing the Day 1 baseline.

We will use a full pair recount while training the Day 1 BPE tokenizer. We can inspect each state
change on this corpus. Its cost grows with both the corpus and the number of merges. We will
replace it only if profiling on the representative corpus shows that it blocks work.

We will not choose the final 16K to 32K vocabulary from two planning documents. Day 1 measures small
attainable vocabularies and records the result. The final choice needs the real code and prose mix.

## What we built and decided

### EXP-001: repository and measurements

We initialized Git and pinned Python 3.14, PyTorch 2.14 CPU, and Ruff with `uv.lock`. The lock-file
SHA-256 is `956ad296f104d9fcf66bde8a8ea5b6d101857c79b6e080c66ea339f483e98909`.
One TOML file holds the Day 1 settings. The dry-run creates a 541,952-parameter model and returns
logits shaped `[1, 128, 1024]`. We kept this setup because it meets the measurement requirement
without a configuration framework.

### EXP-002: hardware limit

The batch-one CPU check completed at every bounded context length. A 128-token forward pass took
4.0 ms. The 256, 512, 1024, and 2048-token passes took 6.6 ms, 15.3 ms, 28.7 ms, and 115.2 ms. Peak
resident memory reached 504 MB by the end of the process. These are smoke measurements from one run,
not stable performance claims. We kept 2048 as the largest verified Day 1 context.

### EXP-003: character tokenizer

The character tokenizer learned 86 entries from `PLAN.md`. It used 13,939 tokens for the 13,939-byte
held-out prose file. The unseen Unicode sample mapped 77.8 percent of its characters to UNK. After 40
steps, its training loss was 3.3950. Validation loss was 3.4011, perplexity was 29.998, and bits per
byte was 4.9068. We kept it as the deliberate baseline.

### EXP-004: byte-level BPE

The 1024-entry BPE tokenizer reduced the held-out prose to 7,319 tokens and the code sample from 49
to 35 tokens. It represented every Unicode sample byte and had no unknown token. The 2048-entry run
stopped at 1,708 entries because no remaining pair occurred twice. This clean stop prevents useless
single-occurrence merges.

After the same 40-step token budget, BPE training loss was 5.5359. Validation loss was 5.4823,
perplexity was 240.391, and bits per byte was 4.1497. The raw perplexity is larger because BPE predicts
a different unit. Bits per byte improved by 15.4 percent against the character run, so we kept the
1024-entry tokenizer for the Day 1 model. We did not choose the final project vocabulary.

### EXP-005: embeddings and positions

The model has separate learned token and position tables. The benchmark emits nearest token and
position IDs without a plotting dependency. The current neighbors only confirm the diagnostic path
because the benchmark model is untrained. Mean pooling changes `[B, T, C]` into `[B, C]`; learned
pooling adds parameters for the same sequence reduction. An autoregressive language model predicts
at each position, so neither pooling method belongs in its forward pass. We kept both as written
exercises and added no unused pooling code.

### EXP-006: causal multi-head attention

The implementation projects Q, K, and V, splits four heads, scales by the square root of the head
width, applies the lower-triangular mask, and joins the heads. The tests compare the four-token mask
with the hand-written matrix. Another test changes the last input token and confirms that all earlier
logits remain byte-identical. We kept the handwritten attention path as the Generation 0 reference.

### EXP-007: baseline block

The baseline has two pre-LayerNorm blocks, learned absolute positions, GELU feed-forward layers, and
a tied output head. The first test run exposed poor learning because PyTorch's default embedding
initialization was too wide for a tied output head. We changed linear and embedding weights to a
normal distribution with standard deviation 0.02. The test then passed. We kept the corrected model.

### EXP-008: training and resume

The trainer uses shifted targets, token-weighted validation loss, AdamW, gradient clipping, linear
warmup, cosine decay, artifact hashes, atomic checkpoints, and JSONL metrics. The checkpoint stores
the model, optimizer, Python random state, PyTorch random state, and sampler state. A split CPU run
and an uninterrupted run finished with identical weights and the same next loss.

On one fixed BPE block, training loss fell from 5.5891 at step 20 to 0.9121 at step 200. Final
perplexity was 2.4830, and bits per byte was 0.6338. Greedy decoding reproduced this training text:

```text
# Miniature modern LLM stack from first principles

## Goal

Build one long runn
```

The result proves that the tokenizer, causal model, loss, optimizer, and generation path compose. It
does not measure generalization.

## What we are not building today

Day 1 excludes RoPE, RMSNorm, SwiGLU, grouped-query attention, production SDPA, FlashAttention, mixed
precision, compilation, quantization, KV caching, distributed training, post-training, retrieval,
tools, and assistant code. It also excludes external datasets and empty directories for later work.

## What failed during setup

The sandbox first blocked writes inside `.git`, so repository initialization required explicit
approval. `uv` could not write its normal cache, so we used `/tmp/octlm-uv-cache`. Package downloads
also needed approved network access.

The first environment used Python 3.13 because the project allowed several versions. We added
`.python-version`, narrowed `requires-python`, and recreated the environment with Python 3.14.7.

PyTorch prints an optional NumPy warning when it imports. Day 1 does not convert tensors to NumPy
arrays, so we did not add an unused runtime dependency to silence the warning. Add NumPy when an
experiment needs that conversion.

## Commands and results

- `uv run python -m unittest` passed all 15 tests in about 0.65 seconds on the final run.
- `uv run ruff check .` passed with the complexity limit set to 10.
- `uv run ruff format --check .` reported that all 12 checked files were formatted.
- `uv run python -m octlm.train --config configs/day1.toml --dry-run` emitted valid JSON.
- `uv run python -m octlm.bench --dummy --contexts 128 256 512 1024 2048` completed every context.
- `uv run python -m octlm.tokenizer --vocab-sizes 512 1024 2048` wrote deterministic artifacts and
  emitted the tokenizer measurements above.
- `uv run python -m octlm.train --tokenizer bpe --overfit` passed the memorization gate.

## Exit check

- [x] Both tokenizers round-trip the required text cases.
- [x] BPE has zero unknown tokens and deterministic merges.
- [x] The causal attention tests pass.
- [x] The baseline overfits one fixed block and generates a recognizable continuation.
- [x] Validation perplexity and bits per byte are logged.
- [x] Checkpoint resume matches uninterrupted CPU training.
- [x] The test, lint, format, dry-run, and benchmark commands pass.
- [x] EXP-001 through EXP-008 contain measurements and keep or revert decisions.

Day 1 is complete. Day 2 starts with fresh research on RoPE, RMSNorm, SwiGLU, MQA, GQA, residual
placement, and SDPA. We will change one component per experiment and compare it with this baseline.
