# octlm

A small language-model engineering lab. It builds a decoder from scratch and trains it properly at
about 20M parameters. Then it loads a small Qwen instruct model into the same code, fine-tunes it
with a LoRA written here, and builds an agent harness that runs the model with tools. Every change
is measured against a baseline. `PLAN.md` defines the roadmap. `AGENTS.md` defines the working rules.

The plan changed on 2026-09-23. The first plan aimed to pretrain 50M to 150M models into a local
assistant. `notes/day3.md` records why it stopped and where each of its experiments went.

## Status

Day 1 is complete: EXP-001 through EXP-008, covering Phase 0 and Phase 1 of nine phases. What runs
today is a character tokenizer, a byte-level BPE tokenizer, a decoder, a training loop with exact
resume, and a benchmark schema.

Day 2 is complete: EXP-009 through EXP-015. RoPE, SwiGLU, and two KV heads are keepers, post-norm is
reverted, and RMSNorm is quality-neutral at this width. Training moved to a Colab T4, because the
laptop overheats under an hour-long grid.

Day 3a stopped on 2026-09-23 before any GPU run: EXP-016 through EXP-019. Multi-token prediction,
sliding-window and strided attention, block-compressed KV, and MLA are built and tested behind flags
that default off. Multi-token prediction runs again as EXP-070. The others are not scheduled.

Day 4 is next: TinyStories, float16 training on a T4, and one properly trained 20M model.

Read `notes/day1.md`, `notes/day2.md`, and `notes/day3.md` for the research sources, the
measurements, and every keep-or-revert decision.

## Requirements

- Python 3.14, pinned by `.python-version`
- [uv](https://docs.astral.sh/uv/) for dependency resolution
- PyTorch 2.14, CPU build, pinned by `uv.lock`

Install the environment:

```sh
uv sync
```

PyTorch prints a NumPy warning on import. Day 1 never converts a tensor to a NumPy array, so the
project does not depend on NumPy. Ignore the warning.

## Commands

Build the model without training it. The dry run reports the config hash, the parameter count, and
the logits shape:

```sh
uv run python -m octlm.train --config configs/day1.toml --dry-run
```

```json
{"config_sha256":"12cc0f1de67889e04411e8d5d78e62ae284318e001f7403f3c8068fb613eae5e","logits_shape":[1,128,1024],"parameters":541952,"type":"dry_run"}
```

Train both tokenizers and write them to `artifacts/day1/`. Each line of output is one JSON record
with token counts, bytes per token, and the unknown rate:

```sh
uv run python -m octlm.tokenizer --vocab-sizes 512 1024 2048
```

Train the decoder. The trainer prints one JSON record per eval interval and one generation sample at
the end:

```sh
uv run python -m octlm.train --tokenizer bpe --metrics runs/day1-bpe.jsonl
```

Prove that the model, the loss, and the optimizer compose by memorizing one fixed block:

```sh
uv run python -m octlm.train --tokenizer bpe --overfit --metrics runs/day1-overfit.jsonl
```

Write a checkpoint, then resume from it. A resumed run produces weights identical to an
uninterrupted run on the same machine, PyTorch build, and device:

```sh
uv run python -m octlm.train --checkpoint artifacts/day1/run.pt --stop-after 100
uv run python -m octlm.train --resume artifacts/day1/run.pt
```

Measure the forward pass at several context lengths. Day 1 supports only `--dummy`:

```sh
uv run python -m octlm.bench --dummy --contexts 128 256 512 1024 2048
```

Build the Day 2 corpus. It reads the Python standard library from this machine and downloads six
public-domain books once, into `data/`:

```sh
uv run python -m octlm.corpus
```

Run the Day 2 experiments. Each stage appends JSONL to `runs/`. The first four are quick. `variants`
and `length` are training runs that saturate every core for tens of minutes, so start them when the
machine is free:

```sh
uv run python -m octlm.day2 tiled         # tiled attention against SDPA
uv run python -m octlm.day2 equivalence   # SDPA against the handwritten path
uv run python -m octlm.day2 sdpa          # math against flash, memory and throughput
uv run python -m octlm.day2 cache         # KV cache bytes per head count
uv run python -m octlm.day2 variants      # the architecture grid, about 50 minutes
uv run python -m octlm.day2 length --config configs/day2-long.toml
uv run python -m octlm.day2 report        # summarize the grid
```

Run the Day 3a experiments. `cache` and `report` are arithmetic and summaries, so they run anywhere.
The four experiment stages train and belong on a GPU. Stage 0 comes first and decides whether the
rest of the day is readable:

```sh
# Stage 0: the seed spread at the Day 3 budget, in its own file
uv run python -m octlm.day2 variants --config configs/day3.toml \
    --variants baseline swiglu modern --out runs/day3-floor.jsonl
uv run python -m octlm.day2 report --out runs/day3-floor.jsonl

uv run python -m octlm.day3 cache --config configs/day3-long.toml   # cache arithmetic, instant
uv run python -m octlm.day3 mtp --config configs/day3.toml          # EXP-016
uv run python -m octlm.day3 sparse --config configs/day3-long.toml       # EXP-017
uv run python -m octlm.day3 compressed --config configs/day3-long.toml   # EXP-018
uv run python -m octlm.day3 mla --config configs/day3-long.toml          # EXP-019
uv run python -m octlm.day3 report
```

Run Day 4 on a GPU. `notebooks/octlm-colab.ipynb` runs the same stages with Drive persistence.
`prepare` downloads 2.2 GB and encodes it once. `train` resumes from `--checkpoint` when the file
exists:

```sh
python -m octlm.day4 prepare                       # EXP-065
python -m octlm.day4 train                         # EXP-066 and EXP-067
python -m octlm.day4 samples --temperature 0       # EXP-068, greedy
python -m octlm.day4 samples                       # EXP-068, temperature 0.8, top-k 40
```

Run the checks:

```sh
uv run python -m unittest
uv run ruff check .
uv run ruff format --check .
```

### Useful flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--config` | `configs/day1.toml` | Model, training, and tokenizer settings |
| `--tokenizer` | `bpe` | `bpe` or `character` |
| `--train` | `PLAN.md` | Training text |
| `--validation` | `day-wise.md` | Held-out text |
| `--checkpoint` | none | Where to write the final checkpoint |
| `--resume` | none | Checkpoint to continue from |
| `--metrics` | none | JSONL file that receives every eval record |
| `--stop-after` | none | Stop before `training.steps` |
| `--overfit` | off | Train and validate on one block |
| `--dry-run` | off | Build the model, print the shape, exit |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N`. `auto` takes the GPU when there is one |
| `--out` | per stage | JSONL that `octlm.day2 variants` and `report` write and read |
| `--seeds` | 3 | Seeds per variant in a grid stage |
| `--probe-step` | 64 | Needle depths the Day 3 copy probe sweeps |

## Running on a GPU

The development machine has no GPU, so every measurement in `notes/` is a CPU measurement.
`--device` on `octlm.train`, `octlm.day2`, and `octlm.bench` moves the model and its batches to
CUDA. It defaults to `auto`, which takes the GPU when the machine has one. Each training record
carries the device it ran on.

`notebooks/octlm-colab.ipynb` runs the training stages on a Colab GPU. Open it in Colab, pick a GPU
runtime, and run the cells in order. It clones this repository and uses Colab's preinstalled
PyTorch, because `uv.lock` pins the CPU build. That means the Python and PyTorch versions differ
from the local environment, so Colab timings cannot be compared against the CPU timings in `notes/`.

Two stages are worth a GPU: `variants` and `length`. The rest of Day 2 measures CPU behavior.
`sdpa` in particular reports process resident memory, which does not describe GPU allocation, so it
stays on CPU until Phase 5 gives `bench.py` a device-aware memory field.

`octlm.corpus` reads the Python standard library of the machine it runs on, so the corpus belongs to
the machine that trains. That is Colab now, and the fingerprints in `notes/day2.md` are the Colab
ones. Check `data/manifest.json` against them before comparing a new run, because Colab upgrades its
Python image without warning and a different image is a different corpus.

## What is in the repository

```text
octlm/
  config.py      TOML settings, validation at the boundary, SHA-256 fingerprint
  tokenizer.py   character and byte-level BPE tokenizers, plus their CLI
  model.py       Generation 0 decoder: learned positions, pre-LayerNorm, MHA, GELU, tied head
  train.py       training loop, evaluation, atomic checkpoints, resume, CLI
  bench.py       forward-pass measurement and the frozen `octlm-bench-v1` record schema
  corpus.py      code and prose corpus builder, manifest with hashes and licenses
  day2.py        Day 2 experiment stages and the tiled attention sketch
  day3.py        Day 3a experiment stages, the copy probe, and the cache arithmetic
  day4.py        TinyStories download, token files, the Day 4 training run, and samples
notebooks/octlm-colab.ipynb     GPU runs on Colab
configs/day1.toml, configs/day2.toml, configs/day2-long.toml
configs/day3.toml, configs/day3-long.toml, configs/day4.toml
tests/test_day1.py, tests/test_day2.py, tests/test_day3.py
notes/day1.md to notes/day4.md    research, decisions, measurements, failures
```

`model.py` carries both generations on one code path. Every Day 2 switch defaults to the Day 1
behavior: `position` picks learned, sinusoidal, or rotary embeddings, `norm` picks LayerNorm or
RMSNorm, `feed_forward` picks GELU or SwiGLU, `attention` picks the handwritten path or SDPA,
`residual` picks pre-norm or post-norm, and `kv_heads` sets the grouped-query head count.

Day 3a adds six more, all defaulting to the earlier behavior. `mtp_depth` adds multi-token
prediction heads, `attention_window` and `attention_stride` narrow the causal mask,
`kv_compress_block` mean-pools the keys and values outside that window, and `mla_rank` with
`mla_rope_dim` switch attention to a low-rank latent cache with a decoupled RoPE key.

Both tokenizers preserve input bytes exactly. Neither lowercases text nor normalizes Unicode. BPE
learns merges from the training corpus only, breaks equal-frequency ties by token ID, and never
merges across a pre-token boundary. Ordinary text containing `<|tool_call|>` encodes as literal
characters, not as the special token.

Every checkpoint stores the config hash, the tokenizer hash, the dataset hash, the optimizer state,
and three random-generator states. A resume with a mismatched hash fails instead of training on the
wrong assumption.

## What Day 1 measured

The training text is `PLAN.md`. The held-out text is `day-wise.md`. Both are small, so treat these
as smoke measurements, not performance claims.

| Run | Tokens on held-out prose | Validation perplexity | Bits per byte |
| --- | ---: | ---: | ---: |
| Character, 86 entries, 40 steps | 13,939 | 29.998 | 4.9068 |
| BPE, 1,024 entries, 40 steps | 7,319 | 240.391 | 4.1497 |
| BPE, 1,024 entries, 200 steps, one block | 128 | 2.4830 | 0.6338 |

Raw perplexity cannot compare two tokenizers, because each predicts a different unit. Bits per UTF-8
byte can. BPE improved bits per byte by 15.4 percent over the character baseline on the same step
budget.

The character tokenizer mapped 77.8 percent of an unseen Unicode sample to UNK. The BPE tokenizer
represented every byte of the same sample with no unknown token. A 2,048-entry BPE request stopped
early at 1,708 entries, because no remaining pair occurred twice.

A batch-one forward pass took 4.0 ms at 128 tokens and 115.2 ms at 2,048 tokens. Peak resident
memory reached 504 MB. The largest verified Day 1 context is 2,048 tokens. The same forward pass takes 9.1 ms
at 2,048 tokens on a Colab T4.

## What Day 1 does not have

These are deferred on purpose, not missing by accident:

- No KV cache, no `torch.compile`, no mixed precision, no quantization. Day 2 added SDPA; Day 1's
  handwritten attention stays as the reference the SDPA path is checked against.
- No sampling. Generation is greedy.
- No GPU path. `nvidia-smi` found no driver on the development machine.
- `bench.py` emits the full `octlm-bench-v1` record but fills only the fields Day 1 can measure.
  Decode throughput, time to first token, and perplexity stay null until Phase 5 fills them.
- BPE training recounts every pair after every merge. The cost grows with both the corpus and the
  merge count. It will be replaced when profiling on a real corpus shows it blocks work.

The repository layout in `PLAN.md` lists directories for every phase. Those directories get created
when their experiment starts. Empty scaffolding is not allowed.

## What Day 2 measured

`tiled_attention` reproduces `scaled_dot_product_attention` to 4.8e-7 at every block size, and the
full model through SDPA matches the handwritten path to about 1e-6 at 8, 4, 2, and 1 KV heads. That
equivalence is what licenses the SDPA path in later runs.

Attention benchmark at batch 1, 8 heads, head width 32, one forward pass per row. Each measurement
runs in its own process, because peak resident memory is a process high-water mark:

| Length | dtype | Backend | Seconds | Tokens/s | RSS growth |
| ---: | --- | --- | ---: | ---: | ---: |
| 1024 | float32 | math | 0.0220 | 46,491 | 80 MB |
| 1024 | float32 | flash | 0.0038 | 268,554 | 9 MB |
| 4096 | float32 | math | 0.3177 | 12,891 | 1.29 GB |
| 4096 | float32 | flash | 0.0428 | 95,673 | 33 MB |
| 8192 | float32 | math | 1.2644 | 6,479 | 5.17 GB |
| 8192 | float32 | flash | 0.1200 | 68,250 | 62 MB |

At 8192 tokens the flash kernel is 10.5x faster on 84x less memory growth. The math backend
allocates the full score matrix, so its footprint grows quadratically. Bfloat16 is slower than
float32 on this CPU at every length, which is the opposite of the GPU case.

The corpus is 6.9 MB of training text and 0.9 MB held out, 60 percent Python standard library and 40
percent public-domain books, built on Colab from Python 3.13.15. A 2,048-entry BPE tokenizer trained
on a tenth of it reaches 0.431 tokens per byte, against 0.525 for the Day 1 tokenizer on the Day 1
corpus.

The variant grid ran 8 architectures across 3 seeds, 400 steps each, on one T4 in 4 minutes. Bits per
byte on held-out code, averaged over seeds, with the seed spread beside it:

| Variant | Code bpb | Seed spread | Cache at 4K | Decision |
| --- | ---: | ---: | ---: | --- |
| baseline | 2.9339 | 0.0434 | 16 MiB | control |
| swiglu | 2.8521 | 0.0085 | 16 MiB | keep |
| rmsnorm | 2.9352 | 0.0390 | 16 MiB | keep only inside `modern` |
| gqa-2 | 2.9449 | 0.0235 | 4 MiB | keep, the cache is the win |
| post-norm | 3.9310 | 0.0047 | 16 MiB | revert |
| modern | 2.8045 | 0.1194 | 4 MiB | best measured, not separated from `swiglu` |

Read the spread before the gap. The baseline moves 0.043 bits per byte across three seeds with
nothing else changed, so SwiGLU's 0.082 is the only single-component result outside the noise.

On the length sweep, RoPE beats sinusoidal positions at every evaluation length, by 0.76 bits per
byte at 2,048. Extrapolation past the 512-token training length degrades rather than collapses, 2.29
at 2,048 against 2.90 at 8,192. Position interpolation pays only past 4x the trained length.

## What Day 3a built

Nothing was measured. The code is built and the checks pass, but no training stage ran before the
plan changed.

Day 2 closed by demanding a bigger step budget: its baseline moved 0.043 bits per byte across three
seeds with nothing else changed, which swallowed every single-component result except SwiGLU's.
`configs/day3.toml` is `configs/day2.toml` with the budget raised from 400 steps to 2000 and nothing
else touched, so the Day 2 rows stay comparable. Stage 0 measures the new spread and freezes it as
the minimum effect the day is allowed to claim.

The cache arithmetic already runs, and it sets up EXP-019. At `d_model` 256 with 8 query heads and
2 KV heads, the present cache holds 128 dimensions per token per layer. MLA holds `rank + rope_dim`,
so it only undercuts GQA-2 below rank 112, against the 10x reductions the papers report for wide
models. Block compression is the larger win here: at 4,096 tokens with a 128-token exact window,
block 8 caches 624 positions instead of 4,096.

Both hypotheses are written into `notes/day3.md` before the runs, along with the predicted negative
for multi-token prediction at this size.

## What is coming

| Day | Experiments | Work |
| --- | --- | --- |
| 4 | EXP-065 to EXP-068 | TinyStories tokenizer, float16 on a T4, the 20M main run, sampling |
| 5 | EXP-069 to EXP-072 | Seed spread at the new scale, multi-token prediction, KV cache, int8 |
| 6 | EXP-073 to EXP-076 | Qwen weights in `model.py`, logit and tokenizer parity, cached generation |
| 7 | EXP-077 to EXP-079 | Tool-call parser, the harness loop, the eval set and the stock-model baseline |
| 8 | EXP-080 to EXP-083 | LoRA from scratch, harness traces, SFT, merge and int8 |
| 9 | optional | GRPO on the task checks, or routing failed tasks to an external API |

## Working on this

Read `PLAN.md` and every completed note in `notes/` before you change code. Take the
lowest unfinished experiment. Write the note first with a hypothesis and a baseline, then implement,
then fill in the measurements, then record keep or revert. Do not start a day until the previous
day's exit check passes.

`AGENTS.md` holds the full rules for research, note keeping, code size, tokenizer behavior, and
phase boundaries.
