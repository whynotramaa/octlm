# octlm

A small language-model engineering lab. The goal is to rebuild the LLM engineering stack from
scratch at 20M to 150M parameters, measure every change against a baseline, and end with a local
coding and writing assistant. `PLAN.md` defines the product and its 64 numbered experiments.
`day-wise.md` defines the reading order. `AGENTS.md` defines the working rules.

The small model is the lab, not the product. Nothing here chases frontier capability.

## Status

Day 1 is complete: EXP-001 through EXP-008, covering Phase 0 and Phase 1 of nine phases. What runs
today is a character tokenizer, a byte-level BPE tokenizer, a decoder, a training loop with exact
resume, and a benchmark schema. Everything is CPU-only.

Day 2 is partly complete. The modern decoder components are implemented and tested, the code and
prose corpus is built, and EXP-014 is measured. EXP-009 to EXP-013 and EXP-015 need their training
runs, which take about an hour on one CPU machine. `notes/day2.md` lists those commands with their
measured cost.

Read `notes/day1.md` and `notes/day2.md` for the research sources, the measurements, and every
keep-or-revert decision.

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

`octlm.corpus` reads the Python standard library of the machine it runs on. A corpus built on Colab
is not the corpus in `data/manifest.json`, so copy `data/` and `artifacts/day2/` across if the
numbers need to line up with an earlier run.

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
notebooks/octlm-colab.ipynb     GPU runs on Colab
configs/day1.toml, configs/day2.toml, configs/day2-long.toml
tests/test_day1.py, tests/test_day2.py
notes/day1.md, notes/day2.md    research, decisions, measurements, failures, exit checks
```

`model.py` carries both generations on one code path. Every Day 2 switch defaults to the Day 1
behavior: `position` picks learned, sinusoidal, or rotary embeddings, `norm` picks LayerNorm or
RMSNorm, `feed_forward` picks GELU or SwiGLU, `attention` picks the handwritten path or SDPA,
`residual` picks pre-norm or post-norm, and `kv_heads` sets the grouped-query head count.

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
memory reached 504 MB. The largest verified Day 1 context is 2,048 tokens.

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

The corpus is 7.0 MB of training text and 0.9 MB held out, 60 percent Python standard library and 40
percent public-domain books. A 2,048-entry BPE tokenizer trained on a tenth of it reaches 0.407
tokens per byte, against 0.525 for the Day 1 tokenizer on the Day 1 corpus.

## What is coming

Each phase changes one component at a time and keeps the previous path behind a config flag.

| Phase | Experiments | Work |
| --- | --- | --- |
| 2. Modern decoder | EXP-009 to EXP-016 | RoPE, RMSNorm, SwiGLU, MQA, GQA, SDPA, residual variants, multi-token prediction |
| 3. Architecture lab | EXP-017 to EXP-025 | Sparse and compressed attention, MLA, MoE with routing collapse and its fix, Muon, mHC |
| 4. Training at scale | EXP-026 to EXP-034 | Mixed precision, gradient accumulation, scheduler sweeps, scaling check, DDP, FSDP, tensor and pipeline parallelism |
| 5. Inference | EXP-035 to EXP-042 | KV cache, prefill and decode split, `torch.compile`, INT8 and INT4, continuous batching, prefix cache, speculative decoding |
| 6. Post training | EXP-043 to EXP-048 | SFT, LoRA, DPO, reward model, GRPO, RLVR |
| 7. Evals | EXP-049 to EXP-053 | Perplexity, coding score, writing preference, regression gates, long-context needle recall |
| 8. Assistant | EXP-054 to EXP-061 | Task classifier, hybrid retrieval, tool loop, cache stack, local-versus-API router |
| 9. Hardening | EXP-062 to EXP-064 | Model card, release, SSM and multimodal survey |

The next milestone is M2: one `modern-50M` config that beats the Generation 0 baseline on both
perplexity and decode speed at equal parameter count.

## Working on this

Read `PLAN.md`, `day-wise.md`, and every completed note in `notes/` before you change code. Take the
lowest unfinished experiment. Write the note first with a hypothesis and a baseline, then implement,
then fill in the measurements, then record keep or revert. Do not start a day until the previous
day's exit check passes.

`AGENTS.md` holds the full rules for research, note keeping, code size, tokenizer behavior, and
phase boundaries.
