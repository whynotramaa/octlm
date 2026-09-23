# octlm plan

## Goal

Build a small decoder from scratch to learn how a language model works. Train it properly once. Then
run a real open model, Qwen, through the same code, fine-tune it for tool use, and build an agent
harness around it.

A harness here means the program around the model. It shows the model a task and a set of tools,
parses each tool call the model emits, runs the tool, feeds the result back, and stops when the
model gives a final answer or hits a step limit.

The project exists to learn ML engineering in depth and to defend that depth in placement
interviews. Every claim in it needs a measurement in `notes/`.

## Why the plan changed on 2026-09-23

The first plan, in git history up to `ab0d49c`, aimed to pretrain a 50M to 150M model and grow it
into a local coding and writing assistant with 64 experiments. Two facts ended it. `notes/day3.md`
records the full reasoning.

1. A model small enough to pretrain on free Colab cannot follow instructions or emit reliable tool
   calls. The harness needs a model that can.
2. The Day 2 grid trained 3.3M-parameter models on about 0.8M tokens each. The compute-optimal
   budget is about 20 tokens per parameter, so those models saw about 1 percent of it. Seed noise
   swallowed most Day 2 results for that reason.

The new plan keeps the from-scratch work, trains it at a budget where results are readable, and
then moves the same code to Qwen.

## What counts as done

1. A decoder of about 20M parameters, trained from scratch in octlm on TinyStories, writes coherent
   short stories. Its note shows samples, the loss curve, and bits per byte.
2. At least two architecture comparisons at that scale, each with a gap larger than the measured
   seed spread, or recorded as unresolved with its numbers.
3. KV-cached generation and int8 quantization written in octlm, measured against naive generation.
4. `octlm/model.py` loads a small Qwen instruct checkpoint and matches reference logits.
5. A LoRA implementation written in octlm, used to fine-tune Qwen for tool calls.
6. A harness that runs the model through octlm, executes its tool calls inside one sandbox
   directory, and attempts every task in a fixed eval set.
7. Eval numbers for the stock model and the fine-tuned model on the same tasks: valid tool-call
   rate, task success rate, mean steps, and tokens per second.

Each item supports the next. The from-scratch model proves the internals. The Qwen parity check
proves the internals are the same ones a real model uses. The harness and its eval prove the
fine-tune did something measurable.

## Hardware

- The laptop runs code edits, tests, dry runs, and short CPU checks. It does not train, because it
  overheats under long all-core load.
- Colab Pro, since 2026-09-23, runs training, the Qwen parity check, and eval runs on a T4. The
  account also offers a TPU v5e-1, which octlm does not use. `notes/day4.md` explains why. The T4
  has no bfloat16, so training uses float16 with `torch.amp.GradScaler`.
- Kaggle notebooks run anything longer than a Colab session survives. The free quota and background
  execution are leads to verify in EXP-066.
- A rented GPU is for a run that blocks a decision, with the blocking evidence in the
  note.

The estimate below uses 6 x parameters x tokens FLOPs and an assumed 15 TFLOPs of useful float16
throughput on a T4. EXP-066 replaces the assumption with a measured rate.

| Model | Tokens | Estimated T4 time |
| ---: | ---: | ---: |
| 20M | 400M | 1 to 3 hours |
| 50M | 1B | 5 to 6 hours |
| 150M | 3B | about 50 hours |

The 150M row is why pretraining stops near 20M.

## Roadmap

Days 1 and 2 are complete. Day 3a built four variants and stopped before any GPU run. New
experiment numbers start at EXP-065. EXP-020 through EXP-064 belonged to the first plan and stay
retired. `notes/day3.md` maps each one to its new number or to the reason it was dropped.

Each day keeps the method from `AGENTS.md`. Write the problem, hypothesis, baseline, measurement,
and stop condition before the run. Record the result, the failures, and the keep or revert decision
after it.

### Day 4. Train the from-scratch model properly

- EXP-065. TinyStories corpus and tokenizer. Download the dataset, split train and validation, and
  train octlm's BPE on a sample. Measure encode throughput first. If the pure-Python encoder cannot
  tokenize about 500M tokens in one session, add a cache from pre-token to IDs. Use an 8,192
  vocabulary. `notes/day4.md` records why the 4,096 comparison moved to Day 5.
- EXP-066. Mixed precision. Add autocast and `GradScaler` on CUDA. Measure tokens per second and
  check that a short mixed-precision run tracks the float32 loss.
- EXP-067. The main run. About 20M parameters, the Day 2 `modern` stack, 512-token context, about
  400M tokens. Checkpoint to Drive and resume across sessions.
- EXP-068. Sampling. Add temperature and top-k to generation, because greedy decoding repeats
  itself on story data.

Exit check. Validation loss has flattened. Ten fixed prompts produce stories a reader judges
coherent, and the note stores the samples. Bits per byte on held-out stories is recorded.

### Day 5. Compare at the new scale and make generation fast

- EXP-069. Seed spread. Three seeds each of `baseline` and `modern` at a shorter budget, about 100M
  tokens. The spread is the smallest effect the day may claim. This replaces Day 3 Stage 0.
- EXP-070. Multi-token prediction at depth 2 against the same control, using the Day 3a code. The
  hypothesis stays negative, as `notes/day3.md` recorded from Gloeckle et al.
- EXP-071. Naive generation against a KV cache. Tokens per second and cache bytes at 64, 256, and
  512 generated tokens.
- EXP-072. Int8 weight-only quantization. Model bytes, tokens per second, and bits-per-byte delta.

Exit check. Both comparisons carry a decision against the measured spread. The KV cache produces
the same tokens as naive generation at temperature 0.

### Day 6. Run Qwen through octlm

Pick the smallest current Qwen instruct model at the start of the day. Qwen3-0.6B is the fallback.
Its details are leads until the checkpoint's `config.json` confirms them.

- EXP-073. Weight loading. Read `safetensors` with the standard library, which is a JSON header
  followed by raw tensor bytes. Map the names into `model.py`. Add only the switches `config.json`
  requires, such as QK-norm, projection bias, and the RoPE base.
- EXP-074. Logit parity. Run Hugging Face `transformers` once on Colab as the reference, save the
  logits for a fixed prompt set, and compare. `transformers` stays out of `pyproject.toml`.
- EXP-075. Tokenizer parity. Either parse Qwen's `tokenizer.json` with octlm's BPE or add the
  `tokenizers` package. Decide by exact ID match on the Day 1 tokenizer test strings.
- EXP-076. KV-cached generation on Qwen. Tokens per second on the T4 and on the laptop CPU, against
  the reference.

Exit check. The largest float32 logit difference sits below a tolerance stated before the run.
Token IDs match exactly. The chat template renders the same text as the reference.

### Day 7. Build the harness and its eval

- EXP-077. Tool-call format. Use the model's native chat-template format for tool calls. Write the
  parser. A malformed call returns an error message to the model instead of crashing the loop.
- EXP-078. Tool loop. Tools are `read_file`, `list_files`, `grep`, `write_file`, and `run_tests`.
  Every path resolves inside one sandbox directory. `run_tests` has a timeout. Keep the KV cache
  across turns, so the system prompt runs once per task.
- EXP-079. Eval set and baseline. Write 30 to 50 tasks in a small fixture repository. Each task has
  a check that passes or fails without judgment, such as a test passing or a file containing a
  string. Run the stock instruct model at temperature 0. Record valid-call rate, task success, mean
  steps, and tokens per second.

Exit check. One command runs the whole eval and writes JSONL. The stock model's numbers are in the
note.

### Day 8. LoRA and tool-use fine-tuning

- EXP-080. LoRA from scratch. Low-rank adapters on the attention and feed-forward projections. The
  B matrix starts at zero, so the adapted model reproduces the base logits at step 0. Measure
  trainable parameters and peak T4 memory.
- EXP-081. Training traces. Collect successful harness runs on tasks disjoint from the eval set.
  The note decides the source, a stronger model through an API or filtered runs of the base model,
  and records the license of any external model output.
- EXP-082. SFT with LoRA. Loss on assistant tokens only. Rerun the EXP-079 eval with the same tasks
  and settings.
- EXP-083. Merge and quantize. Merge the adapters, apply the EXP-072 int8 path, and rerun the eval.

Exit check. The fine-tuned model beats the stock model on task success by more than the spread
across two training seeds, or the note records why it did not.

### Day 9. One optional extension

Pick at most one.

- GRPO on the train-split tasks, with the task checks as the reward.
- Route tasks the local model fails to an external API, and log cost and success against local-only
  handling.

## What was dropped

- MoE, MLA, sparse and compressed attention, Muon, mHC, and the DeepSeek-style runs. None of them
  serve the harness. The Day 3a code for MLA and the two attention variants stays behind flags that
  default off.
- DDP, FSDP, tensor parallelism, and pipeline parallelism. The free tiers give one GPU.
- Pretraining at 50M to 150M, and the scaling check. The compute table above rules them out.
- DPO and the reward model. SFT on harness traces is the post-training that the eval can measure.
- Hybrid retrieval, the writing assistant, the cache stack, continuous batching, and speculative
  decoding. The harness uses `grep` for retrieval and serves one request at a time.

## Risks

- The pure-Python BPE encoder may be too slow for TinyStories. EXP-065 measures it before anything
  depends on it.
- Colab disconnects mid-run. Checkpoints go to Drive on a timer, and exact resume already exists.
- Qwen may differ from `model.py` in more places than expected. EXP-074 gates everything after it.
- A 0.6B model may fail most harness tasks even after SFT. Keep the tasks small. The fallback is
  the next Qwen size with LoRA, if EXP-080 shows it fits in T4 memory.
- Eval tasks can leak into training traces. The eval set and the trace set use separate fixture
  repositories.

## Layout

New code goes in `octlm/` as one module per day, following `day2.py` and `day3.py`. The harness
lives in `octlm/harness.py` until it outgrows one file. Create a directory only when its first file
exists.
