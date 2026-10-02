# Day 8: LoRA and tool-use fine-tuning

Status: complete. Started and finished 2026-10-02.
Experiments: EXP-080 through EXP-083.

## Where we start

Day 7 is complete. Its exit check passed. Stock Qwen3-0.6B in non-thinking mode scores pass^1
0.025 with a seed stdev of 0.025 on the 40 v2 tasks, and pass^3 0. In 110 of 120 runs it answered
on the first turn without a tool call and invented the value. All 16 invalid calls were
`write_file` calls, half with raw newlines inside a JSON string and half cut off at the 512-token
limit. Day 8 has two concrete targets: call a tool before answering, and emit valid JSON for file
content.

## Session on 2026-10-02: research before the code

### Sources

- [LoRA, Hu et al. 2021](https://arxiv.org/abs/2106.09685), section 4.1, computes
  h = W0 x + B A x with A initialized from a Gaussian and B at zero, so B A is zero at the start of
  training. It scales the update by alpha / r and merges W = W0 + B A for inference with no added
  latency. The paper's experiments adapt only the query and value projections. Decision: write
  `LoRALinear` with that formula, B at zero, and a `merge` that folds the update into the base
  weight. The step-0 check compares adapted and base logits and requires exact equality.
- [QLoRA, Dettmers et al. 2023](https://arxiv.org/abs/2305.14314), section 4 and appendix A, found
  that query and value adapters alone could not match full fine-tuning on LLaMA 7B, that "LoRA on
  all linear transformer block layers are required", and that rank r "is unrelated to final
  performance if LoRA is used on all layers". Appendix B.3 found that training on the target only
  beat training on source and target on all four datasets it tried. Its 7B settings are a 2e-4
  learning rate, a constant schedule, Adam beta2 0.999, and max grad norm 0.3. Decision: adapt all
  seven projections in every block (query, key, value, output, gate, up, down), mask the loss to
  assistant tokens, and start from its 7B settings.
- [LoRA Without Regret, Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) reports
  that "attention-only LoRA significantly underperforms MLP-only LoRA", that the best LoRA learning
  rate is about ten times the full fine-tuning rate, and that LoRA loses more than full
  fine-tuning as the batch grows. Decision: the feed-forward adapters stay. Use 8 examples per
  optimizer step, a small batch.
- [PEFT LoRA layer source](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)
  initializes A with `kaiming_uniform_(a=sqrt(5))`, the `nn.Linear` default, and B with zeros. It
  computes the merged delta as `B @ A * scaling` in float32 when the weights are low precision.
  Decision: use the same initialization, seeded from the training seed, and merge in float32
  before casting back to float16.
- [Pinned Qwen3 chat template](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)
  renders the empty `<think>\n\n</think>\n\n` block only on the generation prompt and on the final
  assistant turn. An earlier assistant turn renders without it. A conversation rendered once
  therefore shows every assistant turn but the last in a form the harness never produces.
  Decision: one SFT example per assistant turn. Its prompt is the conversation up to that turn,
  rendered with the generation prompt exactly as the harness renders it. Its target is the reply
  tokens and `<|im_end|>`. The code encodes prompt plus target as one string and checks that the
  prompt's token IDs are a prefix of it, so the boundary cannot shift a merge.
- [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms), section D.4,
  forbid using the services "to build a competing product or service, including to train competing
  AI models". The [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) lists Apache 2.0.
  Filtered runs of the base model cannot work here, because the base model solves 1 task in 40.

### Trace source for EXP-081

PLAN.md offers two sources: a stronger model through an API, or filtered runs of the base model.
Neither fits. The base model's 2.5 percent success would leave about two traces from 80 tasks, and
training a model on an API model's outputs is restricted by the terms above. A larger open Qwen
would need another architecture check and GPU hours for generation.

Decision: a third source, a scripted expert. Each training task carries a reference solution, the
same as the Day 7 tasks. The trace stage replays that solution through the real harness tools in
a fresh sandbox and records the result as a harness message list: the system prompt, the task,
one assistant turn per tool call, each tool result, and the final answer. Every trace must pass
its task check before it is kept. This is behavior cloning from a hand-written policy. It costs
no GPU time, has no license question, and the trace file is byte-for-byte reproducible.

Two rules make the scripted traces teach the right thing:

1. The expert reads before it writes. Day 7's reference solutions write a file without reading
   it, because they were written to validate the checks. A model cannot know a file it has not
   read, so every training edit is preceded by `read_file`.
2. Each assistant turn holds one call in the exact form the template produces for tool calls:
   `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`, serialized with `json.dumps`,
   which escapes newlines. This is the second Day 7 failure.

### Leakage

PLAN.md's risk list requires a separate fixture repository for traces. `fixtures/day8/repo` is a
different package, `library`, with its own config, data, modules, and tests. The training tasks
reuse the Day 7 task families (config lookups, file edits, new functions, planted bugs), because
the point is to teach the tool protocol, not new skills. No training task shares a repository,
file, function name, prompt, or answer with an eval task. The trace stage rejects any training
prompt that equals an eval prompt. The note reports the family overlap as a caveat: the eval
measures transfer to a new repository, not to new task kinds.

### EXP-080: LoRA from scratch

Problem. Full fine-tuning of 596M parameters needs float32 master weights, gradients, and two Adam
moments, about 9.5 GB before activations, on a 15 GB T4.
Hypothesis. Rank-16 adapters on all seven projections train about 10M parameters, under 2 percent
of the model. The adapted model reproduces the base logits exactly at step 0, and one training
step on the longest example fits in T4 memory with room left.
Baseline. The base model's logits on the same tokens.
Measurement. Trainable and total parameters, maximum absolute logit difference at step 0, and
peak CUDA memory during training.
Stop condition. Any nonzero step-0 difference stops the run, because B starts at zero.

### EXP-081: training traces

Problem. SFT needs successful tool-use transcripts on tasks disjoint from the eval.
Hypothesis. A scripted expert over a separate repository yields one passing trace per task.
Baseline. None. This builds data.
Measurement. Task count by kind, trace count, assistant-turn examples, target tokens, and the
longest example in tokens.
Stop condition. A trace that fails its own check, or a prompt that equals an eval prompt, stops
the stage.

### EXP-082: SFT with LoRA

Problem. The stock model answers without tools and writes invalid JSON.
Hypothesis. SFT on the scripted traces raises pass^1 on the Day 7 v2 eval from 0.025 to well
above 0.025 plus the spread across two training seeds, and raises the valid-call rate above
0.333.
Baseline. Day 7 run 2: pass^1 0.025, seed stdev 0.025, pass^3 0, valid-call rate 0.333.
Measurement. Two training seeds, 0 and 1. Each adapter runs the unchanged Day 7 eval: the same
40 v2 tasks, sampling seeds 0, 1, and 2, the same sampling settings, turn, token, and context
limits. The eval reports pass^1, pass^3, valid-call rate, success by kind, mean turns, tokens per
second, and pass^1 counting only runs that made at least one valid call. Training logs the loss
on every optimizer step.
Settings, frozen before the run: rank 16, alpha 32, all seven projections, learning rate 2e-4
constant, AdamW with betas (0.9, 0.999) and no weight decay, 8 examples per optimizer step, 3
epochs, max grad norm 0.3, float16 autocast with `GradScaler`, no LoRA dropout.
Stop condition. Exit check: the mean pass^1 of the two adapters minus 0.025 must exceed the
absolute difference between the two adapters' pass^1. Otherwise record why it did not.

### EXP-083: merge and quantize

Problem. Unmerged adapters add two small matmuls per projection at inference.
Hypothesis. Merging gives the same eval within sampling noise, and the EXP-072 int8 path halves
the stored projection bytes at some cost in speed and success.
Baseline. The unmerged seed-0 adapter's EXP-082 eval.
Measurement. Maximum logit difference between merged and unmerged models on one eval prompt,
stored bytes of the block projections in float16 and int8, and the full eval on the merged int8
model.
Stop condition. A merged-against-unmerged logit difference above the float16 rounding scale
(0.05) means the merge is wrong. Stop and fix it before the int8 eval.

### Implementation

- `octlm/day8.py` holds `LoRALinear`, `add_lora`, `merge_lora`, the trace builder, the SFT loop,
  and the adapted eval. Stages: `traces`, `train`, `eval`.
- `LoRALinear` keeps the frozen base layer and adds `x A^T B^T * alpha / r`. A and B stay in
  float32 while the base weights are float16. Under autocast the adapter matmuls run in float16
  and the gradients land in the float32 parameters. The adapter output is cast back to the input
  dtype before the add, so a zero B adds an exact zero.
- `octlm/harness.py` changed in two ways that leave the eval's behavior the same. The fixture
  directory is now the directory of the tasks file, so `validate` runs on either fixture set.
  `evaluate` is split into model loading and `run_eval`, which takes any loaded model, so Day 8
  passes an adapted or int8 model through the unchanged task loop. The summary gains
  `pass_1_with_valid_call`, which Day 7 asked for. The harness source hash differs from the Day 7
  run for that reason.
- `fixtures/day8/repo` is the `library` package: config, two CSV files, four modules, and 14
  passing tests. `fixtures/day8/tasks.json` holds 74 tasks: 35 `answer`, 15 `file`, 10 `hidden`,
  and 14 `tests`. A throwaway generator built the JSON from short edit specs against the real
  files, so every `write_file` content is the actual file with one change. The JSON is the record.
- The trace expert reads each existing file before writing it. Bug fixes run the tests, read the
  file, write the fix, and run the tests again. New functions read, write, and run the tests.
- Adapters save as a separate file with the base manifest, the tokenizer hash, and the task-file
  hash. `eval` refuses an adapter whose manifest or tokenizer hash differs from the loaded model.

### EXP-081 result: training traces

`python -m octlm.harness validate --tasks fixtures/day8/tasks.json` passed all 74 tasks on the
laptop: each fails on the untouched repository and passes after its scripted solution. The record
is `runs/day8-validate.jsonl`. No training prompt equals an eval prompt.

`python -m octlm.day8 traces` wrote `runs/day8-traces.jsonl`:

| Measure | Value |
| --- | ---: |
| Traces | 74 |
| SFT examples (assistant turns) | 223 |
| Tokens per epoch | 159,856 |
| Target tokens per epoch | 7,287 |
| Longest example | 1,174 tokens |

Targets are 4.6 percent of the tokens. The rest is the system prompt, tool list, task, and tool
results, which the loss mask skips. Each prompt repeats the 516-token tool preamble, which is why
223 short turns cost 160k tokens.

EXP-081 decision: keep. Every trace passes its check by construction, and the trace file is
deterministic.

### Kaggle run

Version 9 of the private `whynotramaa/octlm-code` dataset adds `octlm/day8.py`, the changed
`octlm/harness.py`, and `fixtures/day8` to the previous files. The private
`whynotramaa/octlm-day8` notebook version 1 (`notebooks/octlm-day8.ipynb`) was pushed on a T4.
It validates both task sets, builds the traces, trains seeds 0 and 1, evaluates each adapter on
the Day 7 v2 tasks with sampling seeds 0, 1, and 2, then evaluates the merged int8 seed-0 model.

### EXP-080 and EXP-082 results

The run finished training and both float16 evals, then stopped at the EXP-083 gate. Records are in
`runs/kaggle-day8/`, adapters in `artifacts/day8/`.

EXP-080. Both seeds report a step-0 logit difference of exactly 0.0 on the 1,157-token example.
LoRA trains 10,092,544 of 606,142,464 parameters, 1.7 percent. Peak CUDA memory was 7.9 GB of the
T4's 15 GB. Decision: keep.

Training. Each seed ran 84 optimizer steps in about 270 seconds at about 1,770 tokens per second.
Seed 0 loss fell from 1.75 at step 1 to 0.0006 at step 84. Seed 1 fell from 1.24 to 0.016. Losses
this low on 7,287 target tokens mean the adapters memorized the traces.

| Measure | Day 7 base | Seed 0 | Seed 1 |
| --- | ---: | ---: | ---: |
| pass^1 mean | 0.025 | 0.375 | 0.467 |
| pass^1 per sampling seed | | 0.350, 0.375, 0.400 | 0.450, 0.475, 0.475 |
| pass^3 | 0 | 0.350 | 0.450 |
| Valid-call rate | 0.333 | 0.976 | 0.985 |
| `answer` success | | 0.333 | 0.595 |
| `file` success | | 0.815 | 0.704 |
| `hidden` success | | 0.250 | 0.500 |
| `tests` success | | 0.111 | 0.000 |
| Runs stopped by the turn limit | | 26 of 120 | 46 of 120 |
| Generated tokens per second | | 13.9 | 14.1 |

Exit check. The mean pass^1 is 0.421. Minus 0.025 gives 0.396, which exceeds the 0.092 gap between
the two adapters. EXP-082 passes. Decision: keep.

Both Day 7 failure modes are gone. Valid-call rate rose from 0.333 to above 0.97, and
`pass_1_with_valid_call` equals pass^1, so every pass came from a run that used a tool. The
remaining weak spot is `tests` tasks: 1 pass in 27 tries across both seeds. Seed 1 hit the turn
limit on 46 runs, which suggests it loops on read and test calls instead of writing the fix. The
training seed moves pass^1 by 0.09, about four times the sampling-seed spread.

### EXP-083 failure: the merge gate was set without a floor

The merged float16 model differed from the unmerged one by 0.157 in the largest logit, above the
0.05 tolerance, so the stage raised before the int8 eval. Stored block projections measured
880,932,864 bytes in float16 and 441,907,200 in int8, a ratio of 0.50.

I checked the merge on the laptop with one float32 forward pass of seed 0 on the same prompt:

| Comparison | Max logit difference |
| --- | ---: |
| Float32 merged against float32 unmerged | 0.000115 |
| Merged weights rounded to float16, float32 compute | 0.042 |
| T4, float16 weights and activations | 0.157 |

The merge math is right. The largest logit is about 38, where one float16 step is 0.031, so
rounding the weights alone costs more than one step, and float16 activations across 28 blocks add
the rest. I picked 0.05 as "the float16 rounding scale" without measuring the noise floor, and that
broke the rule in `AGENTS.md`.

Change. `merge_and_quantize` now records the max difference and fails only if the merged and
unmerged models pick a different top-1 token at any position of the prompt. The float32
measurement above is the correctness proof. The laptop run gave top-1 agreement of 1.0 with
float16-rounded weights. The int8 eval still has to run on Kaggle.

### EXP-083 rerun

Version 10 of `whynotramaa/octlm-code` carries the new merge gate. The private
`whynotramaa/octlm-day8-int8` notebook (`notebooks/octlm-day8-int8.ipynb`) retrains seed 0 and runs
only the merged int8 eval. Kaggle refused the errored Day 8 run as an input, and a 40 MB adapter
upload from the laptop stalled at 38.7 MB, so retraining on the T4 was faster. Seed 0's adapter hash
in the first run was `b0327504…`. A different hash in the rerun means T4 training is not
bit-reproducible, and the int8 result should be compared against that rerun's adapter, not the
first one.

### EXP-083 result

The `octlm-day8-int8` run finished on a T4. Records are in `runs/kaggle-day8-int8/`, the retrained
adapter in `artifacts/day8/adapter-0-rerun.pt`. The errored first attempt stays in
`runs/kaggle-day8/eval-0-int8.jsonl` as the record of the 0.05 gate.

The rerun adapter hash is `3ee2a82b…`, not `b0327504…`. The loss curves agree within 0.001 up to
step 10 and within a few percent up to step 20, then drift: step 42 is 0.077 in the first run and
0.045 in the rerun, step 84 is 0.00060 and 0.00073. T4 training is not bit-reproducible, so the int8 eval below runs a different
adapter from the float16 baseline.

Merge gate. The merged and unmerged models picked the same top-1 token at all 516 prompt positions.
The max logit difference was 0.176, close to the 0.157 of the first run. Stored block projections
went from 880,932,864 bytes in float16 to 441,907,200 in int8, a ratio of 0.50.

| Measure | Unmerged float16, adapter `b0327504` | Merged int8, adapter `3ee2a82b` |
| --- | ---: | ---: |
| pass^1 mean | 0.375 | 0.375 |
| pass^1 per sampling seed | 0.350, 0.375, 0.400 | 0.375, 0.400, 0.350 |
| pass^3 | 0.350 | 0.325 |
| Valid-call rate | 0.976 | 0.954 |
| `answer` success | 0.333 | 0.381 |
| `file` success | 0.815 | 0.852 |
| `hidden` success | 0.250 | 0.250 |
| `tests` success | 0.111 | 0.000 |
| Runs stopped by the turn limit | 26 of 120 | 55 of 120 |
| Mean turns | 4.62 | 5.43 |
| Generated tokens per second | 13.9 | 17.1 |

Run by run, 35 of the 120 task and seed pairs pass in both evals, 10 pass only in float16, and 10
pass only in int8. The float16 eval solved 16 distinct tasks at least once and the int8 eval 17.

The turn-limit stops doubled, and most of the rise is `answer` tasks: 1 in float16, 22 in int8. The
int8 model kept reading files after it had the value, and still answered correctly as often. This
run cannot say whether the retrained adapter or the int8 weights cause the looping, because both
changed at once.

Speed rose from 13.9 to 17.1 tokens per second. The Day 7 stock model ran at 18.3. Merging removes
the 14 adapter matmuls per block, and int8 dequantization costs less than they did. Day 5 found int8
slower than float16 on the 20M model, so this gain comes from the merge, not from int8. A float16
merged eval would measure the split. It was not run.

Decision: keep. The merge is correct by the float32 check and the top-1 gate, int8 halves the stored
projections, and pass^1 matches the float16 adapter. Record the confound: a fair comparison needs
the float16 eval of adapter `3ee2a82b`, one more 20-minute T4 run.

### Correction: the traces are not byte-for-byte reproducible

The research section says the trace file is byte-for-byte reproducible. It is not. A diff of the
laptop and Kaggle trace files finds three sources in `run_tests` output:

- unittest's timing line reads `Ran 14 tests in 0.000s` on some calls and `0.001s` on others.
- Tracebacks name the sandbox, a random `tempfile` directory such as `/tmp/tmpro9g_irv/repo`, which
  changes on every run, even on one machine.
- The laptop's Python prints `~~~^^^` marker lines under traceback frames, and Kaggle's Python 3.12
  omits most of them.

The laptop trace summary reports 159,856 tokens with a longest example of 1,174 tokens. The Kaggle
summary reports 158,928 and 1,156. The two Kaggle runs disagree with each other by one token on the
longest example, 1,157 and 1,156, which is the random directory name.

The targets do not change. All three runs have 7,287 target tokens, because the loss mask skips tool
results. The training inputs still differ by a few tokens of context per run, which adds to the
non-reproducible adapter above. The fix is to strip the timing line and the sandbox prefix from
`run_tests` output, which would change the Day 7 harness. It is not done, because Day 7's eval must
stay unchanged.

## Exit check

- [x] EXP-080. B starts at zero, the step-0 logit difference is exactly 0.0 in all three runs,
  10,092,544 parameters train, and peak T4 memory is 7.9 GB.
- [x] EXP-081. 74 traces from a separate repository, each passing its own check, no prompt shared
  with the eval. The source is a scripted expert, with the license reasoning above.
- [x] EXP-082. Both adapters beat the stock model. Mean pass^1 minus the 0.025 baseline is 0.396,
  above the 0.092 gap between training seeds.
- [x] EXP-083. Merge and int8 eval ran, with the adapter confound recorded.

Day 8 is complete. Day 9 may start.

## Session on 2026-10-02: publish the Day 8 explanations

The web journal has one post per Day 8 experiment: `lora-from-scratch` (EXP-080),
`scripted-traces` (EXP-081), `sft-with-lora` (EXP-082), and `merge-and-int8` (EXP-083). Every
number comes from the exported runs, the trace file, or the EXP-083 table in this note.
`web/scripts/export.py` exports both training and eval runs, the failed gate record, the int8
rerun, and `web/src/data/day8.json`, which holds each trace's prompt and target token counts per
example. The exporter recomputes those counts with `octlm.day8.examples` and fails unless the totals
match the trace summary. They matched.

`web/src/lib/lora.ts` ports the LoRA parameter count, merge, and forward pass. A Python fixture of a
small adapted decoder checks all three, and the count for Qwen's shapes at rank 16 matches the
10,092,544 trainable parameters logged on the T4. `EvalGrid` now takes any number of named run sets,
so Day 7 and Day 8 share it.

New widgets: a LoRA parameter and memory counter, a trace reader that marks target turns, a loss
chart, a toy merge with float16 rounding, and a float16 step chart against the 0.05 gate. All 21
web parity checks pass and Astro builds 43 pages. The UI was not inspected in a browser.
