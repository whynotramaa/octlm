# Day 9: GRPO on harness tasks

Status: in progress. Started 2026-10-02.
Experiments: EXP-084 through EXP-086.

## Where we start

Day 8 is complete. LoRA SFT on 74 scripted traces raised pass^1 on the 40 Day 7 eval tasks from
0.025 to 0.375 (seed 0) and 0.467 (seed 1). The valid-call rate rose above 0.97. The gap between
the two training seeds was 0.092. The weak spot is `tests` tasks: 1 pass in 54 runs across both
seeds. Seed 1 hit the 8-turn limit on all 27 of its `tests` runs, reading files and running tests
without writing a fix.

PLAN.md offers one optional Day 9 extension. This day takes GRPO, with the task checks as the
reward. Routing to an external API is skipped.

## Session on 2026-10-02: research before the code

### Sources

- [DeepSeekMath, Shao et al. 2024](https://arxiv.org/abs/2402.03300), section 4, defines GRPO. It
  samples a group of outputs per question, drops the value model, and uses the group-normalized
  reward (r - mean(r)) / std(r) as every token's advantage under outcome supervision. It adds a KL
  penalty to a reference model with the estimator ref/pi - log(ref/pi) - 1. Its settings are a
  1e-6 learning rate and KL coefficient 0.04. Decision: group-relative advantages from a binary
  task-check reward, no value model.
- [DAPO, Yu et al. 2025](https://arxiv.org/abs/2503.14476), section 3, removes the KL term because
  "the model distribution can diverge significantly from the initial model, thus this restriction
  is not necessary". Its dynamic sampling drops groups whose accuracy is 0 or 1, because "a zero
  advantage results in zero policy gradients". It averages the loss over tokens, not samples.
  Decision: no reference model and no KL. Drop uniform groups and keep sampling until the batch
  holds only informative groups. Average the loss over every generated token in the batch.
- [Dr. GRPO, Liu et al. 2025](https://arxiv.org/abs/2503.20783) shows that dividing by the response
  length favors short correct answers and long wrong ones, and that dividing by the group standard
  deviation gives a question-level difficulty bias. Decision: advantage = r - mean(r), with no
  standard-deviation division.
- [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) default to beta 0.0,
  epsilon 0.2, 8 generations, the `dapo` loss, and one optimization pass per batch. With one pass,
  the policy that sampled and the policy being updated are the same, so the clipped ratio equals 1
  and the clip never activates. Decision: one pass per batch, so the loss reduces to
  -A * log pi summed over generated tokens. octlm skips the clipping code because it would compute
  a constant.
- [Search-R1, Jin et al. 2025](https://arxiv.org/abs/2503.09516) masks retrieved tokens out of the
  RL loss so the gradient covers only model-generated tokens. Masking raised its 7B average from
  0.343 to 0.431. It used groups of 5, learning rate 1e-6 and KL 0.001. Decision: the loss covers
  only the tokens the model generated in each assistant turn. Tool results, the system prompt and
  the task are context.
- [LoRA Without Regret, Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) reports
  that rank-1 LoRA matches full fine-tuning in policy-gradient RL, because each episode carries
  O(1) bits, and that the best LoRA learning rate is about 10 times the full fine-tuning rate.
  Decision: continue training the rank-16 Day 8 adapter at 1e-5, ten times the 1e-6 used by
  DeepSeekMath, DAPO and Search-R1.

### Conflict: the plan's train split cannot give a signal

PLAN.md says "GRPO on the train-split tasks". The Day 8 SFT adapter memorized those 74 tasks: its
training loss ended at 0.0006 for seed 0 and 0.016 for seed 1. If every sample in a group passes,
every advantage is zero, and DAPO's filter drops the group. GRPO would spend its rollouts on tasks
it cannot learn from.

Decision: measure it, and train on a new pool. The probe stage runs the SFT adapter on 12 Day 8
training tasks. The GRPO pool is a new task file, `fixtures/day8/grpo-tasks.json`, on the same
`library` repository, so it shares `repo/` and `hidden/` with the SFT tasks. Its prompts, planted
bugs and new functions differ from the SFT tasks and from the eval. The pool leans toward `tests`
and `hidden` tasks, the kinds the eval says are weak. It has no `file` tasks, which already pass
above 0.7.

### Design

Each training step:

1. Take the next task from a shuffled pool and sample a group of 6 rollouts through the unchanged
   harness loop, with the same sampling settings as the eval.
2. Score each rollout with the task check: 1 for a pass, 0 otherwise.
3. If all six rewards are equal, log the group and skip it. Otherwise keep it.
4. Repeat until 4 groups are kept. If 16 groups pass without filling the batch, stop the run and
   record that the signal is too sparse.
5. For each kept rollout and each assistant turn, run the policy on the turn's exact prompt and
   reply tokens, take the log-probabilities of the reply tokens at the sampling temperature, and
   add -A times their sum to the loss. Divide by the batch's generated-token count.
6. Clip the gradient norm to 1.0 and take one AdamW step.

The rollout recomputes the prompt for each turn as the harness renders it, so the tokens in the
loss are the tokens the model saw. The same think-block issue that shaped Day 8's examples applies
here.

Known limitation. The sampler uses top-k 20 and top-p 0.8. The loss uses the full softmax at
temperature 0.7, so the behavior policy and the updated policy differ on the truncated tail. TRL
makes the same approximation.

### EXP-084: GRPO task pool and probe

Problem. GRPO needs tasks where the SFT model sometimes passes and sometimes fails.
Hypothesis. On 12 Day 8 training tasks, at least 10 groups of 4 rollouts are uniform, so the plan's
train split gives almost no gradient. On the new pool, at least a third of the tasks give a mixed
group of 4.
Baseline. None. This measures the signal.
Measurement. Per-task successes out of 4 rollouts on both sets, and the share of mixed tasks.
Stop condition. If fewer than a fifth of the new pool's tasks are mixed, the GRPO run cannot fill
its batches. Record it and stop.

### EXP-085: GRPO training

Problem. SFT taught the tool format but not debugging.
Hypothesis. GRPO raises the pool's pass rate measured by the probe before and after training.
Baseline. The probe of the pool before training, on the same retrained SFT adapter.
Measurement. Every step logs groups tried, groups kept, mean reward of all sampled rollouts, loss,
gradient norm, generated tokens and seconds. The pool probe reruns after training with the same
seeds.
Settings, frozen before the run: start from SFT seed 1 retrained on the T4, LoRA rank 16, learning
rate 1e-5 constant, AdamW betas (0.9, 0.999), no weight decay, group size 6, 4 kept groups per
step, at most 16 groups per step, 30 steps, max grad norm 1.0, float16 autocast with `GradScaler`,
no KL, one pass per batch. A wall-clock budget of 5 hours ends training early and still saves the
adapter.
Stop condition. A step that cannot fill its batch in 16 groups ends the run.

### EXP-086: held-out eval

Problem. A higher pool reward can come from memorizing the pool.
Hypothesis. GRPO raises pass^1 on the 40 Day 7 eval tasks above its SFT starting point by more than
0.092, the Day 8 gap between training seeds, and keeps the valid-call rate above 0.9.
Baseline. The retrained SFT seed-1 adapter, evaluated in the same notebook with the same seeds.
Measurement. The unchanged Day 7 eval: 40 tasks, sampling seeds 0, 1 and 2, pass^1, pass^3,
valid-call rate, success by kind, turn-limit stops.
Stop condition. Exit check: the GRPO adapter's pass^1 minus the SFT adapter's pass^1 must exceed
0.092. Otherwise record why it did not.

### Implementation

- `fixtures/day8/grpo-tasks.json` holds the GRPO pool: 33 tasks, 12 `answer`, 8 `hidden` and 13
  `tests`. It sits next to the SFT tasks so it shares `repo/` and `hidden/`. Eight new hidden tests
  went into `fixtures/day8/hidden/`. A throwaway generator built the JSON from short specs against
  the real files. Each bug fix plants a mutation the SFT tasks did not use, such as
  `MAX_DAYS + days_out` or `max(1, days_late)`. `python -m octlm.harness validate --tasks
  fixtures/day8/grpo-tasks.json` passed all 33 on the laptop: each check fails on the untouched
  repository and passes after its reference solution. The record is `runs/day9-validate.jsonl`.
- `octlm/day9.py` has two stages. `probe` runs a fixed number of seeded rollouts per task and
  records successes. `train` runs GRPO and saves the adapter in the Day 8 format, so
  `python -m octlm.day8 eval --adapter` evaluates it unchanged. `load_pool` refuses a pool whose
  prompts equal any eval or SFT prompt.
- `run_task` in `octlm/harness.py` takes an optional `turns` list and appends each turn's prompt
  and reply token IDs. The eval passes nothing, so its behavior and output are unchanged. The
  harness source hash changes.
- `reply_log_probs` runs the blocks with non-reentrant gradient checkpointing and applies
  `lm_head` only to reply positions. A full-vocabulary float32 log-softmax over a 5,600-token
  prompt would take 3.4 GB alone. A CPU check on a small decoder matched `Decoder.forward` within
  5e-7 and reached every parameter with a gradient.
- `notebooks/octlm-day9.ipynb` runs on a Kaggle T4. It retrains SFT seed 1 with the Day 8 code,
  evaluates it, probes 12 Day 8 training tasks and the GRPO pool, trains GRPO, probes the pool
  again, and evaluates the GRPO adapter. The Day 8 adapters could not be uploaded, so the SFT
  start is retrained in the same notebook, and its eval is the baseline.

### Kaggle run

A new version of the private `whynotramaa/octlm-code` dataset, uploaded 2026-10-02 11:03 UTC, adds
`octlm/day9.py`, the changed harness, the GRPO pool and its hidden tests. The private
`whynotramaa/octlm-day9` notebook version 1 was pushed on a T4 on 2026-10-02.

## Session on 2026-10-03: terminal demos

The project had measurements but nothing a visitor could watch. This session adds two live
terminal demos and recordings of them. It changes no experiment and no eval number.

### Decisions

- One module, `octlm/demo.py`, with two subcommands. Plain ANSI escape codes, no UI library. The
  demos are a stream of colored text, and a TUI framework would add a dependency for no new
  information.
- `agent` runs the real eval loop, `harness.run_task`, not a copy. `run_task` and
  `generate_turn` take an optional `watch` callback that receives each sampled token ID and each
  tool result. The eval passes nothing, so its behavior and records are unchanged. The harness
  source hash changes again.
- Token streaming decodes the whole reply so far and prints the new suffix. It holds output back
  while the text ends in U+FFFD, because one character can span two byte-level BPE tokens.
  Transformers' `TextStreamer` also decodes the whole token cache and prints the new suffix.
- The agent demo uses the eval's sampling settings from `generation_config.json` and the Day 8
  `adapter-0.pt`. The default task is `bug-low-stock`, a held-out eval task that adapter 0 passed
  on all three Kaggle seeds. It shows a failing test, a file read, a fix and a passing test.
- `stories` loads the Day 4 checkpoint through `day4.load_model`, which checks the tokenizer and
  config hashes. It samples with the Day 4 settings, temperature 0.8 and top-k 40, through
  `forward_cached`. It colors each token by the model's own probability for it at temperature 1,
  so the confident and the uncertain choices are visible. The reported tokens per second count
  model time only. The `--delay` pause that makes the stream readable is excluded.

### Harness fix: forced colors in test output

The first local run fed ANSI color codes to the model inside `run_tests` results. Python 3.14's
unittest colors its output when `FORCE_COLOR` is set, and this shell sets `FORCE_COLOR=3`. The
[Python 3.14 docs](https://docs.python.org/3/using/cmdline.html#envvar-PYTHON_COLORS) say
`PYTHON_COLORS` takes precedence over `NO_COLOR` and `FORCE_COLOR`. `test_status` now sets
`PYTHON_COLORS=0`, so tool results no longer depend on the host's environment. The Kaggle runs did
not set `FORCE_COLOR`, so the Day 7 and Day 8 numbers are unaffected. The Day 9 Kaggle run uses
the code uploaded on 2026-10-02 and does not include this change.

### Recordings

Recorded with asciinema 2.4.0 on the laptop CPU, float32, seed 0, idle gaps capped at 1.5
seconds. GIFs rendered with agg 1.9.0. Files are in `web/public/demo/`.

| Recording | Result |
| --- | --- |
| `agent-lora` | PASS in 7 turns, 6 of 6 calls valid, 6.5 tokens/s with prefill |
| `agent-base` | FAIL in 1 turn. It answers without a tool call, the Day 7 failure mode |
| `stories` | 3 stories of 200 tokens, 144 to 161 tokens/s model time, sample perplexity 1.27 to 1.78 |

One CPU run is one sample. It illustrates the Day 8 result. It does not measure it. The pass rates
stay those in `notes/day8.md`. An earlier CPU run of the same command, before the color fix, also
passed, in 8 turns.

The stories recording uses seed 1. Seed 0's third story ended with the dog eating the cat, which is
a fine sample from a 26M model but a poor first impression. Both seeds stop at the 200-token cap
mid-sentence, as the Day 4 samples did.
