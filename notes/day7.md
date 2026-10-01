# Day 7: build the harness and its eval

Status: complete. Started 2026-10-01. Completed 2026-10-02.
Experiments: EXP-077 through EXP-079.

## Decisions carried from Day 6

Day 6 left two open questions: when generation stops, and whether the eval decodes greedily. The
user asked to follow Qwen's published guidance and to choose the best-supported option.

### Sources

- [Pinned generation_config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/generation_config.json)
  lists `eos_token_id` as 151645 (`<|im_end|>`) and 151643 (`<|endoftext|>`), turns sampling on,
  and sets temperature 0.6, top-p 0.95, and top-k 20. Decision: add this file to the pinned
  snapshot and its hash manifest. Stop on either ID, read from the file. Use its sampling values
  for thinking mode.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) recommends temperature 0.6,
  top-p 0.95, top-k 20, and min-p 0 in thinking mode, and says "DO NOT use greedy decoding" there
  because of "performance degradation and endless repetitions". For non-thinking mode it
  recommends temperature 0.7, top-p 0.8, top-k 20, and min-p 0. Decision: replace PLAN.md's
  temperature-0 baseline with the card's sampling. Min-p 0 is off, so octlm needs no min-p.
  The card offers a presence penalty only for repetition problems. None have been measured, so
  octlm adds none.
- [Qwen function-calling guide](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
  warns that stopword-based tool formats such as ReAct break with reasoning models, because "the
  model may output stopwords in the thought section". Its tool-calling example samples at
  temperature 0.7 and top-p 0.8, the non-thinking values. Decision: use the native
  `<tool_call>` template (already planned for EXP-077), stop only on the configured end IDs, and
  run the harness in non-thinking mode by default.
- [Transformers generation utilities](https://huggingface.co/docs/transformers/main/en/internal/generation_utils)
  apply temperature, then top-k, then top-p. Decision: octlm's `sampling_distribution` uses the
  same order. It keeps the smallest set of top tokens whose probability reaches top-p, always
  keeps the top token, and renormalizes.

### Why non-thinking mode is the default

Qwen publishes both modes, so the choice is ours. Non-thinking mode fits a harness better for
three reasons. It is the mode of Qwen's own tool-calling example. It avoids spending the step
budget on reasoning text at the T4's measured 24 to 25 tokens per second. And a thinking block
can contain text that looks like a tool call, which the parser would have to skip. This is a
reasoned default, not a measurement. Thinking mode stays available through
`sampling(directory, thinking=True)`. EXP-079 may add it as a second baseline row if the GPU budget
allows. The model card also says multi-turn history should keep only final answers, not thinking
content. The harness must follow that rule if it ever runs thinking mode.

### Consequence for the eval

Sampling makes each run random, so a single run per task cannot support a claim. EXP-079 must fix
its generator seeds, run each task under several seeds, and report mean success with its
spread. The Day 8 comparison already requires beating the stock model by more than the
seed spread. Greedy decoding remains the parity mode in Day 6 tests and benchmarks.

### Implementation

- `Decoder.generate_cached` accepts `top_p` and `stop_ids`. It stops after emitting a stop ID and
  keeps that ID in its output. Defaults of 1.0 and an empty set keep every earlier call unchanged.
- `sampling_distribution` and `sample_token` accept `top_p` with a default of 1.0. The Day 4
  sampling fixture is unchanged.
- `octlm/day6.py` pins `generation_config.json`. `prepare` downloads only missing files, rejects
  an existing file whose hash changed, and rewrites the manifest. `sampling(directory, thinking)`
  returns the mode's settings and stop IDs. `reply` renders the template, generates, and returns
  the text before the stop ID.
- The local snapshot gained `generation_config.json`. The four existing files kept their stored
  hashes.

All 66 Python tests and 17 web parity checks pass after the change. Ruff lint and format pass.
No new test covers top-p, the stop set, or `reply`, and no real-model reply has run yet. EXP-077
exercises `reply` first.


## Session on 2026-10-02: research before the harness

### Sources

- [Pinned Qwen3 chat template](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)
  renders the tool list into the system turn, asks for
  `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`, and wraps each tool result in
  `<tool_response>` inside a user turn. Consecutive tool messages share one user turn. The
  non-thinking generation prompt ends with an empty `<think>\n\n</think>\n\n` block, but an
  earlier assistant turn with no reasoning renders without that block. Decision: the transcript is
  the message list, and every turn re-renders it through the native template. Appending generated
  IDs directly would give the model a history that differs from the template it was trained on.
- The same template has a consequence for the KV cache. Turn n+1's rendered tokens match turn n's
  only up to the start of the last assistant turn. Decision: keep the cache from the previous turn,
  find the longest common token prefix with the new prompt, slice the cache to that length, and
  prefill only the rest. The system prompt and the tool list are then prefilled once per task.
  Day 6 EXP-074 already measured that a cache built in two chunks matches a full forward pass, so a
  sliced cache needs no new parity claim.
- [vLLM Hermes tool parser](https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/hermes_tool_parser.py),
  which vLLM uses for Qwen tool calls, finds calls with `<tool_call>(.*?)</tool_call>` and also
  accepts an unclosed final tag. On any JSON error it drops every call in the turn and returns the
  raw text. Decision: use the same two patterns, but parse each block on its own. A malformed block
  becomes an error result for the model. The well-formed calls in the same turn still run.
- [Qwen function-calling guide](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
  allows several calls in one turn and returns every result before the next model turn. Decision:
  run all calls of a turn in order and send back one `tool` message for each.
- [BFCL AST evaluation](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
  checks the function name, required parameters, parameter types, and parameters missing from the
  schema. Decision: a call is valid when its block is a JSON object with exactly `name` and
  `arguments`, the name is a known tool, every required argument is present, no unknown argument
  appears, and every argument is a string. Valid-call rate is valid blocks over all blocks.
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) keeps a linear message history and
  runs each action as an independent subprocess, with no persistent shell. Decision: the same. The
  transcript is a plain message list, which Day 8 can reuse as SFT traces without conversion.
- [Python `subprocess`](https://docs.python.org/3/library/subprocess.html) kills only the direct
  child on timeout. `start_new_session` puts the child in its own session. Decision: `run_tests`
  starts unittest in a new session and kills the whole process group on timeout, so a test that
  spawns a child cannot outlive its limit.
- [tau-bench](https://arxiv.org/abs/2406.12045) defines pass^k as the expected value over tasks of
  C(c, k) / C(n, k), the chance that k independent trials all succeed, with n trials and c
  successes. pass^1 equals the mean success rate. Decision: report pass^1 with its spread across
  seeds, and pass^k at the largest k, because a fine-tune that only makes success more
  consistent shows up there first.
- [SWE-bench](https://arxiv.org/abs/2310.06770) grades a fix with tests the model never sees and
  checks every task with its gold patch before using it. Decision: tasks that create code are
  graded with hidden tests copied in after the run. Every task carries a reference solution, and a
  model-free `validate` stage proves two things for each task: the check fails on the untouched
  repository, and it passes after the reference solution runs through the same tool code.

### Design

`octlm/harness.py` holds the tools, the parser, the loop, the checks, and the eval command. The
fixture repository is `fixtures/day7/repo`, a small Python package with passing unittest tests.
`fixtures/day7/tasks.json` lists the tasks. Each task starts from a fresh copy of the repository.
A bug-fix task first applies an exact string replacement that plants one bug.

Task checks:

- `answer`: the final answer contains an expected string, compared case-insensitively.
- `file`: a file matches a regular expression.
- `tests`: the visible tests pass, and no file under `tests/` changed. The second half blocks the
  reward hack of deleting or weakening a failing test.
- `hidden`: a hidden test file, copied in after the run, passes.

The harness does not isolate the operating system. Paths resolve inside the sandbox directory and
symlinks that leave it are rejected, but `run_tests` executes model-written Python. The eval runs
on a disposable Kaggle VM for that reason.

### EXP-077: tool-call format

Problem. The loop needs structured calls from free text, and one malformed call must not end a
task.
Hypothesis. The native `<tool_call>` format with per-block JSON parsing turns every malformed call
into an error message the model can see.
Baseline. The vLLM Hermes parser, which drops the whole turn on one JSON error.
Measurement. Valid-call rate over every emitted block in the EXP-079 run, and the error kinds.
Stop condition. Any exception from the parser or a tool that ends a task is a failure of this
experiment.

### EXP-078: tool loop

Problem. A multi-turn task re-sends the whole conversation every turn.
Hypothesis. Prefix reuse prefills the system prompt and tool list once per task and cuts prefill
tokens by more than half on tasks of three or more turns.
Baseline. Full prefill of every rendered prompt, which equals the sum of rendered prompt lengths.
The harness records both numbers on every turn, so the baseline needs no second run.
Measurement. Rendered prompt tokens, reused tokens, and prefilled tokens per turn.
Stop condition. Stop if a reused prefix ever differs from the new prompt's tokens. The code
compares token IDs before it slices the cache.

### EXP-079: eval set and stock baseline

Problem. Day 8 needs a fixed number to beat.
Hypothesis. Stock Qwen3-0.6B in non-thinking mode solves the lookup tasks often and the code
tasks rarely, so the eval has room in both directions.
Baseline. This run is the baseline for Day 8.
Measurement. 40 tasks, seeds 0, 1, and 2, at most 8 model turns and 6,144 context tokens per task,
at most 512 new tokens per turn. Valid-call rate, pass^1 with its seed spread, pass^3, mean turns,
generated tokens per second, and prefix reuse. Results go to one JSONL file.
Stop condition. The task set, checks, and limits freeze before the run. If the stock model solves
no task or every task, record that. Any change to the task set afterward is a new version with
a new run, not an edit of this one.

### Implementation

- `octlm/harness.py` holds the five tools, the parser, the turn loop, the four checks, a
  `validate` stage, and the `eval` stage. One command runs the whole eval and writes JSONL:
  `python -m octlm.harness eval --device cuda:0`.
- The parser validates each `<tool_call>` block on its own. A bad block returns an error string as
  that call's `tool` message, and the loop continues.
- `generate_turn` keeps the previous turn's token IDs and cache. It slices the cache to the common
  prefix with the newly rendered prompt and prefills the rest. It always prefills at least one token,
  because the next token's logits come from the last prompt position.
- A tool result that contains any of the tokenizer's 26 added-token strings is replaced by an error.
  The rendered conversation is encoded with native special tokens, so such text would otherwise
  turn a file's contents into control tokens, which `AGENTS.md` forbids.
- `fixtures/day7/repo` is a package called `shop` with 12 passing tests. `fixtures/day7/tasks.json`
  holds 40 tasks: 14 `answer`, 9 `file`, 8 `hidden`, and 9 `tests` bug fixes. The hidden tests are
  in `fixtures/day7/hidden`.
- Two counting tasks, `customer-count` and `test-files`, have answers that no tool prints
  verbatim. They carry `derived: true`, so `validate` skips its "answer appears in a tool result"
  check for them.

### Task validation

`python -m octlm.harness validate` ran on the laptop with no model. All 40 tasks fail their check
on the untouched repository and pass after their reference solution runs through the parser and
tools. The record is `runs/day7-validate.jsonl`. The first run caught `test-files`, whose answer
needs counting, which led to the `derived` flag above.

No unit tests were added for the parser, sandbox, or prefix reuse, and the harness has not run
against the real model on the laptop. The first real-model turn is the Kaggle run.

### Kaggle run

A new version of the private `whynotramaa/octlm-code` dataset adds `octlm/harness.py` and
`fixtures/day7` to the same source, test, config, and lock files as before. The private
`whynotramaa/octlm-day7` notebook version 1 (`notebooks/octlm-day7.ipynb`) was pushed on a T4. It
downloads and hash-checks the pinned snapshot, runs `validate`, then runs `eval` with seeds 0, 1,
and 2.

### Run 1 result (task set v1)

`whynotramaa/octlm-day7` version 1 finished with status COMPLETE on a Tesla T4, float16, with the
pinned snapshot, non-thinking sampling (temperature 0.7, top-p 0.8, top-k 20), stop IDs 151643
and 151645, and seeds 0, 1, and 2. `validate` passed all 40 tasks on Kaggle before the eval. The
JSONL and log are in `runs/kaggle-day7/`. No harness exception ended any of the 120 runs, so the
EXP-077 stop condition held.

| Metric | Value |
| --- | ---: |
| Runs | 120 |
| Runs with any tool call | 10 |
| Tool-call blocks | 24 |
| Valid-call rate | 0.333 |
| pass^1 per seed | 0.075, 0.050, 0.025 |
| pass^1 mean and stdev | 0.050 and 0.025 |
| pass^3 | 0.000 |
| Mean turns | 1.18 |
| Generated tokens per second | 19.0 |

The stock model rarely uses the tools. In 110 of 120 runs it answered on the first turn without a
call and invented the value, for example "The server uses port 8080" where the config says 8042.
No `answer` or `tests` task ever made a call. All 10 tool-using runs were `file` or `hidden`
tasks, which name a file to write.

All 16 invalid blocks were `write_file` calls with code in `content`. Eight hold raw newlines
inside a JSON string ("Invalid control character"). Eight are unterminated because the turn hit
the 512-token limit while the model repeated a boolean expression. Both `sku` runs that hit the
8-turn limit resent the same broken call after each error message, so the error message reached
the model but did not change its output.

Prefix reuse worked. Over the multi-turn runs, 29 percent of rendered prompt tokens were
prefilled and 71 percent came from the cache. On a two-turn task, about 586 of 1,101 rendered
tokens were prefilled. The first turn's 516 tokens are system prompt, tool list, and task. The
EXP-078 hypothesis named tasks of three or more turns, and only the two `sku` runs reached that
length, at 22 and 18 percent prefilled. The 19.0 tokens per second includes prefill time, so it
sits below Day 6's 24 to 25 decode-only rate.

### Two check flaws found in run 1

Six runs passed. Reading their transcripts showed that two of those passes are wrong.

- `lamp-stock` passed on seed 2 with "The number of lamps in stock is 123." The `answer` check
  was a substring test, and "123" contains "12". The seed 0 pass ("12") was also a guess, since
  no `answer` run called a tool.
- `add-customer` passed twice after the model overwrote `data/customers.csv` with the header and
  the new row, deleting seven customers. The `file` check only looked for the new line.

Decision, following the stop condition above: keep run 1 as recorded and fix the checks as task
set v2. An answer now has to match as a whole token: no word character, dot, or dash before it,
and no word character, dash, or decimal digit after it. `file` checks gain a `keep` list of
original lines that must survive, on the six edits that modify an existing file. `validate`
passes all 40 v2 tasks. Run 2 on v2 is the Day 8 baseline.

### Run 2 result (task set v2, the Day 8 baseline)

`whynotramaa/octlm-day7` version 2 finished with status COMPLETE on a Tesla T4 with the same
settings. Its environment row carries the v2 `tasks.json` hash
(`dbb2ba8f...`) and the v2 `harness.py` hash (`5fd65da3...`), which match the local files. The
JSONL and log are in `runs/kaggle-day7-v2/`.

| Metric | Value |
| --- | ---: |
| Runs | 120 |
| Valid-call rate | 0.333 (8 of 24 blocks) |
| pass^1 per seed | 0.050, 0.025, 0.000 |
| pass^1 mean and stdev | 0.025 and 0.025 |
| pass^3 | 0.000 |
| Success by kind | answer 0.024, file 0.074, hidden 0, tests 0 |
| Mean turns | 1.18 |
| Stops | 118 final answers, 2 turn limits, 0 context limits |
| Generated tokens per second | 18.3 |
| Prefilled fraction of rendered tokens | 0.28 over all runs, 0.29 multi-turn, 0.21 at 3+ turns |

All 120 transcripts are identical to run 1, token for token. Fixed generator seeds reproduce the
sampled outputs across two T4 sessions, so the only change between the runs is the scoring. The
v2 checks removed exactly the three wrong passes: `lamp-stock` seed 2 ("123") and both
`add-customer` runs. Elapsed generation time differed by 4 percent (437 s and 454 s), which is the
session-to-session timing noise for this throughput figure.

Three passes remain: `todo-file` on seeds 0 and 1, and `lamp-stock` on seed 0. The `lamp-stock`
pass answered "12" without calling a tool, so it is a lucky guess that the check cannot tell
apart from a grounded answer. Day 8 should report successes that called at least one tool next to
the raw pass rate. That needs no rerun, because every row stores its transcript.

EXP-077 decision: keep. The parser turned all 16 malformed blocks into error messages and no run
crashed. The model did not repair a call after seeing its error, so error feedback alone does not
raise the valid-call rate.

EXP-078 decision: keep. Prefix reuse prefilled 21 percent of rendered tokens on the two runs of
three or more turns, which beats the "more than half saved" hypothesis. The two-turn runs save 71
percent. The comparison rests on two long runs only, because the stock model rarely takes a
second turn.

EXP-079 decision: keep v2 as the Day 8 baseline. The stock model solves almost nothing, mostly
because it answers from imagination instead of calling a tool. That leaves Day 8 a large,
specific target: call a tool before answering, and escape newlines inside JSON strings. The
seed spread is 0.025, one task out of 40. Day 8 has to beat 0.025 by more than that spread.

## Exit check

- [x] EXP-077 turns every malformed call into an error message, and no run crashed.
- [x] EXP-078 reuses the cached prefix, measured on every turn of every run.
- [x] EXP-079 has 40 validated tasks and a three-seed stock baseline.
- [x] One command runs the whole eval and writes JSONL:
  `python -m octlm.harness eval --device cuda:0`.
- [x] The stock model's numbers are in this note, run 2 above, with run 1 kept as the record that
  exposed the check flaws.

Day 7 is complete. Day 8 may start.

## Session on 2026-10-02: publish the Day 7 explanations

The web journal has one post per Day 7 experiment: `tool-calls` (EXP-077), `the-agent-loop`
(EXP-078), and `the-day7-eval` (EXP-079). Every number comes from the exported runs or the task
file. `web/scripts/export.py` exports both eval runs, a parser and pass^k fixture, and
`web/src/data/day7.json`, which holds the tool schemas, tasks, and three transcripts with
per-turn token accounting. The exporter recomputes each turn's rendered and reused tokens from the
saved messages and fails unless the totals match the logged run. All three matched.

Four widgets: a live tool-call parser with real model outputs, a turn stepper that shades the
cached prefix, a 40-by-3 run grid that switches between v1 and v2 scoring, and a pass^k against
pass@k calculator. The TypeScript parser, answer check, and pass^k match `octlm.harness` on the
fixture. All 19 web parity checks pass and Astro builds 39 pages. The UI was not inspected in a
browser.
