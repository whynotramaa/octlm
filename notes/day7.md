# Day 7: build the harness and its eval

Status: started 2026-10-01. Decoding decisions made; EXP-077 not started.
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
