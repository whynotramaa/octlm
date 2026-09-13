# Miniature modern LLM stack from first principles

## Goal

Build one long running project that reproduces almost the entire LLM engineering roadmap in code at 20 to 150M parameter scale.

The small model is the lab. We are not chasing frontier capability. We are reproducing the engineering ideas, measuring them, breaking them on purpose, and then using the final model inside our own coding and writing setup.

Start deliberately simple with a 2017 era Transformer. End with a modern decoder, a post trained model, fast local inference, and a local assistant that handles code and prose with retrieval, tools, and caching.

## What counts as done

The project is done when all of these run from this repo:

1. A BPE tokenizer trained on our own code and prose mix.
2. A modern decoder model around 50 to 150M parameters with RoPE, RMSNorm, SwiGLU, and grouped query attention.
3. A training run with logged loss, perplexity, checkpoints, and resumption.
4. Fast inference with KV cache, SDPA, `torch.compile`, mixed precision, and INT8 or INT4 quantization.
5. Post training with SFT, LoRA, DPO, a reward model, and one RL run such as GRPO or RLVR on a verifiable task.
6. A local assistant that answers repo questions with hybrid retrieval, runs tools in a loop, caches aggressively, and routes hard queries to an external API.
7. An eval suite that reports perplexity, coding score, writing preference score, tokens per second, memory, and cost saved by local routing.
8. An experiment log where each change has a baseline, a problem statement, measurements, and a keep or revert decision.

## Guiding rule

Never add an optimization because a paper says it is better. For each technique, record what we believed, what we built, what broke, why it broke, what modern models do about it, our implementation, measurements, and the decision.

## Final shape

```text
                  OUR MODEL LAB
                       |
        +--------------+--------------+
        |                             |
  I. MODEL ENGINEERING         II. POST TRAINING
        |                             |
  tokenizer                       SFT
  transformer                     LoRA
  modern architecture             DPO
  training                        reward model
  inference                       GRPO and RLVR
  optimization                    evals
        |                             |
        +--------------+--------------+
                       |
                       v
               III. LOCAL ASSISTANT
                       |
              context construction
                       |
           +-----------+-----------+
           |           |           |
          RAG       tools        cache
           |           |           |
           +-----------+-----------+
                       |
               coding workspace
                       +
                writing workspace
```

At the end the assistant flow looks like this.

```text
User request
  |
  v
Task classifier, trivial or coding or hard
  |
  +-------------+-------------+
  v             v             v
Local SLM   Local SLM     External API
               + tools
               + RAG
  +-------------+-------------+
  v
Response plus logged cost, latency, and cache hits
```

## Architecture note on DeepSeek V4.1-Flash

The reference model is DeepSeek-V4.1-Flash, released September 10, 2026, not V4.21. Public material describes about 552B total MoE parameters with only about 8B active on input and 16B on output, an asymmetric causal encoder decoder design, and a much smaller KV cache footprint. V4 era work also introduced compressed and sparse attention ideas, multi token prediction, mHC, and the Muon optimizer.

We copy none of the scale. We reproduce the ideas at small scale and ask narrow questions. Does routing work at 50M. Does compressed attention save memory at 8K context. Does predicting two future tokens help. Does Muon converge differently from AdamW.

## Repository layout

Keep the core Transformer clean. Put variants and risky ideas in separate folders so failed versions stay runnable.

```text
octlm/
  PLAN.md
  README.md
  pyproject.toml
  configs/
    tokenizer/
    pretrain/
    sft/
    dpo/
    grpo/
    infer/
  tokenizers/
    char.py
    bpe.py
    train_bpe.py
    eval_tokenizer.py
  models/
    baseline/
    modern/
    moe/
    sparse_attention/
    compressed_attention/
    mtp/
    mla_experimental/
    deepseek_style/
  training/
    datasets.py
    optim.py
    scheduler.py
    trainer.py
    distributed.py
  inference/
    generate_naive.py
    kv_cache.py
    quant.py
    batching.py
    prefix_cache.py
    speculative.py
    bench.py
  posttrain/
    sft.py
    lora.py
    dpo.py
    reward_model.py
    ppo_grpo.py
    rlvr.py
  evals/
    perplexity.py
    coding_eval.py
    writing_eval.py
    regression.py
  assistant/
    classifier.py
    retrieval_bm25.py
    retrieval_embed.py
    retrieval_ast.py
    rerank.py
    context_builder.py
    tools.py
    agent_loop.py
    cache.py
    router.py
    cli.py
  experiments/
    architectures/
      mamba_ssm/
      multimodal_notes/
    log/
      EXP-001-char-tokenizer.md
      EXP-014-mha-to-gqa.md
  scripts/
    prepare_data.py
    train.py
    evaluate.py
    serve.py
```

Multimodal training and Mamba or SSM work live only under `experiments/architectures`. They do not change the core decoder.

## Hardware and cost plan

Develop on a laptop. Train small runs on free GPU time such as Kaggle. Rent a GPU only when free compute blocks a decision. Run final inference locally.

Weight memory alone, ignoring activations and optimizer state:

| Parameters | FP32 | FP16 or BF16 | INT8 | INT4 |
| ---------: | ---: | -----------: | ---: | ---: |
|        30M | 120 MB | 60 MB | 30 MB | 15 MB |
|        50M | 200 MB | 100 MB | 50 MB | 25 MB |
|       100M | 400 MB | 200 MB | 100 MB | 50 MB |
|       300M | 1.2 GB | 600 MB | 300 MB | 150 MB |
|         1B | 4 GB | 2 GB | 1 GB | 500 MB |

Real usage is higher because of activations, KV cache, gradients, optimizer states, and framework overhead. That is why the main target is 50 to 150M. A quantized final model fits easily on a laptop.

Training needs far more memory than inference. Plan for weights plus gradients plus two Adam moments plus activations. Use mixed precision, gradient accumulation, checkpointing, and later FSDP to stay inside free GPU limits.

Flow:

```text
Development, laptop, 0 rupee token cost
  |
  v
Small training, free GPU, 0 rupee
  |
  v
Serious experiment, free GPU first, rented GPU only with evidence
  |
  v
Final inference, laptop, 0 rupee per token plus electricity
```

## Data plan

Specialize. A 50M model cannot cover every language. Target Python, TypeScript, JavaScript, plus natural prose for rewriting, editing, summarization, and tone following.

Suggested splits:

* Code, 50 percent. Python, TypeScript, JavaScript only. Include files, diffs, tests, and short explanations.
* Prose, 30 percent. Clean writing samples, rewrites, summaries, style pairs.
* Instruction mix, 20 percent. Question answer pairs, tool call traces, fix the test traces.

Build three datasets from day one:

1. `pretrain.txt` or sharded JSONL for language modeling.
2. `sft.jsonl` with instruction, input, output triples for coding fixes and writing tasks.
3. `pref.jsonl` with chosen and rejected pairs for DPO and reward model work. Example pair is a generic answer versus a natural answer in the target style.

Record dataset hashes, sizes, licenses, and dedup steps in each experiment note.

## Experiment method

Every numbered experiment lives in `experiments/log/EXP-NNN-name.md` and follows one template.

```text
EXP-014, MHA to GQA

Hypothesis
KV cache shrinks and decode gets faster with little quality loss.

Baseline
MHA, VRAM 2.1 GB, decode 83 tok/s, perplexity 14.3.

Change
Grouped query attention with 8 query heads and 2 KV heads.

Result
VRAM 1.6 GB, decode 112 tok/s, perplexity 14.5.

Decision
Keep GQA. About 35 percent higher decode throughput for small validation loss.

Failure notes
What broke, what the logs showed, what we changed.
```

Numbers above are format examples, not targets. Measure the real numbers with `inference/bench.py`.

Required fields for inference experiments are time to first token, tokens per second, VRAM, RAM, model bytes, KV cache bytes, and perplexity or task score. Required fields for training experiments are loss curves, tokens processed, wall time, memory peak, and eval delta.

## Serial build order

Follow this order. Each item is small enough to finish and measure before moving on. Do not skip ahead to MoE or RL before the decoder and evals work.

### Phase 0, repo and measurement setup

* EXP-001. Scaffold the repo, configs, seeds, logging, and checkpoint format. Define the `bench.py` output schema now so later phases stay comparable.
* EXP-002. Pick the first hardware target and document limits. Record GPU model, VRAM, RAM, CUDA version, PyTorch version, and max sequence length that fits.

Exit check. `train.py --dry-run` and `bench.py --dummy` run and emit JSON logs.

### Phase 1, Generation 0, the deliberately simple Transformer

Goal is to feel each pain before fixing it.

* EXP-003. Character tokenizer. Train the smallest LM on it. Record sequence lengths, compression, and why code becomes slow to model at char level.
* EXP-004. BPE from scratch. Implement byte fallback, merges, vocab size control, and special tokens such as BOS, EOS, PAD, and tool delimiters. Measure compression ratio, tokens per word on code versus prose, vocab size tradeoff, and unknown rate. Keep vocab around 16K to 32K for this scale.
* EXP-005. Embeddings plus absolute positional encoding plus mean or learned pooling sanity checks. Visualize nearest neighbors and position behavior.
* EXP-006. Single head attention from scratch, then multi head attention. Write the naive `softmax(QK^T / sqrt(d))V` version first. Test with masks and check causality by hand.
* EXP-007. Full baseline block. Embedding, absolute position, LayerNorm, MHA, ReLU or GELU FFN, LM head. Train on a tiny corpus until it overfits one file. This proves the pipeline works.
* EXP-008. Training loop v0. Cross entropy LM loss, AdamW, linear or cosine schedule, gradient clipping, validation perplexity, checkpoint save and resume, deterministic seed test.

Exit check. Baseline model generates coherent short completions on the training domain and perplexity is logged.

### Phase 2, Generation 1, modern decoder one change at a time

Change one component per experiment and keep the old code path behind a config flag.

* EXP-009. Absolute position to RoPE. Implement rotary embeddings from scratch. Test position extrapolation to 2K, 4K, 8K. Report perplexity versus length.
* EXP-010. LayerNorm to RMSNorm. Measure speed and stability. Check loss variance across seeds.
* EXP-011. GELU FFN to SwiGLU. Match parameter count roughly when comparing. Report loss and throughput.
* EXP-012. MHA to MQA as an intermediate step. One KV head. Record cache size and quality drop.
* EXP-013. MQA to GQA. Sweep KV head counts such as 1, 2, 4 against 8 or 16 query heads. This is the keeper decision for the main model.
* EXP-014. Naive attention to SDPA and Flash backend. First write a tiled attention sketch to learn the blocking idea. Then benchmark `torch.nn.functional.scaled_dot_product_attention` with flash or memory efficient backends. Do not write a production CUDA kernel here. Measure memory and tokens per second at 1K, 4K, 8K context.
* EXP-015. Residual and placement variants. Compare pre norm baseline against newer residual schemes used in modern decoders. Keep only what improves stability at our scale.
* EXP-016. Multi token prediction head. Predict 1 versus 2 future tokens. Measure loss, extra compute, and whether downstream coding completion improves.

Exit check. One config named `modern-50M` beats the Generation 0 baseline on perplexity and decode speed at equal parameters.

### Phase 3, Generation 2, architecture lab

Each variant is a folder under `models`, not a rewrite of the core.

* EXP-017. Sparse attention prototype. Implement sliding window or strided attention. Measure memory at long context and quality loss on code that needs far references.
* EXP-018. Compressed attention prototype. Implement a simple KV compression or summarization step. Measure cache bytes and retrieval accuracy on a needle style synthetic test.
* EXP-019. MLA experimental. Implement low rank KV compression and expansion. Mark as experimental. Report cache saving versus added compute.
* EXP-020. MoE v0. Replace one SwiGLU FFN with 4 tiny experts, top 1 or top 2 routing, no balancing loss. Log routing histograms. The expected result is collapse where one expert takes most tokens.
* EXP-021. MoE v1. Add auxiliary load balancing loss, expert capacity, and dropped token counting. Sweep expert counts 4 versus 8 and top k 1 versus 2. Report activated versus total parameters, expert utilization, loss, and inference cost.
* EXP-022. DeepSeek style combined run. GQA plus MoE plus MTP in one config. Ask if the combination still trains stably at 50M.
* EXP-023. Asymmetric compute test. Give input and output paths different capacity, for example a lighter encoder style prefill path and a larger decode path, or different expert counts per side. This is a small scale analog of the asymmetric idea, not a copy of the 552B design.
* EXP-024. Optimizer test. AdamW versus Muon on the same seed and data. Compare convergence speed, final loss, and stability.
* EXP-025. mHC or hyper connection variant. Test whether training stability changes. Keep behind a flag and revert if it adds complexity without gain.

Exit check. At least MoE and one long context attention variant have measured cache, speed, and quality numbers. Keep or revert each with reasons.

### Phase 4, training at scale

* EXP-026. Mixed precision. FP32 versus BF16 versus FP16 with loss scaling. Record speed, memory, and any divergence.
* EXP-027. Gradient accumulation and batch size sweep. Find the smallest batch that still trains stably, then simulate larger batches.
* EXP-028. Scheduler sweep. Warmup length, cosine versus linear, and weight decay values for AdamW.
* EXP-029. Checkpointing and resume. Activation checkpointing, dataloader resume, and recovery from a killed run. Measure throughput cost.
* EXP-030. Scaling check. Train 20M, 50M, 100M on the same data budget. Plot loss versus compute. Pick the size that gives the best quality per watt for local inference.
* EXP-031. Data parallelism with DDP on two workers or two GPUs if available. Compare single GPU versus DDP throughput and final loss.
* EXP-032. FSDP sharding test. Measure peak memory versus DDP and note when it matters at our size.
* EXP-033. Tensor parallelism test. Split one FFN across two devices as in the plan sketch below. It is not needed for 50M, the point is to learn the all reduce and combine behavior.

```text
Input [B, T, 512]
        |
        +-------- GPU 0 --------+
        |  W1[:, :1024]         |
        |                       +-- combine
        +-------- GPU 1 --------+
           W1[:, 1024:]         |
                                v
                              output
```

* EXP-034. Pipeline parallelism test. Place different layers on different devices. Measure bubble time and throughput.

Exit check. Document the cheapest setup that reaches target perplexity, plus the single GPU, DDP, FSDP, TP comparison table.

### Phase 5, inference engineering

Start stupid, then fix each bottleneck. Benchmark every step at 512, 2K, 4K, and 8K tokens.

* EXP-035. Naive generate that recomputes the full sequence each step. Record how latency grows with length. This motivates the next step.
* EXP-036. KV cache from scratch. Implement per layer key and value storage, causal update, and cache invalidation on prompt change. Report tokens per second gain and cache bytes.
* EXP-037. Prefill versus decode split. Optimize prompt processing separately from token by token decode. Report time to first token and inter token latency.
* EXP-038. Add SDPA plus GQA plus `torch.compile` plus BF16. This is the fast baseline. Freeze it as a reference config.
* EXP-039. Quantization. FP32 to BF16 to INT8 to INT4. For each level report VRAM, model bytes, speed, and perplexity or task delta. Use existing kernels for INT8 and INT4 rather than writing custom CUDA.
* EXP-040. Continuous batching miniature scheduler. Queue requests, batch decodes, handle early finished sequences. Measure throughput versus naive sequential serving.
* EXP-041. Prefix caching. Cache system prompts and repeated file headers. Measure hit rate and latency on repo style repeated prompts.
* EXP-042. Speculative decoding. Train or distill a tiny draft model, verify with the main model, measure accepted tokens per step and net speedup.

Exit check. One documented inference recipe hits the tokens per second target on the laptop for the chosen model size, with quality loss stated plainly.

### Phase 6, post training

Use the coding and writing tasks as the actual objective, not an isolated notebook demo.

* EXP-043. SFT pipeline. Build instruction triples for code fix, code explain, rewrite, tone follow, and summarize. Train and report task win rate versus base.
* EXP-044. LoRA from scratch. Implement low rank adapters, train only adapter weights, compare full SFT versus LoRA on quality, memory, and merge cost.
* EXP-045. Preference data and DPO loss. Implement DPO without a black box trainer. Train on chosen versus rejected pairs such as natural versus generic AI style writing. Report preference win rate.
* EXP-046. Reward model. Train a scalar scorer on the same preference pairs. Check calibration and over optimization by plotting score versus human or GPT judge preference.
* EXP-047. PPO or GRPO small scale run. Start with GRPO on verifiable tasks because it is simpler to reason about at this scale. Track reward, KL drift, and eval regression.
* EXP-048. RLVR. Use verifiable math and code unit test rewards. Example tasks are pass or fail tests for small Python functions. Report pass rate and reward hacking attempts observed.

Exit check. The post trained model beats base on coding fix rate and writing preference while holding perplexity within an agreed bound.

### Phase 7, evals and regression

* EXP-049. Perplexity suite on held out code and prose splits.
* EXP-050. Coding evals. Small function completion, bug fix with hidden tests, and explanation accuracy. Keep the set tiny but stable so it runs on every commit.
* EXP-051. Writing evals. Style match, rewrite quality, and repetition checks with a fixed judge prompt plus human spot checks.
* EXP-052. Reward hacking log. Record cases where the model games the reward, for example empty tests passing or verbose answers scoring high, and the fix applied.
* EXP-053. Regression gate. `scripts/evaluate.py` fails CI style if perplexity or task score drops past a threshold.

Exit check. One command runs the full eval and writes JSON that the router and cost dashboard can read.

### Phase 8, local assistant

This is where the weak model gets useful. The assistant supplies context the weights lack.

* EXP-054. Intent classifier. Route each request to trivial, coding, or hard. Log accuracy on a labeled set of our own past queries.
* EXP-055. Hybrid code retrieval. Combine BM25, embeddings, and AST or symbol search over filenames, imports, and definitions. Add a reranker and a context builder with token budgets. Measure recall on repo questions such as why auth middleware returns 401.

```text
                    Query
                      |
          +-----------+------------+
          v           v            v
        BM25      embeddings     AST
          |           |            |
          +-----------+------------+
                      v
                   reranker
                      |
                      v
               context builder
```

* EXP-056. RAG quality test. Compare no retrieval versus BM25 only versus hybrid on a fixed set of repo questions. Report answer accuracy and tokens used.
* EXP-057. Tool loop. Implement `read_file`, `write_file`, `grep`, `glob`, `git_diff`, `git_status`, `run_tests`, `python`, and `shell`. Build a ReAct style loop of search, read, patch, run tests, inspect error, revise. Start with a miniature local Codex style CLI.

```text
> slm "fix the failing auth test"

Model
  |
  +-- search repo
  +-- read file
  +-- inspect symbols
  +-- inspect git diff
  +-- propose patch
  +-- apply patch
  +-- run tests
  |
  +-- on failure, inspect error and revise
  |
  +-- return result
```

* EXP-058. Tool calling fine tune. Collect structured call traces and SFT the model to emit valid tool JSON. Report valid call rate and task completion.
* EXP-059. Writing path. Style profile plus retrieved examples plus critic pass for repetition, unnatural tone, style mismatch, and factual drift, then revision.
* EXP-060. Cache stack. Implement response cache, retrieval cache, embedding cache, prefix cache, and KV cache as separate layers. Log hit rate and invalidation rules for each. Code edits must invalidate file derived caches.
* EXP-061. Model router and cost log. Send 80 percent of cheap work local, escape hard work to an external API. Track requests handled locally, tokens avoided, cache hit rate, median time to first token, local decode speed, external spend, and spend without routing.

Example dashboard fields:

```text
Requests                 1,000
Handled locally            781
External API               219
Tokens avoided           2.8M
Cache hit rate             43 pct
Median TTFT               180ms
Local decode             92 tok/s
External spend              $X
Spend without routing       $Y
Savings                    Y-X
```

Exit check. The CLI answers a repo question with cited files, fixes a small failing test through the tool loop, and rewrites a paragraph in the target style.

### Phase 9, integration and hardening

* EXP-062. Long context test. RoPE extrapolation to 8K and 16K with sparse or compressed attention on. Measure needle recall and perplexity slope.
* EXP-063. Stability run. 24 hour local serve test with memory, latency, and crash logging.
* EXP-064. Final model card. Data, size, quantization, eval scores, known failures, license limits, and how to reproduce.

## Concept coverage map

This table maps each roadmap concept to the experiment that owns it and the measurement that proves it.

| Concept | Implement | How | Owned by | Proved by |
| ------- | --------: | --- | -------- | --------- |
| BPE tokenizer | yes | From scratch with byte fallback | EXP-004 | Compression, vocab tradeoff |
| RoPE | yes | From scratch | EXP-009 | Perplexity versus length |
| MHA | yes | From scratch naive | EXP-006 | Correctness and baseline speed |
| GQA and MQA | yes | Implement and benchmark all three | EXP-012, EXP-013 | Cache bytes, tok per sec, perplexity |
| RMSNorm, SwiGLU | yes | From scratch | EXP-010, EXP-011 | Loss and throughput |
| KV cache | yes | From scratch | EXP-036 | Decode speed versus length |
| FlashAttention | partial | Learn tiled idea, benchmark SDPA flash backend | EXP-014 | Memory and speed at 4K, 8K |
| Quantization | yes | INT8 and INT4 runs | EXP-039 | VRAM, speed, quality delta |
| Speculative decoding | yes | Tiny draft model plus verify | EXP-042 | Accepted tokens, net speedup |
| Continuous batching | yes | Miniature scheduler | EXP-040 | Throughput versus sequential |
| LoRA | yes | Own implementation | EXP-044 | Quality versus full SFT |
| SFT | yes | Full pipeline | EXP-043 | Task win rate |
| DPO | yes | Own loss | EXP-045 | Preference win rate |
| Reward model | yes | Train one | EXP-046 | Calibration, over optimization check |
| PPO, GRPO | yes | Small scale | EXP-047 | Reward with KL and eval guard |
| RLVR | yes | Math and code unit rewards | EXP-048 | Pass rate |
| FSDP | yes | If GPUs available | EXP-032 | Peak memory versus DDP |
| Tensor parallelism | yes | Split FFN across devices | EXP-033 | Correctness plus overhead |
| Pipeline parallelism | yes | Layers on devices | EXP-034 | Bubble and throughput |
| Data parallelism | yes | DDP run | EXP-031 | Throughput and loss parity |
| MoE | yes | Small sparse variant | EXP-020, EXP-021 | Utilization, active params, cost |
| RAG | yes | Hybrid retrieval | EXP-055, EXP-056 | Recall and answer accuracy |
| Tool calling | yes | Structured fine tune | EXP-058 | Valid call rate |
| Agent loop | yes | ReAct plus execution | EXP-057 | Task completion |
| Evals | yes | Own harness | EXP-049 to EXP-053 | Regression gate |
| Long context | yes | RoPE plus sparse tests | EXP-062 | Needle recall to 16K |
| Multimodal | deferred | Second project | experiments only | Notes only |
| MLA | experimental | Prototype behind flag | EXP-019 | Cache versus compute |
| SSM, Mamba | experimental | Alt architecture folder | experiments only | Comparison note |

FlashAttention carries a partial mark on purpose. A correct and fast CUDA kernel is its own project. We learn the tiling idea, write a small sketch, and use optimized kernels for real numbers.

## Milestones and gates

* M1. Baseline LM trains and generates. EXP-003 to EXP-008.
* M2. Modern 50M config picked. EXP-009 to EXP-016.
* M3. Architecture lab has numbers. EXP-017 to EXP-025.
* M4. Training recipe is reproducible and distributed runs are measured. EXP-026 to EXP-034.
* M5. Local inference is fast and quantized. EXP-035 to EXP-042.
* M6. Post trained model wins on code and writing tasks. EXP-043 to EXP-048.
* M7. Evals gate every change. EXP-049 to EXP-053.
* M8. Assistant fixes code and rewrites prose locally. EXP-054 to EXP-061.
* M9. Hardened release with model card. EXP-062 to EXP-064.

Do not start post training before M2 and evals exist in at least rough form. Do not start the assistant before SFT and evals exist.

## Risks

* Small model quality will disappoint on general coding. Mitigation is narrow scope to our three languages and heavy reliance on retrieval and tools.
* Expert collapse in MoE without balancing. Mitigation is EXP-020 first without balancing so the failure is visible, then EXP-021 fixes it.
* Quantization hurting a small model more than a large one. Mitigation is per level quality gates in EXP-039.
* Reward hacking in RL. Mitigation is the EXP-052 log plus held out verifiable tests.
* Cache invalidation bugs after file edits. Mitigation is explicit invalidation rules in EXP-060.
* Free GPU limits blocking long runs. Mitigation is small data budgets, accumulation, checkpoint resume, and renting only with blocking evidence.

## How to work from this file

Pick the lowest unfinished EXP number. Read its goal and exit check. Write the note first with hypothesis and baseline. Implement behind a config flag. Run `bench.py` and `evaluate.py`. Fill in measurements. Mark keep or revert. Move to the next EXP.

Keep failed versions runnable. Git history plus the experiment log is the portfolio artifact, not just the final weights.
