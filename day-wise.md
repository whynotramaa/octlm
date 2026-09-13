# Day wise reading schedule

## How to use this file

This is a reading and understanding plan, not a build plan. It covers every concept in `PLAN.md` in 8 days. Each day assumes 6 to 8 hours with breaks. Each day ends with an exit check. If you fail the exit check, repeat the day before moving on.

Daily method for every concept:

1. State the problem it solves in one sentence.
2. Learn the naive version first.
3. Learn the modern fix and why it helps.
4. Sketch the math or algorithm by hand.
5. Write one experiment you would run in `PLAN.md` to prove it.

Keep a single notebook file per day under `experiments/log/READ-DAY-N.md`. Record beliefs, confusions, and sketches there.

## Day 1. Foundations and the Generation 0 Transformer

Goal. Understand why language modeling works and why the 2017 Transformer needed every later fix.

Topics:

* Language modeling objective, next token prediction, cross entropy, perplexity.
* Character tokenizer versus BPE. Merges, vocab size, compression ratio, byte fallback, special tokens, code versus prose behavior.
* Embeddings, absolute positional encoding and its limits.
* Single head attention, multi head attention, causal masking, scaled dot product.
* Full baseline block. Embedding, position, LayerNorm, MHA, ReLU or GELU FFN, LM head.
* Training loop v0. AdamW basics, gradient clipping, validation split, checkpoints, seeds.

Read in order:

* Attention is all you need, sections on architecture, attention, and positions. Focus on the why, skip proofs.
* A BPE explainer with a worked merge example. Trace 20 merges by hand on code text.
* A LayerNorm explainer. Note mean and variance behavior across batch and sequence.

Hands on, on paper:

* Tokenize one Python function by char and by imagined BPE. Count sequence lengths.
* Write `softmax(QK^T / sqrt(d))V` dimensions for B 1, T 8, heads 2.
* Draw the causal mask for T 4.
* List three reasons char tokens hurt code modeling.

Exit check. You can explain BPE merges, attention dimensions, causal masking, and the full forward pass of the baseline without notes. Maps to EXP-003 to EXP-008.

## Day 2. Modern decoder core

Goal. Understand the four changes that define almost every current open decoder.

Topics:

* RoPE. Rotation by position, relative signal, extrapolation to 2K, 4K, 8K.
* RMSNorm versus LayerNorm. What it drops and why training stays stable.
* SwiGLU versus GELU FFN. Gating, parameter matching, throughput effect.
* MHA versus MQA versus GQA. KV head counts, cache math, quality tradeoff.
* Pre norm residuals and newer residual variants.
* SDPA abstraction and what FlashAttention tiling solves. Memory reads and writes, not just FLOPs.

Read in order:

* RoPE paper, intro plus formulation plus extrapolation discussion.
* RMSNorm paper, abstract plus method plus ablation tables.
* SwiGLU or gated linear unit section from the LLaMA era papers.
* GQA paper or LLaMA 2 report section on GQA. Note KV head sweeps.
* FlashAttention paper, only the IO aware motivation and tiling sketch. Skip the full CUDA detail.

Hands on, on paper:

* Compute KV cache bytes for MHA versus GQA with 8 query heads and 2 KV heads. Use layers 12, dim 512, seq 4K, BF16.
* Sketch RoPE rotation for one 2D pair across positions 0, 1, 2.
* Explain why MQA is fastest but often loses quality.
* Write the SDPA call signature and mask behavior from memory.

Exit check. You can derive cache savings for any head count and defend the GQA keeper choice. Maps to EXP-009 to EXP-015.

## Day 3. Architecture lab and DeepSeek style ideas

Goal. Understand sparsity and efficiency ideas without copying frontier scale.

Topics:

* Sparse attention. Sliding window, strided patterns, when far references break.
* Compressed attention. KV summarization, cache saving versus recall loss.
* MLA as experimental low rank KV compression.
* MoE end to end. Router, top k, expert capacity, dropped tokens, auxiliary load balancing loss, active versus total parameters.
* Expert collapse. What the histogram looks like and what fixes it.
* Multi token prediction. Extra heads, extra compute, possible gains for code.
* Muon versus AdamW. Where the update rule differs.
* mHC and hyper connections at a high level.
* DeepSeek V4.1-Flash context. 552B total MoE with 8B active on input and 16B on output, asymmetric encoder decoder idea, reduced cache. Ask only what transfers to 50M.

Read in order:

* A sparse Transformer or Longformer section on windowed attention.
* An MoE survey or the Switch Transformer routing section plus load balancing loss.
* Multi token prediction paper, method plus results on code.
* Muon optimizer notes or release logs, update rule versus AdamW.
* DeepSeek V4 model card and V4.1-Flash announcement. Read for architecture claims only.

Hands on, on paper:

* Draw token to router to 4 experts to top 2 combine.
* Write the auxiliary balancing loss in words, then in math.
* Sketch a collapsed routing histogram and list two fixes.
* List one small scale test for each of MTP, compressed attention, and asymmetric compute.

Exit check. You can explain MoE failure modes and design three small experiments at 50M scale. Maps to EXP-017 to EXP-025.

## Day 4. Training at scale

Goal. Understand what breaks when training gets longer, larger, or distributed.

Topics:

* Mixed precision. FP32, BF16, FP16, loss scaling, divergence signs.
* Batch size, gradient accumulation, warmup, cosine versus linear schedules, weight decay.
* Activation checkpointing, dataloader resume, recovery from kills.
* Scaling intuition. Loss versus compute across 20M, 50M, 100M.
* DDP. Gradients averaged, throughput versus single GPU.
* FSDP. Sharded weights, gradients, optimizer states. When it matters.
* Tensor parallelism. FFN split across devices, all reduce combine.
* Pipeline parallelism. Layers on devices, bubbles, throughput cost.

Read in order:

* Mixed precision training guide from PyTorch. Note autocast and GradScaler behavior.
* DDP and FSDP docs. One page each. Note what is sharded where.
* A tensor parallelism explainer with the column and row parallel FFN split.
* A pipeline parallelism explainer with the bubble diagram.

Hands on, on paper:

* Estimate training memory for 50M params with AdamW in FP32 versus BF16. Include weights, grads, and two moments.
* Draw the FFN split from `PLAN.md` with shapes before and after combine.
* Draw 4 microbatches over 2 pipeline stages and mark idle time.
* Define the cheapest setup question you would answer in EXP-030.

Exit check. You can compare single GPU, DDP, FSDP, TP, and PP on memory and speed and say when each is needed. Maps to EXP-026 to EXP-034.

## Day 5. Inference engineering

Goal. Understand why decoding is slow and how each layer of speed work helps.

Topics:

* Naive generation that recomputes the full sequence. Latency growth with length.
* KV cache. Per layer storage, causal update, invalidation rules.
* Prefill versus decode. Time to first token versus inter token latency.
* SDPA plus GQA plus `torch.compile` plus BF16 as the fast baseline.
* Quantization. INT8 and INT4. VRAM, model bytes, speed, quality loss. Why small models suffer more.
* Continuous batching. Queueing, early finished sequences, throughput versus latency.
* Prefix caching. Repeated system prompts and file headers, hit rate logic.
* Speculative decoding. Draft model, verify step, accepted tokens per step.

Read in order:

* KV cache explainer with a worked decode step.
* FlashAttention and SDPA backend notes, only the inference effect.
* A quantization guide covering dynamic, static, GPTQ, AWQ, and GGUF at a high level. Note calibration data.
* Speculative decoding paper, draft plus verify loop.
* Continuous batching notes from vLLM or TGI docs, scheduler section only.

Hands on, on paper:

* Plot imagined latency for naive versus KV cache from 512 to 8K.
* Compute KV bytes for your 50M config at 8K in BF16 and INT8.
* Work one speculative step with 3 drafted tokens and 2 accepted.
* List what prefix cache must invalidate after a file edit.

Exit check. You can name TTFT, inter token latency, tokens per second, VRAM, cache bytes, and quality delta for every inference change. Maps to EXP-035 to EXP-042.

## Day 6. Post training

Goal. Understand how a base model becomes useful, controlled, and styled.

Topics:

* SFT. Instruction triples, code fix and writing tasks, win rate versus base.
* LoRA. Low rank adapters, frozen base, merge cost, quality versus full SFT.
* Preference data. Chosen versus rejected, style pairs for writing.
* DPO. Own loss intuition, no reward model needed, KL style control.
* Reward models. Scalar scorer, calibration, over optimization.
* PPO and GRPO. Reward plus drift control, why GRPO fits verifiable tasks.
* RLVR. Math and unit test rewards, pass or fail signals, reward hacking.

Read in order:

* InstructGPT or SFT plus RLHF overview, methods sections only.
* LoRA paper, method plus rank sweep.
* DPO paper, loss derivation and implicit reward view.
* GRPO paper or a clear explainer, group baseline idea.
* One RLVR or verifiable reward note on code and math tasks.

Hands on, on paper:

* Write 5 SFT triples, 2 for code fix and 3 for rewrite and tone.
* Write 3 preference pairs where B is natural and A is generic.
* Derive DPO in words. What rises and what falls when the model prefers chosen.
* Design one RLVR task with hidden tests and list two ways the model could cheat.

Exit check. You can run the SFT to DPO to GRPO chain on paper and predict what each stage changes. Maps to EXP-043 to EXP-048.

## Day 7. Evals and long context

Goal. Learn to trust numbers and catch regressions before the assistant hides them.

Topics:

* Perplexity on held out code and prose. What it hides.
* Coding evals. Completion, bug fix with hidden tests, explanation accuracy.
* Writing evals. Style match, repetition, judge prompt plus human spot checks.
* Reward hacking catalog. Empty tests, verbose answers, style gaming.
* Regression gates. Thresholds that fail a run.
* Long context. RoPE extrapolation, needle recall to 8K and 16K, perplexity slope with sparse or compressed attention.

Read in order:

* LM eval harness docs. Note task format and few shot setup.
* A HumanEval or MBPP description. Note pass at k and hidden tests.
* A writing eval or preference judge prompt example. Note bias risks.
* A long context eval note. Needle test plus perplexity versus length.

Hands on, on paper:

* Define the tiny stable eval set you would run on every commit.
* Write thresholds for perplexity, fix rate, and preference win rate that block a merge.
* Sketch needle recall accuracy falling with length for three attention variants.
* Log two reward hacks you expect and the fix for each.

Exit check. You can define `scripts/evaluate.py` inputs, outputs, and fail rules from memory. Maps to EXP-049 to EXP-053 plus EXP-062.

## Day 8. Assistant, retrieval, tools, and the leftover architectures

Goal. Understand how a weak local model becomes useful through context, tools, and routing.

Topics:

* Task classifier. Trivial, coding, hard. Accuracy on your own queries.
* Hybrid retrieval. BM25, embeddings, AST and symbol search, reranker, context builder with token budgets.
* RAG measurement. Recall and answer accuracy with and without retrieval.
* Tool loop. Read, write, grep, glob, git diff, run tests, python, shell. ReAct cycle of search, patch, test, revise.
* Structured tool calling. Valid JSON rate and task completion.
* Writing path. Style profile, retrieved examples, critic pass, revision.
* Cache stack. Response, retrieval, embedding, prefix, and KV caches. Hit rates and invalidation.
* Router and cost log. Local share, tokens avoided, TTFT, external spend, savings.
* Leftover architectures. SSM and Mamba as an alternative track, multimodal as a deferred second project.

Read in order:

* A hybrid search guide covering BM25 plus dense plus rerank.
* Code retrieval notes on AST, symbols, imports, and filenames.
* ReAct paper, loop section only.
* MCP or tool calling schema docs. Note JSON validation.
* A caching systems note. Separate semantic, retrieval, prefix, and KV behavior.
* A Mamba or SSM explainer, scan and state idea at a high level. Keep it in `experiments/architectures` only.

Hands on, on paper:

* Trace the query why auth middleware returns 401 through BM25, embeddings, AST, rerank, context builder, model.
* Write the tool sequence to fix one failing test.
* Define invalidation rules for every cache layer after a file write.
* Fill the cost dashboard in `PLAN.md` with imagined numbers and explain each row.

Exit check. You can whiteboard the full request path from user text to local or external answer with caches, tools, and evals marked. Maps to EXP-054 to EXP-061 plus EXP-063 and EXP-064.

## Coverage check against PLAN.md

* Day 1 covers BPE, MHA, training basics.
* Day 2 covers RoPE, RMSNorm, SwiGLU, GQA and MQA, SDPA.
* Day 3 covers MoE, sparse, compressed, MLA, MTP, Muon, mHC, DeepSeek style.
* Day 4 covers DDP, FSDP, tensor and pipeline parallelism, precision, schedulers.
* Day 5 covers KV cache, FlashAttention use, quantization, batching, prefix cache, speculative decoding.
* Day 6 covers SFT, LoRA, DPO, reward model, PPO, GRPO, RLVR, tool calling fine tune.
* Day 7 covers evals, reward hacking, long context.
* Day 8 covers RAG, tools, agent loop, caches, router, SSM and Mamba survey, multimodal boundary.

## If you have 10 days instead of 8

Split Day 3 into two days, one for attention variants and one for MoE plus MTP plus optimizers. Split Day 5 into two days, one for KV cache plus quantization and one for batching plus prefix plus speculative decoding. Keep the order unchanged.

## If you have only 7 days

Merge Day 7 into Day 6 afternoons and Day 8 mornings. Do one combined eval plus assistant day, but keep the exit checks separate. Do not cut Day 2 or Day 5. Those two carry the most engineering value.
