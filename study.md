# Interview study plan aligned with `day-wise.md`

## Purpose

Use this file to prepare for an LLM engineering interview at the depth expected from someone who
has built and measured the system. The order matches the eight days in `day-wise.md`. Learn the
concepts, implementation choices, failure modes, metrics, and measured project results.

Every topic in this file is required. Finish one day before moving to the next.

## The A+ answer standard

For every topic, prepare an answer with this structure:

1. State the problem.
2. Explain how the mechanism works.
3. Compare it with the main alternative.
4. Name the cost in quality, memory, compute, latency, or complexity.
5. Describe the most likely failure and its visible symptom.
6. Name the metric or experiment that decides whether to keep it.
7. Cite a result from this project when one exists.

A definition alone is not enough. You must handle follow-up questions about tensor shapes, system
behavior, tradeoffs, debugging, and measurement. Use each result only under the conditions that
produced it.

## How to use each day

- Study the topics in the listed order.
- Explain each topic aloud without notes.
- Tie each answer to the experiments listed under the topic.
- Use the measured results in `notes/day1.md`, `notes/day2.md`, and later day notes.
- Repeat the day if you cannot pass its exit check.

Do not memorize claims without their conditions. A result at 3.3M parameters, a 256-token context,
or on a T4 does not automatically transfer to a 50M model, an 8K context, or a laptop CPU.

---

# Day 1. Foundations and the Generation 0 Transformer

Goal: explain the complete path from raw text to next-token logits and the training loop that learns
the model.

Maps to EXP-003 through EXP-008.

## 1. Language modeling objective

Know these topics at interview depth:

- Autoregressive language modeling and next-token prediction.
- Sequence probability as a product of conditional next-token probabilities.
- Teacher forcing during training and autoregressive feedback during generation.
- Shifted inputs and targets.
- Cross-entropy loss and negative log likelihood.
- Perplexity as exponentiated average negative log likelihood.
- Bits per byte for comparisons across tokenizers.
- Causal modeling and future-token leakage.
- Training loss against validation loss.
- Overfitting, underfitting, exposure bias, and data contamination.
- PAD-target exclusion from the loss.

Be ready to explain:

- Why cross entropy fits next-token prediction.
- Why a lower token perplexity does not prove that one tokenizer is better than another.
- Why bits per byte is valid across tokenizers that use different prediction units.
- Why training loss can keep falling while validation loss rises.
- Why an unexpectedly tiny loss often points to target or mask leakage.

Know the failure signatures:

- Future-token leakage causes implausibly fast learning and unrealistically low validation loss.
- An incorrect target shift trains the model to copy the current token.
- PAD targets distort both loss and perplexity.
- A contaminated validation set reports memorization as generalization.

## 2. Character tokenization and byte-level BPE

Know these topics at interview depth:

- Character tokens as the simplest baseline.
- UTF-8 bytes as a lossless base vocabulary.
- BPE vocabulary training through ordered adjacent-pair merges.
- Encoding by replaying learned merges.
- Decoding by joining token byte strings and decoding UTF-8.
- Byte fallback and zero unknown-token behavior.
- Vocabulary size against sequence length and embedding size.
- Compression ratio, tokens per byte, and bits per byte.
- Code against prose tokenization.
- Pre-token boundaries and why merges must not cross them.
- Document boundaries and why training merges must not cross them.
- Deterministic equal-frequency tie-breaking.
- Stable byte IDs, special-token IDs, and merge order.
- BOS, EOS, PAD, and tool delimiters.
- Ordinary text that contains a literal special-token string.
- Training the vocabulary on training data only.
- Dataset deduplication, licenses, hashes, and split integrity.

Know the required tokenizer invariants:

- Preserve exact text. Do not lowercase or normalize Unicode.
- Round-trip tabs, indentation, CRLF, NUL, emoji, combining marks, and non-Latin text.
- Keep special tokens unreachable through ordinary text encoding.
- Refuse a checkpoint if its tokenizer hash differs.
- Compare compression and bits per byte, not raw token perplexity.

Be ready to explain:

- Why character tokenization makes code sequences long.
- Why byte-level BPE does not need an unknown token for ordinary text.
- Why a large vocabulary can hurt a small model even when it shortens sequences.
- Why a prose-heavy tokenizer handles indentation and identifiers poorly.
- Why special tokens need a separate allowed-special path.
- How duplicated training data can make a validation result look better than it is.

## 3. Embeddings and the language-model head

Know these topics at interview depth:

- Token IDs as indices into a learned embedding table.
- Input shape `[batch, sequence]` and hidden shape `[batch, sequence, model_width]`.
- Vocabulary size times model width as the embedding parameter count.
- Weight tying between the input embedding and the output language-model head.
- Output logits shaped `[batch, sequence, vocabulary]`.
- Initialization scale and its effect on early optimization.
- Why autoregressive language modeling needs one output per sequence position.
- Why mean pooling and learned pooling do not belong in the decoder forward path.

Be ready to explain:

- Why weight tying matters more to a 50M model than to a frontier model.
- Why an embedding lookup is mathematically related to a one-hot matrix multiply but implemented as
  indexing.
- How bad embedding initialization can break a tied output head.
- How nearest-neighbor inspection can serve as a diagnostic without proving semantic quality.

## 4. Positional information

Know these topics at interview depth:

- Attention is permutation equivariant without position information.
- Learned absolute position embeddings.
- Fixed sinusoidal position encodings.
- Relative position information.
- Maximum trained length and evaluation length.
- The limitation of a fixed learned position table.
- Position behavior as a measured property, not an assumption.

For Day 1, focus on why position information is necessary and how learned absolute positions work.
RoPE belongs to Day 2.

Be ready to explain:

- What changes if the same tokens appear in a different order.
- Why learned absolute positions cannot run past the size of their table.
- Why a positional method can run at a longer length and still fail in quality.

## 5. Scaled dot-product attention

Know this formula and every named part:

```text
attention(Q, K, V) = softmax(QK^T / sqrt(head_width))V
```

Know these topics at interview depth:

- Query, key, and value projections.
- Attention logits as query-key similarity scores.
- Scaling by the square root of head width to control logit magnitude.
- Row-wise stable softmax.
- Weighted aggregation of values.
- Output projection.
- Tensor shapes before and after the head split.
- Attention compute of `O(sequence_length^2 * model_width)`.
- Score-matrix memory of `O(sequence_length^2)` in the naive implementation.
- Causal masks and padding masks.
- Boolean masks against additive masks.
- Applying the mask before softmax.
- Negative infinity for disallowed positions.

Be ready to explain:

- What Q, K, and V mean operationally.
- Why the scale prevents saturated softmax distributions.
- Why attention becomes expensive at long sequence lengths.
- Why setting masked logits to zero leaks information.
- Why subtracting the row maximum makes softmax numerically stable.
- How to test causality by changing a future token and checking earlier logits.

## 6. Multi-head and causal attention

Know these topics at interview depth:

- Splitting model width into query heads.
- Head width as `model_width / query_heads`.
- Independent attention subspaces followed by concatenation and an output projection.
- Why changing the number of heads does not change the four dense projection parameter count in
  ordinary MHA.
- The lower-triangular causal mask.
- Training with a full sequence against decoding one token at a time.
- Why a single-token KV-cached decode step needs no future-token mask.
- The difference between causal and padding masks.

Know the main failures:

- A head count that does not divide model width is invalid.
- Too many heads leave each head with too little width.
- A mask applied after softmax breaks probability normalization.
- An off-by-one mask lets each target see itself.

## 7. The baseline Transformer block

Know the complete Generation 0 decoder:

```text
token embedding
+ learned absolute position
-> pre-LayerNorm
-> causal multi-head attention
-> residual add
-> pre-LayerNorm
-> GELU feed-forward network
-> residual add
-> final normalization
-> tied language-model head
```

Know these topics at interview depth:

- The residual stream as the shared hidden state across layers.
- Residual connections and the identity gradient path.
- Pre-norm against post-norm placement.
- LayerNorm over the feature dimension for each token.
- LayerNorm gain, bias, and epsilon.
- GELU against ReLU.
- Feed-forward expansion and contraction.
- Final normalization before the language-model head.
- Parameter accounting for embeddings, attention, feed-forward layers, and norms.

Be ready to trace the tensor shapes through the full block for arbitrary batch size, sequence length,
model width, and head count. The interview requirement is an accurate explanation of the data flow.

## 8. Training loop v0

Know these topics at interview depth:

- Training and validation splits.
- Random block sampling without crossing document boundaries.
- Forward pass, shifted targets, cross entropy, backward pass, and optimizer step.
- AdamW and decoupled weight decay.
- Excluding biases and normalization gains from weight decay.
- Linear warmup and cosine or linear decay.
- Gradient clipping and gradient-norm logging.
- Deterministic seeds and the limits of reproducibility across releases and devices.
- Checkpoint contents: model, optimizer, scheduler, random state, sampler state, step, and artifact
  hashes.
- Atomic checkpoint writes.
- Exact resume against approximate resume.
- Validation loss weighted by the number of valid target tokens.

Know the smallest convincing checks:

- Overfit one fixed block.
- Confirm that generated text reproduces the memorized block.
- Save and resume a split run, then compare it with an uninterrupted run.
- Log validation perplexity and bits per byte.
- Verify that causal attention blocks future influence.

## Day 1 project evidence to cite

- The character tokenizer used 13,939 tokens for 13,939 held-out bytes and mapped 77.8 percent of an
  unseen Unicode sample to UNK.
- The 1024-entry byte BPE reduced held-out prose to 7,319 tokens and encoded every Unicode byte.
- BPE improved bits per byte from 4.9068 to 4.1497 after the same 40-step budget. Raw token
  perplexity moved in the opposite direction because the prediction unit changed.
- The baseline overfit one fixed block by step 200. Training loss reached 0.9121, and greedy decode
  reproduced the training text.
- A resumed CPU run matched an uninterrupted run exactly.
- The first overfit attempt exposed an embedding initialization scale that was too wide for the tied
  head. Initializing weights with standard deviation 0.02 fixed it.

## Day 1 exit check

You pass Day 1 when you can explain BPE, tokenizer invariants, attention shapes, causal masking, the
complete baseline forward pass, and the training loop without notes. You must also explain at least
one measured tokenizer result and one measured model failure from this project.

---

# Day 2. Modern decoder core

Goal: explain the changes that define a modern dense decoder and defend each change with its quality,
memory, and speed tradeoff.

Maps to EXP-009 through EXP-015.

## 1. RoPE

Know these topics at interview depth:

- Rotating adjacent query and key channel pairs by a position-dependent angle.
- Multiple rotation frequencies across the head width.
- Relative distance appearing in the dot product between rotated queries and keys.
- Applying RoPE to Q and K, not V.
- Zero learned parameters.
- Plain extrapolation beyond the trained context.
- Position interpolation and other RoPE scaling methods.
- Short-context quality against long-context reach.
- Evaluating perplexity and retrieval behavior as a function of length.

Be ready to explain:

- Why RoPE provides relative position information while keeping the decoder causal.
- Why the query and key receive rotations but the retrieved value does not.
- Why an implementation that accepts 8K tokens has not proved useful 8K context.
- Why interpolation can hurt before plain extrapolation starts to fail.

Project result: a model trained at 512 tokens kept improving through 2048, degraded after 4096, and
benefited from interpolation only at 8192. Treat "apply interpolation past 4x trained length" as a
result for this configuration, not a universal rule.

## 2. RMSNorm against LayerNorm

Know these topics at interview depth:

- LayerNorm subtracts the mean and divides by the standard deviation.
- RMSNorm divides by the root mean square and learns a gain.
- RMSNorm drops mean subtraction and usually drops the bias.
- Both normalize each token over its feature dimension.
- Accumulating the normalization statistic in float32 under lower-precision training.
- Arithmetic savings against actual kernel performance.

Be ready to explain:

- What RMSNorm removes and what invariance it gives up.
- Why fewer arithmetic operations do not guarantee a faster PyTorch implementation.
- Why a fused LayerNorm can beat an unfused RMSNorm at small width.

Project result: RMSNorm quality matched LayerNorm within seed noise but cost 8.9 percent more step
time at width 256 on a T4. Keep it inside the measured modern stack, but do not claim a standalone
speed win.

## 3. SwiGLU against GELU feed-forward layers

Know these topics at interview depth:

- The classic two-matrix feed-forward layer.
- SiLU, also called Swish.
- A gated branch multiplied element-wise by a value branch.
- Three projection matrices in SwiGLU.
- Shrinking hidden width to match the parameter count of the GELU baseline.
- Feed-forward parameter and compute share in a decoder block.

Be ready to explain:

- What the gate changes about information flow.
- Why comparing equal hidden widths gives SwiGLU an unfair parameter advantage.
- Why quality, training throughput, and parameter count must appear in the same comparison.

Project result: matched-parameter SwiGLU improved code bits per byte by 0.082, outside the baseline
seed spread, and cost 11 percent more step time. It was the only isolated Day 2 component with a
clear quality gain.

## 4. MHA, MQA, and GQA

Know these topics at interview depth:

- MHA uses one key head and one value head for every query head.
- MQA shares one key head and one value head across all query heads.
- GQA shares each key-value pair across a group of query heads.
- Query-head count against KV-head count.
- Broadcasting grouped keys and values for attention.
- KV-cache memory scaling linearly with KV-head count, layers, batch, sequence length, head width,
  and bytes per element.
- MQA's maximum cache saving against its possible quality loss.
- GQA as a middle point between MHA quality and MQA memory.
- Why GQA can slow training while helping decode.

Be ready to explain:

- Why decode is sensitive to KV-cache bandwidth.
- Why fewer KV heads reduce cache bytes exactly.
- Why a small undertrained model may not reveal MQA's quality penalty.
- Why an eight-KV-head choice from a 64-query-head model does not transfer literally to an
  eight-query-head model.

Project result: two KV heads cut the 4K cache from 16 MiB to 4 MiB at the Day 2 shape. Its code
quality differed from MHA by 0.011 bits per byte, inside the 0.043 seed spread. The project kept two
KV heads as the risk-aware choice.

## 5. Residual placement

Know these topics at interview depth:

- Pre-norm and post-norm block order.
- Identity paths and gradient flow through depth.
- Warmup requirements.
- Final normalization in a pre-norm decoder.
- Keeping the optimizer and warmup fixed when comparing placement.

Project result: post-norm was 1.00 bits per byte worse on code and 0.70 worse on prose with no speed
benefit. The project reverted it and kept pre-norm.

## 6. SDPA and FlashAttention

Know these topics at interview depth:

- `torch.nn.functional.scaled_dot_product_attention` as the stable PyTorch attention API.
- Math, memory-efficient, and flash backends.
- `is_causal=True` against an explicit attention mask.
- `enable_gqa=True` and grouped query attention.
- Backend constraints from device, dtype, shapes, dropout, and mask type.
- Verifying the backend that ran instead of assuming.
- The naive `sequence x sequence` score and probability matrices.
- FlashAttention tiling and online softmax.
- Keeping blocks in fast on-chip memory.
- Recomputing scores in backward instead of storing them.
- Lower memory traffic with the same mathematical attention result.
- Why a production CUDA kernel is outside this project's scope.

Be ready to explain:

- Why FlashAttention can be faster without reducing FLOPs.
- Why arbitrary masks can disable the flash backend.
- Why CPU flash measurements do not prove a GPU HBM claim.
- How an equivalence test licenses replacing handwritten attention with SDPA.

Project result: at 8192 tokens on CPU, the flash backend used about 62 MB of additional resident
memory against 5.17 GB for math and ran 10.5 times faster in that benchmark. The memory direction is
the transferable result. The CPU speed ratio is not a GPU claim.

## 7. The modern decoder end to end

Know this complete path:

```text
token IDs
-> token embeddings
-> repeated pre-norm blocks
   -> RMSNorm
   -> Q, K, and V projections
   -> RoPE on Q and K
   -> GQA attention through SDPA
   -> output projection and residual add
   -> RMSNorm
   -> SwiGLU and residual add
-> final RMSNorm
-> tied language-model head
```

Be ready to state:

- The tensor shape at every stage.
- Which operations depend quadratically on sequence length.
- Which parameters dominate model size.
- Which tensors enter the KV cache.
- Which changes improve training quality, decode memory, or kernel memory traffic.
- Which claims remain unresolved because their observed difference is inside seed variance.

## Day 2 exit check

You pass Day 2 when you can compare RoPE, RMSNorm, SwiGLU, MHA, MQA, GQA, pre-norm, SDPA, and
FlashAttention without notes. Defend the project's keeper configuration with the measured numbers
and the limitations of those measurements.

---

# Day 3. Architecture lab and DeepSeek-style ideas

Goal: understand the modern architecture experiments, predict their failure modes at small scale,
and keep experimental work separate from the dense core.

Maps to EXP-016 through EXP-025.

## 1. Multi-token prediction

Know these topics at interview depth:

- Predicting tokens at `t+1`, `t+2`, and later offsets from a shared trunk.
- Separate transformations before a shared unembedding matrix.
- Auxiliary losses and their weights.
- Added parameters and training compute.
- Agreement or acceptance rate for future-token heads.
- Reusing an MTP head as a self-draft for speculative decoding.
- Scale dependence and the risk that a tiny model gets no quality gain.
- Parallel heads against sequential DeepSeek-style MTP modules.

Know the project-specific correction:

- Identical tied heads fed the same hidden state produce identical logits. Each future depth needs a
  distinct transformation.
- One linear adapter per extra depth is the minimum meaningful small-scale version.

Be ready to explain why the experiment may be worth running even if next-token quality does not
improve. The acceptance rate can still decide whether self-speculation is useful.

## 2. Sparse attention

Know these topics at interview depth:

- Full causal attention as the quality baseline.
- Sliding-window attention.
- Strided and dilated patterns.
- Global or sink tokens.
- Cost changing from quadratic in length to sequence length times window size.
- Receptive-field growth across layers.
- Far-reference failures in code.
- Mask density and actual backend selection.
- The risk that an arbitrary sparse mask loses the flash kernel.

Be ready to explain:

- Why fewer logical attention scores do not guarantee a faster run.
- Why code can suffer more than prose when distant tokens disappear.
- How a far-reference or copy probe exposes the loss.

## 3. Compressed attention

Know these topics at interview depth:

- Keeping recent tokens exactly and summarizing older blocks.
- Mean-pooled or learned compressed keys and values.
- Cache saving against recall loss.
- Block size as the quality-memory control.
- Rotating compressed keys at a representative position under RoPE.
- Selected blocks and learned gates in fuller sparse-compression designs.
- Why this project implements only the compressed and local branches first.

Know the measurement distinction:

- A standard needle prompt tests both retrieval and instruction following.
- A 3.3M base model may fail the instruction regardless of its attention memory.
- The project therefore uses a repeated-token copy probe to isolate whether a mechanism carries a
  distant reference. Report it as a copy probe, not as a full long-context needle benchmark.

## 4. Multi-head latent attention

Know these topics at interview depth:

- Compressing keys and values into a shared low-rank latent per token.
- Caching the latent instead of expanded K and V.
- Up-projecting the latent for attention.
- A separate decoupled RoPE key.
- Why rotated keys do not pass cleanly through a shared low-rank projection.
- Cache rank against reconstruction quality and added compute.
- Absorbing projections into other matrices as a decode optimization.
- MLA against MHA and against an already-efficient GQA baseline.

Know the project-specific comparison: at width 256 with two KV heads, the GQA cache holds 128
dimensions per token per layer. An MLA rank of 64 plus a 16-dimensional RoPE key holds 80. That is a
1.6 times reduction, not the order-of-magnitude headline measured against wide MHA.

## 5. Mixture of experts

Know these topics at interview depth:

- Replacing a dense feed-forward layer with several experts.
- Router logits and routing probabilities per token.
- Top-1 against top-2 routing.
- Weighted combination of selected expert outputs.
- Total parameters against active parameters per token.
- Capacity factor, expert capacity, and dropped tokens.
- Expert utilization histograms.
- Expert collapse.
- Auxiliary load-balancing losses.
- Noisy routing and capacity limits as other controls.
- Training compute against model memory.
- Communication costs under expert parallelism.

Be ready to explain:

- Why MoE buys capacity per FLOP but not capacity per byte.
- Why all experts must remain resident even when only two run for one token.
- What a collapsed routing histogram looks like.
- Why the roadmap first runs a version without balancing.
- Why a small MoE can lose to a dense model after routing overhead.

## 6. The combined and asymmetric experiments

Know these topics at interview depth:

- Combining GQA, MoE, and MTP only after each isolated experiment exists.
- Interaction effects that do not appear in one-change comparisons.
- Different compute budgets for prefill and decode.
- Different expert counts or path capacity for input and output processing.
- Shared-weight constraints imposed by the KV cache.
- Why a small-scale analog tests an idea but does not reproduce a frontier architecture.

Be ready to explain what the combined run can prove: training stability and interaction at this
scale. It cannot validate a 552B design or its reported production economics.

## 7. AdamW against Muon

Know these topics at interview depth:

- AdamW's per-parameter first and second moments.
- Decoupled weight decay.
- Muon treating matrix-shaped parameters as matrices.
- Orthogonalizing or conditioning the update through a Newton-Schulz-style step.
- Keeping embeddings, normalization gains, and often the language-model head on AdamW.
- Convergence per step against convergence per wall-clock second.
- Extra optimizer compute and optimizer-state memory.
- Separate learning-rate sweeps for a fair comparison.

A fair optimizer experiment holds the initialization, seed, data order, token budget, batch size,
precision, model, and hardware fixed. Report loss against tokens and wall time, peak memory, final
validation score, and variation across seeds.

## 8. mHC and hyperconnections

Know these topics at interview depth:

- Multiple residual paths instead of one residual stream.
- Learned mixing across paths.
- The training-stability motivation at depth.
- Wider residual state and its activation-memory cost.
- Extra parameters and implementation complexity.
- Loss variance and gradient behavior as evidence.
- A predeclared revert rule when the gain is inside measurement noise.

## 9. DeepSeek-style claims and transfer limits

Know how to discuss architecture reports responsibly:

- Separate official model-card claims from independent measurements.
- Distinguish total and active MoE parameters.
- Distinguish prefill and decode compute.
- Compare cache claims against the project's GQA baseline, not only against MHA.
- Treat MTP, compressed attention, asymmetric compute, mHC, and Muon as isolated hypotheses.
- Do not transfer a frontier-scale result directly to 3.3M or 50M parameters.

## Day 3 exit check

You pass Day 3 when you can compare MTP, sparse attention, compressed attention, MLA, MoE, Muon,
mHC, and asymmetric compute. For each one, state its expected benefit, its main cost, its likely
small-scale failure, and the experiment that would keep or revert it.

---

# Day 4. Training at scale

Goal: explain how memory, numerical precision, batch construction, recovery, and distributed
communication change as training grows.

Maps to EXP-026 through EXP-034.

## 1. Training-memory accounting

Know every major memory consumer:

- Model weights.
- Gradients.
- AdamW first and second moments.
- FP32 master weights in mixed-precision recipes that use them.
- Activations saved for backward.
- Temporary attention and feed-forward buffers.
- CUDA context and allocator overhead.
- Dataloader buffers.
- Sharding and communication buffers.

Be ready to estimate which term dominates for a specific model, sequence length, batch size, dtype,
and optimizer. State your assumptions instead of presenting one universal byte count.

## 2. Mixed precision

Know these topics at interview depth:

- FP32, FP16, and BF16 bit layouts at a practical level.
- FP16's narrower exponent range and gradient underflow risk.
- BF16's FP32-sized exponent range and lower mantissa precision.
- Automatic mixed precision.
- FP32 accumulation for sensitive reductions and normalization statistics.
- Static and dynamic loss scaling for FP16.
- NaNs, infinities, zero gradients, and divergence signals.
- Hardware support as the condition for a speed win.

Project evidence: BF16 attention was slower than FP32 on the laptop CPU. Do not sell mixed precision
as a universal optimization. It is primarily a GPU measurement until the target CPU proves otherwise.

## 3. Batch size and gradient accumulation

Know these topics at interview depth:

- Microbatch size.
- Accumulation steps.
- Number of data-parallel workers.
- Effective batch size.
- Loss scaling across accumulated microbatches.
- Gradient zeroing only after an optimizer step.
- Gradient noise and optimization stability.
- Throughput, memory, and optimizer-step frequency.
- Learning-rate changes when effective batch changes.
- Cases where accumulation is not equivalent to one large batch.

Know the silent failure: summing already-averaged microbatch losses without the correct scale changes
the effective learning rate and can make a formerly stable run diverge.

## 4. Warmup, schedules, and weight decay

Know these topics at interview depth:

- Linear warmup.
- Cosine against linear decay.
- Peak and final learning rates.
- AdamW beta values.
- Decoupled weight decay.
- Excluding bias and normalization parameters from decay.
- Scheduler state in checkpoints.
- Token budget against step budget.
- Why stopping a schedule early changes the effective training recipe.

Compare schedules at equal tokens and with learning-rate sweeps. A schedule name alone does not
predict the winner.

## 5. Activation checkpointing and exact resume

Keep these two meanings of checkpoint separate:

- Activation checkpointing saves memory by discarding selected forward activations and recomputing
  them during backward.
- Training checkpoints save persistent state so an interrupted run can resume.

Know the activation-checkpoint tradeoff:

- Lower activation memory.
- Extra forward compute during backward.
- Placement by layer or block.
- No intended change to model quality.

Know complete resume state:

- Model, optimizer, and scheduler.
- Step and token counts.
- Python and PyTorch random states, including CUDA state.
- Sampler and dataloader position.
- Artifact, tokenizer, data, and configuration hashes.
- Mixed-precision scaler state when used.

The correctness test compares a split run with an uninterrupted run. A loss curve that merely looks
similar is weaker evidence.

## 6. Scaling experiments

Know these topics at interview depth:

- Parameter count, training tokens, and compute budget as separate variables.
- Loss against compute.
- Undertraining and overtraining relative to a target model size.
- Compute-optimal training against inference-optimal deployment.
- Why a smaller model trained on more tokens can win for repeated laptop inference.
- Comparing 20M, 50M, and 100M models on the same data and evaluation.
- Quality per watt and quality per model byte.

Be ready to explain why the project targets 50M to 150M even when a larger model would score better.
The product target includes local latency, memory, and ongoing cost.

## 7. Distributed data parallelism

Know these topics at interview depth:

- A full model replica on every worker.
- Different microbatches per worker.
- Gradient AllReduce.
- Gradient buckets and communication overlap with backward.
- Effective global batch size.
- Synchronized optimizer updates.
- Communication overhead against compute per worker.
- Slow interconnects and poor scaling efficiency.

Use DDP when the model fits on one device and data-parallel throughput is the goal.

## 8. Fully sharded data parallelism

Know these topics at interview depth:

- Sharding parameters, gradients, and optimizer states.
- All-gather before parameter use.
- Reduce-scatter for gradients.
- Resharding after a layer or block.
- Lower per-device state against more communication and complexity.
- Wrapping policy and peak-memory behavior.
- Why FSDP is unnecessary for a 50M model that already fits.

The strong interview answer says when not to use FSDP.

## 9. Tensor parallelism

Know these topics at interview depth:

- Splitting large matrix multiplications across devices.
- Column-parallel and row-parallel linear layers.
- Partial outputs and AllReduce combines.
- Communication inside every Transformer block.
- Fast intra-node interconnect as a practical requirement.
- Tensor shape changes across the split.
- Why tensor parallelism solves model-width limits, not data throughput by itself.

## 10. Pipeline parallelism

Know these topics at interview depth:

- Assigning different layer ranges to different devices.
- Microbatches moving through pipeline stages.
- Pipeline bubbles and idle time.
- More microbatches reducing the bubble.
- 1F1B scheduling.
- Activation transfers between stages.
- Stage balancing.
- Throughput against end-to-end latency.

## 11. Choosing the parallelism strategy

Know this decision order:

- If the model fits on one device, start with one device.
- If it fits but needs more throughput, use DDP.
- If optimizer and parameter state do not fit, consider FSDP.
- If individual layers do not fit, use tensor parallelism.
- If layer groups fit on separate devices, pipeline parallelism is an option.
- Large systems often compose these methods, but this project measures them separately first.

For every choice, discuss memory saved, communication added, hardware topology, code complexity, and
the metric that justifies it.

## Day 4 exit check

You pass Day 4 when you can compare single-device training, DDP, FSDP, tensor parallelism, and
pipeline parallelism. Explain mixed precision, accumulation, schedules, activation checkpointing,
and exact resume as parts of one reproducible training recipe.

---

# Day 5. Inference engineering

Goal: explain why autoregressive decoding is slow and how each optimization changes latency, memory,
throughput, or quality.

Maps to EXP-035 through EXP-042.

## 1. Naive generation

Know these topics at interview depth:

- Re-running the full prompt and generated prefix for every new token.
- Recomputing unchanged keys and values.
- Latency growth as the sequence grows.
- Sampling temperature, top-k, top-p, EOS handling, and maximum token limits.
- Correctness before optimization.

The naive path remains the reference for output equivalence and benchmark comparisons.

## 2. KV cache

Know these topics at interview depth:

- Caching keys and values for every layer and previous position.
- Computing only the new token's Q, K, and V during decode.
- Why past queries are not needed again.
- Cache layout by batch, layer, KV head, position, and head width.
- Static preallocation against dynamic growth.
- Maximum context and cache capacity.
- Prompt changes, model changes, and cache invalidation.
- Batched cache slots and releasing finished requests.
- Cache bytes scaling linearly with length and batch.
- GQA, quantization, and MLA as different ways to reduce cache cost.

Be ready to distinguish the per-request KV cache from a reusable cross-request prefix cache.

## 3. Prefill against decode

Know these topics at interview depth:

- Prefill processes prompt tokens in parallel.
- Prefill uses large matrix-matrix operations and often becomes compute-bound.
- Decode processes one new token per sequence at a time.
- Decode repeatedly streams model weights and the KV cache, so it is often bandwidth-bound.
- Time to first token reflects prompt processing and queueing.
- Inter-token latency reflects decode.
- Tokens per second can mean per-request speed or aggregate server throughput.

Be ready to classify every optimization by whether it helps prefill, decode, both, or only aggregate
throughput.

## 4. The fast baseline

Know the role of each part:

- SDPA selects optimized attention kernels.
- GQA reduces KV-cache traffic.
- BF16 can reduce model and activation bytes on supported hardware.
- `torch.compile` captures graphs and fuses operations.
- A static cache avoids shape changes and repeated allocation.

Know `torch.compile` failure modes:

- Graph breaks from data-dependent Python control flow.
- Recompilation from changing shapes.
- Compilation overhead dominating short runs.
- Dynamic cache growth preventing a stable graph.
- A small model becoming slower because launch overhead was not its bottleneck.

## 5. Quantization

Know these topics at interview depth:

- FP32, BF16, INT8, and INT4 deployment choices.
- Scale and zero point.
- Symmetric against asymmetric quantization.
- Per-tensor, per-channel, and per-group scales.
- Weight-only against weight-and-activation quantization.
- Calibration data.
- Activation outliers.
- Dynamic and static quantization.
- GPTQ, AWQ, and GGUF at the level of their purpose and deployment setting.
- Model-byte and bandwidth savings.
- Kernel availability and packing overhead.
- Quality loss measured on tasks as well as perplexity.
- Small-model sensitivity because less redundancy absorbs quantization error.

The shipping decision reports model bytes, RAM or VRAM, time to first token, decode speed, and task
delta at every precision.

## 6. Continuous batching

Know these topics at interview depth:

- A request queue.
- Per-step batch construction.
- Adding new requests while existing requests decode.
- Removing finished sequences immediately.
- Variable prompt and output lengths.
- Cache-slot allocation.
- Padding waste in static batches.
- Maximum batch size against latency and fairness.
- Aggregate throughput against one-user inter-token latency.

Be ready to describe one scheduler step from admitted requests to sampled tokens and cache updates.

## 7. Prefix caching

Know these topics at interview depth:

- Exact repeated token prefixes such as system prompts and file headers.
- Reusing their precomputed KV state across requests.
- Keys that include model, tokenizer, and exact token IDs.
- Content hashes and model-version hashes.
- Memory limits and eviction.
- Hit rate and saved prefill time.
- Invalidation after a source, tokenizer, or model change.

Do not confuse prefix caching with semantic response caching. A prefix-cache hit reuses computation,
not an answer.

## 8. Speculative decoding

Know these topics at interview depth:

- A smaller draft model proposing several tokens.
- The target model verifying them in parallel.
- Accepting the longest valid prefix.
- Corrected sampling after rejection so the target distribution is preserved.
- Acceptance rate.
- Draft cost and target verification cost.
- Draft quality against net speedup.
- Self-speculation with MTP heads.
- Why speculative decoding helps decode but not prompt prefill.

The deciding metric is net tokens per second after draft cost, not accepted tokens alone.

## 9. Inference benchmark discipline

Report all of these for 512, 2K, 4K, and 8K contexts:

- Time to first token.
- Inter-token latency.
- Per-request tokens per second.
- Aggregate throughput under batching.
- RAM and VRAM peak.
- Model bytes.
- KV-cache bytes.
- Prompt and output lengths.
- Backend, dtype, device, batch size, and warmup policy.
- Perplexity or task-score change.

Compare every optimized path against naive generation for correctness and against the previous
keeper for performance.

## Day 5 exit check

You pass Day 5 when you can trace naive decode, KV-cached decode, prefill, continuous batching,
prefix caching, quantization, compilation, and speculative verification. For each optimization,
state which latency or memory metric it changes and what can make it a net loss.

---

# Day 6. Post-training

Goal: explain how a base model becomes an instruction-following model and how preference and
verifiable-reward training can improve or damage it.

Maps to EXP-043 through EXP-048, with structured tool-call SFT feeding EXP-058.

## 1. Supervised fine-tuning

Know these topics at interview depth:

- Instruction, optional input, and target output records.
- Chat templates and exact special-token placement.
- System, user, assistant, and tool roles.
- Loss masking so only intended assistant targets contribute.
- Sequence packing and attention boundaries between packed examples.
- Truncation policy.
- Data quality, mixture weights, deduplication, and eval contamination.
- Code-fix, code-explanation, rewriting, tone, summarization, and tool-call examples.
- Catastrophic forgetting.
- Base-model perplexity against instruction-task win rate.

Know the main failures:

- Training on user tokens teaches the model to generate both sides of the conversation.
- Packing without boundaries leaks information between examples.
- A broken chat template creates a train-serve mismatch.
- Overweighting style data weakens coding or completion ability.

## 2. LoRA

Know these topics at interview depth:

- A frozen base weight plus a trainable low-rank update.
- Rank, alpha, and adapter scaling.
- Two small adapter matrices.
- Zero initialization of the second matrix so the adapter begins as a no-op.
- Target modules such as Q, K, V, output, and feed-forward projections.
- Trainable parameter and optimizer-state savings.
- Rank sweeps.
- Merged inference against unmerged hot-swappable adapters.
- Quality and memory comparison with full SFT.

Be ready to explain that LoRA saves training memory mainly by avoiding gradients and optimizer state
for frozen base parameters. It does not remove the base model from the forward pass.

## 3. Preference data

Know these topics at interview depth:

- Prompt, chosen response, and rejected response.
- Pairwise preferences against scalar ratings.
- Human, synthetic, and model-generated labels.
- Annotator agreement and ambiguous preferences.
- Correctness preferences against style preferences.
- Hard negatives and rejected-response source.
- Position and length bias.
- Training, validation, and held-out preference splits.

## 4. Direct preference optimization

Know these topics at interview depth:

- Increasing the chosen response's log-probability margin over the rejected response.
- Measuring the margin relative to a frozen reference policy.
- Beta as the preference-strength and drift control.
- Sequence log-probabilities and length effects.
- Why DPO needs no separately trained reward model or online rollout loop.
- Stability and cost advantages against PPO.
- The possibility that both chosen and rejected log-probabilities fall while the margin improves.

Log chosen probability, rejected probability, their margin, validation preference win rate, base-task
regression, and divergence from the reference. DPO loss alone is not enough.

## 5. Reward models

Know these topics at interview depth:

- A scalar reward head over a response representation.
- Pairwise ranking with reward differences.
- Why absolute reward values are not intrinsically meaningful.
- Calibration on held-out preferences.
- Position, verbosity, style, and self-preference bias.
- Reward-model overfitting.
- Distribution shift as the policy changes.
- Reward over-optimization.

Know the diagnostic plot: policy reward can keep rising while human or independent-judge quality
peaks and then falls.

## 6. Reinforcement-learning fundamentals for language models

Know these mappings:

- State is the prompt plus generated prefix.
- Action is the next token.
- Policy is the language model.
- A trajectory is the complete generated response.
- Reward often arrives after the complete response.
- Return is the accumulated reward.
- A value estimate predicts expected future reward.
- Advantage compares an action or response with a baseline.
- KL control limits drift from the reference policy.

Know why sparse end-of-sequence rewards have high variance and why verifiable tasks make the signal
more trustworthy without making it dense.

## 7. PPO

Know these topics at interview depth:

- Rollouts from an old policy.
- Probability ratios between new and old policies.
- Clipping large policy changes.
- A value model or value head.
- Advantage estimation.
- Policy loss, value loss, entropy bonus, and KL penalty.
- Policy, old policy, reference policy, and reward model memory.
- Training instability and stale rollouts.
- Reward rise with KL explosion as a failure signal.

Use PPO when the extra machinery earns a measurable gain over simpler offline preference training.

## 8. Group relative policy optimization

Know these topics at interview depth:

- Sampling a group of responses for each prompt.
- Scoring each response.
- Computing a relative group baseline.
- Normalizing rewards within the group.
- PPO-style clipped updates without a learned value model.
- KL control against a reference.
- Group size against rollout cost and variance.
- Identical group rewards producing no useful relative signal.
- Suitability for tasks with reliable automatic graders.

Be ready to compare GRPO and PPO on model memory, rollout compute, baseline quality, and stability.

## 9. RLVR

Know these topics at interview depth:

- Rewards from executable checks rather than a learned judge.
- Math answer verification.
- Code execution and hidden unit tests.
- SQL execution against expected results.
- Sandboxed graders.
- Test leakage.
- Weak tests and false positives.
- Partial credit against pass-or-fail reward.
- Reward hacking through editing tests, hard-coding cases, or exploiting the checker.

For a code-repair task, keep hidden tests outside the model's writable environment. Score only the
solution artifact. Treat any editable reward source as compromised.

## 10. Structured tool-call fine-tuning

Know these topics at interview depth:

- Tool schemas in the training data.
- Assistant tool-call messages and tool-result messages.
- Exact JSON structure and argument types.
- Valid-call rate against end-task completion.
- Negative examples with validation errors and corrections.
- Grammar-constrained decoding as a serving control.
- Separating format competence from tool-choice competence.

## Day 6 exit check

You pass Day 6 when you can explain the path from base model to SFT, LoRA, preference data, DPO,
reward modeling, PPO or GRPO, and RLVR. For each stage, state what data it needs, what objective it
changes, which eval proves a gain, and how it can damage the base model.

---

# Day 7. Evals and long context

Goal: design a small, stable evaluation system that catches quality regressions, reward hacking, and
long-context failures.

Maps to EXP-049 through EXP-053 and EXP-062.

## 1. Perplexity and bits per byte

Know these topics at interview depth:

- Held-out code and prose splits.
- Token-weighted average negative log likelihood.
- Perplexity on a fixed tokenizer.
- Bits per byte across tokenizers.
- Evaluation context length and stride.
- Domain-specific results.
- Confidence intervals or variation across seeds and samples.
- What perplexity misses: instruction following, exact code behavior, style, retrieval, safety, and
  long-range recall.

Do not collapse code and prose into one average that hides a domain regression.

## 2. Coding evals

Know these topics at interview depth:

- Function completion.
- Generation from a specification.
- Bug fixing against hidden tests.
- Explanation accuracy.
- Execution in a filesystem, process, time, memory, and network sandbox.
- `pass@1` and `pass@k`.
- Deterministic tests against stochastic sampling.
- Contamination checks.
- Tiny stable regression set against larger periodic benchmark.
- Confidence intervals on small task counts.

A three-point gain on forty tasks is not automatically a real improvement. Report uncertainty and
inspect which tasks changed.

## 3. Writing evals

Know these topics at interview depth:

- Pairwise preference win rate.
- Style match to a fixed profile.
- Instruction and tone following.
- Repetition and n-gram duplication.
- Factual preservation during rewriting.
- Unsupported additions.
- Fixed judge prompts.
- Position, length, and self-preference bias.
- Swapping response order.
- Human spot checks.

The judge is an instrument that needs calibration. It is not ground truth.

## 4. Regression methodology

Know these topics at interview depth:

- Versioned and hashed evaluation data.
- Fixed prompts and scoring code.
- Seed policy.
- Baseline artifacts.
- Thresholds that fail a run.
- Practical significance against statistical noise.
- Per-domain gates.
- Runtime budget for per-commit checks.
- Larger scheduled evaluations.
- Machine-readable JSON output.

The gate must say what fails, by how much, against which baseline, and under which evaluation
version.

## 5. Reward-hacking catalog

Keep each observed failure with these fields:

- Reward or evaluator rule.
- Exploit the model found.
- Example output.
- Why the evaluator accepted it.
- Independent quality result.
- Fix to the reward, sandbox, prompt, or test.
- Regression case added.

Expected cases include deleted tests, hard-coded outputs, verbosity gaming, judge-style imitation,
empty answers that exploit parsing, and leaked evaluation solutions.

## 6. Long-context evaluation

Know these topics at interview depth:

- Trained context against supported and tested context.
- RoPE extrapolation and interpolation.
- Perplexity as a function of sequence length.
- Needle recall as a function of needle position and context length.
- Copy probes against instruction-following needle tasks.
- Full, sliding-window, and compressed attention comparisons.
- Retrieval accuracy at 8K and 16K.
- Lost-in-the-middle behavior.
- Cache bytes and latency at long context.
- Quality slope, not only the maximum length that runs.

Flat perplexity with falling needle recall means average local prediction remains good while access
to distant information has degraded.

## 7. The evaluation command

Be ready to specify `scripts/evaluate.py`:

- Inputs: model, tokenizer, config, checkpoint, eval version, task selection, seed, device, and output
  path.
- Outputs: per-task records, aggregate scores, uncertainty, runtime, memory, artifact hashes, and
  environment details.
- Failure rules: maximum perplexity or bits-per-byte regression, minimum code-fix score, minimum
  writing preference score, and long-context recall floor.
- Exit status: nonzero when a required gate fails.

## Day 7 exit check

You pass Day 7 when you can design the project's stable eval set, defend each metric, explain the
judge controls, define regression thresholds, and diagnose a long-context model whose perplexity is
stable but recall is not.

---

# Day 8. Assistant, retrieval, tools, caching, routing, and hardening

Goal: explain how retrieval, software tools, caching, and routing make a weak local model useful and
how the system stays correct after files and models change.

Maps to EXP-054 through EXP-061, EXP-063, and EXP-064. Mamba, SSM, and multimodal stay in the
experimental or deferred architecture track.

## 1. Intent classification and routing inputs

Know these topics at interview depth:

- Trivial, coding, and hard request classes.
- A labeled set of real project queries.
- Precision and recall per route.
- Cost of routing a hard task locally against routing an easy task externally.
- Tool availability as a feature.
- Confidence thresholds and abstention.
- Post-answer validation and fallback.
- Classifier drift as the request mix changes.

Bias the classifier according to the more expensive error, not toward a cosmetically balanced
confusion matrix.

## 2. RAG fundamentals

Know these topics at interview depth:

- Parametric knowledge against retrieved external knowledge.
- Retrieval and generation as separate stages with separate failure modes.
- Grounding answers in repository sources.
- Citations to exact files and line locations.
- Why retrieval helps a small model more than adding broad facts to its weights.
- What RAG cannot fix: weak reasoning, a bad instruction policy, or unsafe tool execution.
- No-retrieval, BM25-only, and hybrid retrieval ablations.

When an answer is wrong, first ask whether the correct evidence was retrieved. If it was, diagnose
generation or context construction. If it was not, diagnose indexing, retrieval, or reranking.

## 3. Indexing and chunking

Know these topics at interview depth:

- File discovery and language detection.
- Content hashes.
- Stable chunk IDs.
- Path, symbol, language, imports, and line-range metadata.
- Token-based prose chunks.
- Function, method, class, and module chunks for code.
- AST-aware boundaries.
- Overlap and index-size cost.
- Keeping signatures with bodies.
- Incremental reindexing after a file edit.
- Deduplication.
- Deleted and renamed files.

Fixed-size chunks often separate a function name from its body. Code-aware chunks preserve the unit
that both lexical and semantic retrieval need.

## 4. Lexical, dense, and structural retrieval

Know BM25:

- Term frequency.
- Inverse document frequency.
- Term-frequency saturation.
- Document-length normalization.
- Exact identifier, error string, filename, and configuration-key matches.

Know dense retrieval:

- A retrieval embedding model distinct from the language model's token embeddings.
- Vector normalization.
- Cosine similarity against dot product.
- Embedding dimension, index memory, and query latency.
- Semantic matches that use different words.

Know AST and symbol retrieval:

- Definitions, references, calls, imports, classes, and functions.
- Symbol lookup and caller discovery.
- Static-analysis limits under dynamic dispatch, reflection, and generated code.

Know why the three methods complement one another. BM25 finds exact names, dense retrieval finds
semantic similarity, and AST search finds program structure.

## 5. Approximate nearest-neighbor search

Know these topics at interview depth:

- Exact vector search as the baseline.
- When brute force is sufficient for a small repository.
- HNSW graph search.
- Recall-latency and build-memory tradeoffs.
- Search breadth controls.
- Inverted-file and product-quantization families at a high level.
- Why an ANN dependency is unnecessary until exact search becomes a measured bottleneck.

## 6. Hybrid retrieval and reranking

Know these topics at interview depth:

- Candidate generation from BM25, embeddings, and AST search.
- Deduplication by content or stable chunk ID.
- Reciprocal rank fusion when retriever score scales differ.
- First-stage recall.
- A cross-encoder or model-based reranker.
- Query-document joint scoring.
- Reranking latency inside the time-to-first-token budget.
- Keeping a small final set of high-value chunks.
- Retrieval ablations that remove one source at a time.

Be ready to trace a query such as "why does auth middleware return 401" through every retriever,
the fusion step, the reranker, and the final context.

## 7. Context construction

Know these topics at interview depth:

- Total context-window budget.
- System prompt, conversation history, retrieved evidence, tool results, and output reserve.
- Counting with the serving tokenizer.
- Relevance, recency, source diversity, and dependency relationships.
- Deduplication.
- Chunk ordering and lost-in-the-middle.
- Context compression.
- File and line citations.
- Truncating tool output.
- Refusing unsupported claims when evidence is absent.

Top-k is not a fixed magic number. It follows from the token budget, average chunk size, evidence
quality, and output reserve.

## 8. Tool calling

Know these topics at interview depth:

- JSON Schema tool definitions.
- Required and optional arguments.
- Type, path, and enum validation.
- Tool call, validation, execution, result, and model continuation.
- Returning validation errors as bounded observations.
- Constrained decoding for valid structure.
- Retry and step limits.
- Parallel calls only when operations are independent.
- Output-size limits.
- Separating valid-call rate from successful task completion.

Know the project tool set:

```text
read_file   write_file   grep   glob
git_diff    git_status   run_tests
python      shell
```

For every tool, know whether it is read-only, what it can damage, what permissions it needs, and
what limits its filesystem, network, CPU, memory, and wall-clock access.

## 9. Agent loop and failure recovery

Know the loop:

```text
classify -> retrieve -> inspect -> act -> observe -> revise -> verify -> answer
```

Know these topics at interview depth:

- State carried between tool steps.
- Planning and replanning.
- Tool observations as untrusted input.
- Termination on success, step limit, time limit, or repeated state.
- Repeated-action detection.
- Invalid JSON and invalid arguments.
- Hallucinated paths and symbols.
- Context growth.
- Misread test failures.
- Partial edits.
- Verification after a patch.
- Returning a clear blocker when the task cannot be completed safely.

A coding loop is incomplete until it inspects the diff and runs the smallest relevant verification.

## 10. Coding and writing paths

Know the coding path:

```text
reproduce -> search -> read callers -> inspect symbols -> patch root cause
-> run focused test -> inspect diff -> run broader checks -> report
```

Know the writing path:

```text
identify task -> retrieve target-style examples -> draft -> check factual preservation
-> check tone and repetition -> revise once -> return
```

Know how to measure how much value comes from the model, retrieval, tools, and critic pass. Use
ablations rather than intuition.

## 11. Cache stack and invalidation

Know these cache layers separately:

- Response cache.
- Retrieval-result cache.
- Embedding cache.
- Prefix cache.
- KV cache.

For each cache, know:

- The cached value.
- The exact key.
- The scope and lifetime.
- The memory cost.
- The invalidation event.
- The correctness effect of a stale hit.
- The hit-rate metric.

Know the file-edit chain:

```text
file content changes
-> old embedding is stale
-> retrieval results that reference it are stale
-> built contexts that include it are stale
-> response-cache entries that use those contexts are stale
-> reusable prefixes containing the old content are stale
```

Content-hash keys make many old entries unreachable without an unsafe timestamp-based purge. The
per-request KV cache is discarded or updated according to conversation and prompt changes.

## 12. Model routing and cost

Know these topics at interview depth:

- Routing trivial tasks to the local model.
- Routing coding tasks to the local model with retrieval and tools.
- Routing hard or low-confidence tasks to an external API.
- Pre-answer classification against post-answer validation.
- Fallback after failed execution or low-confidence evidence.
- Local quality against external quality.
- Privacy and offline requirements.
- Route-level latency and cost.

Track these metrics:

- Requests by route.
- Local completion rate.
- External fallback rate.
- Tokens avoided.
- Cache hit rate by layer.
- Time to first token.
- Local decode speed.
- External spend.
- Estimated spend without routing.
- Quality by route.

Report savings only alongside quality. A cheaper wrong answer is not a saving.

## 13. Observability and stability

Know these topics at interview depth:

- Structured per-request logs.
- Prompt and output token counts without storing sensitive text by default.
- Route decision and confidence.
- Retrieval candidates and selected sources.
- Tool calls, durations, exit status, and truncated errors.
- Cache hits and misses.
- Time to first token and inter-token latency.
- Memory, queue depth, and throughput.
- Crash logs and request identifiers.
- Memory leaks from unreleased KV slots, histories, caches, and tensors.
- Latency drift over a 24-hour serve run.
- Client retry and server recovery after a crash.

Define a stability pass with bounded memory growth, bounded latency drift, no lost persistent state,
and successful recovery from controlled failures.

## 14. Model card and reproducibility

Know these required sections:

- Architecture, parameter count, and context length.
- Tokenizer and hash.
- Dataset sources, composition, sizes, hashes, licenses, and deduplication.
- Training hardware, precision, steps, tokens, and optimizer.
- Post-training data and methods.
- Quantization variants.
- Evaluation version and results.
- Intended uses.
- Unsupported uses.
- Known failures and limitations.
- Security and privacy boundaries.
- Exact reproduction commands and artifact hashes.

Record dataset origin and license at collection time. You often cannot reconstruct those facts later.

## 15. SSM, Mamba, and multimodal boundaries

Know these topics at interview depth:

- State-space sequence models as recurrent state updates with parallel scan during training.
- Constant-size recurrent state during inference instead of a growing KV cache.
- Selective, input-dependent state updates in Mamba-style models.
- Long-sequence efficiency against weaker direct token-to-token access.
- Hybrid attention-SSM models.
- A multimodal system using an encoder and projection into the language-model representation.

Know the scope decision:

- The dense decoder remains the core.
- Mamba and SSM stay in an experimental architecture track.
- Multimodal training is a separate project because it adds data, encoders, alignment, and evaluation,
  not one decoder component.

## 16. Final system design

Know the complete request path:

```text
user request
-> intent classifier
-> local model, local model with tools and retrieval, or external API
-> retrieval and context construction when needed
-> tool loop when needed
-> response validation
-> answer with cited sources
-> cache and cost updates
-> evaluation and observability records
```

Be ready to defend every component on:

- Data flow.
- Latency budget.
- Memory and cache cost.
- Quality measurement.
- Failure recovery.
- Security boundary.
- Invalidation behavior.
- Scaling limit.
- Build-versus-buy decision.

## Day 8 exit check

You pass Day 8 when you can trace one repository question and one failing-test request through
classification, retrieval, context construction, tools, caching, routing, validation, and logging.
Explain every invalidation after a file edit and every fallback after a failed local answer.

---

# Coverage against `day-wise.md`

| Roadmap day | Required interview areas | Experiments |
| --- | --- | --- |
| Day 1 | Language modeling, tokenizer, embeddings, positions, MHA, causal masking, baseline block, training loop | EXP-003 to EXP-008 |
| Day 2 | RoPE, RMSNorm, SwiGLU, MHA, MQA, GQA, residual placement, SDPA, FlashAttention | EXP-009 to EXP-015 |
| Day 3 | MTP, sparse attention, compressed attention, MLA, MoE, asymmetric compute, Muon, mHC | EXP-016 to EXP-025 |
| Day 4 | Mixed precision, batching, schedules, recovery, scaling, DDP, FSDP, tensor and pipeline parallelism | EXP-026 to EXP-034 |
| Day 5 | Naive decode, KV cache, prefill, compile, quantization, batching, prefix cache, speculative decoding | EXP-035 to EXP-042 |
| Day 6 | SFT, LoRA, preferences, DPO, reward models, PPO, GRPO, RLVR, tool-call SFT | EXP-043 to EXP-048 and EXP-058 |
| Day 7 | Perplexity, coding and writing evals, regression gates, reward hacking, long context | EXP-049 to EXP-053 and EXP-062 |
| Day 8 | Classifier, hybrid retrieval, tools, agent loop, writing path, caches, routing, cost, hardening | EXP-054 to EXP-061, EXP-063, EXP-064 |

No roadmap topic moves earlier than its prerequisite. The dense decoder and rough evals exist before
post-training. Post-training and evals exist before the assistant.

# Final A+ interview check

Prepare these project stories with real numbers from the day notes:

1. Why byte BPE beat the character baseline even though its raw token perplexity was higher.
2. How a bad embedding initialization blocked the overfit check and how the measured failure led to
   the fix.
3. Why SwiGLU was the only isolated Day 2 quality win outside seed noise.
4. Why the project kept two KV heads even though one KV head looked equally good in a short run.
5. Why post-norm was reverted.
6. Why the CPU flash result is evidence about memory behavior but not a GPU speed claim.
7. Why RoPE interpolation helped only at 8192 in the measured length sweep.
8. How a longer training budget establishes a noise floor before Day 3 architecture comparisons.
9. How you would choose between DDP, FSDP, tensor parallelism, and pipeline parallelism.
10. How you would optimize laptop decode and prove that each step is a net win.
11. How you would detect preference over-optimization and reward hacking.
12. How one file edit invalidates every dependent retrieval and cache artifact.

For every story, use this order:

```text
problem -> baseline -> change -> measurement -> result -> decision -> limitation
```

You are ready when you can answer follow-up questions without changing the conditions of the result,
inventing a measurement, or hiding a failed experiment.
