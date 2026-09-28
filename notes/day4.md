# Day 4: train the from-scratch model properly

Status: code built, nothing run.
Started: 2026-09-23
Experiments: EXP-065 through EXP-068

## What we need to finish

Day 4 ends when one model of about 26M parameters has trained on TinyStories to a flat validation
loss, and ten fixed prompts produce stories a reader judges coherent. `PLAN.md` explains why this
replaces the rest of the first plan.

## Where we start

Nothing trained carries over. The user moved to a new Colab account with a Pro subscription on
2026-09-23, and the old account's Drive files are gone. Nothing needs retraining for that reason.
The Day 2 corpus and grid belong to the first plan, Day 3a never ran, and Day 5 remeasures the
comparisons at the new scale. The Day 2 numbers stay in `notes/day2.md` as the record.

The new Colab Pro account offers two accelerators: a T4 GPU and a TPU v5e-1. No L4 or A100. We
train on the T4, which has no bfloat16, so the trainer uses float16 with `GradScaler`. Every
training record carries the `dtype` that ran.

We rejected the TPU for Day 4. PyTorch reaches a TPU through `torch_xla`, and octlm's device
selection, autocast, `GradScaler`, SDPA kernel choice, and checkpoint saving are all written for
CUDA or CPU. The port would cost more time than it saves on a run estimated at 70 to 180 minutes,
and no later day needs a TPU. On the T4, SDPA uses the memory-efficient kernel, because
FlashAttention needs an sm80 or newer GPU.

## What we read

- [TinyStories on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories), fetched
  2026-09-23. The V2 files are `TinyStoriesV2-GPT4-train.txt` at 2,227,753,162 bytes and
  `TinyStoriesV2-GPT4-valid.txt` at 22.5 MB. The license is CDLA-Sharing-1.0. A line holding only
  `<|endoftext|>` separates stories, and blank lines sometimes follow it. The validation file
  begins in the middle of a story. Decision: split on the delimiter line, strip surrounding
  newlines from each story, drop empty stories, and keep the truncated first story, because
  dropping it would be a special case for one story out of thousands.
- [Eldan and Li, TinyStories](https://arxiv.org/abs/2305.07759), not yet reread for this day. It
  remains the lead from `notes/day3.md` that models under about 30M parameters write coherent
  stories on this data. EXP-067 tests it directly.

## Engineering decisions

**Vocabulary 8,192, without the 4,096 comparison.** `PLAN.md` asked to pick between the two by bits
per byte. Bits per byte is a property of a trained model, so the comparison costs two full runs.
The TinyStories authors used a 10K vocabulary. We take 8,192 and record the compression it reaches.
If Day 5 has spare GPU time, a 4,096 run at the EXP-069 budget answers the question.

**The tokenizer trains on the first 10M characters of the train split.** `ByteBPETokenizer.train`
recounts every pair after every merge, so its cost grows with the sample and the merge count. The
TinyStories vocabulary is small, so 10M characters should cover it. The prepare log records the
training seconds.

**The encoder caches chunk IDs.** BPE encodes each pre-token on its own, so a pre-token always maps
to the same IDs. `ByteBPETokenizer.encode` now keeps a dictionary from chunk bytes to IDs. The
output does not change. EXP-065 measures the speed with and without the cache.

**Token files are flat `uint16`.** Each story is encoded with BOS and EOS and appended to one
stream. Training reads random 513-token windows from a `numpy.memmap`, so a window can span a
story boundary with EOS and BOS between the two stories. BPE merges still never cross a story.

**Validation is the first 256 non-overlapping 512-token blocks of the validation split.** That is
131,328 tokens. `train.evaluate` computes bits per byte from a per-token byte table.

**The trainer writes a checkpoint at every eval.** A Colab disconnect loses at most 1,000 steps.
The checkpoint stores the `GradScaler` state. `octlm.day4 train` resumes when the checkpoint file
exists.

**Mixed precision is opt-in inside `train_model`.** Days 1 to 3 keep float32 on every device, so
their numbers stay comparable. Day 4 turns it on.

## EXP-065: corpus and tokenizer

Problem. The Day 2 corpus had 2.96M training tokens. The Day 4 model needs about 400M.

Hypothesis. The pure-Python encoder is too slow to encode 2.2 GB in one session. The chunk cache
fixes the merge cost. The per-character pre-tokenizer stays the bottleneck.

Baseline. `encode_chunk` on every chunk of the first 2M characters of the validation split, no
cache.

Measurement. The `encode_benchmark` record in `data/tinystories/prepare.jsonl` gives bytes per
second with and without the cache. The `encode` records give tokens, bytes per token, and seconds
per split. `artifacts/day4/bpe.jsonl` gives the tokenizer training time.

Stop condition. If the train split takes longer than two hours to encode, stop and speed up the
pre-tokenizer before training anything.

### Result

Ran on Colab, 2026-09-23. From `prepare.jsonl` and `bpe.jsonl`:

| Measure | Value |
| --- | --- |
| Tokenizer sample | 10,000,355 characters, 12,328 stories |
| Tokenizer training | 303 s |
| Encode without cache | 427 KB/s |
| Encode with cache | 2,559 KB/s, 6.0x faster |
| Train split | 2,717,495 stories, 966,389,646 tokens, 934 s |
| Valid split | 27,630 stories, 9,759,613 tokens, 10 s |
| Bytes per token | 2.262 on both splits |

The first hypothesis is false. With the cache, the train split encoded in 15.6 minutes, well under
the two-hour stop condition. The cache is the fix, so keep it. The pre-tokenizer cost was not
measured separately, so the second half of the hypothesis stays open. The train split holds 966M
tokens, 2.5 times the 393M the main run consumes, so EXP-067 sees each token at most once.

## EXP-066: mixed precision

Problem. Float32 wastes most of the GPU's matrix throughput.

Hypothesis. Mixed precision at least doubles tokens per second, and its validation loss at step
1,000 stays within 0.02 of float32.

Baseline. The same 1,000 steps with `--full-precision`.

Measurement. `train_seconds` and `validation_loss` at step 1,000 in `day4-precision.jsonl`, with
the `dtype` and `device` of each run.

Stop condition. If the gap is larger than 0.02, train the main run in float32 and record why.

### Result

First attempt, 2026-09-23, invalid. `autocast_dtype` chose bfloat16 because
`torch.cuda.is_bf16_supported()` returns `True` on the T4, but the T4 (compute capability 7.5) has
no bfloat16 tensor cores and emulates it. The "mixed" run took 1,529 s against 1,173 s for float32,
with validation loss 1.259 against 1.250. Fix: choose bfloat16 only at compute capability 8.0 or
higher, otherwise float16 with `GradScaler`. The float16 rerun appends to `day4-precision.jsonl`.

Float16 rerun, 2026-09-23, at step 1,000:

| dtype | train_seconds | s/step | validation loss | bits per byte |
| --- | --- | --- | --- | --- |
| float32 | 1,173 | 1.17 | 1.2500 | 0.8001 |
| bfloat16 (emulated) | 1,529 | 1.53 | 1.2590 | 0.8058 |
| float16 | 625 | 0.62 | 1.2513 | 0.8009 |

Float16 is 1.88 times faster than float32, short of the hypothesis's 2x. The loss gap is 0.0013,
far inside the 0.02 stop condition. Keep float16 for the main run.

Measured throughput is 16,384 tokens / 0.625 s, about 26,200 tokens per second, or about 4.1
TFLOPs at 6 x 26.3M FLOPs per token. EXP-067 assumed 15 TFLOPs. The 24,000-step run therefore
takes about 4.2 hours, not 70 minutes.

## EXP-067: the main run

Problem. Every model so far was too undertrained to read an architecture effect or produce text.

Hypothesis. About 26M parameters trained on about 393M tokens writes coherent short stories.

Setup. `configs/day4.toml` is the Day 2 `modern` stack: RoPE, RMSNorm, SwiGLU, and two KV heads,
through SDPA. Width 512, 8 layers, 8 query heads, 512-token context, vocabulary 8,192. The dry run
reports 26,255,872 parameters. 24,000 steps of 32 x 512 tokens is 393,216,000 tokens, about 15
tokens per parameter. The learning rate schedule is the Day 3 one with a longer warmup: peak 6e-4,
floor 6e-5, 500 warmup steps, cosine decay.

Estimate. 6 x 26.3M x 393M is about 6.2e16 FLOPs. At an assumed 15 TFLOPs on a T4 that is about 70
minutes, before attention and data-loading overhead. EXP-066 gives the real rate.

Baseline. None at this scale. The exit check is absolute: a flat validation curve and readable
samples.

Stop condition. If validation loss stops improving for 3,000 steps, stop the run and keep the best
checkpoint. If the loss diverges, halve the learning rate and restart.

### Result

Ran on Kaggle, `octlm-day4` version 4, Tesla T4, PyTorch 2.10.0+cu128, float16. The notebook took
15,852 s end to end. Training took 14,733 s for 24,000 steps, 0.614 s per step against 0.625 s on
Colab. From `metrics.jsonl`:

| Step | Train loss | Validation loss | Bits per byte | Elapsed s |
| --- | --- | --- | --- | --- |
| 1,000 | 1.2874 | 1.2513 | 0.8009 | 608 |
| 3,000 | 0.9606 | 1.0070 | 0.6445 | 1,836 |
| 6,000 | 0.8399 | 0.9180 | 0.5876 | 3,679 |
| 9,000 | 0.8298 | 0.8695 | 0.5565 | 5,522 |
| 12,000 | 0.8413 | 0.8375 | 0.5361 | 7,365 |
| 15,000 | 0.7696 | 0.8069 | 0.5165 | 9,208 |
| 18,000 | 0.7757 | 0.7816 | 0.5003 | 11,049 |
| 21,000 | 0.6988 | 0.7648 | 0.4895 | 12,890 |
| 24,000 | 0.7296 | 0.7561 | 0.4839 | 14,733 |

Final validation loss is 0.7561, perplexity 2.13, and 0.4839 bits per byte on 131,072 held-out
tokens. The step-1,000 validation loss, 1.2513, matches the Colab float16 run to four places, so the
Kaggle path reproduces the Colab one.

The curve has flattened but not stopped. The last four 1,000-step gains are 0.0055, 0.0045, 0.0025,
and 0.0017, while the learning rate sat near its 6e-5 floor. The 3,000-step stop condition never
fired, and the loss never diverged. More tokens would still help a little. The model saw 393M of the
966M encoded training tokens, so a longer run needs no new data. Keep this checkpoint as the Day 4
model.

Kaggle prepare timings: tokenizer training 238 s, encode without cache 462 KB/s, with cache
2,909 KB/s, train split in 738 s against 934 s on Colab. Token counts match Colab exactly.

## EXP-068: sampling

Problem. Greedy decoding on story data repeats phrases, which would make the exit check judge the
decoding instead of the model.

Change. `Decoder.generate` takes `temperature`, `top_k`, and a `torch.Generator`. Temperature 0
keeps greedy decoding, so earlier days' output does not change.

Measurement. `octlm.day4 samples` writes the ten prompts in `day4.PROMPTS`, once greedy and once at
temperature 0.8 with top-k 40, seed 0. Paste both sets into this note.

### Result

Ran on Kaggle after the main run, from `run.pt` at step 24,000. Each sample is cut at the
generation length, so most stop mid-sentence.

My reading, confirmed by the user on 2026-09-28: all twenty are grammatical, stay on the prompt's characters,
and follow a story shape with a problem and a turn. The errors are logic slips, not broken text.
Greedy repeats itself ("a necklace, a necklace, a ring, and a necklace"; the cat dances twice).
Sampling removes the loops but adds odder events ("the dog ate the cat", "Lily thanked Lily", Spot
talks and drives). Both sets pass as coherent for a 26M TinyStories model. Keep temperature 0.8 and
top-k 40 as the default.

#### Greedy

1. Prompt `Once upon a time, there was a little girl named Lily.`

```text
Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat yummy food. One day, Lily found a big box in her room. She was very excited to see what was inside.
Lily opened the box and saw a lot of yummy food. There were apples, bananas, and carrots. She wanted to eat all of them. But her mom said, "No, Lily, you must share with your brother." Lily did not want to share with her brother. She wanted all the food for herself.
Lily was sad and started to cry. Her mom
```

2. Prompt `One day, a boy named Tom found a big red ball`

```text
One day, a boy named Tom found a big red ball. He loved to play with it. He would throw it, kick it, and roll it around. Tom was very happy with his new ball.
Tom saw a girl named Sue. Sue was sad. She had no toys to play with. Tom wanted to help Sue. He said, "Sue, do you want to play with my ball?" Sue smiled and said, "Yes, please!"
Tom and Sue played with the ball together. They had a lot of fun. They laughed and ran around. They became good friends. Tom was not sad
```

3. Prompt `The cat was very hungry.`

```text
The cat was very hungry. He wanted to find some food. He looked around and saw a big bowl of food on the table. He jumped up and grabbed the bowl. He ate until he was full.
The cat was so happy that he started to dance. He danced and danced until he was too tired to dance. He was so happy that he had found food.
The cat was so happy that he started to dance again. He danced and danced until he was tired. Then he went to sleep and had sweet dreams.
```

4. Prompt `Sam wanted to go to the park, but`

```text
Sam wanted to go to the park, but Mom said no. She said they had to stay home and play with their toys. Sam was sad and angry. He did not want to stay home and play with his toys.
"Please, Mom, please!" Sam begged. He tried to resist, but Mom was stronger. She took him by the hand and led him to the car. Sam was happy and excited. He wanted to see the park and the swings and the sandbox.
But when they got to the car, Sam saw something else. He saw a big dog running towards
```

5. Prompt `Mom said, "`

```text
Mom said, "We need to go to the store and get some food." Tom and Lily were sad. They wanted to play outside with their friends. They asked Mom, "Can we go to the store and buy some food?"
Mom said, "No, you have to stay in the car. The car is not for you. The car is for the store. You can play with your toys in the yard. But you have to be good and listen to me."
Tom and Lily did not listen. They waited until Mom was busy in the car
```

6. Prompt `There was a small bird who could not fly.`

```text
There was a small bird who could not fly. He was sad and wanted to fly high in the sky. One day, he saw a big tree with a lot of leaves. He thought it would be fun to play in the leaves.
The small bird flew to the tree and started to play. He was having so much fun. But then, he saw a big cat coming near him. The cat wanted to catch the small bird. The small bird was scared and did not know what to do.
The small bird had an idea. He flew to the big cat and
```

7. Prompt `Tim and his dog went on an adventure`

```text
Tim and his dog went on an adventure. They walked to a big park with many trees and flowers. Tim saw a big tree with a hole in it. He wanted to see what was inside.
"Come on, dog, let's go in the tree!" Tim said. He ran to the tree and looked inside. He saw a lot of leaves and sticks. He also saw a big hole in the tree. He thought it was a good place to hide.
"Wait, Tim, don't go!" his mom said. She was busy in the kitchen. She was making
```

8. Prompt `It was a rainy day and`

```text
It was a rainy day and Tom was outside playing in the rain. He was having so much fun that he didn't notice the rain was getting heavier and heavier. Suddenly, he heard a loud noise and he got scared. He looked around and saw a big truck coming down the street. It was carrying a big box of toys. Tom was so excited that he ran to the truck and asked the driver what was in the box. The driver said it was a surprise for him. Tom was so excited that he jumped up and down and asked the
```

9. Prompt `Anna had a secret.`

```text
Anna had a secret. She had a big box in her room. She wanted to see what was inside. She opened the box and saw a lot of shiny things. She saw a necklace, a necklace, a ring, and a necklace. She wanted to try them on.
She put on the necklace and the necklace. She felt very happy. She looked at herself in the mirror. She smiled. She felt like a princess. She wanted to show her mom. She ran to the kitchen. She said, "Mom, look at me! I am a princess!"
```

10. Prompt `The big tree in the garden`

```text
The big tree in the garden was very tall. It was so tall that it could touch the sky. One day, a little girl named Lucy was playing in the garden. She saw the big tree and wanted to climb it.
Lucy started to climb the tree. She went up and up. When she got to the top, she saw a big bird. The bird was very pretty. Lucy wanted to be friends with the bird. She said, "Hi, bird! I like your big tree. Can we be friends?"
The bird looked at Lucy and said, "
```

#### Temperature 0.8, top-k 40, seed 0

1. Prompt `Once upon a time, there was a little girl named Lily.`

```text
Once upon a time, there was a little girl named Lily. She loved to play with her mom's makeup. One day, Lily found a big box of makeup in the makeup. She was very happy.
Lily put on the makeup and went outside. She saw her friend, Tom, playing with a ball. Tom saw Lily and said, "You look funny, Lily!" Lily smiled and said, "You look silly, Tom!"
Lily and Tom played all day with the makeup. They had a lot of fun. When they were tired, they went back to their homes. When they got home, Lily
```

2. Prompt `One day, a boy named Tom found a big red ball`

```text
One day, a boy named Tom found a big red ball. He was very happy. Tom loved to play with the ball. His mom saw him and said, "Tom, mind your toys and don't play with them."
But Tom did not listen. He took the ball outside to play. The sun was hot and it began to melt the ball. Tom was sad because he lost his ball. He wanted to play with the ball, but he did not want to lose his ball.
Later, Tom found the ball. It was a big rock. He was happy to find it. He
```

3. Prompt `The cat was very hungry.`

```text
The cat was very hungry. He was looking for something to eat. He looked around and saw a can of tuna. He was so happy. He grabbed the can with his paws and started eating.
But then a big dog came out of nowhere. The dog was a threat and it barked loudly. The cat was scared.
He tried to run away, but the dog was too fast. He jumped on the cat and gave it a big kiss.
The cat could not escape. But the dog was still hungry. The dog ate the cat until there was no
```

4. Prompt `Sam wanted to go to the park, but`

```text
Sam wanted to go to the park, but Mom said no. She said there was a big car that was not fast or slow. She said Sam had to wait until Mom was not looking. Sam was sad, but he did what Mom said.
He went to the car with Mom and they drove to the park. They saw the store and the swings and the benches. They also saw some other kids and cars that were fast and loud. Sam thought they looked fun, but he did not want to sit and wait.
They played on the swings and the slide
```

5. Prompt `Mom said, "`

```text
Mom said, "Come on, let's go to the park!" Ben and Lily ran to the door. They wanted to play on the swings and the slide. But Mom was busy in the kitchen. She was working on a big computer.
"Come on, kids, we have to go now," Mom said. "You have to stay here and be good. I will be back soon."
Ben and Lily did not want to go to the park. They wanted to stay and play. They said, "No, Mom, we have to stay here and be
```

6. Prompt `There was a small bird who could not fly.`

```text
There was a small bird who could not fly. He was sad. He wanted to fly high in the sky, but he was too small. He tried and tried, but he could not reach the high clouds.
One day, the bird saw a big storm coming. The wind was very strong. The bird was scared. He tried to fly high to hold on to the lightning. But he could not fly.
A kind girl saw the bird. She wanted to help him. She took him home. The bird was happy. He should be with his friends in the sky.
The bird
```

7. Prompt `Tim and his dog went on an adventure`

```text
Tim and his dog went on an adventure to the park. They liked to play on the swings, the slide and the seesaw. They had a lot of fun together. But after a while, Tim felt sleepy. He yawned and rubbed his eyes.
"Come on, Tim, let's go home," Spot said. He put his hand in his pocket and ran towards the car. He was too sleepy to drive the car with his dog.
Tim did not give up. He got in his car and drove to the park. He saw a big truck with a lot of people
```

8. Prompt `It was a rainy day and`

```text
It was a rainy day and Mommy and Daddy were out in the garden playing. Daddy was playing, and Mommy was having fun. She was running and jumping around. Then Daddy stopped and said, "We need to hurry for dinner. We must come home soon."
Mommy and Daddy started to walk to the house. But suddenly, the door slammed shut and they couldn't get in. Mommy and Daddy were stuck. They were sad and upset.
Mom said, "We need to get the food out of the door!" Daddy got out of the house and said, "Let
```

9. Prompt `Anna had a secret.`

```text
Anna had a secret. She had a big white box that she kept inside with her. She called it a "supply". She kept it in a special box and kept it safe.
One day, she invited her friend Lily to come over and play with her. Lily came and saw the box. She was very happy and excited. She asked Lily to look inside.
Lily opened the box and took out many toys. She said thank you to Anna and gave her a doll. Lily was very happy and thanked Lily.
But then, Lily saw a big
```

10. Prompt `The big tree in the garden`

```text
The big tree in the garden had lots of beautiful birds, and a small pond was a big lake. Every morning the birds would sit and sing together in the pond's cool shade.
One day, a little girl went to the pond and called out to the birds. “Hey, bird! Do you want to play with me?” she asked.
The birds flew down and the girl was so excited. She ran to the edge of the lake and picked up a stick. Then she ran back to the tree and waved it in the air.
The bird waved
```

## What changed in the code

- `octlm/day4.py` has the `prepare`, `train`, and `samples` stages.
- `octlm/tokenizer.py` gained `encode_chunk` and `chunk_cache`. `encode` uses both.
- `octlm/model.py` gained `sample_token` and the sampling arguments on `generate`.
- `octlm/train.py` gained `autocast_dtype`, the `mixed_precision` argument, a checkpoint at every
  eval, and the scaler state in the checkpoint.
- `configs/day4.toml` is new. `notebooks/octlm-colab.ipynb` now runs Day 4 only.

No tests were added and the suite was not run, per the user's standing instruction. `AGENTS.md`
asks for both. That conflict is open.

## Commands

On Colab, in notebook order:

```sh
python -m octlm.day4 prepare
python -m octlm.day4 train --stop-after 1000 --checkpoint /content/mixed.pt ...
python -m octlm.day4 train --stop-after 1000 --full-precision --checkpoint /content/full.pt ...
python -m octlm.day4 train --checkpoint $DRIVE/run.pt --metrics $DRIVE/day4.jsonl ...
python -m octlm.day4 samples --checkpoint $DRIVE/run.pt --temperature 0
python -m octlm.day4 samples --checkpoint $DRIVE/run.pt
```

## Exit check

- [x] Encode throughput with and without the chunk cache is recorded.
- [x] Mixed precision has a speed and loss comparison against float32.
- [x] Validation loss has flattened, and the curve is in this note.
- [x] Bits per byte on the validation blocks is recorded.
- [x] Ten greedy and ten sampled stories are in this note, with a coherence judgment.

## Session on 2026-09-25: move the main run to Kaggle

Problem. The Colab notebook depends on `/content`, a Drive mount, and an interactive run. It cannot
run as a Kaggle notebook. EXP-067 is still the lowest unfinished experiment.

Hypothesis. Kaggle's T4 x2 option can run the unchanged Day 4 model with its measured float16 path.
One full notebook version should finish inside Kaggle's 12-hour GPU limit. The Colab T4 baseline is
0.625 seconds per step, or about 4.2 hours for 24,000 steps, before Kaggle-specific overhead.

Baseline. EXP-066's Colab T4 speed and validation loss. No Kaggle speed or loss has been measured.

Measurement. Record the Kaggle GPU name, PyTorch version, elapsed time, validation loss, and bits per
byte from the first and final `metrics.jsonl` records. Compare the first Kaggle interval with the
Colab T4 baseline before using the four-hour estimate.

Stop condition. If Kaggle does not offer a T4, the first interval is much slower than the Colab
baseline, or the notebook cannot finish inside 12 hours, stop and choose the next run length from
measured Kaggle throughput. Do not change the model or call the Colab timing a Kaggle result.

### Sources and decisions

- [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks), checked 2026-09-25,
  says a GPU session has a 12-hour limit, `/kaggle/working` saves up to 20 GB of notebook output,
  and `Save & Run All` starts a clean session. Decision: the Kaggle notebook runs preparation,
  training, and samples from top to bottom, with files in `/kaggle/working/day4`.
- [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks), checked 2026-09-25,
  lists the T4 x2 option. Decision: select T4 x2 and use `cuda:0`; the code does not use a second
  GPU. The Colab T4 precision result is the starting comparison, not an expected Kaggle speed.
- [Kaggle Datasets documentation](https://www.kaggle.com/docs/datasets), checked 2026-09-25,
  says private datasets can hold uploaded source files and ZIP archives are unpacked for notebooks.
  Decision: attach the repository as a private `octlm-code` input when the GitHub repository is
  private. Attach prepared token files or a checkpoint as a second input when resuming.
- [Kaggle's GPU usage guide](https://www.kaggle.com/docs/efficient-gpu-usage), checked 2026-09-25,
  says the weekly quota is 30 hours or sometimes higher. Decision: check the account's displayed
  quota before starting. The earlier 30-hour figure in `notes/day3.md` was a lead, not a guarantee.

### Operating decision

`PLAN.md` named Colab Pro as the Day 4 training host and Kaggle as an unverified fallback. The
user's 2026-09-25 request makes Kaggle the host for EXP-067. The Kaggle documentation above
confirms a 12-hour run window but does not guarantee that interrupted `Save & Run All` jobs retain
their files. A checkpoint in `/kaggle/working` survives into a later session only after its output
is saved as a notebook version and attached as input. The notebook therefore makes no cross-session
resume promise for a failed version run.

### Result

The Kaggle run has not started. Keep the Day 4 model and trainer unchanged. Use
`notebooks/octlm-kaggle.ipynb` for the next run, and fill this section from its output.

Local checks on 2026-09-25: the notebook JSON and code cells compile. All 54 unit tests pass.
`octlm.train --config configs/day4.toml --dry-run` reports 26,255,872 parameters. Ruff lint and
format pass on the new notebook and `octlm/`. Whole-repository Ruff still fails on a long line in
the older Colab notebook and two long lines in the unrelated, untracked `web/` work; whole-repository
format also flags the latter. No Kaggle measurement has been run.

## Session on 2026-09-25, later: audit Kaggle and restart EXP-067

Kaggle holds two octlm notebooks and one octlm dataset. `octlm-day4` is the EXP-067 run.
`octlm-probe` listed `/kaggle/input` and the GPU once and has no further use. The `octlm-code`
dataset matched the repository byte for byte, and `octlm-day4` matched
`notebooks/octlm-kaggle.ipynb` cell for cell. Nothing on Kaggle departed from `PLAN.md`.

Version 3 of `octlm-day4` looked stuck and was cancelled at 787 s. It was not stuck. The log ends
after the encode benchmark at 305 s, and the next step, encoding the 2.2 GB train split, prints
nothing until it finishes. That step took 934 s on Colab, so the run was about 480 s into it. The
tokenizer trained in 264 s on Kaggle against 303 s on Colab. Encode throughput with the cache was
2,696 KB/s against 2,559 KB/s on Colab.

Fix: `encode_split` now prints the token count at each 16M-token flush. Uploaded as a new
`octlm-code` version and started `octlm-day4` version 4 with the unchanged notebook.

## Session on 2026-09-28: collect the Kaggle run

`octlm-day4` version 4 finished with status COMPLETE. Pulled `metrics.jsonl`, `samples.jsonl`,
`prepare.jsonl`, `bpe.jsonl`, and the kernel log with `kaggle kernels output --file-pattern` into
`runs/kaggle-day4/`. `run.pt`, `train.bin`, and the raw text stay on Kaggle as notebook output.
Attach that output as input to resume or to run Day 5 from this checkpoint.

Deleted the `octlm-probe` Kaggle notebook, which had no further use.

The user accepted the coherence judgment on 2026-09-28. Every exit check passes. Day 4 is closed.
