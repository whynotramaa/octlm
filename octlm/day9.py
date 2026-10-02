from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from octlm.day2 import _write
from octlm.day6 import QwenTokenizer, file_hash, load_qwen, sampling
from octlm.day8 import TRAIN_TASKS, dtype_for, load_adapter
from octlm.harness import FIXTURES, load_tasks, passes, run_task, sandbox
from octlm.model import Decoder, rope_tables
from octlm.train import resolve_device

POOL = TRAIN_TASKS.with_name("grpo-tasks.json")
LEARNING_RATE, GROUP, KEEP, MAX_GROUPS, STEPS, MAX_GRAD_NORM = 1e-5, 6, 4, 16, 30, 1.0


def load_pool(path: Path) -> list[dict]:
    tasks = load_tasks(path)
    if path.resolve() != TRAIN_TASKS.resolve():
        seen = {t["prompt"] for p in (FIXTURES / "tasks.json", TRAIN_TASKS) for t in load_tasks(p)}
        leaked = [task["id"] for task in tasks if task["prompt"] in seen]
        if leaked:
            raise ValueError(f"GRPO prompts equal eval or SFT prompts: {leaked}")
    return tasks


def rollouts(model, tokenizer, settings, task: dict, fixtures: Path, seeds: list[int]) -> list:
    group = []
    for seed in seeds:
        turns: list = []
        with sandbox(task, fixtures) as root:
            record = run_task(model, tokenizer, settings, task, root, seed, turns)
            reward = float(passes(task, root, record["answer"], fixtures))
        group.append((reward, turns, record))
    return group


def reply_log_probs(model: Decoder, ids: list[int], start: int, temperature: float):
    device = model.token_embedding.weight.device
    tokens = torch.tensor([ids], device=device)
    config = model.config
    rope = rope_tables(len(ids) - 1, config.head_size, config.rope_scale, device, config.rope_base)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        x = model._embed(tokens[:, :-1])
        for block in model.blocks:
            x = checkpoint(block, x, rope, use_reentrant=False)
        logits = model.lm_head(model.final_norm(x[:, start - 1 :]))
    log_probs = (logits.float() / temperature).log_softmax(-1)
    return log_probs.gather(-1, tokens[:, start:, None]).squeeze(-1)


def backward_group(model, group: list, temperature: float, total: int, scaler) -> float:
    mean = statistics.mean(reward for reward, _, _ in group)
    loss = 0.0
    for reward, turns, _ in group:
        advantage = reward - mean
        for prompt, new in turns:
            log_probs = reply_log_probs(model, prompt + new, len(prompt), temperature)
            value = -advantage * log_probs.sum() / total
            scaler.scale(value).backward()
            loss += value.item()
    return loss


def stage_probe(args: argparse.Namespace) -> None:
    model, tokenizer, settings = load_policy(args)
    tasks = load_pool(args.tasks)
    limit = min(args.limit or len(tasks), len(tasks))
    tasks = [tasks[i * len(tasks) // limit] for i in range(limit)]
    counts = []
    for task in tasks:
        group = rollouts(model, tokenizer, settings, task, args.tasks.parent, args.seeds)
        successes = int(sum(reward for reward, _, _ in group))
        counts.append(successes)
        row = {"type": "probe", "task": task["id"], "kind": task["check"]["kind"]}
        turns = statistics.mean(record["turns"] for _, _, record in group)
        _write(args.out, row | {"successes": successes, "samples": len(group), "mean_turns": turns})
    samples = len(args.seeds)
    _write(
        args.out,
        {
            "type": "probe_summary",
            "tasks": len(tasks),
            "samples": samples,
            "pass_rate": sum(counts) / (samples * len(tasks)),
            "mixed": sum(0 < c < samples for c in counts),
            "adapter_hash": file_hash(args.adapter),
            "tasks_hash": file_hash(args.tasks),
        },
    )


def load_policy(args: argparse.Namespace):
    device = resolve_device(args.device)
    model, manifest = load_qwen(args.model, str(device), dtype_for(device))
    tokenizer = QwenTokenizer(args.model)
    load_adapter(model, manifest, tokenizer, args.adapter)
    model.eval()
    args.manifest = manifest
    return model, tokenizer, sampling(args.model, thinking=False)


def fill_batch(model, tokenizer, settings, args, step: int, cursor) -> tuple:
    kept, tried, rewards = [], 0, []
    while len(kept) < KEEP and tried < MAX_GROUPS:
        task = next(cursor)
        seeds = [10_000 * step + GROUP * tried + i for i in range(GROUP)]
        group = rollouts(model, tokenizer, settings, task, args.tasks.parent, seeds)
        tried += 1
        rewards += [reward for reward, _, _ in group]
        if len({reward for reward, _, _ in group}) > 1:
            kept.append((task["id"], group))
    return kept, tried, rewards


def cycle(tasks: list[dict], seed: int):
    order = random.Random(seed)
    while True:
        shuffled = tasks[:]
        order.shuffle(shuffled)
        yield from shuffled


def stage_train(args: argparse.Namespace) -> None:
    model, tokenizer, settings = load_policy(args)
    device = model.token_embedding.weight.device
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    cursor = cycle(load_pool(args.tasks), args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, stop = time.perf_counter(), "steps"
    for step in range(1, STEPS + 1):
        if time.perf_counter() - started > args.hours * 3600:
            stop = "time"
            break
        began = time.perf_counter()
        kept, tried, rewards = fill_batch(model, tokenizer, settings, args, step, cursor)
        row = {"type": "grpo_step", "step": step, "groups_tried": tried, "groups_kept": len(kept)}
        row |= {"mean_reward": statistics.mean(rewards), "rollouts": len(rewards)}
        if len(kept) < KEEP:
            _write(args.out, row | {"seconds": time.perf_counter() - began})
            stop = "sparse"
            break
        total = sum(len(new) for _, g in kept for _, turns, _ in g for _, new in turns)
        loss = sum(
            backward_group(model, g, settings["temperature"], total, scaler) for _, g in kept
        )
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        row |= {"kept_tasks": [task for task, _ in kept], "loss": loss, "grad_norm": norm}
        _write(args.out, row | {"tokens": total, "seconds": time.perf_counter() - began})
    save(model, tokenizer, args, step - (stop != "steps"))
    _write(
        args.out,
        {
            "type": "grpo_summary",
            "stop": stop,
            "steps": step - (stop != "steps"),
            "seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "start_adapter_hash": file_hash(args.adapter),
            "adapter_hash": file_hash(args.save),
            "settings": [LEARNING_RATE, GROUP, KEEP, MAX_GROUPS, STEPS, MAX_GRAD_NORM],
        },
    )


def save(model: Decoder, tokenizer: QwenTokenizer, args: argparse.Namespace, steps: int) -> None:
    start = torch.load(args.adapter, weights_only=True)
    adapters = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_" in k}
    torch.save(
        start
        | {
            "adapters": adapters,
            "grpo_steps": steps,
            "grpo_seed": args.seed,
            "grpo_tasks_hash": file_hash(args.tasks),
            "start_adapter_hash": file_hash(args.adapter),
        },
        args.save,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 9 GRPO stages.")
    parser.add_argument("stage", choices=("probe", "train"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/day6/qwen"))
    parser.add_argument("--tasks", type=Path, default=POOL)
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/day8/adapter-1.pt"))
    parser.add_argument("--save", type=Path, default=Path("artifacts/day9/grpo.pt"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds + [args.seed]) < 0:
        parser.error("seeds must be distinct and nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.hours <= 0:
        parser.error("hours must be positive")
    if not args.adapter.is_file():
        parser.error(f"adapter not found: {args.adapter}")
    args.out = args.out or Path(f"runs/day9-{args.stage}.jsonl")
    if args.out.exists():
        parser.error(f"{args.out} already contains a measurement")
    if args.stage == "train":
        args.save.parent.mkdir(parents=True, exist_ok=True)
    {"probe": stage_probe, "train": stage_train}[args.stage](args)


if __name__ == "__main__":
    main()
