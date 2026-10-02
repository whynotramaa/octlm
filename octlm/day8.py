from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from octlm.day2 import _write
from octlm.day5 import quantize_int8
from octlm.day6 import QwenTokenizer, file_hash, load_qwen
from octlm.harness import (
    FIXTURES,
    SYSTEM,
    TOOLS,
    execute,
    load_tasks,
    parse_calls,
    passes,
    run_eval,
    sandbox,
)
from octlm.model import Decoder, parameter_count
from octlm.train import resolve_device

TRAIN_TASKS = FIXTURES.parent / "day8" / "tasks.json"
TARGETS = {
    "attention": ("query", "key", "value", "output"),
    "feed_forward": ("gate", "up", "down"),
}
RANK, ALPHA = 16, 32
LEARNING_RATE, BATCH, EPOCHS, MAX_GRAD_NORM = 2e-4, 8, 3, 0.3
END = "<|im_end|>"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, generator: torch.Generator):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        a = torch.empty(rank, base.in_features)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        device = base.weight.device
        self.lora_a = nn.Parameter(a.to(device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = x.to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + (update * self.scale).to(x.dtype)

    def merged(self) -> nn.Linear:
        weight = self.base.weight
        linear = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        delta = self.lora_b.float() @ self.lora_a.float() * self.scale
        linear.weight.data.copy_(weight.float() + delta)
        return linear


def projections(model: Decoder):
    for block in model.blocks:
        for part, names in TARGETS.items():
            parent = getattr(block, part)
            for name in names:
                yield parent, name


def add_lora(model: Decoder, rank: int, alpha: float, seed: int) -> list[nn.Parameter]:
    model.requires_grad_(False)
    generator = torch.Generator().manual_seed(seed)
    for parent, name in projections(model):
        setattr(parent, name, LoRALinear(getattr(parent, name), rank, alpha, generator))
    return [p for p in model.parameters() if p.requires_grad]


def merge_lora(model: Decoder) -> None:
    for parent, name in projections(model):
        setattr(parent, name, getattr(parent, name).merged())


def call_text(call: dict) -> str:
    body = json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


def trace(task: dict, fixtures: Path) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    with sandbox(task, fixtures) as root:
        for call in task["solution"]["calls"]:
            text = call_text(call)
            if parse_calls(text) != [call]:
                raise ValueError(f"trace call does not round-trip in task {task['id']}")
            result, ok = execute(root, call)
            if not ok:
                raise ValueError(f"trace call failed in task {task['id']}: {result}")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "tool", "content": result})
        messages.append({"role": "assistant", "content": task["solution"]["answer"]})
        if not passes(task, root, task["solution"]["answer"], fixtures):
            raise ValueError(f"trace fails its own check in task {task['id']}")
    return messages


def traces(path: Path) -> list[tuple[dict, list[dict]]]:
    tasks = load_tasks(path)
    held_out = {task["prompt"] for task in load_tasks(FIXTURES / "tasks.json")}
    leaked = [task["id"] for task in tasks if task["prompt"] in held_out]
    if leaked:
        raise ValueError(f"training prompts equal eval prompts: {leaked}")
    return [(task, trace(task, path.parent)) for task in tasks]


def examples(messages: list[dict], tokenizer: QwenTokenizer) -> list[tuple[list[int], int]]:
    rows = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt_text = tokenizer.chat(messages[:index], tools=TOOLS)
        prompt = tokenizer.encode(prompt_text, allow_special=True)
        ids = tokenizer.encode(prompt_text + message["content"] + END, allow_special=True)
        if ids[: len(prompt)] != prompt or ids[-1] != tokenizer.native.token_to_id(END):
            raise ValueError("assistant target does not extend the rendered prompt")
        rows.append((ids, len(prompt)))
    return rows


def stage_traces(args: argparse.Namespace) -> None:
    tokenizer = QwenTokenizer(args.model)
    kinds, rows = {}, []
    for task, messages in traces(args.tasks):
        kinds[task["check"]["kind"]] = kinds.get(task["check"]["kind"], 0) + 1
        found = examples(messages, tokenizer)
        rows += found
        _write(args.out, {"type": "day8_trace", "task": task["id"], "messages": messages})
    _write(
        args.out,
        {
            "type": "day8_trace_summary",
            "tasks": sum(kinds.values()),
            "by_kind": kinds,
            "examples": len(rows),
            "target_tokens": sum(len(ids) - start for ids, start in rows),
            "total_tokens": sum(len(ids) for ids, _ in rows),
            "longest": max(len(ids) for ids, _ in rows),
            "tasks_hash": file_hash(args.tasks),
        },
    )


def example_loss(model: Decoder, ids: list[int], start: int, total: int, device) -> torch.Tensor:
    tokens = torch.tensor([ids], device=device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(tokens[:, :-1])
    loss = F.cross_entropy(logits[0, start - 1 :].float(), tokens[0, start:], reduction="sum")
    return loss / total


@torch.no_grad()
def step_zero(model: Decoder, ids: list[int], device: torch.device, seed: int) -> dict:
    tokens = torch.tensor([ids], device=device)
    base = model(tokens)
    trainable = add_lora(model, RANK, ALPHA, seed)
    difference = (model(tokens).float() - base.float()).abs().max().item()
    if difference != 0:
        raise RuntimeError(f"adapted logits differ from base at step 0: {difference}")
    return {
        "type": "lora_step_zero",
        "tokens": len(ids),
        "max_absolute_difference": difference,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": parameter_count(model),
    }


def stage_train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, manifest = load_qwen(args.model, str(device), dtype_for(device))
    tokenizer = QwenTokenizer(args.model)
    data = [row for _, messages in traces(args.tasks) for row in examples(messages, tokenizer)]
    longest = max(data, key=lambda row: len(row[0]))[0]
    _write(args.out, step_zero(model, longest, device, args.seed))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    started, step = time.perf_counter(), 0
    for epoch in range(EPOCHS):
        order = torch.randperm(len(data), generator=generator).tolist()
        for first in range(0, len(order), BATCH):
            batch = [data[i] for i in order[first : first + BATCH]]
            total = sum(len(ids) - start for ids, start in batch)
            loss = 0.0
            for ids, start in batch:
                value = example_loss(model, ids, start, total, device)
                scaler.scale(value).backward()
                loss += value.item()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            row = {"type": "sft_step", "step": step, "epoch": epoch, "loss": loss}
            _write(args.out, row | {"grad_norm": norm, "target_tokens": total})
    seconds = time.perf_counter() - started
    adapters = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_" in k}
    torch.save(
        {
            "adapters": adapters,
            "rank": RANK,
            "alpha": ALPHA,
            "seed": args.seed,
            "manifest": manifest,
            "tokenizer_hash": tokenizer.fingerprint,
            "tasks_hash": file_hash(args.tasks),
        },
        args.adapter,
    )
    _write(
        args.out,
        {
            "type": "sft_summary",
            "seed": args.seed,
            "examples": len(data),
            "steps": step,
            "seconds": seconds,
            "tokens_per_second": EPOCHS * sum(len(ids) for ids, _ in data) / seconds,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "adapter_hash": file_hash(args.adapter),
            "settings": [RANK, ALPHA, LEARNING_RATE, BATCH, EPOCHS, MAX_GRAD_NORM],
        },
    )


def dtype_for(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def load_adapter(model: Decoder, manifest: dict, tokenizer: QwenTokenizer, path: Path) -> None:
    saved = torch.load(path, weights_only=True)
    if saved["manifest"] != manifest or saved["tokenizer_hash"] != tokenizer.fingerprint:
        raise ValueError("adapter was trained on a different checkpoint or tokenizer")
    add_lora(model, saved["rank"], saved["alpha"], 0)
    expected = {k for k in model.state_dict() if ".lora_" in k}
    if set(saved["adapters"]) != expected:
        raise ValueError("adapter tensors do not match the LoRA targets")
    model.load_state_dict(saved["adapters"], strict=False)


def projection_bytes(model: Decoder) -> int:
    return sum(t.numel() * t.element_size() for t in model.blocks.state_dict().values())


@torch.no_grad()
def merge_and_quantize(model: Decoder, tokenizer: QwenTokenizer, task: dict, out: Path) -> None:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    ids = tokenizer.encode(tokenizer.chat(messages, tools=TOOLS), allow_special=True)
    tokens = torch.tensor([ids], device=model.token_embedding.weight.device)
    unmerged = model(tokens).float()
    merge_lora(model)
    merged = model(tokens).float()
    difference = (merged - unmerged).abs().max().item()
    agreement = (merged.argmax(-1) == unmerged.argmax(-1)).float().mean().item()
    float_bytes = projection_bytes(model)
    quantize_int8(model)
    row = {"type": "merge_quantize", "tokens": len(ids), "max_absolute_difference": difference}
    row |= {"float_block_bytes": float_bytes, "int8_block_bytes": projection_bytes(model)}
    _write(out, row | {"top_1_agreement": agreement})
    if agreement < 1:
        raise RuntimeError(f"merged top-1 tokens differ from unmerged on {1 - agreement:.2%}")


def stage_eval(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, manifest = load_qwen(args.model, str(device), dtype_for(device))
    tokenizer = QwenTokenizer(args.model)
    load_adapter(model, manifest, tokenizer, args.adapter)
    model.eval()
    if args.int8:
        merge_and_quantize(model, tokenizer, load_tasks(args.tasks)[0], args.out)
    extra = {"adapter_hash": file_hash(args.adapter), "merged_int8": args.int8}
    run_eval(model, manifest, args, extra)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 8 LoRA fine-tuning stages.")
    parser.add_argument("stage", choices=("traces", "train", "eval"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/day6/qwen"))
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/day8/adapter-0.pt"))
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds + [args.seed]) < 0:
        parser.error("seeds must be distinct and nonnegative")
    args.tasks = args.tasks or (FIXTURES / "tasks.json" if args.stage == "eval" else TRAIN_TASKS)
    args.out = args.out or Path(f"runs/day8-{args.stage}.jsonl")
    if args.out.exists():
        parser.error(f"{args.out} already contains a measurement")
    if args.stage == "eval" and not args.adapter.is_file():
        parser.error(f"adapter not found: {args.adapter}")
    if args.stage == "train":
        args.adapter.parent.mkdir(parents=True, exist_ok=True)
    {"traces": stage_traces, "train": stage_train, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    main()
