from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

from octlm.day4 import PROMPTS, SAMPLE_TOKENS, load_model
from octlm.day6 import QwenTokenizer, load_qwen, sampling
from octlm.day8 import dtype_for, load_adapter
from octlm.harness import FIXTURES, load_tasks, passes, run_task, sandbox
from octlm.model import parameter_count, sample_token
from octlm.train import resolve_device

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, BLUE, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[36m"
SHADES = ((0.9, GREEN), (0.5, CYAN), (0.2, YELLOW), (0.0, RED))
TOOL_LINES = 12


def say(text: str, color: str = "") -> None:
    sys.stdout.write(f"{color}{text}{RESET}" if color else text)
    sys.stdout.flush()


class Stream:
    def __init__(self, decode) -> None:
        self.decode, self.ids, self.shown = decode, [], ""

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.decode(self.ids)
        if text.endswith("�"):
            return ""
        new, self.shown = text[len(self.shown) :], text
        return new


def show_tool(result: str) -> None:
    lines = result.splitlines() or [""]
    hidden = len(lines) - TOOL_LINES
    body = "\n".join(f"  │ {line}" for line in lines[:TOOL_LINES])
    say(f"\n  tool result\n{body}\n", YELLOW)
    if hidden > 0:
        say(f"  │ ... {hidden} more lines sent to the model\n", YELLOW)


def agent_watch(tokenizer: QwenTokenizer, stop_ids: set[int]):
    state = {"stream": Stream(tokenizer.decode), "turn": 0}

    def watch(event: str, value) -> None:
        stream = state["stream"]
        if event == "tool":
            show_tool(value)
            state["stream"] = Stream(tokenizer.decode)
            return
        if not stream.ids:
            state["turn"] += 1
            say(f"\n▸ model, turn {state['turn']}\n", BOLD)
        if value in stop_ids:
            state["stream"] = Stream(tokenizer.decode)
            say("\n")
            return
        chunk = stream.push(value)
        say(chunk, CYAN if "<tool_call>" in stream.shown else GREEN + BOLD)

    return watch


def agent(args: argparse.Namespace) -> None:
    tasks = {task["id"]: task for task in load_tasks(args.tasks)}
    if args.list:
        for task in tasks.values():
            print(f"{task['id']:24} {task['check']['kind']:7} {task['prompt']}")
        return
    if args.task not in tasks:
        raise SystemExit(f"unknown task {args.task!r}, see --list")
    task, device = tasks[args.task], resolve_device(args.device)
    model, manifest = load_qwen(args.model, str(device), dtype_for(device))
    tokenizer, settings = QwenTokenizer(args.model), sampling(args.model, thinking=False)
    label = "Qwen3-0.6B, no adapter"
    if not args.base:
        load_adapter(model, manifest, tokenizer, args.adapter)
        label = f"Qwen3-0.6B + LoRA {args.adapter.name}"
    model.eval()
    say(f"octlm agent · {label} · {device} · seed {args.seed}\n", DIM)
    say(f"task {task['id']}: {task['prompt']}\n", BOLD)
    with sandbox(task, args.tasks.parent) as root:
        watch = agent_watch(tokenizer, settings["stop_ids"])
        record = run_task(model, tokenizer, settings, task, root, args.seed, watch=watch)
        success = passes(task, root, record["answer"], args.tasks.parent)
    verdict = ("PASS", GREEN) if success else ("FAIL", RED)
    say(f"\n{verdict[0]}", verdict[1] + BOLD)
    say(
        f"  check {task['check']['kind']} · stop {record['stop']} · turns {record['turns']}"
        f" · valid calls {record['valid_calls']}/{record['calls']}"
        f" · {record['generated_tokens'] / record['seconds']:.1f} tokens/s with prefill\n",
        DIM,
    )


def shade(probability: float) -> str:
    return next(color for floor, color in SHADES if probability >= floor)


@torch.inference_mode()
def tell(model, tokenizer, prompt: str, args, generator: torch.Generator) -> dict:
    ids = [tokenizer.bos_id, *tokenizer.encode(prompt)]
    budget = min(args.tokens, model.config.context_length - len(ids))
    device = model.token_embedding.weight.device
    stream = Stream(lambda tokens: tokenizer.decode(tokens, errors="replace"))
    say(prompt, BOLD)
    started = time.perf_counter()
    logits, cache = model.forward_cached(torch.tensor([ids], device=device))
    seconds, logprobs = time.perf_counter() - started, []
    for _ in range(budget):
        last = logits[:, -1].float()
        token = sample_token(last, args.temperature, args.top_k, generator)
        if token.item() == tokenizer.eos_id:
            break
        logprobs.append(torch.log_softmax(last, -1)[0, token.item()].item())
        say(stream.push(token.item()), shade(math.exp(logprobs[-1])))
        started = time.perf_counter()
        logits, cache = model.forward_cached(token, cache)
        seconds += time.perf_counter() - started
        time.sleep(args.delay)
    count = max(len(logprobs), 1)
    return {"tokens": len(logprobs), "seconds": seconds, "nll": -sum(logprobs) / count}


def stories(args: argparse.Namespace) -> None:
    model, tokenizer = load_model(args)
    config = model.config
    say(
        f"octlm TinyStories · {parameter_count(model) / 1e6:.1f}M parameters, trained from"
        f" scratch · {config.n_layers} layers · width {config.d_model} · {config.position}"
        f" · {config.feed_forward} · {config.key_value_heads} KV heads · {model.lm_head.weight.device}\n",
        DIM,
    )
    say("p(token):")
    for floor, color in SHADES:
        say(f" ■ ≥{floor:.1f}", color)
    say(f"   temperature {args.temperature}, top-k {args.top_k}, KV cache\n", DIM)
    generator = torch.Generator().manual_seed(args.seed)
    for prompt in args.prompt or PROMPTS[:3]:
        say("\n")
        result = tell(model, tokenizer, prompt, args, generator)
        say(
            f"\n  {result['tokens']} tokens · {result['tokens'] / result['seconds']:.0f} tokens/s"
            f" model time · sample perplexity {math.exp(result['nll']):.2f}\n",
            DIM,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live terminal demos of the octlm models.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("agent", help="Run Qwen3-0.6B through the tool harness live.")
    run.add_argument("--task", default="bug-low-stock")
    run.add_argument("--list", action="store_true")
    run.add_argument("--base", action="store_true", help="Skip the LoRA adapter.")
    run.add_argument("--adapter", type=Path, default=Path("artifacts/day8/adapter-0.pt"))
    run.add_argument("--model", type=Path, default=Path("artifacts/day6/qwen"))
    run.add_argument("--tasks", type=Path, default=FIXTURES / "tasks.json")
    tell_parser = commands.add_parser("stories", help="Stream the Day 4 TinyStories model.")
    tell_parser.add_argument("--prompt", action="append")
    tell_parser.add_argument("--tokens", type=int, default=SAMPLE_TOKENS)
    tell_parser.add_argument("--temperature", type=float, default=0.8)
    tell_parser.add_argument("--top-k", type=int, default=40)
    tell_parser.add_argument("--delay", type=float, default=0.03)
    tell_parser.add_argument("--config", type=Path, default=Path("configs/day4.toml"))
    tell_parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/day4/bpe.json"))
    tell_parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/day4/run.pt"))
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.command == "agent" and not args.base and not args.adapter.is_file():
        parser.error(f"adapter not found: {args.adapter}")
    if args.command == "stories":
        if args.tokens < 1 or args.temperature < 0 or args.top_k < 0 or args.delay < 0:
            parser.error("tokens must be positive; temperature, top-k, delay nonnegative")
        for path in (args.tokenizer, args.checkpoint):
            if not path.is_file():
                parser.error(f"not found: {path}")
    {"agent": agent, "stories": stories}[args.command](args)


if __name__ == "__main__":
    main()
