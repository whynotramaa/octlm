from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from octlm.config import ProjectConfig
from octlm.day2 import _write
from octlm.model import Decoder
from octlm.tokenizer import ByteBPETokenizer, pretokenize
from octlm.train import (
    TokenBlocks,
    data_fingerprint,
    decoder_config,
    resolve_device,
    train_model,
)

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{}"
SOURCES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}
DELIMITER = "<|endoftext|>"
TOKENIZER_SAMPLE_CHARS = 10_000_000
BENCH_CHARS = 2_000_000
FLUSH_TOKENS = 1 << 24
EVAL_BLOCKS = 256
SAMPLE_TOKENS = 200
PROMPTS = (
    "Once upon a time, there was a little girl named Lily.",
    "One day, a boy named Tom found a big red ball",
    "The cat was very hungry.",
    "Sam wanted to go to the park, but",
    'Mom said, "',
    "There was a small bird who could not fly.",
    "Tim and his dog went on an adventure",
    "It was a rainy day and",
    "Anna had a secret.",
    "The big tree in the garden",
)


@dataclass
class TokenStream:
    tokens: np.ndarray
    context_length: int

    def batch(self, size: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        width = self.context_length + 1
        starts = torch.randint(len(self.tokens) - width, (size,), generator=generator).tolist()
        rows = np.stack([self.tokens[start : start + width] for start in starts])
        block = torch.from_numpy(rows.astype(np.int64))
        return block[:, :-1], block[:, 1:], block[:, 1:]


def download(split: str, data_dir: Path) -> Path:
    path = data_dir / SOURCES[split]
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(URL.format(SOURCES[split]), timeout=60) as response:
        with partial.open("wb") as file:
            shutil.copyfileobj(response, file, 1 << 20)
    os.replace(partial, path)
    return path


def stories(path: Path) -> Iterator[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8", newline="") as file:
        for line in file:
            if line.rstrip("\r\n") != DELIMITER:
                lines.append(line)
                continue
            story = "".join(lines).strip("\n")
            lines = []
            if story:
                yield story
    story = "".join(lines).strip("\n")
    if story:
        yield story


def leading_stories(path: Path, characters: int) -> list[str]:
    sample: list[str] = []
    total = 0
    for story in stories(path):
        if total >= characters:
            break
        sample.append(story)
        total += len(story)
    return sample


def train_tokenizer(path: Path, config: ProjectConfig, target: Path) -> ByteBPETokenizer:
    if target.exists():
        return ByteBPETokenizer.load(target)
    sample = leading_stories(path, TOKENIZER_SAMPLE_CHARS)
    started = time.perf_counter()
    tokenizer = ByteBPETokenizer.train(
        sample, config.tokenizer.vocab_size, config.tokenizer.min_frequency
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(target)
    record = {
        "fingerprint": tokenizer.fingerprint,
        "sample_characters": sum(len(story) for story in sample),
        "sample_stories": len(sample),
        "seconds": time.perf_counter() - started,
        "type": "tokenizer",
        "vocab_size": tokenizer.vocab_size,
    }
    _write(target.with_suffix(".jsonl"), record)
    return tokenizer


def encode_benchmark(tokenizer_path: Path, path: Path) -> dict[str, object]:
    sample = leading_stories(path, BENCH_CHARS)
    byte_count = sum(len(story.encode()) for story in sample)
    uncached = ByteBPETokenizer.load(tokenizer_path)
    started = time.perf_counter()
    for story in sample:
        for chunk in pretokenize(story):
            uncached.encode_chunk(chunk)
    plain = time.perf_counter() - started
    cached = ByteBPETokenizer.load(tokenizer_path)
    started = time.perf_counter()
    for story in sample:
        cached.encode(story)
    cold_cache = time.perf_counter() - started
    return {
        "bytes": byte_count,
        "cached_bytes_per_second": byte_count / cold_cache,
        "plain_bytes_per_second": byte_count / plain,
        "type": "encode_benchmark",
    }


def encode_split(tokenizer: ByteBPETokenizer, source: Path, target: Path) -> dict[str, object]:
    if tokenizer.vocab_size > 1 << 16:
        raise ValueError("token IDs must fit in uint16")
    started = time.perf_counter()
    buffer = array("H")
    counts = {"bytes": 0, "stories": 0, "tokens": 0}
    partial = target.with_suffix(".part")
    with partial.open("wb") as file:
        for story in stories(source):
            buffer.extend(tokenizer.encode(story, add_special_tokens=True))
            counts["bytes"] += len(story.encode())
            counts["stories"] += 1
            if len(buffer) >= FLUSH_TOKENS:
                counts["tokens"] += len(buffer)
                buffer.tofile(file)
                buffer = array("H")
        counts["tokens"] += len(buffer)
        buffer.tofile(file)
    os.replace(partial, target)
    with target.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    seconds = time.perf_counter() - started
    return {
        **counts,
        "bytes_per_token": counts["bytes"] / counts["tokens"],
        "seconds": seconds,
        "sha256": digest,
        "split": target.stem,
        "tokens_per_second": counts["tokens"] / seconds,
        "type": "encode",
    }


def prepare(args: argparse.Namespace) -> None:
    config = ProjectConfig.load(args.config)
    paths = {split: download(split, args.data) for split in SOURCES}
    tokenizer = train_tokenizer(paths["train"], config, args.tokenizer)
    log = args.data / "prepare.jsonl"
    _write(log, encode_benchmark(args.tokenizer, paths["valid"]))
    for split, path in paths.items():
        target = args.data / f"{split}.bin"
        if not target.exists():
            _write(log, encode_split(tokenizer, path, target))


def validation_blocks(
    tokens: np.ndarray, tokenizer: ByteBPETokenizer, context_length: int
) -> TokenBlocks:
    width = context_length + 1
    if len(tokens) < EVAL_BLOCKS * width:
        raise ValueError("validation split is shorter than the evaluation window")
    rows = torch.from_numpy(tokens[: EVAL_BLOCKS * width].astype(np.int64)).view(-1, width)
    table = torch.tensor([tokenizer.token_bytes(i) for i in range(tokenizer.vocab_size)])
    return TokenBlocks(rows[:, :-1], rows[:, 1:], table[rows[:, 1:]])


def token_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"{path} is missing, run the prepare stage first")
    return np.memmap(path, dtype=np.uint16, mode="r")


def train(args: argparse.Namespace) -> None:
    config = ProjectConfig.load(args.config)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    paths = [args.data / "train.bin", args.data / "valid.bin"]
    train_tokens, valid_tokens = (token_file(path) for path in paths)
    context = config.model.context_length
    resume = args.checkpoint if args.checkpoint.exists() else None
    model, _, _ = train_model(
        config,
        tokenizer,
        TokenStream(train_tokens, context),
        validation_blocks(valid_tokens, tokenizer, context),
        data_fingerprint(paths),
        args.stop_after,
        args.checkpoint,
        resume,
        args.metrics,
        args.device,
        mixed_precision=not args.full_precision,
    )
    write_samples(model.eval(), tokenizer, args)


def load_model(args: argparse.Namespace) -> tuple[Decoder, ByteBPETokenizer]:
    config = ProjectConfig.load(args.config)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("tokenizer_hash") != tokenizer.fingerprint:
        raise ValueError("checkpoint tokenizer does not match")
    if state.get("config_hash") != config.fingerprint:
        raise ValueError("checkpoint config does not match")
    model = Decoder(decoder_config(config, tokenizer.vocab_size))
    model.load_state_dict(state["model"])
    return model.to(resolve_device(args.device)).eval(), tokenizer


def write_samples(model: Decoder, tokenizer: ByteBPETokenizer, args: argparse.Namespace) -> None:
    generator = torch.Generator().manual_seed(args.seed)
    device = next(model.parameters()).device
    for prompt in PROMPTS:
        prompt_ids = [tokenizer.bos_id, *tokenizer.encode(prompt)]
        inputs = torch.tensor([prompt_ids], device=device)
        output = model.generate(inputs, SAMPLE_TOKENS, args.temperature, args.top_k, generator)
        new_ids = output[0, len(prompt_ids) :].tolist()
        if tokenizer.eos_id in new_ids:
            new_ids = new_ids[: new_ids.index(tokenizer.eos_id)]
        text = tokenizer.decode(new_ids, errors="replace", skip_special_tokens=True)
        record = {
            "prompt": prompt,
            "seed": args.seed,
            "temperature": args.temperature,
            "text": prompt + text,
            "top_k": args.top_k,
            "type": "sample",
        }
        _write(args.samples_out, record)


def samples(args: argparse.Namespace) -> None:
    model, tokenizer = load_model(args)
    write_samples(model, tokenizer, args)


STAGES = {"prepare": prepare, "train": train, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 4 TinyStories experiments.")
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--config", type=Path, default=Path("configs/day4.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/day4/bpe.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/day4/run.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("runs/day4.jsonl"))
    parser.add_argument("--samples-out", type=Path, default=Path("runs/day4-samples.jsonl"))
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
