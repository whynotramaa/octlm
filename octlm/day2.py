from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from octlm.config import ProjectConfig
from octlm.corpus import load_documents
from octlm.model import Decoder, DecoderConfig, kv_cache_bytes, parameter_count
from octlm.tokenizer import ByteBPETokenizer
from octlm.train import (
    TokenBlocks,
    _json,
    data_fingerprint,
    decoder_config,
    evaluate,
    make_blocks,
    train_model,
)

ARTIFACTS = Path("artifacts/day2")
RUNS = Path("runs")
TRAIN_SPLIT = Path("data/train.jsonl")
VALIDATION_SPLIT = Path("data/val.jsonl")
TOKENIZER_SAMPLE = 10
SEEDS = (1337, 1338, 1339)
EVAL_BLOCKS = 96
FINAL_BLOCKS = 192
BACKENDS = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}

VARIANTS: dict[str, dict[str, object]] = {
    "baseline": {},
    "rmsnorm": {"norm": "rmsnorm"},
    "swiglu": {"feed_forward": "swiglu"},
    "post-norm": {"residual": "post"},
    "gqa-4": {"kv_heads": 4},
    "gqa-2": {"kv_heads": 2},
    "mqa-1": {"kv_heads": 1},
    "modern": {
        "position": "rope",
        "norm": "rmsnorm",
        "feed_forward": "swiglu",
        "kv_heads": 2,
    },
}


def _peak_rss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _write(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(_json(record) + "\n")
    print(_json(record), flush=True)


def build_tokenizer(vocab_size: int) -> ByteBPETokenizer:
    """Train once on every tenth training document, then reuse the artifact."""
    path = ARTIFACTS / f"bpe-{vocab_size}.json"
    if path.exists():
        return ByteBPETokenizer.load(path)
    documents = load_documents(TRAIN_SPLIT)[::TOKENIZER_SAMPLE]
    tokenizer = ByteBPETokenizer.train(documents, vocab_size)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    return tokenizer


def blocks_for(
    split: Path, tokenizer: ByteBPETokenizer, context_length: int, kind: str | None = None
) -> TokenBlocks:
    name = f"{split.stem}-{kind or 'all'}-{context_length}-{tokenizer.vocab_size}.pt"
    path = ARTIFACTS / name
    if path.exists():
        return TokenBlocks(*torch.load(path, weights_only=True))
    blocks = make_blocks(load_documents(split, kind), tokenizer, context_length)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save((blocks.inputs, blocks.targets, blocks.target_bytes), path)
    return blocks


def head(blocks: TokenBlocks, count: int) -> TokenBlocks:
    return TokenBlocks(blocks.inputs[:count], blocks.targets[:count], blocks.target_bytes[:count])


def tiled_attention(query: Tensor, key: Tensor, value: Tensor, block: int = 128) -> Tensor:
    """FlashAttention's blocking. One key block at a time, rescaling by a running maximum."""
    length, head_size = query.shape[-2], query.shape[-1]
    rows = torch.arange(length, device=query.device).unsqueeze(1)
    output = torch.zeros_like(query)
    running_max = torch.full(query.shape[:-1] + (1,), float("-inf"), device=query.device)
    running_sum = torch.zeros_like(running_max)
    for start in range(0, length, block):
        stop = min(start + block, length)
        scores = query @ key[..., start:stop, :].transpose(-2, -1) / math.sqrt(head_size)
        columns = torch.arange(start, stop, device=query.device).unsqueeze(0)
        scores = scores.masked_fill(columns > rows, float("-inf"))
        block_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
        block_max = torch.where(block_max.isinf(), torch.zeros_like(block_max), block_max)
        correction = (running_max - block_max).exp()
        weights = (scores - block_max).exp()
        running_sum = correction * running_sum + weights.sum(dim=-1, keepdim=True)
        output = correction * output + weights @ value[..., start:stop, :]
        running_max = block_max
    return output / running_sum


def stage_tiled(_: argparse.Namespace) -> None:
    torch.manual_seed(0)
    path = RUNS / "day2-exp014-tiled.jsonl"
    for length in (128, 512, 1024):
        query, key, value = (torch.randn(2, 8, length, 32) for _ in range(3))
        reference = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        for block in (64, 128, 256):
            tiled = tiled_attention(query, key, value, block)
            _write(
                path,
                {
                    "block": block,
                    "length": length,
                    "max_abs_error": float((tiled - reference).abs().max()),
                    "type": "tiled_attention",
                },
            )


def _time_attention(shape: tuple[int, ...], dtype: torch.dtype, backend: SDPBackend) -> dict:
    query, key, value = (torch.randn(*shape, dtype=dtype) for _ in range(3))
    with torch.inference_mode(), sdpa_kernel(backend):
        F.scaled_dot_product_attention(query, key, value, is_causal=True)
        started = time.perf_counter()
        F.scaled_dot_product_attention(query, key, value, is_causal=True)
        elapsed = time.perf_counter() - started
    return {"seconds": elapsed, "tokens_per_second": shape[2] / elapsed}


def _measure_one(backend: str, dtype_name: str, length: int) -> dict[str, object]:
    dtype = getattr(torch, dtype_name)
    baseline = _peak_rss()
    result = _time_attention((1, 8, length, 32), dtype, BACKENDS[backend])
    return {
        "backend": backend,
        "dtype": dtype_name,
        "length": length,
        "peak_rss_bytes": _peak_rss(),
        "rss_growth_bytes": _peak_rss() - baseline,
        "type": "sdpa_benchmark",
        **result,
    }


def stage_sdpa(args: argparse.Namespace) -> None:
    """Each measurement runs in a fresh process, because peak RSS is a process high-water mark."""
    torch.manual_seed(0)
    if args.one:
        backend, dtype_name, length = args.one
        print(_json(_measure_one(backend, dtype_name, int(length))), flush=True)
        return
    path = RUNS / "day2-exp014-sdpa.jsonl"
    for length in (1024, 2048, 4096, 8192):
        for dtype_name in ("float32", "bfloat16"):
            for backend in BACKENDS:
                command = [sys.executable, "-m", "octlm.day2", "sdpa", "--one"]
                command += [backend, dtype_name, str(length)]
                output = subprocess.run(command, capture_output=True, text=True, check=True)
                _write(path, json.loads(output.stdout))


def stage_equivalence(args: argparse.Namespace) -> None:
    """SDPA must match the Day 1 handwritten path before we trust it in every later run."""
    torch.manual_seed(0)
    path = RUNS / "day2-exp014-equivalence.jsonl"
    base = ProjectConfig.load(args.config)
    tokens = torch.randint(0, 256, (2, 64))
    for kv_heads in (8, 4, 2, 1):
        outputs = {}
        for attention in ("naive", "sdpa"):
            settings = replace(base.model, attention=attention, kv_heads=kv_heads)
            torch.manual_seed(0)
            model = Decoder(decoder_config(replace(base, model=settings), 256)).eval()
            with torch.inference_mode():
                outputs[attention] = model(tokens)
        difference = (outputs["naive"] - outputs["sdpa"]).abs().max()
        _write(
            path,
            {
                "kv_heads": kv_heads,
                "max_abs_error": float(difference),
                "type": "attention_equivalence",
            },
        )


def _variant_config(base: ProjectConfig, overrides: dict[str, object], seed: int) -> ProjectConfig:
    return replace(
        base,
        model=replace(base.model, **overrides),
        training=replace(base.training, seed=seed),
    )


def _run_one(
    name: str,
    config: ProjectConfig,
    tokenizer: ByteBPETokenizer,
    blocks: dict[str, TokenBlocks],
    dataset_hash: str,
    device: str | None = None,
) -> tuple[Decoder, dict[str, object]]:
    model, _, records = train_model(
        config, tokenizer, blocks["train"], blocks["validation"], dataset_hash, device=device
    )
    decoder = decoder_config(config, tokenizer.vocab_size)
    result: dict[str, object] = {
        "cache_bytes_4k": kv_cache_bytes(decoder, 4096),
        "device": str(next(model.parameters()).device),
        "kv_heads": decoder.key_value_heads,
        "parameters": parameter_count(model),
        "seconds_per_step": float(records[-1]["train_seconds"]) / config.training.steps,
        "seed": config.training.seed,
        "train_loss": records[-1]["train_loss"],
        "type": "variant",
        "variant": name,
    }
    for kind in ("code", "prose"):
        measured = evaluate(model, blocks[kind], tokenizer.pad_id)
        result[f"{kind}_loss"] = measured["loss"]
        result[f"{kind}_bits_per_byte"] = measured["bits_per_byte"]
    return model, result


def stage_variants(args: argparse.Namespace) -> None:
    base = ProjectConfig.load(args.config)
    tokenizer = build_tokenizer(base.tokenizer.vocab_size)
    length = base.model.context_length
    blocks = {
        "train": blocks_for(TRAIN_SPLIT, tokenizer, length),
        "validation": head(blocks_for(VALIDATION_SPLIT, tokenizer, length), EVAL_BLOCKS),
        "code": head(blocks_for(VALIDATION_SPLIT, tokenizer, length, "code"), FINAL_BLOCKS),
        "prose": head(blocks_for(VALIDATION_SPLIT, tokenizer, length, "prose"), FINAL_BLOCKS),
    }
    dataset_hash = data_fingerprint([TRAIN_SPLIT, VALIDATION_SPLIT])
    path = args.out or RUNS / "day2-variants.jsonl"
    selected = args.variants or list(VARIANTS)
    for name in selected:
        for seed in SEEDS[: args.seeds]:
            config = _variant_config(base, VARIANTS[name], seed)
            _, record = _run_one(name, config, tokenizer, blocks, dataset_hash, args.device)
            _write(path, record)


def _length_evaluation(
    model: Decoder, tokenizer: ByteBPETokenizer, length: int, windows: int, scale: float
) -> dict[str, float]:
    model.config = replace(model.config, rope_scale=scale)
    blocks = head(blocks_for(VALIDATION_SPLIT, tokenizer, length), windows)
    measured = evaluate(model, blocks, tokenizer.pad_id)
    return {"bits_per_byte": measured["bits_per_byte"], "loss": measured["loss"]}


def stage_length(args: argparse.Namespace) -> None:
    base = ProjectConfig.load(args.config)
    tokenizer = build_tokenizer(base.tokenizer.vocab_size)
    trained = base.model.context_length
    blocks = {
        "train": blocks_for(TRAIN_SPLIT, tokenizer, trained),
        "validation": head(blocks_for(VALIDATION_SPLIT, tokenizer, trained), EVAL_BLOCKS),
    }
    dataset_hash = data_fingerprint([TRAIN_SPLIT, VALIDATION_SPLIT])
    path = RUNS / "day2-exp009-length.jsonl"
    for position in ("sinusoidal", "rope"):
        config = _variant_config(base, {"position": position}, SEEDS[0])
        model, _, _ = train_model(
            config,
            tokenizer,
            blocks["train"],
            blocks["validation"],
            dataset_hash,
            device=args.device,
        )
        for length in args.lengths:
            for interpolated in (False, True):
                if interpolated and position != "rope":
                    continue
                scale = max(1.0, length / trained) if interpolated else 1.0
                measured = _length_evaluation(model, tokenizer, length, args.windows, scale)
                _write(
                    path,
                    {
                        "interpolated": interpolated,
                        "length": length,
                        "position": position,
                        "rope_scale": scale,
                        "trained_length": trained,
                        "type": "length_sweep",
                        **measured,
                    },
                )


def stage_cache(args: argparse.Namespace) -> None:
    base = ProjectConfig.load(args.config)
    path = RUNS / "day2-exp013-cache.jsonl"
    for kv_heads in (8, 4, 2, 1):
        decoder = DecoderConfig(
            vocab_size=base.tokenizer.vocab_size, **{**vars(base.model), "kv_heads": kv_heads}
        )
        for length in (1024, 4096):
            _write(
                path,
                {
                    "bytes": kv_cache_bytes(decoder, length),
                    "kv_heads": kv_heads,
                    "length": length,
                    "type": "kv_cache",
                },
            )


def stage_report(args: argparse.Namespace) -> None:
    """Summarize the variant grid: mean and spread across seeds, in a stable order."""
    path = args.out or RUNS / "day2-variants.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for name in VARIANTS:
        group = [record for record in records if record["variant"] == name]
        if not group:
            continue
        summary: dict[str, object] = {
            "cache_bytes_4k": group[0]["cache_bytes_4k"],
            "kv_heads": group[0]["kv_heads"],
            "parameters": group[0]["parameters"],
            "seeds": len(group),
            "type": "summary",
            "variant": name,
        }
        for field in (
            "code_loss",
            "prose_loss",
            "code_bits_per_byte",
            "prose_bits_per_byte",
            "seconds_per_step",
        ):
            values = [record[field] for record in group]
            summary[f"{field}_mean"] = statistics.fmean(values)
            summary[f"{field}_spread"] = max(values) - min(values)
        print(_json(summary), flush=True)


STAGES = {
    "tiled": stage_tiled,
    "equivalence": stage_equivalence,
    "sdpa": stage_sdpa,
    "variants": stage_variants,
    "length": stage_length,
    "cache": stage_cache,
    "report": stage_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 2 experiments.")
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--config", type=Path, default=Path("configs/day2.toml"))
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS))
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--one", nargs=3, metavar=("BACKEND", "DTYPE", "LENGTH"))
    parser.add_argument("--out", type=Path, help="JSONL for variant records")
    args = parser.parse_args()
    if not 1 <= args.seeds <= len(SEEDS):
        raise SystemExit(f"seeds must be between 1 and {len(SEEDS)}")
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
