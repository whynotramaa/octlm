from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from octlm.config import ProjectConfig
from octlm.day2 import SEEDS, _write
from octlm.day3 import mtp_agreement
from octlm.day4 import TokenStream, load_model, token_file, validation_blocks
from octlm.model import Decoder, kv_cache_bytes, parameter_count
from octlm.tokenizer import ByteBPETokenizer
from octlm.train import TokenBlocks, data_fingerprint, evaluate, train_model


def variant_config(base: ProjectConfig, variant: str, seed: int) -> ProjectConfig:
    model = base.model
    if variant == "baseline":
        model = replace(
            model, position="learned", norm="layernorm", feed_forward="gelu", kv_heads=0
        )
    elif variant == "mtp":
        model = replace(model, mtp_depth=2)
    elif variant != "modern":
        raise ValueError(f"unknown variant: {variant}")
    return replace(base, model=model, training=replace(base.training, seed=seed))


def spread(records: list[dict[str, object]], variant: str) -> dict[str, float]:
    values = [float(row["bits_per_byte"]) for row in records if row["variant"] == variant]
    if not values:
        raise ValueError(f"no completed {variant} runs")
    return {"mean": statistics.fmean(values), "spread": max(values) - min(values)}


def read_runs(
    path: Path,
    base: ProjectConfig,
    data_hash: str,
    tokenizer_hash: str,
    variants: tuple[str, ...],
) -> tuple[list[dict[str, object]], set[tuple[str, int]]]:
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed: set[tuple[str, int]] = set()
    for row in records:
        variant, seed = row.get("variant"), row.get("seed")
        if variant not in variants or seed not in SEEDS:
            raise ValueError("output contains an unknown variant or seed")
        key = (variant, seed)
        if key in completed:
            raise ValueError("output contains a duplicate run")
        if (
            row.get("config_hash") != variant_config(base, variant, seed).fingerprint
            or row.get("data_hash") != data_hash
            or row.get("tokenizer_hash") != tokenizer_hash
        ):
            raise ValueError("output belongs to different inputs or settings")
        completed.add(key)
    return records, completed


def inputs(
    args: argparse.Namespace,
) -> tuple[ProjectConfig, ByteBPETokenizer, TokenStream, TokenBlocks, str]:
    base = ProjectConfig.load(args.config)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != base.tokenizer.vocab_size:
        raise ValueError("tokenizer vocabulary does not match configuration")
    paths = [args.data / "train.bin", args.data / "valid.bin"]
    train_tokens, valid_tokens = (token_file(path) for path in paths)
    train = TokenStream(train_tokens, base.model.context_length)
    valid = validation_blocks(valid_tokens, tokenizer, base.model.context_length)
    return base, tokenizer, train, valid, data_fingerprint(paths)


def train_variant(
    base: ProjectConfig,
    tokenizer: ByteBPETokenizer,
    train: TokenStream,
    valid: TokenBlocks,
    dataset_hash: str,
    variant: str,
    seed: int,
    device: str,
) -> dict[str, object]:
    config = variant_config(base, variant, seed)
    model, _, records = train_model(
        config, tokenizer, train, valid, dataset_hash, device=device, mixed_precision=True
    )
    final = records[-1]
    row: dict[str, object] = {
        "bits_per_byte": final["validation_bits_per_byte"],
        "config_hash": config.fingerprint,
        "data_hash": dataset_hash,
        "device": final["device"],
        "loss": final["validation_loss"],
        "parameters": parameter_count(model),
        "pytorch": torch.__version__,
        "seconds_per_step": float(final["train_seconds"]) / config.training.steps,
        "seed": seed,
        "tokenizer_hash": tokenizer.fingerprint,
        "tokens": config.training.steps * config.training.batch_size * config.model.context_length,
        "type": "mtp_run" if variant == "mtp" else "seed_run",
        "variant": variant,
    }
    if variant == "mtp":
        row["depth2_top1_accuracy"] = mtp_agreement(model, valid, tokenizer.pad_id)
    return row


def stage_spread(args: argparse.Namespace) -> None:
    base, tokenizer, train, valid, dataset_hash = inputs(args)
    existing, completed = read_runs(
        args.out, base, dataset_hash, tokenizer.fingerprint, ("baseline", "modern")
    )
    for variant in ("baseline", "modern"):
        for seed in SEEDS[: args.seeds]:
            if (variant, seed) in completed:
                continue
            row = train_variant(
                base, tokenizer, train, valid, dataset_hash, variant, seed, args.device
            )
            _write(args.out, row)
            existing.append(row)
            completed.add((variant, seed))
    for variant in ("baseline", "modern"):
        if all((variant, seed) in completed for seed in SEEDS[: args.seeds]):
            print(json.dumps({"variant": variant, **spread(existing, variant)}, sort_keys=True))


def stage_mtp(args: argparse.Namespace) -> None:
    base, tokenizer, train, valid, dataset_hash = inputs(args)
    controls, completed = read_runs(
        args.spread_in, base, dataset_hash, tokenizer.fingerprint, ("baseline", "modern")
    )
    if any(
        (variant, seed) not in completed
        for variant in ("baseline", "modern")
        for seed in SEEDS[: args.seeds]
    ):
        raise ValueError("EXP-069 must finish before EXP-070 starts")
    threshold = max(spread(controls, variant)["spread"] for variant in ("baseline", "modern"))
    existing, completed = read_runs(args.out, base, dataset_hash, tokenizer.fingerprint, ("mtp",))
    for seed in SEEDS[: args.seeds]:
        if ("mtp", seed) in completed:
            continue
        row = train_variant(base, tokenizer, train, valid, dataset_hash, "mtp", seed, args.device)
        _write(args.out, row)
        existing.append(row)
        completed.add(("mtp", seed))
    if all(("mtp", seed) in completed for seed in SEEDS[: args.seeds]):
        print(
            json.dumps(
                {
                    "control_mean": spread(controls, "modern")["mean"],
                    "mtp_mean": spread(existing, "mtp")["mean"],
                    "minimum_detectable_effect": threshold,
                },
                sort_keys=True,
            )
        )


def timed_generation(
    model: Decoder, prompt: torch.Tensor, length: int, cached: bool
) -> tuple[torch.Tensor, float]:
    if prompt.device.type == "cuda":
        torch.cuda.synchronize(prompt.device)
    started = time.perf_counter()
    output = model.generate_cached(prompt, length) if cached else model.generate(prompt, length)
    if prompt.device.type == "cuda":
        torch.cuda.synchronize(prompt.device)
    return output, time.perf_counter() - started


def stage_cache(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"{args.out} already contains a measurement")
    model, tokenizer = load_model(args)
    if model.config.attention != "sdpa" or any(
        length > model.config.context_length for length in args.lengths
    ):
        raise ValueError("cache benchmark requires SDPA and lengths within context")
    device = next(model.parameters()).device
    if device.type == "cuda":
        model.half()
    prompt = torch.tensor([[tokenizer.bos_id]], device=device)
    model.generate(prompt, 2)
    model.generate_cached(prompt, 2)
    for length in args.lengths:
        naive, naive_seconds = timed_generation(model, prompt, length, False)
        cached, cached_seconds = timed_generation(model, prompt, length, True)
        _, cache = model.forward_cached(cached[:, :-1])
        cache_bytes = sum(
            (key.numel() + value.numel()) * key.element_size() for key, value in cache
        )
        expected = kv_cache_bytes(model.config, length, next(model.parameters()).element_size())
        if cache_bytes != expected:
            raise RuntimeError("actual cache bytes differ from the model formula")
        row = {
            "cache_bytes": cache_bytes,
            "cached_tokens_per_second": length / cached_seconds,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "greedy_equal": bool(torch.equal(naive, cached)),
            "length": length,
            "naive_tokens_per_second": length / naive_seconds,
            "pytorch": torch.__version__,
            "type": "cache_benchmark",
        }
        _write(args.out, row)
        if not row["greedy_equal"]:
            raise RuntimeError(f"cached generation changed greedy output at length {length}")


class Int8Linear(nn.Module):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        weight = source.weight.detach().float()
        scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127
        self.register_buffer(
            "qweight", (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        )
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if source.bias is None else source.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.to(x.dtype)
        weight.mul_(self.scale.to(x.dtype).unsqueeze(1))
        return F.linear(x, weight, None if self.bias is None else self.bias.to(x.dtype))


def quantize_int8(model: Decoder) -> Decoder:
    if model.training:
        raise ValueError("quantization requires eval mode")
    for parent in model.blocks.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, Int8Linear(child))
    return model


def stage_quant(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"{args.out} already contains a measurement")
    model, tokenizer = load_model(args)
    device = next(model.parameters()).device
    if device.type == "cuda":
        model.half()
    valid_tokens = token_file(args.data / "valid.bin")
    valid = validation_blocks(valid_tokens, tokenizer, model.config.context_length)
    baseline_bpb = evaluate(model, valid, tokenizer.pad_id)["bits_per_byte"]
    model.eval()
    quantized = quantize_int8(copy.deepcopy(model))
    quantized_bpb = evaluate(quantized, valid, tokenizer.pad_id)["bits_per_byte"]
    quantized.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    baseline_path = args.out.with_suffix(".float.pt")
    quantized_path = args.out.with_suffix(".int8.pt")
    torch.save(model.state_dict(), baseline_path)
    torch.save(quantized.state_dict(), quantized_path)
    prompt = torch.tensor([[tokenizer.bos_id]], device=device)
    model.generate_cached(prompt, 2)
    quantized.generate_cached(prompt, 2)
    _, baseline_seconds = timed_generation(model, prompt, args.quant_tokens, True)
    _, quantized_seconds = timed_generation(quantized, prompt, args.quant_tokens, True)
    _write(
        args.out,
        {
            "baseline_bits_per_byte": baseline_bpb,
            "baseline_model_bytes": baseline_path.stat().st_size,
            "baseline_tokens_per_second": args.quant_tokens / baseline_seconds,
            "bits_per_byte_delta": quantized_bpb - baseline_bpb,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "generated_tokens": args.quant_tokens,
            "int8_bits_per_byte": quantized_bpb,
            "int8_model_bytes": quantized_path.stat().st_size,
            "int8_tokens_per_second": args.quant_tokens / quantized_seconds,
            "pytorch": torch.__version__,
            "type": "int8_benchmark",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 5 experiments.")
    parser.add_argument("stage", choices=("spread", "mtp", "cache", "quant"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/day4/bpe.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/day4/run.pt"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--spread-in", type=Path, default=Path("runs/day5-spread.jsonl"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lengths", type=int, nargs="+", default=(64, 256, 512))
    parser.add_argument("--quant-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 1 <= args.seeds <= len(SEEDS):
        parser.error(f"seeds must be between 1 and {len(SEEDS)}")
    if any(length < 1 for length in args.lengths):
        parser.error("lengths must be positive")
    if args.quant_tokens < 1:
        parser.error("quant-tokens must be positive")
    args.config = args.config or Path(
        "configs/day4.toml" if args.stage in ("cache", "quant") else "configs/day5.toml"
    )
    args.out = args.out or Path(f"runs/day5-{args.stage}.jsonl")
    {"spread": stage_spread, "mtp": stage_mtp, "cache": stage_cache, "quant": stage_quant}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
