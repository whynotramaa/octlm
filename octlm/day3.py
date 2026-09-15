from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from octlm.config import ProjectConfig
from octlm.day2 import (
    EVAL_BLOCKS,
    FINAL_BLOCKS,
    RUNS,
    SEEDS,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    VARIANTS,
    _run_one,
    _variant_config,
    _write,
    blocks_for,
    build_tokenizer,
    head,
)
from octlm.model import Decoder, attention_mask, kv_cache_bytes, mask_density
from octlm.tokenizer import ByteBPETokenizer
from octlm.train import TokenBlocks, _json, data_fingerprint, decoder_config

MODERN = VARIANTS["modern"]
PROBE_SEED = 20260915
NEEDLE_LENGTH = 8

GRIDS: dict[str, dict[str, dict[str, object]]] = {
    "mtp": {
        "depth-1": {"mtp_depth": 1},
        "depth-2": {"mtp_depth": 2},
        "depth-3": {"mtp_depth": 3},
    },
    "sparse": {
        "full": {},
        "window-256": {"attention_window": 256},
        "window-128": {"attention_window": 128},
        "window-128-stride-64": {"attention_window": 128, "attention_stride": 64},
    },
    "compressed": {
        "window-128": {"attention_window": 128},
        "block-4": {"attention_window": 128, "kv_compress_block": 4},
        "block-8": {"attention_window": 128, "kv_compress_block": 8},
    },
    "mla": {
        "gqa-2": {},
        "mla-32": {"attention": "mla", "mla_rank": 32, "mla_rope_dim": 16},
        "mla-64": {"attention": "mla", "mla_rank": 64, "mla_rope_dim": 16},
        "mla-128": {"attention": "mla", "mla_rank": 128, "mla_rope_dim": 16},
    },
}


def _corpus(args: argparse.Namespace) -> tuple[ProjectConfig, ByteBPETokenizer, dict, str]:
    base = ProjectConfig.load(args.config)
    tokenizer = build_tokenizer(base.tokenizer.vocab_size)
    length = base.model.context_length
    blocks = {
        "train": blocks_for(TRAIN_SPLIT, tokenizer, length),
        "validation": head(blocks_for(VALIDATION_SPLIT, tokenizer, length), EVAL_BLOCKS),
        "code": head(blocks_for(VALIDATION_SPLIT, tokenizer, length, "code"), FINAL_BLOCKS),
        "prose": head(blocks_for(VALIDATION_SPLIT, tokenizer, length, "prose"), FINAL_BLOCKS),
    }
    return base, tokenizer, blocks, data_fingerprint([TRAIN_SPLIT, VALIDATION_SPLIT])


def mtp_agreement(model: Decoder, blocks: TokenBlocks, pad_id: int) -> float:
    """How often the depth-2 head's argmax is the true t+2 token. The Phase 5 draft signal."""
    if model.config.mtp_depth < 2:
        return 0.0
    model.eval()
    device = next(model.parameters()).device
    hits = total = 0
    with torch.inference_mode():
        for inputs, targets in zip(blocks.inputs, blocks.targets, strict=True):
            stack = model(inputs.unsqueeze(0).to(device), all_depths=True)
            predicted = stack[1, 0, :-1].argmax(dim=-1)
            actual = targets[1:].to(device)
            valid = actual != pad_id
            hits += int((predicted[valid] == actual[valid]).sum())
            total += int(valid.sum())
    model.train()
    return hits / max(1, total)


def _span_nll(model: Decoder, sequence: torch.Tensor, start: int, stop: int) -> float:
    device = next(model.parameters()).device
    inputs = sequence[:-1].unsqueeze(0).to(device)
    targets = sequence[1:].to(device)
    with torch.inference_mode():
        logits = model(inputs)[0]
    losses = F.cross_entropy(logits, targets, reduction="none")
    return float(losses[start:stop].mean())


def copy_probe(
    model: Decoder, blocks: TokenBlocks, vocab_size: int, depths: list[int]
) -> list[dict[str, object]]:
    """Plant a random span at `depth`, repeat it at the end, and read the NLL on the repeat.

    A prompt-and-answer needle test measures instruction following, which a base model at this
    scale does not have. Recalling a span it has already seen is the part it can do.
    """
    generator = torch.Generator().manual_seed(PROBE_SEED)
    filler = blocks.inputs[0].clone()
    length = filler.shape[0]
    results = []
    for depth in depths:
        if depth + NEEDLE_LENGTH > length - NEEDLE_LENGTH:
            continue
        needle = torch.randint(256, vocab_size, (NEEDLE_LENGTH,), generator=generator)
        decoy = torch.randint(256, vocab_size, (NEEDLE_LENGTH,), generator=generator)
        start = length - NEEDLE_LENGTH
        planted, control = filler.clone(), filler.clone()
        planted[depth : depth + NEEDLE_LENGTH] = needle
        control[depth : depth + NEEDLE_LENGTH] = decoy
        planted[start:] = needle
        control[start:] = needle
        # `_span_nll` reads targets, which are the inputs shifted by one.
        span = (start - 1, length - 1)
        recalled = _span_nll(model, planted, *span)
        unseen = _span_nll(model, control, *span)
        results.append(
            {
                "depth": depth,
                "gain": unseen - recalled,
                "needle_nll_recalled": recalled,
                "needle_nll_unseen": unseen,
                "type": "copy_probe",
            }
        )
    return results


def flash_accepts(mask: torch.Tensor | None, device: torch.device) -> bool:
    """Does the flash kernel take this mask, or does SDPA fall back to a slower backend?

    Device-dependent, and the device is the answer: PyTorch's CPU flash path accepts an explicit
    mask, the CUDA kernel does not. Probing on CPU for a CUDA run would report the wrong backend.
    """
    rows = mask.shape[-2] if mask is not None else 64
    columns = mask.shape[-1] if mask is not None else 64
    query = torch.randn(1, 2, rows, 32, device=device)
    key = torch.randn(1, 2, columns, 32, device=device)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(query, key, key, attn_mask=mask, is_causal=mask is None)
    except RuntimeError:
        return False
    return True


def _extras(stage: str, name: str, model: Decoder, blocks: dict, tokenizer: ByteBPETokenizer):
    config = model.config
    if stage == "mtp":
        return {
            "mtp_depth": config.mtp_depth,
            "depth2_top1_agreement": mtp_agreement(model, blocks["validation"], tokenizer.pad_id),
        }
    if stage == "sparse":
        device = next(model.parameters()).device
        mask = attention_mask(
            config.context_length, config.attention_window, config.attention_stride, device
        )
        explicit = bool(config.attention_window)
        return {
            "attention_window": config.attention_window,
            "attention_stride": config.attention_stride,
            "flash_accepts_mask": flash_accepts(mask if explicit else None, device),
            "flash_probe_device": str(device),
            "mask_density": mask_density(config, config.context_length),
        }
    if stage == "compressed":
        return {
            "attention_window": config.attention_window,
            "cache_dims": config.cache_dims,
            "cached_positions_4k": config.cached_positions(4096),
            "kv_compress_block": config.kv_compress_block,
        }
    return {"cache_dims": config.cache_dims, "mla_rank": config.mla_rank}


def _sweep(args: argparse.Namespace, stage: str) -> None:
    base, tokenizer, blocks, dataset_hash = _corpus(args)
    path = RUNS / f"day3-{stage}.jsonl"
    depths = list(range(0, base.model.context_length - 2 * NEEDLE_LENGTH, args.probe_step))
    for name, overrides in GRIDS[stage].items():
        for index, seed in enumerate(SEEDS[: args.seeds]):
            config = _variant_config(base, {**MODERN, **overrides}, seed)
            model, record = _run_one(name, config, tokenizer, blocks, dataset_hash, args.device)
            record["stage"] = stage
            record.update(_extras(stage, name, model, blocks, tokenizer))
            _write(path, record)
            if index == 0 and stage in ("sparse", "compressed"):
                for probe in copy_probe(model, blocks["validation"], tokenizer.vocab_size, depths):
                    _write(path, {"seed": seed, "stage": stage, "variant": name, **probe})


def stage_cache(args: argparse.Namespace) -> None:
    """Arithmetic only. No training, so this one runs on the laptop."""
    base = ProjectConfig.load(args.config)
    path = RUNS / "day3-cache.jsonl"
    for stage in ("compressed", "mla"):
        for name, overrides in GRIDS[stage].items():
            config = _variant_config(base, {**MODERN, **overrides}, SEEDS[0])
            decoder = decoder_config(config, base.tokenizer.vocab_size)
            for length in (1024, 4096):
                _write(
                    path,
                    {
                        "bytes": kv_cache_bytes(decoder, length),
                        "cache_dims": decoder.cache_dims,
                        "cached_positions": decoder.cached_positions(length),
                        "length": length,
                        "stage": stage,
                        "type": "kv_cache",
                        "variant": name,
                    },
                )


def stage_report(args: argparse.Namespace) -> None:
    """Mean and spread per variant, in the grid's order, one block per stage that has records."""
    for stage, grid in GRIDS.items():
        path = RUNS / f"day3-{stage}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for name in grid:
            group = [r for r in records if r.get("variant") == name and r.get("type") == "variant"]
            if not group:
                continue
            summary: dict[str, object] = {
                "cache_bytes_4k": group[0]["cache_bytes_4k"],
                "parameters": group[0]["parameters"],
                "seeds": len(group),
                "stage": stage,
                "type": "summary",
                "variant": name,
            }
            for field in ("code_bits_per_byte", "prose_bits_per_byte", "seconds_per_step"):
                values = [r[field] for r in group]
                summary[f"{field}_mean"] = statistics.fmean(values)
                summary[f"{field}_spread"] = max(values) - min(values)
            for field in ("depth2_top1_agreement", "mask_density", "flash_accepts_mask"):
                if field in group[0]:
                    summary[field] = group[0][field]
            print(_json(summary), flush=True)
        for record in (r for r in records if r.get("type") == "copy_probe"):
            print(_json(record), flush=True)


STAGES = {
    "mtp": lambda args: _sweep(args, "mtp"),
    "sparse": lambda args: _sweep(args, "sparse"),
    "compressed": lambda args: _sweep(args, "compressed"),
    "mla": lambda args: _sweep(args, "mla"),
    "cache": stage_cache,
    "report": stage_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 3a experiments.")
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--config", type=Path, default=Path("configs/day3.toml"))
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--probe-step", type=int, default=64, help="needle depths to sweep")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    if not 1 <= args.seeds <= len(SEEDS):
        raise SystemExit(f"seeds must be between 1 and {len(SEEDS)}")
    if args.probe_step < 1:
        raise SystemExit("probe-step must be positive")
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
