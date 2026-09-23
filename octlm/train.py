from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from octlm.config import ProjectConfig
from octlm.corpus import load_documents
from octlm.model import Decoder, DecoderConfig, parameter_count
from octlm.tokenizer import ByteBPETokenizer, CharacterTokenizer

Tokenizer: TypeAlias = CharacterTokenizer | ByteBPETokenizer


@dataclass
class TokenBlocks:
    inputs: Tensor
    targets: Tensor
    target_bytes: Tensor

    def batch(self, size: int, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        indices = torch.randint(len(self.inputs), (size,), generator=generator)
        return self.inputs[indices], self.targets[indices], self.target_bytes[indices]


def _json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def data_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_device(name: str | None) -> torch.device:
    """`auto` and `None` take the GPU when the machine has one."""
    if name in (None, "auto"):
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def decoder_config(config: ProjectConfig, vocab_size: int) -> DecoderConfig:
    return DecoderConfig(vocab_size=vocab_size, **asdict(config.model))


def read_documents(path: Path) -> list[str]:
    """A .jsonl path is a corpus split, anything else is one document."""
    return load_documents(path) if path.suffix == ".jsonl" else [path.read_text()]


def make_blocks(texts: str | list[str], tokenizer: Tokenizer, context_length: int) -> TokenBlocks:
    documents = [texts] if isinstance(texts, str) else texts
    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document, add_special_tokens=True))
    width = context_length + 1
    if len(token_ids) < width:
        token_ids.extend([tokenizer.pad_id] * (width - len(token_ids)))
    block_count = len(token_ids) // width
    tokens = torch.tensor(token_ids[: block_count * width]).view(block_count, width)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    byte_lengths = torch.tensor(
        [[tokenizer.token_bytes(int(token)) for token in row] for row in targets]
    )
    return TokenBlocks(inputs, targets, byte_lengths)


def learning_rate(step: int, config: ProjectConfig) -> float:
    settings = config.training
    if step < settings.warmup_steps:
        return settings.learning_rate * (step + 1) / max(1, settings.warmup_steps)
    progress = (step - settings.warmup_steps) / max(1, settings.steps - settings.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return settings.min_learning_rate + cosine * (
        settings.learning_rate - settings.min_learning_rate
    )


def optimizer_for(model: nn.Module, config: ProjectConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.training.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.training.learning_rate, betas=(0.9, 0.95))


def mtp_loss(model: Decoder, inputs: Tensor, targets: Tensor, pad_id: int) -> Tensor:
    """Depth k predicts token t+k, so depth k reads targets shifted by k and loses k positions."""
    depth = model.config.mtp_depth
    if depth == 1:
        logits = model(inputs)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=pad_id)
    stack = model(inputs, all_depths=True)
    losses = []
    for k, logits in enumerate(stack):
        span = targets.shape[1] - k
        shifted = targets[:, k:]
        losses.append(
            F.cross_entropy(logits[:, :span].flatten(0, 1), shifted.flatten(), ignore_index=pad_id)
        )
    return torch.stack(losses).mean()


def evaluate(model: Decoder, blocks: TokenBlocks, pad_id: int) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    nll_sum = 0.0
    token_count = 0
    byte_count = 0
    with torch.inference_mode():
        for inputs, targets, byte_lengths in zip(
            blocks.inputs, blocks.targets, blocks.target_bytes, strict=True
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            byte_lengths = byte_lengths.to(device)
            logits = model(inputs.unsqueeze(0))
            losses = F.cross_entropy(
                logits.flatten(0, 1), targets, ignore_index=pad_id, reduction="none"
            )
            valid = targets != pad_id
            nll_sum += losses[valid].sum().item()
            token_count += valid.sum().item()
            byte_count += byte_lengths[valid].sum().item()
    model.train()
    nll = nll_sum / max(1, token_count)
    return {
        "bits_per_byte": nll_sum / (math.log(2) * max(1, byte_count)),
        "loss": nll,
        "perplexity": math.exp(nll),
        "tokens": token_count,
    }


def save_checkpoint(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: Decoder,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    expected: dict[str, str],
    scaler: torch.amp.GradScaler | None = None,
) -> int:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"cannot load checkpoint: {error}") from error
    if state.get("format") != "octlm-checkpoint-v1":
        raise ValueError("unsupported checkpoint format")
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(f"checkpoint {name} does not match")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    generator.set_state(state["sampler_rng"])
    if scaler is not None:
        scaler.load_state_dict(state.get("scaler", {}))
    return int(state["step"])


def checkpoint_state(
    model: Decoder,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    step: int,
    config: ProjectConfig,
    tokenizer: Tokenizer,
    dataset_hash: str,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, object]:
    return {
        "config_hash": config.fingerprint,
        "data_hash": dataset_hash,
        "format": "octlm-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "sampler_rng": generator.get_state(),
        "scaler": scaler.state_dict() if scaler else {},
        "step": step,
        "tokenizer_hash": tokenizer.fingerprint,
        "tokens_processed": step * config.training.batch_size * config.model.context_length,
        "torch_rng": torch.get_rng_state(),
    }


def train_model(
    config: ProjectConfig,
    tokenizer: Tokenizer,
    train_blocks: TokenBlocks,
    validation_blocks: TokenBlocks,
    dataset_hash: str,
    stop_after: int | None = None,
    checkpoint: Path | None = None,
    resume: Path | None = None,
    metrics_path: Path | None = None,
    device: str | None = None,
    mixed_precision: bool = False,
) -> tuple[Decoder, torch.optim.AdamW, list[dict[str, float | int | str]]]:
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    generator = torch.Generator().manual_seed(config.training.seed + 1)
    target = resolve_device(device)
    model = Decoder(decoder_config(config, _vocab_size(tokenizer))).to(target)
    optimizer = optimizer_for(model, config)
    dtype = autocast_dtype(target) if mixed_precision else None
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    expected = {
        "config_hash": config.fingerprint,
        "data_hash": dataset_hash,
        "tokenizer_hash": tokenizer.fingerprint,
    }
    start_step = (
        load_checkpoint(resume, model, optimizer, generator, expected, scaler) if resume else 0
    )
    final_step = min(stop_after or config.training.steps, config.training.steps)
    records: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    train_seconds = 0.0
    for step in range(start_step, final_step):
        tick = time.perf_counter()
        rate = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        inputs, targets, _ = train_blocks.batch(config.training.batch_size, generator)
        inputs, targets = inputs.to(target), targets.to(target)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target.type, dtype=dtype, enabled=dtype is not None):
            loss = mtp_loss(model, inputs, targets, tokenizer.pad_id)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        train_seconds += time.perf_counter() - tick
        if (step + 1) % config.training.eval_interval == 0 or step + 1 == final_step:
            validation = evaluate(model, validation_blocks, tokenizer.pad_id)
            record: dict[str, float | int | str] = {
                "device": str(target),
                "dtype": str(dtype or torch.float32),
                "elapsed_seconds": time.perf_counter() - started,
                "gradient_norm": float(gradient_norm),
                "learning_rate": rate,
                "step": step + 1,
                "train_loss": loss.item(),
                "train_seconds": train_seconds,
                "type": "training",
                **{f"validation_{key}": value for key, value in validation.items()},
            }
            records.append(record)
            print(_json(record), flush=True)
            if metrics_path:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a") as file:
                    file.write(_json(record) + "\n")
            if checkpoint:
                save_checkpoint(
                    checkpoint,
                    checkpoint_state(
                        model,
                        optimizer,
                        generator,
                        step + 1,
                        config,
                        tokenizer,
                        dataset_hash,
                        scaler,
                    ),
                )
    return model, optimizer, records


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _vocab_size(tokenizer: Tokenizer) -> int:
    return (
        len(tokenizer.vocabulary)
        if isinstance(tokenizer, CharacterTokenizer)
        else tokenizer.vocab_size
    )


def decode_generation(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    if isinstance(tokenizer, ByteBPETokenizer):
        return tokenizer.decode(token_ids, errors="replace", skip_special_tokens=True)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def load_or_train_tokenizer(kind: str, documents: list[str], config: ProjectConfig) -> Tokenizer:
    if kind == "character":
        return CharacterTokenizer.train(documents)
    return ByteBPETokenizer.train(
        documents, config.tokenizer.vocab_size, config.tokenizer.min_frequency
    )


def dry_run(config: ProjectConfig) -> None:
    model = Decoder(decoder_config(config, config.tokenizer.vocab_size))
    inputs = torch.zeros((1, config.model.context_length), dtype=torch.long)
    logits = model(inputs)
    print(
        _json(
            {
                "config_sha256": config.fingerprint,
                "logits_shape": list(logits.shape),
                "parameters": parameter_count(model),
                "type": "dry_run",
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Day 1 decoder.")
    parser.add_argument("--config", type=Path, default=Path("configs/day1.toml"))
    parser.add_argument("--tokenizer", choices=("character", "bpe"), default="bpe")
    parser.add_argument("--train", type=Path, default=Path("PLAN.md"))
    parser.add_argument("--validation", type=Path, default=Path("day-wise.md"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    if args.dry_run:
        dry_run(config)
        return
    train_documents = read_documents(args.train)
    validation_documents = read_documents(args.validation)
    tokenizer = load_or_train_tokenizer(args.tokenizer, train_documents, config)
    train_blocks = make_blocks(train_documents, tokenizer, config.model.context_length)
    validation_blocks = make_blocks(validation_documents, tokenizer, config.model.context_length)
    if args.overfit:
        train_blocks = TokenBlocks(
            train_blocks.inputs[:1], train_blocks.targets[:1], train_blocks.target_bytes[:1]
        )
        validation_blocks = train_blocks
    dataset_hash = data_fingerprint([args.train, args.validation])
    model, _, _ = train_model(
        config,
        tokenizer,
        train_blocks,
        validation_blocks,
        dataset_hash,
        args.stop_after,
        args.checkpoint,
        args.resume,
        args.metrics,
        args.device,
    )
    prompt = train_blocks.inputs[0, :8].unsqueeze(0).to(next(model.parameters()).device)
    generated = model.generate(prompt, 32)[0].tolist()
    print(_json({"text": decode_generation(tokenizer, generated), "type": "generation"}))
    print(
        _json(
            {
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "type": "resources",
            }
        )
    )


if __name__ == "__main__":
    main()
