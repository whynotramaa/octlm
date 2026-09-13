from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSettings:
    context_length: int
    d_model: int
    n_heads: int
    n_layers: int
    ff_multiplier: int
    dropout: float


@dataclass(frozen=True)
class TrainingSettings:
    seed: int
    batch_size: int
    steps: int
    eval_interval: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float


@dataclass(frozen=True)
class TokenizerSettings:
    vocab_size: int
    min_frequency: int


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelSettings
    training: TrainingSettings
    tokenizer: TokenizerSettings

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
        try:
            config = cls(
                model=ModelSettings(**data["model"]),
                training=TrainingSettings(**data["training"]),
                tokenizer=TokenizerSettings(**data["tokenizer"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid configuration: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        model = self.model
        training = self.training
        tokenizer = self.tokenizer
        if min(model.context_length, model.d_model, model.n_heads, model.n_layers) < 1:
            raise ValueError("model dimensions must be positive")
        if model.d_model % model.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if model.ff_multiplier < 1 or not 0 <= model.dropout < 1:
            raise ValueError("ff_multiplier and dropout are invalid")
        if min(training.batch_size, training.steps, training.eval_interval) < 1:
            raise ValueError("training counts must be positive")
        if not 0 <= training.warmup_steps <= training.steps:
            raise ValueError("warmup_steps must be between zero and steps")
        if min(training.learning_rate, training.min_learning_rate, training.grad_clip) <= 0:
            raise ValueError("learning rates and grad_clip must be positive")
        if tokenizer.vocab_size < 261 or tokenizer.min_frequency < 1:
            raise ValueError("tokenizer settings are invalid")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
