from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

POSITIONS = ("learned", "sinusoidal", "rope")
NORMS = ("layernorm", "rmsnorm")
FEED_FORWARDS = ("gelu", "swiglu")
ATTENTIONS = ("naive", "sdpa")
RESIDUALS = ("pre", "post")
ROPE_BASE = 10000.0


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.0
    position: str = "learned"
    norm: str = "layernorm"
    feed_forward: str = "gelu"
    attention: str = "naive"
    residual: str = "pre"
    kv_heads: int = 0
    rope_scale: float = 1.0

    @property
    def head_size(self) -> int:
        return self.d_model // self.n_heads

    @property
    def key_value_heads(self) -> int:
        return self.kv_heads or self.n_heads

    @property
    def hidden_size(self) -> int:
        """SwiGLU spends three matrices instead of two, so 2/3 holds the parameter count."""
        full = self.d_model * self.ff_multiplier
        if self.feed_forward == "gelu":
            return full
        return max(8, round(full * 2 / 3 / 8) * 8)

    def validate(self) -> None:
        if min(self.vocab_size, self.context_length, self.d_model, self.n_heads, self.n_layers) < 1:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ff_multiplier < 1 or not 0 <= self.dropout < 1:
            raise ValueError("ff_multiplier and dropout are invalid")
        for name, value, allowed in (
            ("position", self.position, POSITIONS),
            ("norm", self.norm, NORMS),
            ("feed_forward", self.feed_forward, FEED_FORWARDS),
            ("attention", self.attention, ATTENTIONS),
            ("residual", self.residual, RESIDUALS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.kv_heads < 0 or self.n_heads % self.key_value_heads:
            raise ValueError("kv_heads must divide n_heads")
        if self.position == "rope" and self.head_size % 2:
            raise ValueError("RoPE needs an even head size")
        if self.rope_scale <= 0:
            raise ValueError("rope_scale must be positive")


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.epsilon)
        return (x.float() * scale).type_as(x) * self.weight


def make_norm(config: DecoderConfig) -> nn.Module:
    return nn.LayerNorm(config.d_model) if config.norm == "layernorm" else RMSNorm(config.d_model)


def sinusoidal_positions(length: int, width: int, device: torch.device) -> Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, width, 2, device=device, dtype=torch.float32)
    angle = position / torch.pow(ROPE_BASE, index / width)
    table = torch.zeros(length, width, device=device)
    table[:, 0::2] = angle.sin()
    table[:, 1::2] = angle.cos()
    return table


def rope_tables(
    length: int, head_size: int, scale: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    index = torch.arange(0, head_size, 2, device=device, dtype=torch.float32)
    inverse_frequency = 1.0 / torch.pow(ROPE_BASE, index / head_size)
    position = torch.arange(length, device=device, dtype=torch.float32) / scale
    angle = torch.outer(position, inverse_frequency)
    return angle.cos(), angle.sin()


def apply_rope(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate each adjacent channel pair of [batch, heads, length, head_size] by its angle."""
    pairs = x.float().unflatten(-1, (-1, 2))
    left, right = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((left * cosine - right * sine, left * sine + right * cosine), dim=-1)
    return rotated.flatten(-2).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.kv_heads = config.key_value_heads
        self.head_size = config.head_size
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, self.kv_heads * self.head_size, bias=False)
        self.value = nn.Linear(config.d_model, self.kv_heads * self.head_size, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask.view(1, 1, *mask.shape), persistent=False)

    def causal_mask_for(self, length: int) -> Tensor:
        if length <= self.causal_mask.shape[-1]:
            return self.causal_mask[:, :, :length, :length]
        mask = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=self.query.weight.device)
        )
        return mask.view(1, 1, length, length)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        query = self.query(x).view(batch, length, self.n_heads, self.head_size).transpose(1, 2)
        key = self.key(x).view(batch, length, self.kv_heads, self.head_size).transpose(1, 2)
        value = self.value(x).view(batch, length, self.kv_heads, self.head_size).transpose(1, 2)
        return query, key, value

    def _naive(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        groups = self.n_heads // self.kv_heads
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        length = query.shape[2]
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.causal_mask_for(length), float("-inf"))
        return self.dropout(scores.softmax(dim=-1)) @ value

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        batch, length, width = x.shape
        query, key, value = self._project(x)
        if rope is not None:
            query = apply_rope(query, *rope)
            key = apply_rope(key, *rope)
        if self.config.attention == "sdpa":
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0,
                enable_gqa=self.kv_heads != self.n_heads,
            )
        else:
            attended = self._naive(query, key, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.gated = config.feed_forward == "swiglu"
        self.up = nn.Linear(config.d_model, hidden, bias=False)
        self.gate = nn.Linear(config.d_model, hidden, bias=False) if self.gated else None
        self.down = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.gate is None:
            return self.dropout(self.down(F.gelu(self.up(x))))
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.pre_norm = config.residual == "pre"
        self.attention_norm = make_norm(config)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = make_norm(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        if self.pre_norm:
            x = x + self.attention(self.attention_norm(x), rope)
            return x + self.feed_forward(self.feed_forward_norm(x))
        x = self.attention_norm(x + self.attention(x, rope))
        return self.feed_forward_norm(x + self.feed_forward(x))


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.context_length, config.d_model)
            if config.position == "learned"
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = make_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _embed(self, token_ids: Tensor) -> Tensor:
        x = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            return x + self.position_embedding(positions)
        if self.config.position == "sinusoidal":
            table = sinusoidal_positions(token_ids.shape[1], self.config.d_model, token_ids.device)
            return x + table
        return x

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token IDs must have shape [batch, sequence]")
        length = token_ids.shape[1]
        if self.position_embedding is not None and length > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        rope = None
        if self.config.position == "rope":
            rope = rope_tables(
                length, self.config.head_size, self.config.rope_scale, token_ids.device
            )
        x = self.dropout(self._embed(token_ids))
        for block in self.blocks:
            x = block(x, rope)
        return self.lm_head(self.final_norm(x))

    @torch.inference_mode()
    def generate(self, token_ids: Tensor, max_new_tokens: int) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("generation accepts one sequence")
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.context_length :]
            next_token = self(context)[:, -1].argmax(dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def kv_cache_bytes(config: DecoderConfig, length: int, element_bytes: int = 2) -> int:
    """Bytes held by the key and value cache for one sequence, all layers."""
    per_token = 2 * config.key_value_heads * config.head_size * element_bytes
    return per_token * config.n_layers * length
