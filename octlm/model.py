from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

POSITIONS = ("learned", "sinusoidal", "rope")
NORMS = ("layernorm", "rmsnorm")
FEED_FORWARDS = ("gelu", "swiglu")
ATTENTIONS = ("naive", "sdpa", "mla")
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
    mtp_depth: int = 1
    attention_window: int = 0
    attention_stride: int = 0
    kv_compress_block: int = 0
    mla_rank: int = 0
    mla_rope_dim: int = 0

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

    @property
    def cache_dims(self) -> int:
        """Dimensions the cache holds per token per layer. MLA caches a latent, not heads."""
        if self.attention == "mla":
            return self.mla_rank + self.mla_rope_dim
        return 2 * self.key_value_heads * self.head_size

    def cached_positions(self, length: int) -> int:
        """Compression replaces each block outside the local window with one pooled entry."""
        if not self.kv_compress_block:
            return length
        window = min(self.attention_window, length)
        return window + (length - window) // self.kv_compress_block

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
        self._validate_day3()

    def _validate_day3(self) -> None:
        """One table, so the Day 3 switches stay readable as they accumulate."""
        mla = self.attention == "mla"
        packed = bool(self.kv_compress_block)
        problems = (
            (self.mtp_depth < 1, "mtp_depth must be at least one"),
            (
                not 0 <= self.attention_window <= self.context_length,
                "attention_window must be between zero and context_length",
            ),
            (self.attention_stride < 0, "attention_stride must not be negative"),
            (
                bool(self.attention_stride) and not self.attention_window,
                "attention_stride needs a window, otherwise it changes nothing",
            ),
            (
                packed and not self.attention_window,
                "kv_compress_block needs a window to keep exact",
            ),
            (
                packed and self.context_length % max(1, self.kv_compress_block) != 0,
                "kv_compress_block must divide context_length",
            ),
            (packed and mla, "kv_compress_block and mla are separate experiments"),
            (
                mla and (self.mla_rank < 1 or self.mla_rope_dim < 1),
                "mla needs a positive rank and rope dimension",
            ),
            (mla and self.mla_rope_dim % 2 != 0, "mla_rope_dim must be even"),
            (mla and self.position != "rope", "mla decouples RoPE, so it needs position = rope"),
            (
                not mla and bool(self.mla_rank or self.mla_rope_dim),
                "mla_rank and mla_rope_dim need attention = mla",
            ),
        )
        for failed, message in problems:
            if failed:
                raise ValueError(message)


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


def attention_mask(
    length: int, window: int, stride: int, device: torch.device | None = None
) -> Tensor:
    """Causal mask, optionally narrowed to a sliding window with strided global columns."""
    rows = torch.arange(length, device=device).unsqueeze(1)
    columns = torch.arange(length, device=device).unsqueeze(0)
    distance = rows - columns
    allowed = distance >= 0
    if window:
        local = distance < window
        if stride:
            local = local | (columns % stride == 0)
        allowed = allowed & local
    return allowed.view(1, 1, length, length)


def compressed_mask(length: int, window: int, block: int, device: torch.device | None) -> Tensor:
    """Columns are [pooled blocks, exact tokens]. A block is visible once it clears the window."""
    rows = torch.arange(length, device=device).unsqueeze(1)
    blocks = torch.arange(length // block, device=device).unsqueeze(0)
    block_visible = (blocks + 1) * block <= rows - window + 1
    exact = attention_mask(length, window, 0, device)[0, 0]
    return torch.cat((block_visible, exact), dim=1).view(1, 1, length, -1)


def pool_blocks(x: Tensor, block: int) -> Tensor:
    """Mean-pool [batch, heads, length, width] into length // block entries."""
    count = x.shape[2] // block
    return x[:, :, : count * block, :].unflatten(2, (count, block)).mean(dim=3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.kv_heads = config.n_heads if config.attention == "mla" else config.key_value_heads
        self.head_size = config.head_size
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        if config.attention == "mla":
            self.kv_down = nn.Linear(config.d_model, config.mla_rank, bias=False)
            self.kv_up = nn.Linear(config.mla_rank, 2 * config.d_model, bias=False)
            self.k_rope = nn.Linear(config.d_model, config.mla_rope_dim, bias=False)
            self.q_rope = nn.Linear(
                config.d_model, config.n_heads * config.mla_rope_dim, bias=False
            )
        else:
            width = self.kv_heads * self.head_size
            self.key = nn.Linear(config.d_model, width, bias=False)
            self.value = nn.Linear(config.d_model, width, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        mask = attention_mask(
            config.context_length, config.attention_window, config.attention_stride
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def causal_mask_for(self, length: int) -> Tensor:
        if length <= self.causal_mask.shape[-1]:
            return self.causal_mask[:, :, :length, :length]
        return attention_mask(
            length,
            self.config.attention_window,
            self.config.attention_stride,
            self.query.weight.device,
        )

    def _heads(self, x: Tensor, heads: int, width: int) -> Tensor:
        return x.unflatten(-1, (heads, width)).transpose(1, 2)

    def _project(
        self, x: Tensor, rope: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.attention == "mla":
            return self._project_mla(x, rope)
        query = self._heads(self.query(x), self.n_heads, self.head_size)
        key = self._heads(self.key(x), self.kv_heads, self.head_size)
        value = self._heads(self.value(x), self.kv_heads, self.head_size)
        if self.config.kv_compress_block:
            return self._compress(query, key, value, rope)
        if rope is not None:
            query, key = apply_rope(query, *rope), apply_rope(key, *rope)
        return query, key, value

    def _project_mla(
        self, x: Tensor, rope: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Content rides a shared low-rank latent, position rides a separate shared RoPE key."""
        rope_dim = self.config.mla_rope_dim
        content = self.kv_up(self.kv_down(x))
        key, value = content.chunk(2, dim=-1)
        key = self._heads(key, self.n_heads, self.head_size)
        value = self._heads(value, self.n_heads, self.head_size)
        query = self._heads(self.query(x), self.n_heads, self.head_size)
        cosine, sine = rope
        rope_slice = (cosine[..., : rope_dim // 2], sine[..., : rope_dim // 2])
        query_rope = apply_rope(self._heads(self.q_rope(x), self.n_heads, rope_dim), *rope_slice)
        key_rope = apply_rope(self.k_rope(x).unsqueeze(1), *rope_slice)
        key_rope = key_rope.expand(-1, self.n_heads, -1, -1)
        return torch.cat((query, query_rope), -1), torch.cat((key, key_rope), -1), value

    def _compress(
        self, query: Tensor, key: Tensor, value: Tensor, rope: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pool before rotating, then rotate each pooled block at its midpoint position."""
        block = self.config.kv_compress_block
        pooled_key, pooled_value = pool_blocks(key, block), pool_blocks(value, block)
        if rope is not None:
            cosine, sine = rope
            midpoints = torch.arange(pooled_key.shape[2], device=key.device) * block + block // 2
            midpoints = midpoints.clamp(max=cosine.shape[0] - 1)
            pooled_key = apply_rope(pooled_key, cosine[midpoints], sine[midpoints])
            query, key = apply_rope(query, cosine, sine), apply_rope(key, cosine, sine)
        return query, torch.cat((pooled_key, key), 2), torch.cat((pooled_value, value), 2)

    def _naive(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        groups = self.n_heads // key.shape[1]
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
        scores = scores.masked_fill(~mask, float("-inf"))
        return self.dropout(scores.softmax(dim=-1)) @ value

    def _mask_for(self, length: int, device: torch.device) -> Tensor | None:
        """None means plain causal, which lets SDPA keep its flash kernel."""
        if self.config.kv_compress_block:
            return compressed_mask(
                length, self.config.attention_window, self.config.kv_compress_block, device
            )
        if self.config.attention_window:
            return self.causal_mask_for(length)
        return None

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        batch, length, width = x.shape
        query, key, value = self._project(x, rope)
        mask = self._mask_for(length, x.device)
        if self.config.attention == "naive":
            attended = self._naive(
                query,
                key,
                value,
                mask if mask is not None else attention_mask(length, 0, 0, x.device),
            )
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                is_causal=mask is None,
                dropout_p=self.dropout.p if self.training else 0.0,
                enable_gqa=key.shape[1] != self.n_heads,
            )
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
        # One adapter per extra depth. Heads sharing both the trunk state and the tied
        # unembedding would compute identical logits, so each depth needs its own map.
        self.mtp_adapters = nn.ModuleList(
            nn.Linear(config.d_model, config.d_model, bias=False)
            for _ in range(config.mtp_depth - 1)
        )
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

    def forward(self, token_ids: Tensor, all_depths: bool = False) -> Tensor:
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
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        if not all_depths:
            return logits
        return torch.stack([logits, *(self.lm_head(a(hidden)) for a in self.mtp_adapters)])

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
    per_token = config.cache_dims * element_bytes
    return per_token * config.n_layers * config.cached_positions(length)


def mask_density(config: DecoderConfig, length: int) -> float:
    """Fraction of the causal triangle a sparse mask admits."""
    mask = attention_mask(length, config.attention_window, config.attention_stride)
    causal = attention_mask(length, 0, 0)
    return float(mask.sum()) / float(causal.sum())
