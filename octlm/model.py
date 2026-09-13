from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        if min(self.vocab_size, self.context_length, self.d_model, self.n_heads, self.n_layers) < 1:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ff_multiplier < 1 or not 0 <= self.dropout < 1:
            raise ValueError("ff_multiplier and dropout are invalid")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask.view(1, 1, *mask.shape), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.n_heads, self.head_size)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.causal_mask[:, :, :length, :length], float("-inf"))
        weights = self.dropout(scores.softmax(dim=-1))
        attended = (weights @ value).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        hidden = config.d_model * config.ff_multiplier
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token IDs must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.dropout(self.token_embedding(token_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
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
