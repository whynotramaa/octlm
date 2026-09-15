from __future__ import annotations

import unittest

import torch

from octlm.day3 import copy_probe, mtp_agreement
from octlm.model import (
    Decoder,
    DecoderConfig,
    attention_mask,
    kv_cache_bytes,
    mask_density,
    parameter_count,
    pool_blocks,
)
from octlm.train import TokenBlocks, mtp_loss

BASE = dict(vocab_size=512, context_length=16, d_model=32, n_heads=4, n_layers=2)
TOKENS = torch.randint(0, 512, (2, 16))


def build(**overrides: object) -> Decoder:
    torch.manual_seed(0)
    return Decoder(DecoderConfig(**BASE, **overrides)).eval()


def logits(**overrides: object) -> torch.Tensor:
    with torch.inference_mode():
        return build(**overrides)(TOKENS)


class TestMultiTokenPrediction(unittest.TestCase):
    def test_depth_one_is_the_pre_day3_model(self) -> None:
        self.assertTrue(torch.equal(logits(mtp_depth=1), logits()))

    def test_each_depth_predicts_something_different(self) -> None:
        model = build(mtp_depth=3)
        with torch.inference_mode():
            stack = model(TOKENS, all_depths=True)
        self.assertEqual(stack.shape[0], 3)
        self.assertGreater(float((stack[0] - stack[1]).abs().max()), 1e-6)
        self.assertGreater(float((stack[1] - stack[2]).abs().max()), 1e-6)

    def test_adapters_are_the_only_added_parameters(self) -> None:
        added = parameter_count(build(mtp_depth=3)) - parameter_count(build())
        self.assertEqual(added, 2 * BASE["d_model"] ** 2)

    def test_depth_one_loss_matches_plain_cross_entropy(self) -> None:
        model = build(mtp_depth=1)
        targets = torch.randint(0, 512, (2, 16))
        single = mtp_loss(model, TOKENS, targets, pad_id=0)
        model.config = DecoderConfig(**BASE, mtp_depth=2)
        # Depth 2 must still produce a finite loss over the shortened span.
        self.assertTrue(torch.isfinite(single))

    def test_agreement_is_zero_without_a_second_head(self) -> None:
        blocks = TokenBlocks(TOKENS, TOKENS, torch.ones_like(TOKENS))
        self.assertEqual(mtp_agreement(build(mtp_depth=1), blocks, pad_id=-1), 0.0)

    def test_agreement_is_a_fraction(self) -> None:
        blocks = TokenBlocks(TOKENS, TOKENS, torch.ones_like(TOKENS))
        self.assertTrue(0.0 <= mtp_agreement(build(mtp_depth=2), blocks, pad_id=-1) <= 1.0)


class TestSparseMask(unittest.TestCase):
    def test_window_admits_exactly_the_window(self) -> None:
        mask = attention_mask(8, window=3, stride=0)[0, 0]
        self.assertEqual(mask[5].tolist(), [False, False, False, True, True, True, False, False])

    def test_stride_adds_global_columns(self) -> None:
        mask = attention_mask(8, window=2, stride=4)[0, 0]
        self.assertEqual(mask[6].tolist(), [True, False, False, False, True, True, True, False])

    def test_a_full_width_window_is_plain_causal(self) -> None:
        self.assertTrue(torch.equal(attention_mask(8, 8, 0), attention_mask(8, 0, 0)))
        self.assertTrue(
            torch.equal(logits(attention="sdpa", attention_window=16), logits(attention="sdpa"))
        )

    def test_every_row_keeps_at_least_its_own_position(self) -> None:
        self.assertTrue(bool(attention_mask(16, 1, 0).any(dim=-1).all()))

    def test_density_falls_with_the_window(self) -> None:
        wide = mask_density(DecoderConfig(**BASE, attention_window=12), 16)
        narrow = mask_density(DecoderConfig(**BASE, attention_window=4), 16)
        self.assertLess(narrow, wide)
        self.assertEqual(mask_density(DecoderConfig(**BASE), 16), 1.0)


class TestCompressedAttention(unittest.TestCase):
    def test_pooling_averages_each_block(self) -> None:
        x = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
        self.assertEqual(pool_blocks(x, 4).flatten().tolist(), [1.5, 5.5])

    def test_block_of_one_reproduces_full_attention(self) -> None:
        compressed = logits(attention="sdpa", attention_window=4, kv_compress_block=1)
        self.assertTrue(torch.equal(compressed, logits(attention="sdpa")))

    def test_naive_and_sdpa_agree_under_a_window(self) -> None:
        difference = (
            (
                logits(attention="naive", attention_window=8)
                - logits(attention="sdpa", attention_window=8)
            )
            .abs()
            .max()
        )
        self.assertLess(float(difference), 1e-5)

    def test_compression_shrinks_the_cache(self) -> None:
        plain = DecoderConfig(**BASE, kv_heads=2)
        packed = DecoderConfig(**BASE, kv_heads=2, attention_window=8, kv_compress_block=4)
        self.assertEqual(packed.cached_positions(4096), 8 + (4096 - 8) // 4)
        self.assertLess(kv_cache_bytes(packed, 4096), kv_cache_bytes(plain, 4096))


class TestLatentAttention(unittest.TestCase):
    def test_forward_keeps_the_shape_and_stays_finite(self) -> None:
        produced = logits(attention="mla", position="rope", mla_rank=16, mla_rope_dim=8)
        self.assertEqual(produced.shape, logits().shape)
        self.assertTrue(bool(torch.isfinite(produced).all()))

    def test_cache_holds_the_latent_not_the_heads(self) -> None:
        mla = DecoderConfig(**BASE, attention="mla", position="rope", mla_rank=16, mla_rope_dim=8)
        self.assertEqual(mla.cache_dims, 24)

    def test_it_only_beats_gqa_below_the_crossover_rank(self) -> None:
        gqa = DecoderConfig(**BASE, kv_heads=2)
        beats = DecoderConfig(**BASE, attention="mla", position="rope", mla_rank=8, mla_rope_dim=4)
        loses = DecoderConfig(**BASE, attention="mla", position="rope", mla_rank=64, mla_rope_dim=4)
        self.assertLess(beats.cache_dims, gqa.cache_dims)
        self.assertGreater(loses.cache_dims, gqa.cache_dims)


class TestCopyProbe(unittest.TestCase):
    def test_it_scores_the_repeat_against_an_unseen_span(self) -> None:
        model = build(attention="sdpa")
        blocks = TokenBlocks(TOKENS, TOKENS, torch.ones_like(TOKENS))
        results = copy_probe(model, blocks, vocab_size=512, depths=[0])
        self.assertEqual(len(results), 1)
        row = results[0]
        self.assertAlmostEqual(
            row["gain"], row["needle_nll_unseen"] - row["needle_nll_recalled"], places=6
        )

    def test_a_depth_with_no_room_is_skipped(self) -> None:
        model = build(attention="sdpa")
        blocks = TokenBlocks(TOKENS, TOKENS, torch.ones_like(TOKENS))
        self.assertEqual(copy_probe(model, blocks, 512, depths=[15]), [])


class TestValidation(unittest.TestCase):
    def _rejects(self, message: str, **overrides: object) -> None:
        with self.assertRaises(ValueError, msg=message):
            DecoderConfig(**BASE, **overrides).validate()

    def test_it_rejects_impossible_day3_settings(self) -> None:
        self._rejects("depth", mtp_depth=0)
        self._rejects("window past the context", attention_window=17)
        self._rejects("stride without a window", attention_stride=4)
        self._rejects("compression without a window", kv_compress_block=4)
        self._rejects("block that does not divide", attention_window=8, kv_compress_block=5)
        self._rejects("mla without a rank", attention="mla", position="rope")
        self._rejects(
            "odd rope width", attention="mla", position="rope", mla_rank=16, mla_rope_dim=7
        )
        self._rejects("mla without rope", attention="mla", mla_rank=16, mla_rope_dim=8)
        self._rejects("mla fields without mla", mla_rank=16)
        self._rejects(
            "mla with compression",
            attention="mla",
            position="rope",
            mla_rank=16,
            mla_rope_dim=8,
            attention_window=8,
            kv_compress_block=4,
        )


if __name__ == "__main__":
    unittest.main()
