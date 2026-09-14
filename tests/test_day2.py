from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from octlm.corpus import chunk, strip_gutenberg
from octlm.day2 import tiled_attention
from octlm.model import (
    Decoder,
    DecoderConfig,
    FeedForward,
    RMSNorm,
    apply_rope,
    kv_cache_bytes,
    parameter_count,
    rope_tables,
)

BASE = DecoderConfig(vocab_size=64, context_length=16, d_model=32, n_heads=4, n_layers=2)


class TestRMSNorm(unittest.TestCase):
    def test_output_has_unit_root_mean_square(self) -> None:
        x = torch.randn(3, 5, 32) * 7 + 2
        normalized = RMSNorm(32)(x)
        self.assertTrue(
            torch.allclose(normalized.pow(2).mean(-1).sqrt(), torch.ones(3, 5), atol=1e-4)
        )

    def test_it_does_not_re_center(self) -> None:
        x = torch.randn(1, 1, 32)
        shifted = x + 5.0
        self.assertFalse(torch.allclose(RMSNorm(32)(shifted).mean(), torch.zeros(()), atol=1e-3))
        self.assertTrue(
            torch.allclose(nn.LayerNorm(32)(shifted).mean(), torch.zeros(()), atol=1e-5)
        )


class TestRope(unittest.TestCase):
    def test_rotation_preserves_length(self) -> None:
        x = torch.randn(1, 2, 8, 16)
        cosine, sine = rope_tables(8, 16, 1.0, x.device)
        rotated = apply_rope(x, cosine, sine)
        self.assertTrue(torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5))

    def test_scores_depend_only_on_relative_distance(self) -> None:
        query = torch.randn(1, 1, 1, 16).expand(1, 1, 8, 16).contiguous()
        key = torch.randn(1, 1, 1, 16).expand(1, 1, 8, 16).contiguous()
        cosine, sine = rope_tables(8, 16, 1.0, query.device)
        scores = apply_rope(query, cosine, sine) @ apply_rope(key, cosine, sine).transpose(-2, -1)
        self.assertAlmostEqual(float(scores[0, 0, 5, 3]), float(scores[0, 0, 3, 1]), places=4)
        self.assertAlmostEqual(float(scores[0, 0, 7, 2]), float(scores[0, 0, 6, 1]), places=4)

    def test_interpolation_shrinks_the_angle(self) -> None:
        full, _ = rope_tables(8, 16, 1.0, torch.device("cpu"))
        halved, _ = rope_tables(8, 16, 2.0, torch.device("cpu"))
        self.assertTrue(torch.allclose(halved[4], full[2], atol=1e-6))


class TestAttentionPaths(unittest.TestCase):
    def test_sdpa_matches_the_handwritten_path(self) -> None:
        tokens = torch.randint(0, 64, (2, 16))
        for kv_heads in (4, 2, 1):
            outputs = []
            for attention in ("naive", "sdpa"):
                torch.manual_seed(0)
                config = DecoderConfig(
                    **{**vars(BASE), "attention": attention, "kv_heads": kv_heads}
                )
                with torch.inference_mode():
                    outputs.append(Decoder(config).eval()(tokens))
            self.assertTrue(torch.allclose(outputs[0], outputs[1], atol=1e-4))

    def test_tiled_attention_matches_sdpa(self) -> None:
        query, key, value = (torch.randn(1, 2, 96, 16) for _ in range(3))
        reference = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        for block in (16, 32, 96, 128):
            self.assertTrue(
                torch.allclose(tiled_attention(query, key, value, block), reference, atol=1e-5)
            )


class TestFeedForward(unittest.TestCase):
    def test_swiglu_holds_the_feed_forward_parameter_count(self) -> None:
        """Parity belongs to the block, so compare blocks, not embedding-heavy totals."""
        wide = DecoderConfig(**{**vars(BASE), "d_model": 256, "n_heads": 8})
        gelu = parameter_count(FeedForward(wide))
        swiglu = parameter_count(
            FeedForward(DecoderConfig(**{**vars(wide), "feed_forward": "swiglu"}))
        )
        self.assertLess(abs(swiglu - gelu) / gelu, 0.01)

    def test_swiglu_width_rounds_to_a_multiple_of_eight(self) -> None:
        swiglu = DecoderConfig(**{**vars(BASE), "feed_forward": "swiglu"})
        self.assertEqual(swiglu.hidden_size % 8, 0)
        self.assertLess(swiglu.hidden_size, BASE.d_model * BASE.ff_multiplier)


class TestCache(unittest.TestCase):
    def test_cache_bytes_scale_with_kv_heads(self) -> None:
        mha = kv_cache_bytes(BASE, 1024)
        gqa = kv_cache_bytes(DecoderConfig(**{**vars(BASE), "kv_heads": 2}), 1024)
        self.assertEqual(mha, 2 * BASE.n_layers * BASE.d_model * 1024 * 2)
        self.assertEqual(mha // gqa, 2)


class TestConfigValidation(unittest.TestCase):
    def test_it_rejects_unknown_and_mismatched_choices(self) -> None:
        for override in (
            {"norm": "batchnorm"},
            {"position": "alibi"},
            {"kv_heads": 3},
            {"rope_scale": 0.0},
        ):
            with self.assertRaises(ValueError):
                DecoderConfig(**{**vars(BASE), **override}).validate()

    def test_rope_needs_an_even_head_size(self) -> None:
        odd = DecoderConfig(vocab_size=8, d_model=6, n_heads=2, position="rope")
        with self.assertRaises(ValueError):
            odd.validate()


class TestLongContext(unittest.TestCase):
    def test_learned_positions_stop_at_the_table(self) -> None:
        model = Decoder(BASE)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, BASE.context_length + 1), dtype=torch.long))

    def test_rope_runs_past_the_trained_length(self) -> None:
        config = DecoderConfig(**{**vars(BASE), "position": "rope", "attention": "sdpa"})
        logits = Decoder(config)(torch.zeros((1, BASE.context_length * 3), dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, BASE.context_length * 3, BASE.vocab_size))


class TestResidualPlacement(unittest.TestCase):
    def test_post_norm_differs_from_pre_norm(self) -> None:
        tokens = torch.randint(0, 64, (1, 8))
        outputs = []
        for residual in ("pre", "post"):
            torch.manual_seed(0)
            outputs.append(Decoder(DecoderConfig(**{**vars(BASE), "residual": residual}))(tokens))
        self.assertFalse(torch.allclose(outputs[0], outputs[1], atol=1e-3))


class TestCorpus(unittest.TestCase):
    def test_strip_gutenberg_keeps_only_the_work(self) -> None:
        text = (
            "header noise\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
            "the work\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
            "footer noise\n"
        )
        self.assertEqual(strip_gutenberg(text), "the work")

    def test_strip_gutenberg_rejects_missing_markers(self) -> None:
        with self.assertRaises(ValueError):
            strip_gutenberg("no markers here")

    def test_chunks_cover_the_input_and_split_on_lines(self) -> None:
        text = "".join(f"line {index}\n" for index in range(200))
        pieces = chunk(text, 64)
        self.assertEqual("".join(pieces), text)
        self.assertTrue(all(piece.endswith("\n") for piece in pieces))
        self.assertGreater(len(pieces), 1)


if __name__ == "__main__":
    unittest.main()
