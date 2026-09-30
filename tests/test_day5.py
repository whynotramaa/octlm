import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from octlm.config import ProjectConfig
from octlm.day5 import Int8Linear, quantize_int8, read_runs, spread, variant_config
from octlm.model import Decoder, DecoderConfig, kv_cache_bytes


class TestSeedSpread(unittest.TestCase):
    def test_variants_and_spread(self) -> None:
        base = ProjectConfig.load("configs/day5.toml")
        baseline = variant_config(base, "baseline", 1338)
        modern = variant_config(base, "modern", 1338)
        mtp = variant_config(base, "mtp", 1338)
        self.assertEqual(baseline.model.kv_heads, 0)
        self.assertEqual(modern.model.kv_heads, 2)
        self.assertEqual(mtp.model.mtp_depth, 2)
        self.assertEqual(baseline.training.seed, modern.training.seed)
        rows = [
            {"variant": "modern", "bits_per_byte": 0.4},
            {"variant": "modern", "bits_per_byte": 0.5},
            {"variant": "baseline", "bits_per_byte": 0.6},
        ]
        self.assertAlmostEqual(spread(rows, "modern")["mean"], 0.45)
        self.assertAlmostEqual(spread(rows, "modern")["spread"], 0.1)
        with self.assertRaises(ValueError):
            spread(rows, "missing")

    def test_resume_rejects_other_inputs(self) -> None:
        base = ProjectConfig.load("configs/day5.toml")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            path.write_text(
                '{"variant":"modern","seed":1337,"config_hash":"wrong",'
                '"data_hash":"data","tokenizer_hash":"tokenizer"}\n'
            )
            with self.assertRaisesRegex(ValueError, "different inputs"):
                read_runs(path, base, "data", "tokenizer", ("modern",))


class TestKvCache(unittest.TestCase):
    def test_chunk_and_window_match_naive(self) -> None:
        tokens = torch.arange(1, 7).unsqueeze(0)
        for position in ("rope", "learned", "sinusoidal"):
            config = DecoderConfig(
                vocab_size=64,
                context_length=8,
                d_model=32,
                n_heads=4,
                n_layers=2,
                attention="sdpa",
                kv_heads=2,
                position=position,
            )
            torch.manual_seed(0)
            model = Decoder(config).eval()
            with torch.inference_mode():
                expected = model(tokens)
                first, cache = model.forward_cached(tokens[:, :3])
                second, cache = model.forward_cached(tokens[:, 3:], cache)
                torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
                self.assertEqual(
                    sum((key.numel() + value.numel()) * key.element_size() for key, value in cache),
                    kv_cache_bytes(config, 6, 4),
                )
                self.assertTrue(
                    torch.equal(
                        model.generate(tokens[:, :1], 12), model.generate_cached(tokens[:, :1], 12)
                    )
                )

    def test_invalid_cache_is_rejected(self) -> None:
        model = Decoder(DecoderConfig(vocab_size=64, attention="sdpa")).eval()
        with self.assertRaisesRegex(ValueError, "one entry per layer"):
            model.forward_cached(torch.ones(1, 1, dtype=torch.long), [])


class TestInt8Weights(unittest.TestCase):
    def test_per_row_scale_and_tied_embedding(self) -> None:
        source = torch.nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            source.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, -1.0]]))
            source.bias.copy_(torch.tensor([0.5, -0.5]))
        layer = Int8Linear(source)
        self.assertEqual(layer.qweight.dtype, torch.int8)
        torch.testing.assert_close(layer(torch.ones(1, 2)), source(torch.ones(1, 2)))
        model = Decoder(DecoderConfig(vocab_size=64, attention="sdpa")).eval()
        quantize_int8(model)
        self.assertIsInstance(model.blocks[0].attention.query, Int8Linear)
        self.assertIsInstance(model.lm_head, torch.nn.Linear)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)


if __name__ == "__main__":
    unittest.main()
