from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from octlm.bench import measure
from octlm.config import ModelSettings, ProjectConfig, TokenizerSettings, TrainingSettings
from octlm.model import CausalSelfAttention, Decoder, DecoderConfig
from octlm.tokenizer import BPE_SPECIALS, ByteBPETokenizer, CharacterTokenizer, pretokenize
from octlm.train import make_blocks, train_model

ROUND_TRIPS = (
    "",
    "plain ASCII",
    "def f(x):\n    return x + 1\n",
    "\tindent\r\nnext\x00",
    "emoji 👩🏽‍💻",
    "composed é and split e\u0301",
    "नमस्ते दुनिया",
    "literal <|tool_call|> text",
)


def tiny_config(steps: int = 8) -> ProjectConfig:
    return ProjectConfig(
        ModelSettings(8, 16, 2, 1, 2, 0.0),
        TrainingSettings(7, 2, steps, steps, 0.01, 0.001, 1, 0.0, 1.0),
        TokenizerSettings(300, 2),
    )


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = "".join(ROUND_TRIPS) * 3

    def test_pretokenize_preserves_exact_bytes(self) -> None:
        for text in ROUND_TRIPS:
            self.assertEqual(b"".join(pretokenize(text)), text.encode())

    def test_character_round_trip_and_unknown(self) -> None:
        tokenizer = CharacterTokenizer.train([self.corpus])
        for text in ROUND_TRIPS:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        held_out = CharacterTokenizer.train(["abc"])
        self.assertEqual(held_out.encode("z"), [3])

    def test_bpe_round_trip_has_no_unknown(self) -> None:
        tokenizer = ByteBPETokenizer.train([self.corpus], 320)
        for text in ROUND_TRIPS:
            token_ids = tokenizer.encode(text)
            self.assertEqual(tokenizer.decode(token_ids), text)
            self.assertTrue(all(token_id != 3 for token_id in token_ids))

    def test_bpe_is_deterministic(self) -> None:
        first = ByteBPETokenizer.train([self.corpus], 320)
        second = ByteBPETokenizer.train([self.corpus], 320)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_special_ids_are_stable_and_not_activated_by_text(self) -> None:
        tokenizer = ByteBPETokenizer.train([self.corpus], 300)
        self.assertEqual(
            tokenizer.special_to_id, {token: 256 + i for i, token in enumerate(BPE_SPECIALS)}
        )
        literal = "<|tool_call|>"
        encoded = tokenizer.encode(literal)
        self.assertNotIn(tokenizer.special_to_id[literal], encoded)
        self.assertEqual(tokenizer.decode(encoded), literal)

    def test_tokenizers_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            character = CharacterTokenizer.train([self.corpus])
            character.save(root / "char.json")
            self.assertEqual(CharacterTokenizer.load(root / "char.json"), character)
            bpe = ByteBPETokenizer.train([self.corpus], 320)
            bpe.save(root / "bpe.json")
            self.assertEqual(ByteBPETokenizer.load(root / "bpe.json"), bpe)

    def test_strict_decode_rejects_partial_utf8(self) -> None:
        tokenizer = ByteBPETokenizer.train(["text"], 270)
        with self.assertRaises(UnicodeDecodeError):
            tokenizer.decode([0xF0])
        self.assertEqual(tokenizer.decode([0xF0], errors="replace"), "�")


class ModelTests(unittest.TestCase):
    def test_attention_shape_and_mask(self) -> None:
        config = DecoderConfig(32, context_length=4, d_model=16, n_heads=2, n_layers=1)
        attention = CausalSelfAttention(config)
        output = attention(torch.randn(1, 4, 16))
        self.assertEqual(output.shape, (1, 4, 16))
        self.assertTrue(torch.isfinite(output).all())
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(attention.causal_mask[0, 0], expected))

    def test_future_token_does_not_change_earlier_logits(self) -> None:
        torch.manual_seed(1)
        model = Decoder(
            DecoderConfig(32, context_length=4, d_model=16, n_heads=2, n_layers=1)
        ).eval()
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 3, 9]])
        with torch.inference_mode():
            first_logits = model(first)
            second_logits = model(second)
        torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3], rtol=0, atol=0)

    def test_model_rejects_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            Decoder(DecoderConfig(32, d_model=15, n_heads=2))
        model = Decoder(DecoderConfig(32, context_length=4, d_model=16, n_heads=2))
        with self.assertRaises(ValueError):
            model(torch.zeros(5, dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 5), dtype=torch.long))


class TrainingTests(unittest.TestCase):
    def test_blocks_shift_targets_and_mask_padding(self) -> None:
        tokenizer = CharacterTokenizer.train(["abc"])
        blocks = make_blocks("a", tokenizer, 4)
        self.assertEqual(
            blocks.inputs[0].tolist(),
            [tokenizer.bos_id, tokenizer.encode("a")[0], tokenizer.eos_id, tokenizer.pad_id],
        )
        self.assertEqual(
            blocks.targets[0].tolist(),
            [tokenizer.encode("a")[0], tokenizer.eos_id, tokenizer.pad_id, tokenizer.pad_id],
        )
        logits = torch.randn(4, len(tokenizer.vocabulary))
        loss = F.cross_entropy(logits, blocks.targets[0], ignore_index=tokenizer.pad_id)
        self.assertTrue(torch.isfinite(loss))

    def test_resume_matches_uninterrupted_training(self) -> None:
        config = tiny_config()
        tokenizer = CharacterTokenizer.train(["abcdefgh"])
        blocks = make_blocks("abcdefgh" * 5, tokenizer, config.model.context_length)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            checkpoint = Path(directory) / "checkpoint.pt"
            full, _, full_records = train_model(config, tokenizer, blocks, blocks, "data")
            train_model(
                config, tokenizer, blocks, blocks, "data", stop_after=3, checkpoint=checkpoint
            )
            resumed, _, resumed_records = train_model(
                config, tokenizer, blocks, blocks, "data", resume=checkpoint
            )
        for name, tensor in full.state_dict().items():
            torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)
        self.assertLess(full_records[-1]["train_loss"], 3.0)
        self.assertEqual(full_records[-1]["train_loss"], resumed_records[-1]["train_loss"])

    def test_checkpoint_rejects_mismatched_tokenizer(self) -> None:
        config = tiny_config(steps=2)
        first = CharacterTokenizer.train(["abcdefgh"])
        second = CharacterTokenizer.train(["abcdefghi"])
        blocks = make_blocks("abcdefgh" * 3, first, config.model.context_length)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            checkpoint = Path(directory) / "checkpoint.pt"
            train_model(config, first, blocks, blocks, "data", stop_after=1, checkpoint=checkpoint)
            with self.assertRaisesRegex(ValueError, "tokenizer_hash"):
                train_model(config, second, blocks, blocks, "data", resume=checkpoint)

    def test_corrupt_checkpoint_fails_clearly(self) -> None:
        config = tiny_config(steps=2)
        tokenizer = CharacterTokenizer.train(["abcdefgh"])
        blocks = make_blocks("abcdefgh" * 3, tokenizer, config.model.context_length)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_text("not a checkpoint")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "cannot load checkpoint"):
                    train_model(config, tokenizer, blocks, blocks, "data", resume=checkpoint)

    def test_benchmark_schema_is_json_safe(self) -> None:
        result = measure(8)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual(result["schema"], "octlm-bench-v1")


if __name__ == "__main__":
    unittest.main()
