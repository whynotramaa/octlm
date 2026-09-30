from __future__ import annotations

import io
import json
import struct
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers

from octlm.day6 import (
    FILES,
    MODEL_ID,
    REVISION,
    QwenTokenizer,
    file_hash,
    load_qwen,
    qwen_config,
    read_header,
    read_json,
    safetensors,
    tensor_spec,
    verify_snapshot,
    weight_names,
)
from octlm.model import Decoder, DecoderConfig, apply_rope, rope_tables


def settings() -> dict:
    return {
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_scaling": None,
        "use_sliding_window": False,
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-6,
        "vocab_size": 64,
        "max_position_embeddings": 40960,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 24,
        "rope_theta": 1000000,
    }


def write_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header, payload = {}, bytearray()
    for name, tensor in tensors.items():
        raw = bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())
        dtype = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}[tensor.dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def manifest(directory: Path) -> None:
    for name in FILES:
        if not (directory / name).exists():
            (directory / name).write_text("{}")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "sha256": {name: file_hash(directory / name) for name in FILES},
            }
        )
    )


class TestSafetensors(unittest.TestCase):
    def test_supported_floats_and_empty_tensor(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            tensors = {
                str(dtype): torch.arange(6).to(dtype).reshape(2, 3)
                for dtype in (torch.float32, torch.float16, torch.bfloat16)
            }
            tensors["empty"] = torch.empty(0)
            write_tensors(path, tensors)
            for name, actual in safetensors(path):
                torch.testing.assert_close(actual, tensors[name], rtol=0, atol=0)

    def test_rejects_invalid_header_and_payload(self) -> None:
        valid = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        for header in (
            {"x": {**valid["x"], "data_offsets": [1, 5]}},
            {"x": {**valid["x"], "shape": [2]}},
            {"x": valid["x"], "y": valid["x"]},
            {"x": {**valid["x"], "dtype": "I64"}},
            {"__metadata__": {"bad": 1}, **valid},
            [],
        ):
            encoded = json.dumps(header).encode()
            data = struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 4
            with self.subTest(header=header), self.assertRaises(ValueError):
                read_header(io.BytesIO(data), len(data))
        for data in (b"", struct.pack("<Q", 1000), struct.pack("<Q", 0)):
            with self.assertRaises(ValueError):
                read_header(io.BytesIO(data), len(data))
        encoded = json.dumps(valid).encode()
        data = struct.pack("<Q", len(encoded)) + encoded
        with self.assertRaisesRegex(ValueError, "payload"):
            read_header(io.BytesIO(data), len(data))

    def test_rejects_invalid_entries_and_duplicate_json(self) -> None:
        for entry in (
            {},
            [],
            {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]},
            {"dtype": "F32", "shape": [1], "data_offsets": [True, 4]},
        ):
            with self.assertRaises(ValueError):
                tensor_spec(entry)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"a":1,"a":2}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_json(path)
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "object"):
                read_json(path)


class TestQwenModel(unittest.TestCase):
    def test_config_shapes_rope_and_cache(self) -> None:
        config = qwen_config(settings())
        torch.manual_seed(0)
        model = Decoder(config).eval()
        attention = model.blocks[0].attention
        self.assertEqual(attention.query.weight.shape, (32, 16))
        self.assertEqual(attention.output.weight.shape, (16, 32))
        self.assertEqual(model.blocks[0].feed_forward.up.weight.shape, (24, 16))
        self.assertEqual(attention.causal_mask.numel(), 1)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.inference_mode():
            full = model(tokens)
            first, cache = model.forward_cached(tokens[:, :2])
            rest, _ = model.forward_cached(tokens[:, 2:], cache)
            torch.testing.assert_close(torch.cat((first, rest), 1), full, atol=1e-6, rtol=1e-5)
            self.assertTrue(
                torch.equal(
                    model.generate(tokens[:, :2], 4), model.generate_cached(tokens[:, :2], 4)
                )
            )
        x = torch.randn(1, 2, 6, 8)
        cosine, sine = rope_tables(6, 8, 1.0, x.device, config.rope_base)
        left, right = x.chunk(2, -1)
        expected = torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1)
        torch.testing.assert_close(apply_rope(x, cosine, sine, split=True), expected)

    def test_unsupported_settings_are_rejected(self) -> None:
        for key, value in (
            ("model_type", "qwen3_5"),
            ("attention_bias", True),
            ("head_dim", 0),
            ("num_hidden_layers", True),
            ("rope_theta", float("nan")),
            ("rms_norm_eps", 1e-5),
            ("rope_scaling", {"type": "yarn"}),
            ("tie_word_embeddings", False),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                qwen_config({**settings(), key: value})
        for changes in ({"head_dim": -1}, {"ff_hidden": -1}, {"rope_base": 0}):
            with self.assertRaises(ValueError):
                replace(DecoderConfig(vocab_size=64), **changes).validate()

    def test_load_hashes_shapes_missing_and_unknown_weights(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            config = qwen_config(settings())
            source = Decoder(config).eval()
            names = weight_names(config)
            state = source.state_dict()
            tensors = {name: state[target] for name, target in names.items()}
            (path / "config.json").write_text(json.dumps(settings()))
            write_tensors(path / "model.safetensors", tensors)
            manifest(path)
            loaded, _ = load_qwen(path)
            tokens = torch.tensor([[1, 2, 3]])
            with torch.inference_mode():
                torch.testing.assert_close(loaded(tokens), source(tokens), rtol=0, atol=0)
            self.assertIs(loaded.lm_head.weight, loaded.token_embedding.weight)
            for altered, error in (
                ({k: v for k, v in tensors.items() if k != "model.norm.weight"}, "missing"),
                ({**tensors, "unknown": torch.ones(1)}, "unexpected"),
                ({**tensors, "model.norm.weight": torch.ones(1)}, "shape"),
                (
                    {**tensors, "lm_head.weight": torch.ones_like(state["lm_head.weight"])},
                    "tied weights",
                ),
            ):
                write_tensors(path / "model.safetensors", altered)
                manifest(path)
                with self.assertRaisesRegex(ValueError, error):
                    load_qwen(path)
            (path / "tokenizer.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch: tokenizer"):
                load_qwen(path)
            record = json.loads((path / "manifest.json").read_text())
            record["revision"] = "wrong"
            (path / "manifest.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "revision"):
                verify_snapshot(path)


class TestNativeTokenizer(unittest.TestCase):
    def test_literal_controls_roundtrip_and_native_template(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            alphabet = pre_tokenizers.ByteLevel.alphabet()
            tokenizer = Tokenizer(
                models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
            )
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            tokenizer.normalizer = normalizers.NFC()
            tokenizer.add_special_tokens([AddedToken("<|im_start|>", special=True)])
            tokenizer.save(str(path / "tokenizer.json"))
            (path / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "bos_token": None,
                        "eos_token": "<|im_start|>",
                        "pad_token": "<|im_start|>",
                        "chat_template": (
                            "{{ eos_token }}{{ messages[0].content }}"
                            "{% if not enable_thinking %}plain{% endif %}"
                        ),
                    }
                )
            )
            native = QwenTokenizer(path)
            for text in ("\t  \r\n\x00", "🙂 cafe\u0301 日本語", "<|im_start|>"):
                ids = native.encode(text)
                self.assertFalse(set(ids) & native.special_ids)
                self.assertEqual(native.decode(ids), text)
            self.assertEqual(native.encode("<|im_start|>", allow_special=True), [256])
            messages = [{"role": "user", "content": "hello"}]
            self.assertEqual(native.chat(messages), "<|im_start|>helloplain")
            self.assertEqual(native.chat(messages, thinking=True), "<|im_start|>hello")
            with self.assertRaises(ValueError):
                native.chat([])
            with self.assertRaises(ValueError):
                native.chat([{"role": "unknown"}])


if __name__ == "__main__":
    unittest.main()
