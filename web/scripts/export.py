from __future__ import annotations

import json
from pathlib import Path

import torch

from octlm.config import ProjectConfig
from octlm.day2 import VARIANTS, tiled_attention
from octlm.day5 import Int8Linear, spread
from octlm.model import (
    CausalSelfAttention,
    Decoder,
    DecoderConfig,
    apply_rope,
    attention_mask,
    compressed_mask,
    kv_cache_bytes,
    mask_density,
    parameter_count,
    rope_tables,
    sampling_distribution,
    sinusoidal_positions,
)
from octlm.tokenizer import ByteBPETokenizer, pretokenize
from octlm.train import _json, learning_rate

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
CPU = torch.device("cpu")
SHAPE_KEYS = (
    "vocab_size",
    "context_length",
    "d_model",
    "n_heads",
    "n_layers",
    "ff_multiplier",
    "position",
    "norm",
    "feed_forward",
    "kv_heads",
)

PRETOKENIZE_CASES = (
    "def total_number(values):\n\treturn sum(values)\r\n",
    "x1 = y_2 + 30.5",
    "naïve café 🙂👍🏽",
    "日本語のテキスト, Привет мир!",
    "a\x00b  <|eos|>",
)

BPE_TEXT = (
    "def total_number(values):\n"
    "    total = 0\n"
    "    for value in values:\n"
    "        total = total + value\n"
    "    return total\n"
) * 3


def write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(data) + "\n", encoding="utf-8")


def export_jsonl(source: Path, name: str) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    write(WEB / "src/data/runs" / f"{name}.json", records)


def export_runs() -> None:
    for source in sorted((ROOT / "runs").glob("*.jsonl")):
        export_jsonl(source, source.stem)
    for source in sorted((ROOT / "runs/kaggle-day4/day4").rglob("*.jsonl")):
        export_jsonl(source, f"day4-{source.stem}")
    for name in ("inference", "spread", "mtp"):
        for source in sorted((ROOT / f"runs/kaggle-day5-{name}/day5").glob("*.jsonl")):
            export_jsonl(source, source.stem)


def tokenizer_fixture() -> dict:
    tokenizer = ByteBPETokenizer.train([BPE_TEXT], vocab_size=261 + 40)
    return {
        "pretokenize": [
            {"text": text, "chunks": [chunk.decode("utf-8") for chunk in pretokenize(text)]}
            for text in PRETOKENIZE_CASES
        ],
        "bpe": {
            "text": BPE_TEXT,
            "vocab_size": 261 + 40,
            "merges": [list(merge) for merge in tokenizer.merges],
            "encoded": tokenizer.encode(BPE_TEXT),
        },
    }


def attention_fixture() -> dict:
    torch.manual_seed(0)
    config = DecoderConfig(vocab_size=16, context_length=8, d_model=16, n_heads=2)
    module = CausalSelfAttention(config).eval()
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        output = module(x)
    return {
        "n_heads": 2,
        "x": x[0].tolist(),
        "query": module.query.weight.tolist(),
        "key": module.key.weight.tolist(),
        "value": module.value.weight.tolist(),
        "output_weight": module.output.weight.tolist(),
        "output": output[0].tolist(),
    }


def tiled_fixture() -> dict:
    torch.manual_seed(1)
    query, key, value = (torch.randn(12, 4) for _ in range(3))
    return {
        "query": query.tolist(),
        "key": key.tolist(),
        "value": value.tolist(),
        "block": 4,
        "output": tiled_attention(query, key, value, block=4).tolist(),
    }


def positions_fixture() -> dict:
    cosine, sine = rope_tables(16, 8, 1.0, CPU)
    scaled_cosine, scaled_sine = rope_tables(16, 8, 4.0, CPU)
    torch.manual_seed(2)
    x = torch.randn(1, 1, 16, 8)
    return {
        "sinusoidal": sinusoidal_positions(16, 16, CPU).tolist(),
        "rope": {"length": 16, "head_size": 8, "cos": cosine.tolist(), "sin": sine.tolist()},
        "rope_scaled": {"scale": 4.0, "cos": scaled_cosine.tolist(), "sin": scaled_sine.tolist()},
        "apply": {"x": x[0, 0].tolist(), "rotated": apply_rope(x, cosine, sine)[0, 0].tolist()},
    }


def masks_fixture() -> dict:
    cases = [(12, 0, 0), (12, 4, 0), (12, 4, 3), (16, 5, 4)]
    masks = [
        {
            "length": length,
            "window": window,
            "stride": stride,
            "mask": attention_mask(length, window, stride)[0, 0].int().tolist(),
            "density": mask_density(
                DecoderConfig(
                    vocab_size=16,
                    context_length=length,
                    attention_window=window,
                    attention_stride=stride,
                ),
                length,
            ),
        }
        for length, window, stride in cases
    ]
    compressed = [
        {
            "length": length,
            "window": window,
            "block": block,
            "mask": compressed_mask(length, window, block, None)[0, 0].int().tolist(),
        }
        for length, window, block in [(12, 4, 2), (16, 4, 4)]
    ]
    return {"masks": masks, "compressed": compressed}


def cache_fixture() -> list[dict]:
    shapes = [
        {"n_layers": 4, "d_model": 256, "n_heads": 8, "kv_heads": 8},
        {"n_layers": 4, "d_model": 256, "n_heads": 8, "kv_heads": 2},
        {"n_layers": 8, "d_model": 512, "n_heads": 8, "kv_heads": 2},
        {"n_layers": 4, "d_model": 256, "n_heads": 8, "kv_heads": 1},
        {
            "n_layers": 4,
            "d_model": 256,
            "n_heads": 8,
            "kv_heads": 2,
            "context_length": 4096,
            "attention_window": 256,
            "kv_compress_block": 16,
        },
        {
            "n_layers": 4,
            "d_model": 256,
            "n_heads": 8,
            "attention": "mla",
            "position": "rope",
            "mla_rank": 64,
            "mla_rope_dim": 16,
        },
    ]
    rows = []
    for shape in shapes:
        config = DecoderConfig(vocab_size=16, **shape)
        for length in (1024, 4096):
            rows.append(
                {"config": shape, "length": length, "bytes": kv_cache_bytes(config, length)}
            )
    return rows


def training_fixture() -> dict:
    schedules = {}
    for name in ("day1", "day4"):
        config = ProjectConfig.load(ROOT / "configs" / f"{name}.toml")
        settings = config.training
        steps = sorted(
            {0, 1, settings.warmup_steps - 1, settings.warmup_steps, settings.steps // 2}
        )
        steps += [settings.steps - 1, settings.steps, settings.steps + 10]
        schedules[name] = {
            "learning_rate": settings.learning_rate,
            "min_learning_rate": settings.min_learning_rate,
            "warmup_steps": settings.warmup_steps,
            "steps": settings.steps,
            "points": [[step, learning_rate(step, config)] for step in steps],
        }
    feed_forward = [
        {"d_model": width, "ff_multiplier": 4, "feed_forward": kind, "hidden_size": hidden}
        for width in (256, 512)
        for kind in ("gelu", "swiglu")
        for hidden in [DecoderConfig(vocab_size=16, d_model=width, feed_forward=kind).hidden_size]
    ]
    return {"schedules": schedules, "hidden_size": feed_forward}


def sampling_fixture() -> list[dict]:
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.5, -1.0, 3.0, 0.0, -2.0]])
    return [
        {
            "logits": logits[0].tolist(),
            "temperature": temperature,
            "top_k": top_k,
            "probabilities": sampling_distribution(logits, temperature, top_k)[0].tolist(),
        }
        for temperature in (0.5, 0.8, 1.0, 1.5)
        for top_k in (0, 1, 3, 8)
    ]


def parameters_fixture() -> list[dict]:
    rows = []
    for name, vocab, overrides in (
        ("day1", 1024, {}),
        ("day2", 2048, {}),
        ("day2", 2048, VARIANTS["swiglu"]),
        ("day2", 2048, VARIANTS["modern"]),
        ("day4", 8192, {}),
    ):
        settings = ProjectConfig.load(ROOT / "configs" / f"{name}.toml").model
        config = DecoderConfig(vocab_size=vocab, **{**settings.__dict__, **overrides})
        shape = {key: getattr(config, key) for key in SHAPE_KEYS}
        rows.append({"config": shape, "parameters": parameter_count(Decoder(config))})
    return rows


def floats_fixture() -> list[dict]:
    values = [0.1, 1 / 3, 3.14159265, 1000.7, 65504.0, 65519.0, 1e-5, 3e-8, 1e-9, -2.5e-6, 12345.6]
    source = torch.tensor(values, dtype=torch.float32)
    return [
        {"value": value, "float16": half, "bfloat16": brain}
        for value, half, brain in zip(
            values,
            source.to(torch.float16).float().tolist(),
            source.to(torch.bfloat16).float().tolist(),
            strict=True,
        )
    ]


def day5_fixture() -> dict:
    weights = [-0.8, -0.35, 0.0, 0.32, 1.0]
    linear = torch.nn.Linear(len(weights), 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([weights]))
    quantized = Int8Linear(linear)
    rows = [{"variant": "baseline", "bits_per_byte": value} for value in (0.48, 0.5, 0.52)]
    return {
        "spread": spread(rows, "baseline"),
        "weights": weights,
        "scale": quantized.scale.item(),
        "codes": quantized.qweight[0].tolist(),
    }


def main() -> None:
    export_runs()
    fixtures = WEB / "fixtures"
    write(fixtures / "tokenizer.json", tokenizer_fixture())
    write(fixtures / "attention.json", attention_fixture())
    write(fixtures / "tiled.json", tiled_fixture())
    write(fixtures / "positions.json", positions_fixture())
    write(fixtures / "masks.json", masks_fixture())
    write(fixtures / "cache.json", cache_fixture())
    write(fixtures / "training.json", training_fixture())
    write(fixtures / "sampling.json", sampling_fixture())
    write(fixtures / "parameters.json", parameters_fixture())
    write(fixtures / "floats.json", floats_fixture())
    write(fixtures / "day5.json", day5_fixture())


if __name__ == "__main__":
    main()
