from __future__ import annotations

import json
from pathlib import Path

import torch

from octlm.config import ProjectConfig
from octlm.day2 import VARIANTS, tiled_attention
from octlm.day5 import Int8Linear, spread
from octlm.day6 import CHATS, QwenTokenizer, qwen_config, read_header, read_json, weight_names
from octlm.day8 import TARGETS, add_lora, examples
from octlm.harness import (
    SYSTEM,
    TOOLS,
    answer_matches,
    common_prefix,
    load_tasks,
    parse_calls,
    pass_hat,
)
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
    export_jsonl(ROOT / "runs/kaggle-day6/day6/results.jsonl", "day6-t4")
    export_jsonl(ROOT / "runs/day6-cpu-parity/results.jsonl", "day6-cpu-parity")
    export_jsonl(ROOT / "runs/kaggle-day7/day7/eval.jsonl", "day7-eval-v1")
    export_jsonl(ROOT / "runs/kaggle-day7-v2/day7/eval.jsonl", "day7-eval")
    for name in ("train-0", "train-1", "eval-0", "eval-1"):
        export_jsonl(ROOT / f"runs/kaggle-day8/{name}.jsonl", f"day8-{name}")
    export_jsonl(ROOT / "runs/kaggle-day8/eval-0-int8.jsonl", "day8-int8-gate")
    export_jsonl(ROOT / "runs/kaggle-day8-int8/eval-0-int8.jsonl", "day8-eval-0-int8")
    export_jsonl(ROOT / "runs/kaggle-day8-int8/train-0.jsonl", "day8-train-0-rerun")


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


def day6_fixture() -> dict:
    cosine, sine = rope_tables(16, 8, 1.0, CPU, 1e6)
    torch.manual_seed(3)
    x = torch.randn(1, 1, 16, 8)
    return {
        "base": 1e6,
        "x": x[0, 0].tolist(),
        "rotated": apply_rope(x, cosine, sine, split=True)[0, 0].tolist(),
    }


def qwen_data() -> dict:
    directory = ROOT / "artifacts/day6/qwen"
    path = directory / "model.safetensors"
    with path.open("rb") as stream:
        entries, data_start = read_header(stream, path.stat().st_size)
    config = qwen_config(read_json(directory / "config.json"))
    names = weight_names(config)
    tensors = [
        {"name": name, "octlm": names[name], "shape": shape, "start": start, "end": end}
        for name, (_, shape, start, end) in sorted(entries.items(), key=lambda x: x[1][2])
    ]
    tokenizer = QwenTokenizer(directory)
    records = [json.loads(line) for line in (ROOT / "runs/kaggle-day6/day6/results.jsonl").open()]
    generations = [
        {
            "case": row["case"],
            "prompt": tokenizer.chat(CHATS[row["case"]]),
            "pieces": [tokenizer.decode([token]) for token in row["generated_ids"][0]],
            "ids": row["generated_ids"][0],
        }
        for row in records
        if row["type"] == "qwen_generation"
    ]
    return {
        "file_bytes": path.stat().st_size,
        "header_bytes": data_start,
        "dtype": "BF16",
        "tensors": tensors,
        "generations": generations,
        "im_end": tokenizer.native.token_to_id("<|im_end|>"),
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "kv_heads": config.kv_heads,
            "head_size": config.head_size,
            "ff_hidden": config.hidden_size,
            "rope_base": config.rope_base,
        },
    }


PARSE_CASES = (
    "The server uses port 8080.",
    '<tool_call>\n{"name": "read_file", "arguments": {"path": "README.md"}}\n</tool_call>',
    '<tool_call>\n{"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1\ny"}}'
    "\n</tool_call>",
    '<tool_call>\n{"name": "write_file", "arguments": {"path": "a.py", "content": "def f(',
    '<tool_call>\n{"name": "delete_file", "arguments": {"path": "a.py"}}\n</tool_call>',
    '<tool_call>\n{"name": "grep", "arguments": {"pattern": "x", "limit": "3"}}\n</tool_call>',
    '<tool_call>\n{"name": "run_tests", "arguments": {}}\n</tool_call>\n'
    '<tool_call>\n{"name": "read_file", "arguments": {"path": 7}}\n</tool_call>',
    '<tool_call>["read_file"]</tool_call>',
    "<tool_call></tool_call>",
)
ANSWER_CASES = (
    ("12", "The number of lamps in stock is 123."),
    ("12", "There are 12 lamps."),
    ("1.4.2", "Version 1.4.2."),
    ("inventory.py", "It is in shop/inventory.py"),
    ("cart", "shop/cart.py imports it"),
    ("5", "The timeout is 45"),
    ("0.08", "The rate is 0.085"),
    ("Apache-2.0", "The Apache-2.0 license"),
)


def day7_fixture() -> dict:
    return {
        "parse": [{"text": text, "calls": parse_calls(text)} for text in PARSE_CASES],
        "answers": [
            {"expected": e, "answer": a, "match": answer_matches(e, a)} for e, a in ANSWER_CASES
        ],
        "pass_hat": [
            {"successes": c, "trials": 3, "k": k, "value": pass_hat(c, 3, k)}
            for c in ([0, 1, 2, 3], [3, 3, 0], [1, 1, 1, 1])
            for k in (1, 2, 3)
        ],
    }


def turn_tokens(tokenizer: QwenTokenizer, messages: list[dict]) -> list[dict]:
    turns, cached = [], []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        text = tokenizer.chat(messages[:index], tools=TOOLS, thinking=False)
        prompt = tokenizer.encode(text, allow_special=True)
        keep = min(common_prefix(cached, prompt), len(prompt) - 1)
        reused = tokenizer.decode(prompt[:keep])
        turns.append(
            {"text": text, "reused_chars": len(reused), "rendered": len(prompt), "keep": keep}
        )
        cached = prompt + tokenizer.encode(message["content"], allow_special=True)
    return turns


def day7_data() -> dict:
    tokenizer = QwenTokenizer(ROOT / "artifacts/day6/qwen")
    rows = [json.loads(line) for line in (ROOT / "runs/kaggle-day7-v2/day7/eval.jsonl").open()]
    runs = [row for row in rows if row["type"] == "day7_run"]
    traces = []
    for task, seed in (("todo-file", 0), ("bump-version", 2), ("sku", 1)):
        run = next(r for r in runs if r["task"] == task and r["seed"] == seed)
        turns = turn_tokens(tokenizer, run["messages"])
        rendered = sum(t["rendered"] for t in turns)
        prefilled = sum(t["rendered"] - t["keep"] for t in turns)
        if (rendered, prefilled) != (run["rendered_tokens"], run["prefilled_tokens"]):
            raise RuntimeError(f"token accounting for {task} differs from the recorded run")
        traces.append({"task": task, "seed": seed, "turns": turns, "messages": run["messages"]})
    first = tokenizer.encode(
        tokenizer.chat([{"role": "system", "content": SYSTEM}], tools=TOOLS), allow_special=True
    )
    tasks = load_tasks(ROOT / "fixtures/day7/tasks.json")
    return {
        "system": SYSTEM,
        "tools": TOOLS,
        "system_tokens": len(first),
        "tasks": [{k: t[k] for k in ("id", "prompt", "check")} for t in tasks],
        "traces": traces,
    }


def day8_fixture() -> dict:
    config = DecoderConfig(
        vocab_size=32,
        context_length=8,
        d_model=8,
        n_heads=2,
        n_layers=2,
        kv_heads=1,
        position="rope",
        norm="rmsnorm",
        feed_forward="swiglu",
    )
    model = Decoder(config).eval()
    trainable = sum(p.numel() for p in add_lora(model, 4, 8, seed=0))
    layer = model.blocks[0].attention.query
    generator = torch.Generator().manual_seed(1)
    layer.lora_b.data = torch.randn(layer.lora_b.shape, generator=generator)
    x = torch.randn(3, layer.base.in_features, generator=generator)
    return {
        "shapes": [
            [getattr(block, name).base.in_features, getattr(block, name).base.out_features]
            for part, names in TARGETS.items()
            for name in names
            for block in [getattr(model.blocks[0], part)]
        ],
        "layers": config.n_layers,
        "rank": 4,
        "trainable": trainable,
        "base": layer.base.weight.tolist(),
        "a": layer.lora_a.tolist(),
        "b": layer.lora_b.tolist(),
        "scale": layer.scale,
        "x": x.tolist(),
        "output": layer(x).tolist(),
        "merged": layer.merged().weight.tolist(),
    }


def day8_data() -> dict:
    tokenizer = QwenTokenizer(ROOT / "artifacts/day6/qwen")
    rows = [json.loads(line) for line in (ROOT / "runs/day8-traces.jsonl").open()]
    summary = rows[-1]
    tasks = {task["id"]: task for task in load_tasks(ROOT / "fixtures/day8/tasks.json")}
    traces = []
    for row in rows[:-1]:
        found = examples(row["messages"], tokenizer)
        turns = [[start, len(ids) - start] for ids, start in found]
        traces.append(
            {"task": row["task"], "kind": tasks[row["task"]]["check"]["kind"], "turns": turns}
        )
    counts = [turn for item in traces for turn in item["turns"]]
    totals = (len(counts), sum(p + t for p, t in counts), sum(t for _, t in counts))
    if totals != (summary["examples"], summary["total_tokens"], summary["target_tokens"]):
        raise RuntimeError("Day 8 trace token counts differ from the recorded trace summary")
    return {"traces": traces}


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
    write(fixtures / "day6.json", day6_fixture())
    write(fixtures / "day7.json", day7_fixture())
    write(fixtures / "day8.json", day8_fixture())
    write(WEB / "src/data/qwen.json", qwen_data())
    write(WEB / "src/data/day7.json", day7_data())
    write(WEB / "src/data/day8.json", day8_data())


if __name__ == "__main__":
    main()
