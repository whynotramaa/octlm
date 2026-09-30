from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
import statistics
import struct
import time
import urllib.request
from pathlib import Path

import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from octlm.day2 import _write
from octlm.model import Decoder, DecoderConfig, kv_cache_bytes, parameter_count
from octlm.train import resolve_device

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
LOGIT_TOLERANCE = 1e-3
TEXTS = (
    "Once upon a time, a little robot found a key.",
    "def add(a, b):\n\treturn a + b\r\n",
    "a\x00b 🙂👍🏽 cafe\u0301 日本語 Привет",
    "  if x:\n    print(x)\n\n<|im_start|> <|im_end|> <|endoftext|>",
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
CHATS = (
    [{"role": "user", "content": "Reply with one short sentence about a robot."}],
    [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "What is 2 + 2?"}],
    [
        {"role": "user", "content": "Read hello.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "hello.py"}},
                }
            ],
        },
        {"role": "tool", "content": "print('hello')"},
    ],
)
DTYPES = {"F32": (torch.float32, 4), "F16": (torch.float16, 2), "BF16": (torch.bfloat16, 2)}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return result


def prepare(directory: Path) -> None:
    if (directory / "manifest.json").exists():
        verify_snapshot(directory)
        return
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = directory / name
        temporary = path.with_suffix(path.suffix + ".part")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(path)
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "sha256": {name: file_hash(directory / name) for name in FILES},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    verify_snapshot(directory)


def verify_snapshot(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION:
        raise ValueError("checkpoint identity or revision differs from the pinned snapshot")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(FILES):
        raise ValueError("checkpoint manifest must hash every required file")
    for name in FILES:
        if hashes[name] != file_hash(directory / name):
            raise ValueError(f"checkpoint hash mismatch: {name}")
    return manifest


def positive_int(data: dict, name: str) -> int:
    value = data.get(name)
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def qwen_config(data: dict) -> DecoderConfig:
    supported = {
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_scaling": None,
        "use_sliding_window": False,
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-6,
    }
    for key, value in supported.items():
        if data.get(key) != value:
            raise ValueError(f"unsupported Qwen setting: {key}")
    config = DecoderConfig(
        vocab_size=positive_int(data, "vocab_size"),
        context_length=positive_int(data, "max_position_embeddings"),
        d_model=positive_int(data, "hidden_size"),
        n_heads=positive_int(data, "num_attention_heads"),
        n_layers=positive_int(data, "num_hidden_layers"),
        kv_heads=positive_int(data, "num_key_value_heads"),
        head_dim=positive_int(data, "head_dim"),
        ff_hidden=positive_int(data, "intermediate_size"),
        rope_base=float(data["rope_theta"]),
        position="rope",
        norm="rmsnorm",
        feed_forward="swiglu",
        attention="sdpa",
        rope_split=True,
        qk_norm=True,
    )
    config.validate()
    return config


def tensor_spec(entry: dict) -> tuple[torch.dtype, list[int], int, int]:
    if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
        raise ValueError("invalid safetensors entry")
    shape, offsets = entry["shape"], entry["data_offsets"]
    if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("invalid tensor shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(x) is not int for x in offsets)
    ):
        raise ValueError("invalid tensor offsets")
    if not isinstance(entry["dtype"], str) or entry["dtype"] not in DTYPES:
        raise ValueError("unsupported tensor dtype")
    dtype, size = DTYPES[entry["dtype"]]
    start, end = offsets
    if start < 0 or end - start != math.prod(shape) * size:
        raise ValueError("tensor byte count differs from shape")
    return dtype, shape, start, end


def read_header(stream, file_size: int) -> tuple[dict, int]:
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated safetensors header length")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= min(100_000_000, file_size - 8):
        raise ValueError("invalid safetensors header length")
    header = json.loads(stream.read(length), object_pairs_hook=unique_keys)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("invalid safetensors metadata")
    cursor = 0
    entries = {name: tensor_spec(entry) for name, entry in header.items()}
    for _, _, start, end in sorted(entries.values(), key=lambda x: (x[2], x[3])):
        if start != cursor:
            raise ValueError("tensor offsets overlap or leave a gap")
        cursor = end
    if cursor != file_size - 8 - length:
        raise ValueError("tensor offsets do not cover the complete payload")
    return entries, 8 + length


def safetensors(path: Path):
    with path.open("rb") as stream:
        entries, data_start = read_header(stream, path.stat().st_size)
        for name, (dtype, shape, start, end) in entries.items():
            stream.seek(data_start + start)
            buffer = bytearray(stream.read(end - start))
            if len(buffer) != end - start:
                raise ValueError("truncated tensor payload")
            tensor = (
                torch.frombuffer(buffer, dtype=dtype).reshape(shape)
                if buffer
                else torch.empty(shape, dtype=dtype)
            )
            yield name, tensor


def weight_names(config: DecoderConfig) -> dict[str, str]:
    names = {
        "model.embed_tokens.weight": "token_embedding.weight",
        "model.norm.weight": "final_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    layer_names = {
        "input_layernorm": "attention_norm",
        "post_attention_layernorm": "feed_forward_norm",
        "self_attn.q_proj": "attention.query",
        "self_attn.k_proj": "attention.key",
        "self_attn.v_proj": "attention.value",
        "self_attn.o_proj": "attention.output",
        "self_attn.q_norm": "attention.query_norm",
        "self_attn.k_norm": "attention.key_norm",
        "mlp.gate_proj": "feed_forward.gate",
        "mlp.up_proj": "feed_forward.up",
        "mlp.down_proj": "feed_forward.down",
    }
    for index in range(config.n_layers):
        for source, target in layer_names.items():
            names[f"model.layers.{index}.{source}.weight"] = f"blocks.{index}.{target}.weight"
    return names


def load_qwen(
    directory: Path, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> tuple[Decoder, dict]:
    manifest = verify_snapshot(directory)
    config = qwen_config(read_json(directory / "config.json"))
    with torch.device("meta"):
        model = Decoder(config)
    model = model.to(dtype=dtype).to_empty(device=resolve_device(device))
    model.lm_head.weight = model.token_embedding.weight
    state, names, seen = model.state_dict(), weight_names(config), set()
    with torch.no_grad():
        for name, tensor in safetensors(directory / "model.safetensors"):
            if name not in names:
                raise ValueError(f"unexpected checkpoint tensor: {name}")
            target = state[names[name]]
            if tensor.shape != target.shape:
                raise ValueError(f"checkpoint tensor shape mismatch: {name}")
            tied = {"model.embed_tokens.weight", "lm_head.weight"}
            if name in tied and seen & tied:
                if not torch.equal(target, tensor.to(device=target.device, dtype=target.dtype)):
                    raise ValueError("checkpoint tied weights differ")
            target.copy_(tensor)
            seen.add(name)
    missing = set(names) - seen - {"lm_head.weight"}
    if missing:
        raise ValueError(f"missing checkpoint tensors: {sorted(missing)}")
    return model.eval(), manifest


def template_json(value: object, **kwargs) -> str:
    return json.dumps(value, ensure_ascii=False, **kwargs)


class QwenTokenizer:
    def __init__(self, directory: Path) -> None:
        self.native = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.literal = Tokenizer.from_str(self.native.to_str())
        self.literal.encode_special_tokens = True
        self.literal.normalizer = None
        self.fingerprint = file_hash(directory / "tokenizer.json")
        self.settings = read_json(directory / "tokenizer_config.json")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = template_json
        self.template = environment.from_string(self.settings["chat_template"])
        self.special_ids = {
            token["id"]
            for token in json.loads(self.native.to_str())["added_tokens"]
            if token["special"]
        }

    def encode(self, text: str, *, allow_special: bool = False) -> list[int]:
        encoder = self.native if allow_special else self.literal
        return encoder.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.native.decode(ids, skip_special_tokens=False)

    def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, thinking: bool = False
    ) -> str:
        if not messages or any(
            message.get("role") not in ("system", "user", "assistant", "tool")
            for message in messages
        ):
            raise ValueError("chat needs nonempty messages with supported roles")
        return self.template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=thinking,
            **{key: self.settings[key] for key in ("bos_token", "eos_token", "pad_token")},
        )


def dry_run(directory: Path, out: Path) -> None:
    config = qwen_config(read_json(directory / "config.json"))
    with torch.device("meta"):
        model = Decoder(config)
    _write(
        out,
        {
            "type": "qwen_dry_run",
            "parameters": parameter_count(model),
            "head_size": config.head_size,
            "attention_width": config.n_heads * config.head_size,
            "ff_hidden": config.hidden_size,
            "revision": REVISION,
        },
    )


def tokenizer_parity(tokenizer: QwenTokenizer, reference, out: Path) -> list[list[int]]:
    literal_reference = copy.deepcopy(reference)
    literal_reference.backend_tokenizer.normalizer = None
    literal_reference.split_special_tokens = True
    cases = []
    for text in TEXTS:
        ids = tokenizer.encode(text)
        expected = literal_reference(text, add_special_tokens=False)["input_ids"]
        if ids != expected or tokenizer.decode(ids) != text or set(ids) & tokenizer.special_ids:
            raise RuntimeError("ordinary tokenizer parity or literal preservation failed")
        native = tokenizer.encode(text, allow_special=True)
        if native != reference(text, add_special_tokens=False)["input_ids"]:
            raise RuntimeError("native tokenizer ID parity failed")
        cases.append(ids)
        _write(
            out,
            {
                "type": "tokenizer_parity",
                "text": text,
                "ids": ids,
                "native_ids": native,
                "equal": True,
            },
        )
    for index, messages in enumerate(CHATS):
        for thinking in (False, True):
            tools = TOOLS if index == 2 else None
            text = tokenizer.chat(messages, tools=tools, thinking=thinking)
            expected = reference.apply_chat_template(
                messages,
                tools=tools,
                enable_thinking=thinking,
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tokenizer.encode(text, allow_special=True)
            if (
                text != expected
                or ids != reference(expected, add_special_tokens=False)["input_ids"]
            ):
                raise RuntimeError("native chat-template parity failed")
            cases.append(ids)
            _write(
                out,
                {
                    "type": "template_parity",
                    "chat": index,
                    "thinking": thinking,
                    "text": text,
                    "ids": ids,
                    "equal": True,
                },
            )
    return cases


@torch.inference_mode()
def logits_parity(model: Decoder, reference, cases: list[list[int]], out: Path) -> None:
    device = next(model.parameters()).device
    for index, ids in enumerate(cases):
        tokens = torch.tensor([ids], device=device)
        expected = reference(tokens, use_cache=False).logits.float()
        torch.save(
            {"ids": ids, "logits": expected.cpu()}, out.parent / f"reference-logits-{index}.pt"
        )
        full = model(tokens)
        split = max(1, tokens.shape[1] // 2)
        first, cache = model.forward_cached(tokens[:, :split])
        second, _ = model.forward_cached(tokens[:, split:], cache)
        cached = torch.cat((first, second), dim=1)
        for path, actual in (("full", full), ("cached", cached)):
            delta = (actual.float() - expected).abs()
            maximum = delta.max().item()
            row = {
                "type": "logit_parity",
                "case": index,
                "path": path,
                "max_absolute_error": maximum,
                "mean_absolute_error": delta.mean().item(),
                "argmax_agreement": (actual.argmax(-1) == expected.argmax(-1))
                .float()
                .mean()
                .item(),
                "tolerance": LOGIT_TOLERANCE,
            }
            _write(out, row)
            if not torch.isfinite(actual).all() or maximum > LOGIT_TOLERANCE:
                raise RuntimeError(f"float32 {path} logit parity failed on case {index}: {maximum}")


@torch.inference_mode()
def reference_generate(reference, prompt: torch.Tensor, count: int) -> torch.Tensor:
    result = reference(prompt, use_cache=True)
    output = prompt
    for step in range(count):
        token = result.logits[:, -1].argmax(-1, keepdim=True)
        output = torch.cat((output, token), 1)
        if step + 1 < count:
            result = reference(token, past_key_values=result.past_key_values, use_cache=True)
    return output


def timed(call, device: torch.device) -> tuple[torch.Tensor, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - started


def generation_benchmark(
    model: Decoder, reference, tokenizer: QwenTokenizer, count: int, repeats: int, out: Path
) -> None:
    device = next(model.parameters()).device
    for index, messages in enumerate(CHATS[:2]):
        prompt = torch.tensor(
            [tokenizer.encode(tokenizer.chat(messages), allow_special=True)], device=device
        )
        model.generate_cached(prompt, 2)
        reference_generate(reference, prompt, 2)
        ours_times, reference_times = [], []
        for _ in range(repeats):
            ours, elapsed = timed(lambda: model.generate_cached(prompt, count), device)
            expected, reference_elapsed = timed(
                lambda: reference_generate(reference, prompt, count), device
            )
            ours_times.append(elapsed)
            reference_times.append(reference_elapsed)
            if not torch.equal(ours, expected):
                _write(
                    out,
                    {
                        "type": "generation_mismatch",
                        "case": index,
                        "actual_ids": ours.tolist(),
                        "reference_ids": expected.tolist(),
                    },
                )
                raise RuntimeError("cached generation differs from the reference")
        _, cache = model.forward_cached(ours[:, :-1])
        cache_bytes = sum((k.numel() + v.numel()) * k.element_size() for k, v in cache)
        expected_bytes = kv_cache_bytes(
            model.config, ours.shape[1] - 1, next(model.parameters()).element_size()
        )
        if cache_bytes != expected_bytes:
            raise RuntimeError("actual Qwen cache bytes differ from the formula")
        _write(
            out,
            {
                "type": "qwen_generation",
                "case": index,
                "device": str(device),
                "dtype": str(next(model.parameters()).dtype),
                "new_tokens": count,
                "seconds": ours_times,
                "reference_seconds": reference_times,
                "tokens_per_second": count / statistics.median(ours_times),
                "reference_tokens_per_second": count / statistics.median(reference_times),
                "cache_bytes": cache_bytes,
                "greedy_equal": True,
                "generated_ids": ours[:, prompt.shape[1] :].tolist(),
                "text": tokenizer.decode(ours[0, prompt.shape[1] :].tolist()),
            },
        )


def run(args: argparse.Namespace) -> None:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.out.exists():
        raise FileExistsError(f"{args.out} already contains a measurement")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, manifest = load_qwen(args.model, str(device))
    tokenizer = QwenTokenizer(args.model)
    reference_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    reference = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    _write(
        args.out,
        {
            "type": "qwen_environment",
            "manifest": manifest,
            "parameters": parameter_count(model),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "threads": args.threads,
            "tokenizer_hash": tokenizer.fingerprint,
            "source_hashes": {
                name: file_hash(Path(__file__).with_name(name)) for name in ("day6.py", "model.py")
            },
        },
    )
    cases = tokenizer_parity(tokenizer, reference_tokenizer, args.out)
    if not args.benchmark_only:
        logits_parity(model, reference, cases, args.out)
    if device.type == "cuda":
        model.half()
        reference.half()
    gc.collect()
    generation_benchmark(model, reference, tokenizer, args.tokens, args.repeats, args.out)
    _write(
        args.out,
        {
            "type": "day6_exit",
            "logit_parity": not args.benchmark_only,
            "tokenizer_parity": True,
            "template_parity": True,
            "generation_parity": True,
            "device": str(device),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and measure the pinned Qwen checkpoint.")
    parser.add_argument("stage", choices=("prepare", "dry-run", "run"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/day6/qwen"))
    parser.add_argument("--out", type=Path, default=Path("runs/day6.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.tokens, args.repeats) < 1:
        parser.error("threads, tokens, and repeats must be positive")
    if args.stage == "prepare":
        prepare(args.model)
    elif args.stage == "dry-run":
        dry_run(args.model, args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()
