from __future__ import annotations

import argparse
import json
import platform
import resource
import time

import torch
from torch.nn import functional as F

from octlm.model import Decoder, DecoderConfig, parameter_count
from octlm.train import resolve_device


def measure(context_length: int, device: torch.device) -> dict[str, object]:
    torch.manual_seed(0)
    config = DecoderConfig(
        vocab_size=512,
        context_length=context_length,
        d_model=128,
        n_heads=4,
        n_layers=2,
    )
    model = Decoder(config).eval().to(device)
    inputs = torch.zeros((1, context_length), dtype=torch.long, device=device)
    with torch.inference_mode():
        if device.type == "cuda":
            model(inputs)  # first call loads kernels, so time the second
            torch.cuda.synchronize()
        started = time.perf_counter()
        model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    count = parameter_count(model)
    token_vectors = F.normalize(model.token_embedding.weight.detach(), dim=1)
    position_vectors = F.normalize(model.position_embedding.weight.detach(), dim=1)
    token_neighbor = int((token_vectors[0] @ token_vectors[1:].T).argmax()) + 1
    position_neighbor = (
        int((position_vectors[0] @ position_vectors[1:].T).argmax()) + 1
        if context_length > 1
        else None
    )
    return {
        "batch_size": 1,
        "config_sha256": None,
        "context_length": context_length,
        "decode_tokens_per_second": None,
        "device": str(device),
        "dtype": "float32",
        "elapsed_seconds": elapsed,
        "experiment_id": "EXP-002",
        "kv_cache_bytes": 0,
        "loss": None,
        "model_bytes": count * 4,
        "nearest_position_to_zero": position_neighbor,
        "nearest_token_to_zero": token_neighbor,
        "parameter_count": count,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "perplexity": None,
        "python": platform.python_version(),
        "run_id": f"{device.type}-context-{context_length}",
        "schema": "octlm-bench-v1",
        "time_to_first_token_ms": None,
        "tokenizer_sha256": None,
        "tokens_processed": context_length,
        "torch": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the Day 1 baseline.")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--contexts", type=int, nargs="+", default=[128])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    if not args.dummy:
        parser.error("Day 1 supports only --dummy")
    device = resolve_device(args.device)
    for context_length in args.contexts:
        if context_length < 1 or context_length > 2048:
            parser.error("context lengths must be between 1 and 2048")
        print(json.dumps(measure(context_length, device), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
