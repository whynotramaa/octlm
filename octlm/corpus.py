from __future__ import annotations

import argparse
import hashlib
import json
import random
import sysconfig
import urllib.request
from pathlib import Path

from octlm.tokenizer import _document_hash, _json

GUTENBERG_BOOKS = {
    84: "Frankenstein; or, The Modern Prometheus",
    1228: "On the Origin of Species",
    1342: "Pride and Prejudice",
    1497: "The Republic",
    2701: "Moby Dick; or, The Whale",
    2814: "Dubliners",
}
GUTENBERG_LICENSE = "Public domain in the United States (Project Gutenberg)"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{0}/pg{0}.txt"
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"
SKIP_PREFIXES = ("test_", "__")
CHUNK_BYTES = 65536


def strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg header and footer, keeping the work itself."""
    start = text.find(START_MARKER)
    if start >= 0:
        start = text.find("\n", start) + 1
    end = text.find(END_MARKER)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Project Gutenberg markers are missing or out of order")
    return text[start:end].strip("\n")


def fetch_book(book_id: int, cache: Path) -> str:
    cached = cache / f"pg{book_id}.txt"
    if not cached.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GUTENBERG_URL.format(book_id), timeout=60) as response:
            cached.write_bytes(response.read())
    return cached.read_text(encoding="utf-8-sig")


def python_sources() -> list[tuple[str, str]]:
    """Top-level standard-library modules, sorted, excluding tests and dunder files."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    sources = []
    for path in sorted(stdlib.glob("*.py")):
        if path.name.startswith(SKIP_PREFIXES):
            continue
        sources.append((path.name, path.read_text(encoding="utf-8", errors="strict")))
    return sources


def chunk(text: str, limit: int = CHUNK_BYTES) -> list[str]:
    """Split on line boundaries so no chunk ends mid-line."""
    chunks, current, size = [], [], 0
    for line in text.splitlines(keepends=True):
        current.append(line)
        size += len(line.encode())
        if size >= limit:
            chunks.append("".join(current))
            current, size = [], 0
    if current:
        chunks.append("".join(current))
    return chunks


def _take(sources: list[dict], budget: int) -> list[dict]:
    taken, used = [], 0
    for document in sources:
        if used >= budget:
            break
        taken.append(document)
        used += document["bytes"]
    return taken


def _documents(name: str, kind: str, license_name: str, text: str) -> list[dict]:
    return [
        {
            "bytes": len(piece.encode()),
            "kind": kind,
            "license": license_name,
            "source": name,
            "text": piece,
        }
        for piece in chunk(text)
    ]


def collect(budget_bytes: int, code_fraction: float, cache: Path) -> list[dict]:
    code: list[dict] = []
    for name, text in python_sources():
        code += _documents(name, "code", f"Python {sysconfig.get_python_version()} stdlib", text)
    prose: list[dict] = []
    for book_id, title in GUTENBERG_BOOKS.items():
        body = strip_gutenberg(fetch_book(book_id, cache))
        prose += _documents(f"gutenberg-{book_id} {title}", "prose", GUTENBERG_LICENSE, body)
    code_budget = int(budget_bytes * code_fraction)
    return _take(code, code_budget) + _take(prose, budget_bytes - code_budget)


def split(documents: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)
    cut = round(len(order) * val_fraction)
    validation = {index for index in order[:cut]}
    train = [document for index, document in enumerate(documents) if index not in validation]
    held_out = [document for index, document in enumerate(documents) if index in validation]
    return train, held_out


def write_split(path: Path, documents: list[dict]) -> dict:
    path.write_text("".join(_json(document) + "\n" for document in documents), encoding="utf-8")
    totals: dict[str, int] = {}
    for document in documents:
        totals[document["kind"]] = totals.get(document["kind"], 0) + document["bytes"]
    return {
        "bytes": sum(totals.values()),
        "bytes_by_kind": totals,
        "documents": len(documents),
        "fingerprint": _document_hash(tuple(document["text"] for document in documents)),
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_documents(path: Path, kind: str | None = None) -> list[str]:
    """Read one split back as plain document strings, in file order."""
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [record["text"] for record in records if kind is None or record["kind"] == kind]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Day 2 code and prose corpus.")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--budget-mb", type=float, default=8.0)
    parser.add_argument("--code-fraction", type=float, default=0.625)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if not 0.0 < args.code_fraction < 1.0 or not 0.0 < args.val_fraction < 0.5:
        raise SystemExit("code-fraction must be in (0, 1) and val-fraction in (0, 0.5)")
    args.output.mkdir(parents=True, exist_ok=True)
    documents = collect(round(args.budget_mb * 1024 * 1024), args.code_fraction, args.cache)
    train, held_out = split(documents, args.val_fraction, args.seed)
    manifest = {
        "code_fraction": args.code_fraction,
        "seed": args.seed,
        "sources": sorted({(document["kind"], document["source"]) for document in documents}),
        "train": write_split(args.output / "train.jsonl", train),
        "validation": write_split(args.output / "val.jsonl", held_out),
    }
    (args.output / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    print(_json({key: manifest[key] for key in ("train", "validation")}))


if __name__ == "__main__":
    main()
