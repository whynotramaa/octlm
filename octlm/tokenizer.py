from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable

CHAR_SPECIALS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")
BPE_SPECIALS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|tool_call|>", "<|tool_result|>")


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(data: object) -> str:
    return hashlib.sha256(_json(data).encode()).hexdigest()


def _document_hash(documents: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        encoded = document.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _kind(character: str) -> int:
    if character.isspace():
        return 0
    category = unicodedata.category(character)
    if character == "_" or category[0] in {"L", "M"}:
        return 1
    if category[0] == "N":
        return 2
    return 3


def pretokenize(text: str) -> list[bytes]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    previous = _kind(text[0])
    for index, character in enumerate(text[1:], 1):
        current = _kind(character)
        if current != previous:
            chunks.append(text[start:index])
            start = index
            previous = current
    chunks.append(text[start:])
    return [chunk.encode("utf-8") for chunk in chunks]


@dataclass(frozen=True)
class CharacterTokenizer:
    vocabulary: tuple[str, ...]

    @classmethod
    def train(cls, texts: Iterable[str]) -> CharacterTokenizer:
        characters = sorted(set().union(*(set(text) for text in texts)))
        return cls(CHAR_SPECIALS + tuple(characters))

    @cached_property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabulary)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        lookup = self.token_to_id
        tokens = [lookup.get(character, 3) for character in text]
        return [self.bos_id, *tokens, self.eos_id] if add_special_tokens else tokens

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        output = []
        for token_id in token_ids:
            if not 0 <= token_id < len(self.vocabulary):
                raise ValueError(f"unknown token ID: {token_id}")
            token = self.vocabulary[token_id]
            if token_id < len(CHAR_SPECIALS) and skip_special_tokens:
                continue
            output.append(token)
        return "".join(output)

    def token_bytes(self, token_id: int) -> int:
        if not 0 <= token_id < len(self.vocabulary):
            raise ValueError(f"unknown token ID: {token_id}")
        return 0 if token_id < len(CHAR_SPECIALS) else len(self.vocabulary[token_id].encode())

    def to_dict(self) -> dict[str, object]:
        return {"format": "octlm-char-v1", "vocabulary": list(self.vocabulary)}

    @property
    def fingerprint(self) -> str:
        return _hash(self.to_dict())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(_json(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        data = json.loads(Path(path).read_text())
        if data.get("format") != "octlm-char-v1":
            raise ValueError("unsupported character tokenizer format")
        tokenizer = cls(tuple(data["vocabulary"]))
        if tokenizer.vocabulary[: len(CHAR_SPECIALS)] != CHAR_SPECIALS:
            raise ValueError("character tokenizer special IDs changed")
        return tokenizer


def _merge(sequence: tuple[int, ...], pair: tuple[int, int], token_id: int) -> tuple[int, ...]:
    output: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index : index + 2] == pair:
            output.append(token_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return tuple(output)


@dataclass(frozen=True)
class ByteBPETokenizer:
    merges: tuple[tuple[int, int, int], ...]
    corpus_hash: str
    requested_vocab_size: int
    min_frequency: int

    @classmethod
    def train(
        cls, texts: Iterable[str], vocab_size: int, min_frequency: int = 2
    ) -> ByteBPETokenizer:
        documents = tuple(texts)
        base_size = 256 + len(BPE_SPECIALS)
        if vocab_size < base_size or min_frequency < 1:
            raise ValueError(f"vocab_size must be at least {base_size}")
        sequences = Counter(tuple(chunk) for text in documents for chunk in pretokenize(text))
        merges: list[tuple[int, int, int]] = []
        while base_size + len(merges) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence, frequency in sequences.items():
                counts.update({pair: count * frequency for pair, count in _pairs(sequence).items()})
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            if counts[pair] < min_frequency:
                break
            token_id = base_size + len(merges)
            merged: Counter[tuple[int, ...]] = Counter()
            for sequence, frequency in sequences.items():
                merged[_merge(sequence, pair, token_id)] += frequency
            sequences = merged
            merges.append((*pair, token_id))
        corpus_hash = _document_hash(documents)
        return cls(tuple(merges), corpus_hash, vocab_size, min_frequency)

    @property
    def pad_id(self) -> int:
        return 256

    @property
    def bos_id(self) -> int:
        return 257

    @property
    def eos_id(self) -> int:
        return 258

    @property
    def vocab_size(self) -> int:
        return 256 + len(BPE_SPECIALS) + len(self.merges)

    @property
    def special_to_id(self) -> dict[str, int]:
        return {token: 256 + index for index, token in enumerate(BPE_SPECIALS)}

    @cached_property
    def merge_ranks(self) -> dict[tuple[int, int], tuple[int, int]]:
        return {
            (left, right): (rank, token_id)
            for rank, (left, right, token_id) in enumerate(self.merges)
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ranks = self.merge_ranks
        output: list[int] = []
        for chunk in pretokenize(text):
            sequence = tuple(chunk)
            while len(sequence) > 1:
                candidates = (pair for pair in _pairs(sequence) if pair in ranks)
                pair = min(candidates, key=lambda item: ranks[item][0], default=None)
                if pair is None:
                    break
                sequence = _merge(sequence, pair, ranks[pair][1])
            output.extend(sequence)
        return [self.bos_id, *output, self.eos_id] if add_special_tokens else output

    def decode(
        self,
        token_ids: Iterable[int],
        errors: str = "strict",
        skip_special_tokens: bool = False,
    ) -> str:
        table = self.byte_table
        specials = {token_id: token for token, token_id in self.special_to_id.items()}
        output = bytearray()
        for token_id in token_ids:
            if token_id in specials:
                if not skip_special_tokens:
                    output.extend(specials[token_id].encode())
            elif token_id in table:
                output.extend(table[token_id])
            else:
                raise ValueError(f"unknown token ID: {token_id}")
        return output.decode("utf-8", errors=errors)

    def token_bytes(self, token_id: int) -> int:
        if token_id in self.special_to_id.values():
            return 0
        try:
            return len(self.byte_table[token_id])
        except KeyError as error:
            raise ValueError(f"unknown token ID: {token_id}") from error

    @cached_property
    def byte_table(self) -> dict[int, bytes]:
        table = {index: bytes([index]) for index in range(256)}
        for offset, (left, right, token_id) in enumerate(self.merges):
            if token_id != 256 + len(BPE_SPECIALS) + offset:
                raise ValueError("merge IDs are not sequential")
            try:
                table[token_id] = table[left] + table[right]
            except KeyError as error:
                raise ValueError("merge references an unknown token") from error
        return table

    def to_dict(self) -> dict[str, object]:
        return {
            "actual_vocab_size": self.vocab_size,
            "corpus_sha256": self.corpus_hash,
            "format": "octlm-byte-bpe-v1",
            "merges": [list(merge) for merge in self.merges],
            "min_frequency": self.min_frequency,
            "pretokenizer": "unicode-category-runs-v1",
            "requested_vocab_size": self.requested_vocab_size,
            "special_tokens": self.special_to_id,
        }

    @property
    def fingerprint(self) -> str:
        return _hash(self.to_dict())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(_json(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        data = json.loads(Path(path).read_text())
        if data.get("format") != "octlm-byte-bpe-v1":
            raise ValueError("unsupported BPE tokenizer format")
        if data.get("pretokenizer") != "unicode-category-runs-v1":
            raise ValueError("unsupported pre-tokenizer")
        if data.get("special_tokens") != {token: 256 + i for i, token in enumerate(BPE_SPECIALS)}:
            raise ValueError("BPE special IDs changed")
        tokenizer = cls(
            tuple(tuple(merge) for merge in data["merges"]),
            data["corpus_sha256"],
            data["requested_vocab_size"],
            data["min_frequency"],
        )
        tokenizer.byte_table
        if tokenizer.vocab_size != data["actual_vocab_size"]:
            raise ValueError("BPE vocabulary size does not match its merges")
        return tokenizer


def _pairs(sequence: tuple[int, ...]) -> Counter[tuple[int, int]]:
    return Counter(zip(sequence, sequence[1:]))


def tokenizer_metrics(
    tokenizer: CharacterTokenizer | ByteBPETokenizer, text: str
) -> dict[str, float | int]:
    token_ids = tokenizer.encode(text)
    byte_count = len(text.encode())
    unknown_count = token_ids.count(3) if isinstance(tokenizer, CharacterTokenizer) else 0
    return {
        "bytes": byte_count,
        "bytes_per_token": byte_count / len(token_ids) if token_ids else 0.0,
        "characters": len(text),
        "tokens": len(token_ids),
        "unknown_rate": unknown_count / len(token_ids) if token_ids else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and measure the Day 1 tokenizers.")
    parser.add_argument("--train", type=Path, default=Path("PLAN.md"))
    parser.add_argument("--eval", type=Path, default=Path("day-wise.md"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/day1"))
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[512, 1024, 2048])
    args = parser.parse_args()
    train_text = args.train.read_text()
    samples = {
        "code": "def add(x: int, y: int) -> int:\n    return x + y\n",
        "prose": args.eval.read_text(),
        "unicode": "नमस्ते 👩🏽‍💻 e\u0301\r\n\t\x00",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    character = CharacterTokenizer.train([train_text])
    character.save(args.output / "character.json")
    print(
        _json(
            {
                "fingerprint": character.fingerprint,
                "kind": "character",
                "vocab_size": len(character.vocabulary),
            }
        )
    )
    for name, sample in samples.items():
        print(_json({"kind": "character", "sample": name, **tokenizer_metrics(character, sample)}))
    for size in args.vocab_sizes:
        bpe = ByteBPETokenizer.train([train_text], size)
        bpe.save(args.output / f"bpe-{size}.json")
        print(_json({"fingerprint": bpe.fingerprint, "kind": "bpe", "vocab_size": bpe.vocab_size}))
        for name, sample in samples.items():
            print(
                _json(
                    {
                        "kind": "bpe",
                        "requested_vocab": size,
                        "sample": name,
                        **tokenizer_metrics(bpe, sample),
                    }
                )
            )


if __name__ == "__main__":
    main()
