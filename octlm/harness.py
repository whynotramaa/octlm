from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

from octlm.day2 import _write
from octlm.day6 import QwenTokenizer, file_hash, load_qwen, sampling, timed
from octlm.model import Decoder, parameter_count, sample_token
from octlm.train import resolve_device

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "day7"
SYSTEM = (
    "You are a coding agent working in a small Python repository. Use the tools to inspect and "
    "change files. Paths are relative to the repository root. When the task is done, reply with "
    "a short final answer and no tool call."
)
TEST_COMMAND = (sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".")
TEST_TIMEOUT = 30
OUTPUT_LIMIT = 4000
MAX_TURNS = 8
MAX_NEW_TOKENS = 512
CONTEXT_LIMIT = 6144
CHECKS = {"answer", "file", "tests", "hidden"}
CALL = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)


def tool(name: str, description: str, required: tuple[str, ...], **properties: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "description": text} for key, text in properties.items()
                },
                "required": list(required),
            },
        },
    }


TOOLS = [
    tool("read_file", "Read a UTF-8 text file.", ("path",), path="File path."),
    tool(
        "list_files",
        "List files under a directory, recursively.",
        (),
        path="Directory path. Defaults to the repository root.",
    ),
    tool(
        "grep",
        "Search file lines with a Python regular expression.",
        ("pattern",),
        pattern="Regular expression.",
        path="File or directory to search. Defaults to the repository root.",
    ),
    tool(
        "write_file",
        "Create or overwrite a file with the given content.",
        ("path", "content"),
        path="File path.",
        content="Complete new file content.",
    ),
    tool("run_tests", "Run the repository's unittest suite.", ()),
]
SCHEMAS = {entry["function"]["name"]: entry["function"]["parameters"] for entry in TOOLS}


def resolve(root: Path, path: str) -> Path:
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path leaves the repository: {path}")
    return target


def files(root: Path, start: Path) -> list[Path]:
    found = [start] if start.is_file() else start.rglob("*")
    return sorted(
        path
        for path in found
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts
    )


def clip(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    return text[:OUTPUT_LIMIT] + f"\n[truncated {len(text) - OUTPUT_LIMIT} characters]"


def read_file(root: Path, path: str) -> str:
    return resolve(root, path).read_text(encoding="utf-8")


def list_files(root: Path, path: str = ".") -> str:
    start = resolve(root, path)
    if not start.is_dir():
        raise ValueError(f"not a directory: {path}")
    return "\n".join(str(item.relative_to(root)) for item in files(root, start)) or "(empty)"


def grep(root: Path, pattern: str, path: str = ".") -> str:
    expression, lines = re.compile(pattern), []
    for item in files(root, resolve(root, path)):
        text = item.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if expression.search(line):
                lines.append(f"{item.relative_to(root)}:{number}: {line}")
    return "\n".join(lines) or "no matches"


def write_file(root: Path, path: str, content: str) -> str:
    target = resolve(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {target.relative_to(root)}"


def test_status(root: Path) -> tuple[int, str]:
    process = subprocess.Popen(
        TEST_COMMAND,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        output, _ = process.communicate(timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return 1, f"tests timed out after {TEST_TIMEOUT} seconds"
    return process.returncode, output


def run_tests(root: Path) -> str:
    code, output = test_status(root)
    return f"exit code {code}\n{output}"


IMPLEMENTATIONS = {
    "read_file": read_file,
    "list_files": list_files,
    "grep": grep,
    "write_file": write_file,
    "run_tests": run_tests,
}


def execute(root: Path, call: dict) -> tuple[str, bool]:
    try:
        return clip(IMPLEMENTATIONS[call["name"]](root, **call["arguments"])), True
    except (OSError, ValueError, re.error) as error:
        return f"error: {error}", False


def validate_call(block: str) -> dict | str:
    try:
        call = json.loads(block)
    except json.JSONDecodeError as error:
        return f"error: tool call is not valid JSON: {error}"
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return 'error: a tool call must be a JSON object with exactly "name" and "arguments"'
    schema = SCHEMAS.get(call["name"]) if isinstance(call["name"], str) else None
    if schema is None:
        return f"error: unknown tool {call['name']!r}"
    arguments = call["arguments"]
    if not isinstance(arguments, dict):
        return "error: arguments must be a JSON object"
    missing = set(schema["required"]) - set(arguments)
    unknown = set(arguments) - set(schema["properties"])
    if missing or unknown:
        return f"error: missing arguments {sorted(missing)}, unknown arguments {sorted(unknown)}"
    if any(not isinstance(value, str) for value in arguments.values()):
        return "error: every argument must be a string"
    return call


def parse_calls(text: str) -> list[dict | str]:
    return [
        validate_call(match[1] if match[1] is not None else match[2])
        for match in CALL.finditer(text)
    ]


def load_tasks(path: Path) -> list[dict]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks file must hold a nonempty list")
    for task in tasks:
        if not {"id", "prompt", "check", "solution"} <= set(task):
            raise ValueError(f"task lacks a required field: {task.get('id')}")
        if task["check"].get("kind") not in CHECKS:
            raise ValueError(f"unknown check kind in task {task['id']}")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique")
    return tasks


@contextmanager
def sandbox(task: dict):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory, "repo").resolve()
        shutil.copytree(FIXTURES / "repo", root, ignore=shutil.ignore_patterns("__pycache__"))
        for edit in task.get("setup", []):
            path = root / edit["path"]
            text = path.read_text(encoding="utf-8")
            if text.count(edit["old"]) != 1:
                raise ValueError(f"setup text must occur once in {edit['path']}")
            path.write_text(text.replace(edit["old"], edit["new"]), encoding="utf-8")
        yield root


def tests_pass(root: Path, hidden: str | None = None) -> bool:
    fixture = FIXTURES / "repo"
    for path in files(fixture, fixture / "tests"):
        copy = root / path.relative_to(fixture)
        if not copy.is_file() or copy.read_bytes() != path.read_bytes():
            return False
    if hidden:
        shutil.copy(FIXTURES / "hidden" / hidden, root / "tests" / hidden)
    return test_status(root)[0] == 0


def answer_matches(expected: str, answer: str) -> bool:
    pattern = rf"(?<![\w.-]){re.escape(expected.casefold())}(?![\w-]|\.\d)"
    return re.search(pattern, answer.casefold()) is not None


def passes(task: dict, root: Path, answer: str | None) -> bool:
    check = task["check"]
    if check["kind"] == "answer":
        return answer is not None and answer_matches(check["expected"], answer)
    if check["kind"] == "file":
        path = root / check["path"]
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        kept = all(line in text for line in check.get("keep", ()))
        return kept and re.search(check["pattern"], text) is not None
    return tests_pass(root, check.get("test"))


def common_prefix(first: list[int], second: list[int]) -> int:
    pairs = enumerate(zip(first, second))
    return next((index for index, (a, b) in pairs if a != b), min(len(first), len(second)))


@torch.inference_mode()
def generate_turn(
    model: Decoder, prompt: list[int], state: dict, settings: dict, generator: torch.Generator
) -> tuple[list[int], int]:
    keep = min(common_prefix(state["ids"], prompt), len(prompt) - 1)
    cache = [(k[:, :, :keep], v[:, :, :keep]) for k, v in state["cache"]] if keep else None
    device = model.token_embedding.weight.device
    logits, cache = model.forward_cached(torch.tensor([prompt[keep:]], device=device), cache)
    new = []
    for step in range(MAX_NEW_TOKENS):
        token = sample_token(
            logits[:, -1].float(),
            settings["temperature"],
            settings["top_k"],
            generator,
            settings["top_p"],
        )
        new.append(token.item())
        if new[-1] in settings["stop_ids"] or step + 1 == MAX_NEW_TOKENS:
            break
        logits, cache = model.forward_cached(token, cache)
    state["ids"], state["cache"] = prompt + new[:-1], cache
    return new, keep


def respond(root: Path, call: dict | str, record: dict, reserved: list[str]) -> str:
    record["calls"] += 1
    if isinstance(call, str):
        record["call_errors"].append(call)
        return call
    record["valid_calls"] += 1
    result, ok = execute(root, call)
    if any(token in result for token in reserved):
        result, ok = "error: the result contains a reserved chat token string", False
    record["tool_errors"] += not ok
    return result


def run_task(
    model: Decoder, tokenizer: QwenTokenizer, settings: dict, task: dict, root: Path, seed: int
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    device = model.token_embedding.weight.device
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    state = {"ids": [], "cache": []}
    reserved = [token.content for token in tokenizer.native.get_added_tokens_decoder().values()]
    record = {
        "stop": "turns",
        "turns": 0,
        "calls": 0,
        "valid_calls": 0,
        "tool_errors": 0,
        "call_errors": [],
        "truncated_turns": 0,
        "generated_tokens": 0,
        "rendered_tokens": 0,
        "prefilled_tokens": 0,
        "seconds": 0.0,
        "answer": None,
    }
    for _ in range(MAX_TURNS):
        text = tokenizer.chat(messages, tools=TOOLS, thinking=settings["thinking"])
        prompt = tokenizer.encode(text, allow_special=True)
        if len(prompt) + MAX_NEW_TOKENS > CONTEXT_LIMIT:
            record["stop"] = "context"
            break
        (new, keep), seconds = timed(
            lambda: generate_turn(model, prompt, state, settings, generator), device
        )
        ended = new[-1] in settings["stop_ids"]
        record["turns"] += 1
        record["truncated_turns"] += not ended
        record["generated_tokens"] += len(new)
        record["rendered_tokens"] += len(prompt)
        record["prefilled_tokens"] += len(prompt) - keep
        record["seconds"] += seconds
        reply = tokenizer.decode(new[:-1] if ended else new)
        messages.append({"role": "assistant", "content": reply})
        calls = parse_calls(reply)
        if not calls:
            record["stop"], record["answer"] = "answer", reply
            break
        messages.extend(
            {"role": "tool", "content": respond(root, c, record, reserved)} for c in calls
        )
    return {**record, "messages": messages}


def pass_hat(successes: list[int], trials: int, k: int) -> float:
    return statistics.mean(math.comb(c, k) / math.comb(trials, k) for c in successes)


def summarize(rows: list[dict], seeds: list[int]) -> dict:
    per_seed = [statistics.mean(r["success"] for r in rows if r["seed"] == s) for s in seeds]
    per_task: dict[str, int] = {}
    per_kind: dict[str, list[bool]] = {}
    for row in rows:
        per_task[row["task"]] = per_task.get(row["task"], 0) + row["success"]
        per_kind.setdefault(row["kind"], []).append(row["success"])
    calls = sum(row["calls"] for row in rows)
    return {
        "type": "day7_summary",
        "runs": len(rows),
        "valid_call_rate": sum(row["valid_calls"] for row in rows) / calls if calls else None,
        "calls": calls,
        "pass_1_per_seed": per_seed,
        "pass_1_mean": statistics.mean(per_seed),
        "pass_1_stdev": statistics.stdev(per_seed) if len(seeds) > 1 else 0.0,
        f"pass_hat_{len(seeds)}": pass_hat(list(per_task.values()), len(seeds), len(seeds)),
        "success_by_kind": {kind: statistics.mean(v) for kind, v in sorted(per_kind.items())},
        "mean_turns": statistics.mean(row["turns"] for row in rows),
        "stops": {s: sum(row["stop"] == s for row in rows) for s in ("answer", "turns", "context")},
        "generated_tokens_per_second": sum(row["generated_tokens"] for row in rows)
        / sum(row["seconds"] for row in rows),
        "prefill_fraction": sum(row["prefilled_tokens"] for row in rows)
        / sum(row["rendered_tokens"] for row in rows),
    }


def evaluate(args: argparse.Namespace) -> None:
    tasks = load_tasks(args.tasks)
    device = resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, manifest = load_qwen(args.model, str(device), dtype)
    tokenizer, settings = QwenTokenizer(args.model), sampling(args.model, thinking=False)
    _write(
        args.out,
        {
            "type": "day7_environment",
            "manifest": manifest,
            "parameters": parameter_count(model),
            "pytorch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": str(dtype),
            "sampling": {k: v for k, v in settings.items() if k != "stop_ids"},
            "stop_ids": sorted(settings["stop_ids"]),
            "seeds": args.seeds,
            "limits": [MAX_TURNS, MAX_NEW_TOKENS, CONTEXT_LIMIT, TEST_TIMEOUT, OUTPUT_LIMIT],
            "tasks_hash": file_hash(args.tasks),
            "source_hashes": {
                name: file_hash(Path(__file__).with_name(name))
                for name in ("harness.py", "day6.py", "model.py")
            },
        },
    )
    rows = []
    for task in tasks:
        for seed in args.seeds:
            with sandbox(task) as root:
                record = run_task(model, tokenizer, settings, task, root, seed)
                success = passes(task, root, record["answer"])
            row = {"type": "day7_run", "task": task["id"], "kind": task["check"]["kind"]}
            rows.append({**row, "seed": seed, "success": success, **record})
            _write(args.out, rows[-1])
    _write(args.out, summarize(rows, args.seeds))


def replay(task: dict, root: Path) -> list[str]:
    results = []
    for call in task["solution"]["calls"]:
        parsed = parse_calls(f"<tool_call>\n{json.dumps(call)}\n</tool_call>")
        if len(parsed) != 1 or isinstance(parsed[0], str):
            raise ValueError(f"reference call is invalid in task {task['id']}: {parsed}")
        result, ok = execute(root, parsed[0])
        if not ok:
            raise ValueError(f"reference call failed in task {task['id']}: {result}")
        results.append(result)
    return results


def validate(args: argparse.Namespace) -> None:
    tasks = load_tasks(args.tasks)
    for task in tasks:
        with sandbox(task) as root:
            before = passes(task, root, "")
        with sandbox(task) as root:
            results = replay(task, root)
            after = passes(task, root, task["solution"]["answer"])
        check = task["check"]
        found = (
            check["kind"] != "answer"
            or check.get("derived")
            or any(check["expected"].casefold() in result.casefold() for result in results)
        )
        row = {"type": "task_validation", "task": task["id"], "kind": check["kind"]}
        _write(args.out, {**row, "fails_before": not before, "passes_after": after, "found": found})
        if before or not after or not found:
            raise RuntimeError(f"task {task['id']} failed validation")
    _write(args.out, {"type": "validation_summary", "tasks": len(tasks), "valid": True})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 7 tool harness and its eval.")
    parser.add_argument("stage", choices=("validate", "eval"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/day6/qwen"))
    parser.add_argument("--tasks", type=Path, default=FIXTURES / "tasks.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be distinct and nonnegative")
    args.out = args.out or Path(f"runs/day7-{args.stage}.jsonl")
    if args.out.exists():
        parser.error(f"{args.out} already contains a measurement")
    {"validate": validate, "eval": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
