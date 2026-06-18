from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParsedTrainingCommand:
    raw_text: str
    out_dir: str = ""
    train_py: str = ""
    python_executable: str = "python"
    python_flags: List[str] = None
    env_vars: Dict[str, str] = None
    taskset_cpu: Optional[str] = None
    args: Dict[str, Any] = None
    ordered_tokens: List[str] = None
    log_path: str = ""

    def __post_init__(self) -> None:
        self.python_flags = self.python_flags or []
        self.env_vars = self.env_vars or {}
        self.args = self.args or {}
        self.ordered_tokens = self.ordered_tokens or []


_NUMERIC_RE = re.compile(r"^-?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?$")


def _coerce_value(value: str) -> Any:
    value = str(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    if _NUMERIC_RE.match(value):
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        except Exception:
            return value
    return value


def _shell_var(text: str, name: str) -> str:
    # Handles OUT_DIR="clean/x" and OUT_DIR='clean/x'.
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(['\"]?)(.*?)\1\s*$"
    for line in text.splitlines():
        m = re.match(pattern, line.strip())
        if m:
            return m.group(2)
    return ""


def _extract_inner_command(text: str) -> str:
    """Extract the command inside `bash -c '...'` when present."""
    marker = "bash -c"
    if marker not in text:
        return text

    idx = text.find(marker) + len(marker)
    tail = text[idx:].lstrip()
    if not tail:
        return text

    quote = tail[0]
    if quote not in {"'", '"'}:
        return text

    chars: List[str] = []
    escaped = False
    for ch in tail[1:]:
        if escaped:
            chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            chars.append(ch)
            continue
        if ch == quote:
            break
        chars.append(ch)
    return "".join(chars)


def _normalize_command(text: str) -> str:
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")
    text = text.replace("\r\n", "\n")
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _strip_wrappers(tokens: List[str]) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    env_vars: Dict[str, str] = {}
    taskset_cpu: Optional[str] = None
    cleaned = list(tokens)

    # Drop leading nohup if user passes a full command.
    if cleaned and cleaned[0] == "nohup":
        cleaned = cleaned[1:]

    # Collect ENV=VALUE tokens before python/taskset.
    while cleaned and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", cleaned[0]):
        key, val = cleaned.pop(0).split("=", 1)
        env_vars[key] = val

    # Optional taskset -c CPU.
    if len(cleaned) >= 3 and cleaned[0] == "taskset" and cleaned[1] == "-c":
        taskset_cpu = cleaned[2]
        cleaned = cleaned[3:]

    return cleaned, env_vars, taskset_cpu


def parse_arg_tokens(tokens: List[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args[key] = _coerce_value(tokens[i + 1])
            i += 2
        else:
            args[key] = True
            i += 1
    return args


def parse_training_shell(text: str) -> ParsedTrainingCommand:
    out_dir = _shell_var(text, "OUT_DIR")
    inner = _extract_inner_command(text)
    normalized = _normalize_command(inner)

    try:
        tokens = shlex.split(normalized)
    except ValueError:
        tokens = normalized.split()

    tokens, env_vars, taskset_cpu = _strip_wrappers(tokens)

    python_exe = "python"
    python_flags: List[str] = []
    train_py = ""
    remaining: List[str] = []

    if tokens:
        python_exe = tokens.pop(0)

    while tokens and tokens[0].startswith("-") and not tokens[0].startswith("--"):
        python_flags.append(tokens.pop(0))

    if tokens:
        train_py = tokens.pop(0)
        remaining = tokens

    args = parse_arg_tokens(remaining)
    if not out_dir:
        out_dir = str(args.get("out_dir", ""))

    log_path = str(Path(out_dir) / "train.log") if out_dir else ""

    return ParsedTrainingCommand(
        raw_text=text,
        out_dir=out_dir,
        train_py=train_py,
        python_executable=python_exe,
        python_flags=python_flags,
        env_vars=env_vars,
        taskset_cpu=taskset_cpu,
        args=args,
        ordered_tokens=remaining,
        log_path=log_path,
    )


def parsed_to_dataframe_rows(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, value in args.items():
        rows.append({"parameter": key, "value": value, "type": type(value).__name__})
    return rows


def load_run_args(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_train_command(
    parsed: ParsedTrainingCommand,
    args: Dict[str, Any],
    use_taskset: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    command: List[str] = []
    env = dict(parsed.env_vars or {})

    if use_taskset and parsed.taskset_cpu:
        command.extend(["taskset", "-c", str(parsed.taskset_cpu)])

    command.append(parsed.python_executable or "python")
    command.extend(parsed.python_flags or ["-u"])

    if not parsed.train_py:
        raise ValueError("Could not find train.py path in the shell command.")
    command.append(parsed.train_py)

    for key, value in args.items():
        cli_key = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(cli_key)
        elif value is None or str(value) == "":
            continue
        else:
            command.extend([cli_key, str(value)])

    return command, env
