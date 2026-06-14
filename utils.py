"""
utils.py
--------
Pure utility functions shared between file1.py and feedback_loop.py.
No imports from either of those modules — safe to import from both
without causing a circular dependency.
"""

import os
import ast
import sys
import tempfile
import subprocess
import traceback
from pathlib import Path
from typing import Optional

# ── resource is POSIX-only — guard so utils.py imports cleanly on Windows ─────
if os.name != "nt":
    import resource
else:  # Windows
    resource = None


def _posix_resource_limits(cpu_seconds: int, memory_mb: Optional[int]):
    """
    preexec_fn that caps CPU time (and optionally address space) for the child.
    POSIX ONLY — never pass this on Windows.

    memory_mb is OFF by default: RLIMIT_AS caps *virtual* memory, and numpy /
    torch / networkx reserve large virtual ranges, so a tight cap can break
    legitimate imports. Enable it only if you know the footprint.
    """
    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if memory_mb:
            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return _apply


def extract_code_block(text: str) -> str:
    """
    Extracts Python code from a markdown code block if present.
    Otherwise returns the raw text.
    """
    text = str(text).strip()

    if "```" not in text:
        return text

    parts = text.split("```")

    for part in parts:
        stripped = part.strip()
        if stripped.lower().startswith("python"):
            return stripped[len("python"):].strip()

    if len(parts) >= 2:
        return parts[1].strip()

    return text


def build_executable_code(
    generated_code: str,
    helper_functions: list[dict],
) -> str:
    """
    Prepends helper function code to the generated code so the combined
    script is self-contained and executable.

    Class methods are wrapped in their parent class so indentation is valid
    and pytest can collect the file without an IndentationError.
    """
    helper_parts = []
    for fn in helper_functions:
        code = fn.get("code", "")
        if not code:
            continue
        class_name = fn.get("class_name")
        if class_name:
            # Method body stored without the class header — re-wrap it.
            indented = "\n".join("    " + line for line in code.splitlines())
            helper_parts.append(f"class {class_name}:\n{indented}")
        else:
            helper_parts.append(code)

    helper_code = "\n\n".join(helper_parts)
    return helper_code + "\n\n" + extract_code_block(generated_code)


def run_code_safely(
    code: str,
    test_code: str = "",
    timeout_seconds: int = 5,
    memory_mb: Optional[int] = None,
) -> dict:
    """
    Run code (optionally with appended pytest tests) in a temporary subprocess.

      * POSIX   -> own session + CPU-time cap (and optional memory cap).
      * Windows -> wall-clock timeout only (resource limits unavailable),
                   CREATE_NEW_PROCESS_GROUP for clean termination.

    If test_code is given, runs pytest; otherwise runs the script directly so a
    plain call-site can be executed and its stdout captured.

    Returns: {"success": bool, "returncode": int|None, "stdout": str, "stderr": str}
    Minimal safeguard, not a sandbox — use a container/jail for untrusted code.
    """
    cleaned = extract_code_block(code)

    # Syntax-check before writing to disk so errors are clear immediately.
    try:
        ast.parse(cleaned)
    except SyntaxError as e:
        return {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": f"SyntaxError in code before execution: {e}",
        }

    full_code = cleaned
    if test_code:
        full_code += "\n\n" + extract_code_block(test_code)

    use_pytest = bool(test_code.strip())

    # Generated test snippets routinely call pytest.raises(...) without importing
    # pytest (they're written as bare functions). Guarantee it's importable when
    # we actually run under pytest.
    if use_pytest and "import pytest" not in full_code:
        full_code = "import pytest\n" + full_code

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(full_code, encoding="utf-8")

        cmd = (
            [sys.executable, "-m", "pytest", str(script_path), "-v", "--tb=short"]
            if use_pytest else
            [sys.executable, str(script_path)]
        )

        run_kwargs = dict(
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=tmpdir,
            env=os.environ.copy(),
        )

        if os.name != "nt":
            run_kwargs["start_new_session"] = True
            run_kwargs["preexec_fn"] = _posix_resource_limits(
                cpu_seconds=timeout_seconds + 1, memory_mb=memory_mb
            )
        else:
            run_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            completed = subprocess.run(cmd, **run_kwargs)
            return {
                "success": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": None,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds.",
            }
        except Exception:
            return {
                "success": False,
                "returncode": None,
                "stdout": "",
                "stderr": traceback.format_exc(),
            }