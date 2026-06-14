"""
code_executor.py
----------------
Standalone module for safely running generated Python code and unit tests
inside a temporary subprocess, with an autonomous LLM-driven repair loop that
retries on test failures.

Public API
----------
run_code_safely(code, test_code, timeout_seconds)          -> ExecutionResult
build_executable_code(generated_code, helper_functions)    -> str
extract_code_block(text)                                   -> str
repair_and_run(...)                                        -> RepairResult

Warning
-------
run_code_safely() uses a subprocess with a timeout but provides NO network
isolation or resource-limit sandbox.  For production use, wrap the subprocess
inside Docker / firejail with --no-net and memory limits.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Structured result returned by run_code_safely()."""
    success: bool
    returncode: int | None
    stdout: str
    stderr: str

    def summary(self) -> str:
        status = "PASSED" if self.success else "FAILED"
        rc = self.returncode if self.returncode is not None else "timeout"
        return f"[{status}] returncode={rc}"


@dataclass
class RepairAttempt:
    """Record of a single generate -> run cycle inside the repair loop."""
    attempt_number: int
    code: str
    test_code: str
    result: ExecutionResult


@dataclass
class RepairResult:
    """
    Returned by repair_and_run().

    Attributes
    ----------
    success       : True if any attempt produced a passing test run.
    final_code    : The code from the last attempt (passing or best-effort).
    final_test_code: The test code used in the last attempt.
    attempts      : Full history of every generate -> run cycle.
    total_attempts: Convenience count equal to len(attempts).
    """
    success: bool
    final_code: str
    final_test_code: str
    attempts: list = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

    def summary(self) -> str:
        status = "PASSED" if self.success else "FAILED"
        last = self.attempts[-1].result.summary() if self.attempts else "none"
        return f"[{status}] {self.total_attempts} attempt(s). Last: {last}"


# ── Repair prompt ─────────────────────────────────────────────────────────────

_REPAIR_PROMPT = """You are an expert Python debugging assistant for a TSN (Time-Sensitive Networking) codebase.

The code below was generated to satisfy the following user request, but its unit tests failed.
Your job is to fix ALL bugs so that every test passes.

USER REQUEST
{query}

ATTEMPT {attempt_number} / {max_attempts}

FAILING CODE
{failing_code}

FAILING TESTS
{test_code}

PYTEST OUTPUT
stdout:
{stdout}

stderr:
{stderr}

PREVIOUS REPAIR ATTEMPTS (for context)
{previous_attempts}

INSTRUCTIONS
1. Carefully read every failed assertion and traceback above.
2. Identify the exact root causes.
3. Rewrite ONLY the implementation code (not the tests) to fix all failures.
4. Preserve the original function signatures exactly.
5. Do not remove or weaken any test cases.

Return ONLY a JSON object with this exact schema (no markdown fences, no preamble):
{{"diagnosis": "one paragraph explaining what was wrong and why", "fixed_code": "complete corrected Python source as a single string"}}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> str:
    """
    Strips a markdown fenced code block and returns the raw Python source.
    Returns the text unchanged if no fences are present.
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
    helper_functions: list,
) -> str:
    """
    Prepends helper function source code to the generated code so the combined
    script can be executed as a single self-contained file.
    """
    helper_code = "\n\n".join(
        fn.get("code", "")
        for fn in helper_functions
        if fn.get("code")
    )
    return helper_code + "\n\n" + extract_code_block(generated_code)


def _format_previous_attempts(attempts: list) -> str:
    if not attempts:
        return "None - this is the first attempt."
    lines = []
    for a in attempts:
        lines.append(f"Attempt {a.attempt_number}: {a.result.summary()}")
        if a.result.stdout:
            lines.append(f"  stdout: {a.result.stdout[:600]}")
        if a.result.stderr:
            lines.append(f"  stderr: {a.result.stderr[:600]}")
    return "\n".join(lines)


def _call_llm_for_repair(
    llm_complete: Callable,
    prompt: str,
) -> Optional[str]:
    """
    Calls the LLM and extracts fixed_code from its JSON response.
    Returns None if the response cannot be parsed.
    """
    raw = llm_complete(prompt).strip()

    # Strip optional markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            print(f"[Repair] LLM response could not be parsed.\n{raw[:400]}")
            return None
        try:
            parsed = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            print(f"[Repair] JSON extraction also failed.\n{raw[:400]}")
            return None

    diagnosis = parsed.get("diagnosis", "")
    fixed_code = parsed.get("fixed_code", "")

    if diagnosis:
        print(f"[Repair] Diagnosis: {diagnosis}")

    return extract_code_block(fixed_code) if fixed_code else None


# ── Core executor ─────────────────────────────────────────────────────────────

def run_code_safely(
    code: str,
    test_code: str = "",
    timeout_seconds: int = 5,
) -> ExecutionResult:
    """
    Writes code (and optionally test_code) to a temp file and executes it via
    pytest in a child process.  The temp directory is deleted automatically.

    Parameters
    ----------
    code             : Python source to execute. Markdown fences are stripped.
    test_code        : Optional pytest-compatible source appended after code.
    timeout_seconds  : Hard wall-clock limit. Defaults to 5 s.

    Returns
    -------
    ExecutionResult — success is True only when pytest exits with returncode 0.
    On timeout returncode is None.
    """
    cleaned_code = extract_code_block(code)

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        full_code = cleaned_code
        if test_code:
            full_code += "\n\n" + extract_code_block(test_code)
        script_path.write_text(full_code, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", str(script_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
                env=os.environ.copy(),
            )
            return ExecutionResult(
                success=completed.returncode == 0,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                returncode=None,
                stdout="",
                stderr=f"Execution timed out after {timeout_seconds} seconds.",
            )
        except Exception:
            return ExecutionResult(
                success=False,
                returncode=None,
                stdout="",
                stderr=traceback.format_exc(),
            )


# ── Repair loop ───────────────────────────────────────────────────────────────

def repair_and_run(
    query: str,
    initial_code: str,
    test_code: str,
    llm_complete: Callable,
    max_attempts: int = 3,
    timeout_seconds: int = 5,
) -> RepairResult:
    """
    Runs initial_code against test_code and, on failure, asks the LLM to
    repair the code.  Retries up to max_attempts times in total (the initial
    run counts as attempt 1).

    Parameters
    ----------
    query           : Original user request injected into the repair prompt.
    initial_code    : First generated code to try. Markdown fences are stripped.
    test_code       : The pytest tests to run on every candidate. Never modified.
    llm_complete    : Callable(prompt: str) -> str.  Pass the LLM like this:
                          lambda p: Settings.llm.complete(p).text
    max_attempts    : Total generate->run cycles allowed. Defaults to 3.
    timeout_seconds : Per-attempt subprocess timeout.

    Returns
    -------
    RepairResult with success, final_code, and the full attempt history.

    Usage in file1.py (inside answer_query_deterministic)
    -----------------------------------------------------
        from code_executor import repair_and_run, RepairResult

        repair_result = repair_and_run(
            query=query,
            initial_code=code_for_tests,
            test_code="\n\n".join(generated_unit_tests),
            llm_complete=lambda p: Settings.llm.complete(p).text,
            max_attempts=3,
        )

        execution_result = {
            "success":     repair_result.success,
            "returncode":  repair_result.attempts[-1].result.returncode,
            "stdout":      repair_result.attempts[-1].result.stdout,
            "stderr":      repair_result.attempts[-1].result.stderr,
            "attempts":    repair_result.total_attempts,
        }
        # Use repair_result.final_code as the definitive code to display/save.
    """
    current_code = extract_code_block(initial_code)
    current_tests = extract_code_block(test_code)
    attempts: list = []

    for attempt_num in range(1, max_attempts + 1):
        print(f"\n[Repair loop] Attempt {attempt_num}/{max_attempts} ...")

        result = run_code_safely(
            code=current_code,
            test_code=current_tests,
            timeout_seconds=timeout_seconds,
        )

        attempts.append(RepairAttempt(
            attempt_number=attempt_num,
            code=current_code,
            test_code=current_tests,
            result=result,
        ))

        print(f"[Repair loop] {result.summary()}")

        if result.success:
            print(f"[Repair loop] All tests passed on attempt {attempt_num}.")
            return RepairResult(
                success=True,
                final_code=current_code,
                final_test_code=current_tests,
                attempts=attempts,
            )

        if attempt_num == max_attempts:
            print(f"[Repair loop] Max attempts ({max_attempts}) reached. Giving up.")
            break

        print(f"[Repair loop] Tests failed. Sending to LLM for repair ...")

        prompt = _REPAIR_PROMPT.format(
            query=query,
            attempt_number=attempt_num,
            max_attempts=max_attempts,
            failing_code=current_code,
            test_code=current_tests,
            stdout=result.stdout[:2000],
            stderr=result.stderr[:2000],
            previous_attempts=_format_previous_attempts(attempts[:-1]),
        )

        repaired_code = _call_llm_for_repair(llm_complete, prompt)

        if repaired_code:
            current_code = repaired_code
        else:
            print("[Repair loop] LLM repair parse failed; retrying with unchanged code.")

    return RepairResult(
        success=False,
        final_code=current_code,
        final_test_code=current_tests,
        attempts=attempts,
    )