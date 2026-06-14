"""
================
Self-correcting execution loop for the TSN coding agent.

Responsibilities
----------------
1. Run generated code + unit tests via run_code_safely().
2. Parse the execution result to decide: PASS / FIXABLE_FAILURE / GIVE_UP.
3. On failure, ask the LLM to repair the code, then re-run.
4. Return a FeedbackResult that summarises every attempt for the caller.

Design decisions
----------------
- Fully decoupled from file1.py: only imports helpers that are pure functions
  (run_code_safely, extract_code_block, build_executable_code) and the LLM
  singleton (Settings.llm).  No retriever or index dependency.
- The LLM repair prompt receives the *original* user query, the failing code,
  all previously seen errors, and any helper function code so the model has
  full context when it rewrites.
- A hard cap (MAX_REPAIR_ATTEMPTS) prevents infinite loops; the last error
  is always preserved for the caller to surface.
- Each attempt is logged to a FeedbackAttempt dataclass so the outer pipeline
  can display a clear audit trail (attempt N → error → fixed? → …).
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ── Re-use helpers and the LLM from the main agent file ──────────────────────
# We import only the pure, side-effect-free helpers so this module remains
# independently testable.  The LLM is accessed through the shared Settings
# object that file1.py already initialised.
from llama_index.core import Settings  # shared LLM singleton

# These helpers live in file1.py.  A clean alternative would be to move them
# into a shared utils.py, but the import below keeps the change minimal.
from utils import (
    run_code_safely,
    extract_code_block,
    build_executable_code,
)

# Spec-first triage: decide whether a failure is the implementation's fault or
# the test's fault, so the loop repairs the code instead of weakening the tests.
from spec_test_engine import triage_failure, should_repair_implementation

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_REPAIR_ATTEMPTS: int = 1   # how many LLM repair rounds before giving up
TIMEOUT_SECONDS:    int = 10   # per-attempt subprocess timeout

# ── Data model ────────────────────────────────────────────────────────────────

class FeedbackStatus(str, Enum):
    PASS             = "PASS"            # all tests green on some attempt
    REPAIRED         = "REPAIRED"        # failed at first but passed after repair
    GIVE_UP          = "GIVE_UP"         # exhausted attempts, still failing
    NO_TESTS         = "NO_TESTS"        # no test code supplied — nothing to verify
    EXECUTION_ERROR  = "EXECUTION_ERROR" # hard crash before any test ran
    NEEDS_REVIEW     = "NEEDS_REVIEW"    # triage blamed the test/spec, not the code —
                                         # surfaced for a human instead of weakening tests


@dataclass
class FeedbackAttempt:
    """One round of execute → observe."""
    attempt_number: int
    code_snapshot:  str          # the exact code that was executed
    test_snapshot:  str          # the exact test code that was executed
    success:        bool
    returncode:     Optional[int]
    stdout:         str
    stderr:         str
    error_summary:  str          # human-readable one-liner extracted from output


@dataclass
class FeedbackResult:
    """
    Returned by run_feedback_loop().  Contains the final (possibly repaired)
    code, the verdict, and the full attempt history for audit / display.
    """
    status:          FeedbackStatus
    final_code:      str                        # best code produced
    final_tests:     str                        # tests that were last run
    attempts:        List[FeedbackAttempt] = field(default_factory=list)
    total_attempts:  int = 0
    repair_count:    int = 0                    # how many LLM repair rounds ran

    # Convenience helpers -------------------------------------------------------

    def passed(self) -> bool:
        return self.status in {FeedbackStatus.PASS, FeedbackStatus.REPAIRED}

    def summary(self) -> str:
        lines = [
            f"Status        : {self.status.value}",
            f"Total attempts: {self.total_attempts}",
            f"Repair rounds : {self.repair_count}",
        ]
        for a in self.attempts:
            tick = "✓" if a.success else "✗"
            lines.append(
                f"  Attempt {a.attempt_number} [{tick}]  {a.error_summary or 'OK'}"
            )
        return "\n".join(lines)


# ── Error parsing ─────────────────────────────────────────────────────────────

def _extract_error_summary(stdout: str, stderr: str) -> str:
    """
    Pulls the most informative single line out of pytest / Python output.
    Priority: FAILED lines > ERROR lines > last non-empty stderr line.
    """
    combined = stdout + "\n" + stderr

    # pytest FAILED lines  →  "FAILED test_..py::test_foo - AssertionError: …"
    for line in combined.splitlines():
        if line.strip().startswith("FAILED"):
            return line.strip()[:200]

    # Standard Python exception  →  last "Error:" line
    for line in reversed(combined.splitlines()):
        stripped = line.strip()
        if "Error" in stripped and stripped:
            return stripped[:200]

    # Fallback: last non-empty stderr line
    for line in reversed(stderr.splitlines()):
        if line.strip():
            return line.strip()[:200]

    return "unknown error"


def _has_import_error(stderr: str) -> bool:
    """Returns True if the failure is purely a missing module — not worth retrying."""
    return bool(re.search(r"ModuleNotFoundError|ImportError|No module named", stderr))


# ── LLM repair prompt ─────────────────────────────────────────────────────────

REPAIR_PROMPT_TEMPLATE = """\
You are an expert Python debugger for a TSN (Time-Sensitive Networking) codebase.

The code below was generated to satisfy the user's request but failed its unit tests.
Your job is to fix the code so that all tests pass.

STRICT RULES:
- Return ONLY the corrected Python code, nothing else.
- Do NOT include markdown fences (```python ... ```).
- Do NOT change function signatures or names.
- Do NOT remove, skip, or weaken any test; fix the implementation instead.
- The UNIT TESTS were written from the BEHAVIORAL CONTRACT below BEFORE this
  code existed. Treat the contract + tests as the source of truth. Make the
  implementation satisfy them; never edit them.
- If a helper function stub is needed for the tests, inline it before the main function.
- All necessary imports must be at the top of the returned code.

User request:
{query}

=== BEHAVIORAL CONTRACT (source of truth — the code must satisfy this) ===
{contract}

=== RETRIEVED CODE (from codebase — read-only, do not modify) ===
{retrieved_code}

=== HELPER FUNCTIONS (read-only — do not modify) ===
{helpers_full_code}

=== UNIT TESTS (read-only — do not modify) ===
{test_code}

=== PREVIOUS ERRORS (most recent last) ===
{errors_history}

=== FAILING CODE ===
{failing_code}

Corrected code:
"""


def _build_repair_prompt(
    query: str,
    failing_code: str,
    test_code: str,
    errors_history: List[str],
    helper_functions: List[dict],
    retrieved_code: Optional[str] = None,
    contract: Optional[dict] = None,
) -> str:
    # ── Behavioral contract (the spec the code must meet) ─────────────────────
    contract_section = (
        json.dumps(contract, indent=2)
        if contract and contract.get("behaviors")
        else "# no formal contract available"
    )
    # ── Retrieved code (verbatim source from the index) ───────────────────────
    retrieved_section = retrieved_code.strip() if retrieved_code and retrieved_code.strip() else "# none"

    # ── Helper functions: full source where available ─────────────────────────
    helper_blocks = []
    for fn in helper_functions:
        name = fn.get("name", "")
        code = fn.get("code", "").strip()
        if not name:
            continue
        if code:
            helper_blocks.append(f"# --- helper: {name} ---\n{code}")
        else:
            # Fallback to signature + purpose when full source is absent.
            params = ", ".join(
                p.get("name", "") for p in fn.get("parameters", [])
                if isinstance(p, dict)
            )
            purpose = fn.get("purpose", fn.get("description", ""))
            helper_blocks.append(f"# --- helper: {name} (source unavailable) ---\n"
                                  f"# {purpose}\n"
                                  f"def {name}({params}): ...")

    helpers_full_code = "\n\n".join(helper_blocks) or "# no helpers"

    errors_text = "\n---\n".join(
        f"Attempt {i + 1}:\n{err}" for i, err in enumerate(errors_history)
    )

    return REPAIR_PROMPT_TEMPLATE.format(
        query=query,
        contract=contract_section,
        retrieved_code=retrieved_section,
        helpers_full_code=helpers_full_code,
        test_code=test_code or "# no tests",
        errors_history=errors_text,
        failing_code=failing_code,
    )


# ── Core feedback loop ────────────────────────────────────────────────────────

def run_feedback_loop(
    query:            str,
    generated_code:   str,
    unit_tests:       List[str],
    helper_functions: Optional[List[dict]] = None,
    retrieved_code:   Optional[str] = None,
    contract:         Optional[dict] = None,
    max_attempts:     int = MAX_REPAIR_ATTEMPTS,
    timeout_seconds:  int = TIMEOUT_SECONDS,
    verbose:          bool = True,
) -> FeedbackResult:
    """
    Execute generated_code + unit_tests, repair on failure, repeat.

    Parameters
    ----------
    query            : original user request (used in repair prompt for context)
    generated_code   : raw generated code string (may contain markdown fences)
    unit_tests       : list of pytest-style test function source strings
    helper_functions : helper function dicts (used to build executable code +
                       give the repair LLM context)
    retrieved_code   : raw source retrieved from the codebase index — sent
                       verbatim to the LLM so it can see all relevant context
    max_attempts     : maximum number of LLM repair rounds (default 3)
    timeout_seconds  : subprocess timeout per attempt
    verbose          : print progress to stdout

    Returns
    -------
    FeedbackResult
    """
    helpers = helper_functions or []
    test_code_str = "\n\n".join(unit_tests)

    # ── Guard: nothing to test ────────────────────────────────────────────────
    if not unit_tests or not test_code_str.strip():
        if verbose:
            print("[FeedbackLoop] No unit tests provided — skipping execution.")
        return FeedbackResult(
            status=FeedbackStatus.NO_TESTS,
            final_code=generated_code,
            final_tests="",
            total_attempts=0,
        )

    # ── Attempt loop ──────────────────────────────────────────────────────────
    current_code  = extract_code_block(generated_code)
    errors_history: List[str] = []
    attempts:       List[FeedbackAttempt] = []
    repair_count  = 0

    for attempt_num in range(1, max_attempts + 2):  # +2: initial + max_attempts repairs
        # Build the full executable blob (helpers prepended)
        executable = build_executable_code(
            generated_code=current_code,
            helper_functions=helpers,
        )

        if verbose:
            print(f"\n[FeedbackLoop] Attempt {attempt_num} ...")

        result = run_code_safely(
            code=executable,
            test_code=test_code_str,
            timeout_seconds=timeout_seconds,
        )

        error_summary = _extract_error_summary(
            result.get("stdout", ""),
            result.get("stderr", ""),
        ) if not result["success"] else ""

        attempt = FeedbackAttempt(
            attempt_number=attempt_num,
            code_snapshot=current_code,
            test_snapshot=test_code_str,
            success=result["success"],
            returncode=result.get("returncode"),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            error_summary=error_summary,
        )
        attempts.append(attempt)

        if verbose:
            tick = " PASS" if result["success"] else f" FAIL  {error_summary}"
            print(f"[FeedbackLoop]   -> {tick}")

        # ── SUCCESS ──────────────────────────────────────────────────────────
        if result["success"]:
            status = FeedbackStatus.PASS if repair_count == 0 else FeedbackStatus.REPAIRED
            return FeedbackResult(
                status=status,
                final_code=current_code,
                final_tests=test_code_str,
                attempts=attempts,
                total_attempts=attempt_num,
                repair_count=repair_count,
            )

        # ── FAILURE ───────────────────────────────────────────────────────────
        full_error = (result.get("stdout", "") + "\n" + result.get("stderr", "")).strip()
        errors_history.append(full_error)

        # Hard stop: import errors are environment problems, not code bugs.
        if _has_import_error(result.get("stderr", "")):
            if verbose:
                print(
                    "[FeedbackLoop]   -> Import/module error detected — "
                    "stopping (environment issue, not a code bug)."
                )
            return FeedbackResult(
                status=FeedbackStatus.EXECUTION_ERROR,
                final_code=current_code,
                final_tests=test_code_str,
                attempts=attempts,
                total_attempts=attempt_num,
                repair_count=repair_count,
            )

        # Hard stop: no more repair budget.
        if attempt_num > max_attempts:
            if verbose:
                print(
                    f"[FeedbackLoop]   -> Exhausted {max_attempts} repair attempt(s). Giving up."
                )
            return FeedbackResult(
                status=FeedbackStatus.GIVE_UP,
                final_code=current_code,
                final_tests=test_code_str,
                attempts=attempts,
                total_attempts=attempt_num,
                repair_count=repair_count,
            )

        # ── TRIAGE: implementation's fault or test's fault? ───────────────────
        # The tests came from the contract before this code existed, so the
        # default is to blame the implementation. We only stop (and flag for a
        # human) when triage is HIGH-confidence that the test/spec is wrong —
        # we never auto-edit a contract test to force a pass.
        if contract and contract.get("behaviors"):
            triage = triage_failure(
                contract=contract,
                test_code=test_code_str,
                impl_code=current_code,
                failure_output=full_error,
            )
            if verbose:
                print(
                    f"[FeedbackLoop]   -> Triage: {triage.get('verdict')} "
                    f"({triage.get('confidence')}) — {triage.get('explanation', '')[:120]}"
                )
            if not should_repair_implementation(triage):
                if verbose:
                    print(
                        "[FeedbackLoop]   -> Triage blames the test/spec, not the "
                        "code. Not weakening tests — flagging for human review."
                    )
                return FeedbackResult(
                    status=FeedbackStatus.NEEDS_REVIEW,
                    final_code=current_code,
                    final_tests=test_code_str,
                    attempts=attempts,
                    total_attempts=attempt_num,
                    repair_count=repair_count,
                )

        # ── LLM REPAIR ───────────────────────────────────────────────────────
        repair_count += 1
        if verbose:
            print(f"[FeedbackLoop]   -> Requesting LLM repair (round {repair_count}) ...")

        repair_prompt = _build_repair_prompt(
            query=query,
            failing_code=current_code,
            test_code=test_code_str,
            errors_history=errors_history,
            helper_functions=helpers,
            retrieved_code=retrieved_code,
            contract=contract,
        )

        repaired_raw = Settings.llm.complete(repair_prompt).text.strip()
        repaired_code = extract_code_block(repaired_raw)

        if not repaired_code.strip():
            if verbose:
                print("[FeedbackLoop]   -> LLM returned empty repair — giving up.")
            return FeedbackResult(
                status=FeedbackStatus.GIVE_UP,
                final_code=current_code,
                final_tests=test_code_str,
                attempts=attempts,
                total_attempts=attempt_num,
                repair_count=repair_count,
            )

        current_code = repaired_code

    # Should never reach here (loop always returns inside), but satisfies type checker.
    return FeedbackResult(
        status=FeedbackStatus.GIVE_UP,
        final_code=current_code,
        final_tests=test_code_str,
        attempts=attempts,
        total_attempts=max_attempts + 1,
        repair_count=repair_count,
    )


# ── Convenience wrapper for answer_query_deterministic ───────────────────────

def run_feedback_loop_for_subquery(
    query:            str,
    generated_code:   str,
    unit_tests:       List[str],
    helper_functions: Optional[List[dict]] = None,
    retrieved_code:   Optional[str] = None,
    contract:         Optional[dict] = None,
    verbose:          bool = True,
) -> dict:
    """
    Thin wrapper that calls run_feedback_loop() and returns a plain dict
    that is directly compatible with the `execution_result` slot consumed
    by format_subquery_answer() in file1.py.

    The dict has the same shape as the one produced by run_code_safely()
    (keys: success, returncode, stdout, stderr) plus two extra keys:
        feedback_status  : FeedbackStatus value string
        repair_count     : int — how many repair rounds were used
        attempt_log      : list of dicts for display / audit

    Usage in answer_query_deterministic (replace the existing block):
    ----------------------------------------------------------------
        from feedback_loop import run_feedback_loop_for_subquery

        if run_generated_code and generated_unit_tests:
            execution_result = run_feedback_loop_for_subquery(
                query=query,
                generated_code=code_for_tests,
                unit_tests=generated_unit_tests,
                helper_functions=functions,
                retrieved_code=retrieved_source,   # pass the raw retrieved text
            )
    """
    fb = run_feedback_loop(
        query=query,
        generated_code=generated_code,
        unit_tests=unit_tests,
        helper_functions=helper_functions,
        retrieved_code=retrieved_code,
        contract=contract,
        verbose=verbose,
    )

    last = fb.attempts[-1] if fb.attempts else None
    return {
        # ── run_code_safely-compatible keys ──────────────────────────────────
        "success":    fb.passed(),
        "returncode": last.returncode if last else None,
        "stdout":     last.stdout     if last else "",
        "stderr":     last.stderr     if last else "",
        # ── extra feedback keys ───────────────────────────────────────────────
        "feedback_status": fb.status.value,
        "repair_count":    fb.repair_count,
        "final_code":      fb.final_code,     # NEW: the code that actually ran/passed
        "final_tests":     fb.final_tests,    # NEW: optional, the tests last run
        "attempt_log": [
            {
                "attempt":       a.attempt_number,
                "success":       a.success,
                "error_summary": a.error_summary,
            }
            for a in fb.attempts
        ],
    }