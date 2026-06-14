"""
spec_test_engine.py
===================
Specification-first unit-test generation for the TSN coding agent.

WHY THIS EXISTS
---------------
The original `unit_test_generator.py` shows the implementation to the LLM and
asks it to write tests *for that code*. A model that can see the implementation
writes tests that mirror it, so a buggy implementation gets buggy tests that
pass ("fitted to pass"). This module breaks that loop:

    1. Derive a BEHAVIORAL CONTRACT from the *intent only*:
         - for RETRIEVED functions  -> from signature/description/pre/post/raises
         - for GENERATED code        -> from the user query + missing requirements
                                        + extracted parameters
       In neither case is the implementation body shown to the contract builder.

    2. Write tests strictly FROM THE CONTRACT. The test writer is told the
       implementation does not exist yet, so it cannot copy it.

    3. When a test fails, TRIAGE decides whether the implementation or the test
       is at fault. Default assumption: the implementation is wrong (the test
       was written before seeing it). This is what stops the repair loop from
       "passing" by weakening the tests.

DECOUPLING
----------
Like feedback_loop.py, this module imports only the shared LLM singleton and
stays independent of file1.py (which has heavy import-time side effects). The
three prompts that used to sit dead in file1.py are owned here.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from llama_index.core import Settings  # shared LLM singleton


# ── JSON helper (self-contained copy) ─────────────────────────────────────────

def _extract_json_object(text: str) -> dict:
    """Extracts the first JSON object from an LLM response (raw or fenced)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(text[start:end + 1])


# ── Prompts ───────────────────────────────────────────────────────────────────

# Contract from an EXISTING function's declared interface (no body shown).
_SPEC_FROM_FUNCTION_PROMPT = """\
You are a software specification analyst. Produce a behavioral contract
for the function described below. You are NOT given the implementation,
and you must NOT guess at implementation details.

Derive the contract ONLY from the description, signature, preconditions,
postconditions, and declared exceptions.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
  "target_name": "{name}",
  "behaviors": [
    {{
      "id": "B1",
      "description": "observable behavior in plain English",
      "given": "input condition / state",
      "when": "the function is called with ...",
      "then": "the observable, assertable outcome",
      "category": "happy_path | edge_case | error_case | invariant"
    }}
  ],
  "input_domains": {{
    "<param_name>": {{
      "valid": ["representative valid values"],
      "boundary": ["boundary values to test"],
      "invalid": ["values that should trigger errors"]
    }}
  }},
  "invariants": ["properties that must always hold across all inputs"]
}}

Function name: {name}
Signature: {signature}
Description: {description}
Parameters: {parameters}
Preconditions: {preconditions}
Postconditions: {postconditions}
Raises: {raises}

JSON output:
"""

# Contract from the USER REQUIREMENTS, for code that does not exist yet.
# This is the key path for GENERATE_STANDALONE / GENERATE_WITH_HELPERS: the
# contract is anchored to what the user asked for, never to generated code.
_SPEC_FROM_REQUIREMENTS_PROMPT = """\
You are a software specification analyst for a TSN (Time-Sensitive Networking)
system. Code is about to be written to satisfy a user request. Your job is to
write the behavioral contract the code MUST satisfy — BEFORE the code exists.

You must NOT invent functionality the request does not ask for, and you must
NOT describe implementation details (no algorithms, no internal data
structures). Describe only observable, assertable behavior.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
  "target_name": "{target_name}",
  "behaviors": [
    {{
      "id": "B1",
      "description": "observable behavior in plain English",
      "given": "input condition / state",
      "when": "the entry point is called with ...",
      "then": "the observable, assertable outcome",
      "category": "happy_path | edge_case | error_case | invariant"
    }}
  ],
  "input_domains": {{
    "<param_name>": {{
      "valid": ["representative valid values"],
      "boundary": ["boundary values to test"],
      "invalid": ["values that should trigger errors"]
    }}
  }},
  "invariants": ["properties that must always hold across all inputs"]
}}

Intended entry point name: {target_name}
Intended signature (may be approximate): {signature}

User request:
{query}

Sub-task being implemented:
{sub_query}

Requirements this code MUST cover (gaps the existing codebase could not satisfy):
{missing_requirements}

Concrete parameter values mentioned by the user (use as boundary/valid inputs):
{params_json}

JSON output:
"""

# Tests written FROM THE CONTRACT, run in the same file as the code under test.
_SPEC_BASED_TEST_PROMPT = """\
You are a test engineer writing pytest tests from a BEHAVIORAL CONTRACT.

CRITICAL RULES:
- You are given a contract, NOT an implementation. Assume the implementation
  may be wrong. Your tests are the source of truth for correctness.
- Write one test per behavior in the contract.
- Name each test test_<behavior_id>_<slug>, where <slug> is a short snake_case
  phrase you derive from THIS behavior's "then" outcome. Do NOT copy the example
  slug from the schema below — it is a placeholder, not a real name.
- Assert ONLY what the contract states. Do NOT invent behavior.
- Do NOT weaken assertions to make tests pass. Do NOT wrap the whole body in
  try/except that swallows failures. Do NOT use trivially-true assertions
  (assert True, assert x == x).
- For error_case behaviors, use pytest.raises with the declared exception.
- Include a docstring in each test quoting the behavior it verifies.

EXECUTION MODEL (important):
- Your tests will be appended to the SAME source file as the code under test.
- Therefore `{target_name}` and any helper functions are ALREADY DEFINED above
  your tests. Reference them DIRECTLY. Do NOT write `from ... import {target_name}`
  or `import` the module under test — that will fail.
- You MAY import stdlib modules and pytest, and you MAY define minimal inline
  stubs for genuinely external libraries (networkx, numpy) only if a test
  cannot call the entry point without them. NEVER stub the entry point itself.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
  "tests": [
    {{
      "behavior_id": "B1",
      "test_name": "test_B1_<short_slug_from_this_behavior>",
      "code": "def test_B1_<short_slug_from_this_behavior>():\\n    \\"\\"\\"B1: <quote the behavior>\\"\\"\\"\\n    ..."
    }}
  ]
}}

Entry point already defined in the file: {target_name}

Behavioral contract:
{contract_json}

JSON output:
"""

# Triage: when a contract-derived test fails, who is wrong?
_TRIAGE_PROMPT = """\
You are a test-failure triage analyst.

A test derived from a behavioral contract failed against an implementation.
Decide the ROOT CAUSE. Be skeptical: the default assumption is that the
IMPLEMENTATION is wrong, because the test was written from the contract
before the implementation was seen.

Only declare TEST_INCORRECT if you can prove the test contradicts the
contract or asserts something the contract does not support. Quote the
contradicting part of the contract as evidence.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
  "verdict": "IMPL_INCORRECT" | "TEST_INCORRECT" | "SPEC_AMBIGUOUS" | "FLAKY_OR_ENV",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "explanation": "...",
  "evidence_from_contract": "quote the relevant behavior, or '' if none",
  "suggested_fix_target": "implementation" | "test" | "spec" | "none"
}}

Behavioral contract:
{contract_json}

Test(s) that failed:
{test_code}

Implementation under test:
{impl_code}

Failure output (stdout + stderr):
{failure_output}

JSON output:
"""


# ── Contract builders ─────────────────────────────────────────────────────────

_EMPTY_CONTRACT = {"behaviors": [], "input_domains": {}, "invariants": []}


def build_contract_for_function(fn: dict) -> dict:
    """
    Build a behavioral contract for a RETRIEVED function from its declared
    interface only. The implementation body is never sent to the LLM.
    """
    name = fn.get("name") or "<unknown>"
    prompt = _SPEC_FROM_FUNCTION_PROMPT.format(
        name=name,
        signature=fn.get("signature") or "",
        description=fn.get("description") or fn.get("purpose") or "",
        parameters=json.dumps(fn.get("parameters") or [], indent=2),
        preconditions=json.dumps(fn.get("preconditions") or [], indent=2),
        postconditions=json.dumps(fn.get("postconditions") or [], indent=2),
        raises=json.dumps(fn.get("raises") or [], indent=2),
    )
    raw = Settings.llm.complete(prompt).text.strip()
    try:
        contract = _extract_json_object(raw)
        contract.setdefault("target_name", name)
        return contract
    except Exception as e:  # noqa: BLE001
        print(f"[spec_test_engine] contract (function) parse failed ({e}).")
        return {**_EMPTY_CONTRACT, "target_name": name}


def build_contract_from_requirements(
    query: str,
    sub_query: str,
    missing_requirements: List[str],
    extracted_params: Optional[dict] = None,
    target_name: str = "solution",
    signature: str = "",
) -> dict:
    """
    Build a behavioral contract for code that does NOT exist yet, anchored to
    the user's request and the requirements the codebase could not satisfy.
    Used for GENERATE_STANDALONE / GENERATE_WITH_HELPERS.
    """
    prompt = _SPEC_FROM_REQUIREMENTS_PROMPT.format(
        target_name=target_name,
        signature=signature or "(unknown — infer from the request)",
        query=query,
        sub_query=sub_query or query,
        missing_requirements=json.dumps(missing_requirements or [], indent=2),
        params_json=json.dumps(extracted_params or {}, indent=2),
    )
    raw = Settings.llm.complete(prompt).text.strip()
    try:
        contract = _extract_json_object(raw)
        contract.setdefault("target_name", target_name)
        return contract
    except Exception as e:  # noqa: BLE001
        print(f"[spec_test_engine] contract (requirements) parse failed ({e}).")
        return {**_EMPTY_CONTRACT, "target_name": target_name}


def contract_is_empty(contract: dict) -> bool:
    return not (contract or {}).get("behaviors")


# ── Spec-based test generation ─────────────────────────────────────────────────

def generate_spec_based_tests(contract: dict) -> List[str]:
    """
    Generate pytest test functions strictly from the contract. The test writer
    never sees the implementation. Returns a list of test-function source
    strings, ready to be appended to the code under test in the same file.
    """
    if contract_is_empty(contract):
        return []

    target_name = contract.get("target_name", "solution")
    prompt = _SPEC_BASED_TEST_PROMPT.format(
        target_name=target_name,
        contract_json=json.dumps(contract, indent=2),
    )
    raw = Settings.llm.complete(prompt).text.strip()

    try:
        parsed = _extract_json_object(raw)
        tests = parsed.get("tests", [])
        out: List[str] = []
        for t in tests:
            if isinstance(t, dict):
                code = t.get("code", "")
            else:
                code = t  # tolerate a bare string
            if isinstance(code, str) and code.strip():
                out.append(code)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[spec_test_engine] test parse failed ({e}). Raw:\n{raw[:800]}")
        return []


# ── Triage ─────────────────────────────────────────────────────────────────────

def triage_failure(
    contract: dict,
    test_code: str,
    impl_code: str,
    failure_output: str,
) -> dict:
    """
    Decide whether a contract-derived test failure is the implementation's
    fault or the test's fault. Returns the parsed verdict dict (see schema).
    Defaults to blaming the implementation when parsing fails — the safe choice,
    because it keeps the test honest.
    """
    prompt = _TRIAGE_PROMPT.format(
        contract_json=json.dumps(contract, indent=2),
        test_code=test_code,
        impl_code=impl_code,
        failure_output=(failure_output or "")[:4000],
    )
    raw = Settings.llm.complete(prompt).text.strip()
    try:
        verdict = _extract_json_object(raw)
    except Exception:  # noqa: BLE001
        verdict = {}

    verdict.setdefault("verdict", "IMPL_INCORRECT")
    verdict.setdefault("confidence", "LOW")
    verdict.setdefault("explanation", "Triage parse failed; defaulting to IMPL_INCORRECT.")
    verdict.setdefault("evidence_from_contract", "")
    verdict.setdefault("suggested_fix_target", "implementation")
    return verdict


def should_repair_implementation(triage: dict) -> bool:
    """
    Gate for the feedback loop. Repair the implementation UNLESS triage is
    confident the test (or spec) is the problem. We never auto-edit a
    contract-derived test to make it pass — that is the failure mode we are
    avoiding. A HIGH-confidence TEST_INCORRECT/SPEC_AMBIGUOUS is surfaced for
    human review instead.
    """
    verdict = (triage or {}).get("verdict", "IMPL_INCORRECT")
    confidence = (triage or {}).get("confidence", "LOW")
    if verdict in {"TEST_INCORRECT", "SPEC_AMBIGUOUS"} and confidence == "HIGH":
        return False
    if verdict == "FLAKY_OR_ENV" and confidence == "HIGH":
        return False
    return True