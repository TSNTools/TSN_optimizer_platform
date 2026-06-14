"""
Standalone unit test generation module for the TSN coding assistant.

Generates pytest-style unit tests for BOTH:
  - Retrieved (existing) functions from the codebase
  - Generated (new) functions produced by the code generator

Usage:
    from unit_test_generator import generate_unit_tests_for_functions

    tests = generate_unit_tests_for_functions(
        query=query,
        functions=functions,          # list of function dicts (retrieved or generated)
        extracted_params=params,      # dict of extracted parameter values
        source="retrieved",           # "retrieved" | "generated"
    )
    # tests is a list of test-function source strings
"""

from __future__ import annotations

import json
from typing import Optional

from llama_index.core import Settings


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_code_block(text: str) -> str:
    """Strips markdown fences from a code string."""
    text = str(text).strip()
    if "```" not in text:
        return text

    parts = text.split("```")
    for part in parts:
        stripped = part.strip()
        if stripped.lower().startswith("python"):
            return stripped[len("python") :].strip()

    return parts[1].strip() if len(parts) >= 2 else text


def _extract_json_object(text: str) -> dict:
    """
    Extracts the first JSON object from an LLM response.
    Handles raw JSON and markdown-fenced JSON.
    """
    import json
    import re

    text = text.strip()
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

    return json.loads(text[start : end + 1])


# ── Prompts ───────────────────────────────────────────────────────────────────

_UNIT_TEST_PROMPT_RETRIEVED = """\
You are an expert Python test engineer.

You are given one or more EXISTING functions retrieved from a TSN codebase.
Your job is to write pytest-style unit tests that verify each function behaves correctly.

Rules:
- Return ONLY valid JSON. No markdown fences. No preamble.
- The JSON must have a single key "unit_tests" whose value is a list of strings.
- Each string is a complete, self-contained Python test function (def test_...).
- Each test function must:
    * import everything it needs at the top of the function body
    * define or inline any stubs/mocks for external dependencies
      (networkx, numpy, etc.) when the real library is unavailable
    * contain at least one assertion
- Cover at minimum: normal cases, edge cases (empty input, zero, None), and
  expected exceptions where documented.
- Use concrete parameter values extracted from the user query wherever possible.
- Do NOT rely on external files or network access.
- If a function belongs to a class, instantiate a minimal stub of the class inline.
- Test each retrieved function independently.

Schema:
{{
  "unit_tests": [
    "def test_<function_name>_normal():\\n    ...\\n    assert ...",
    "def test_<function_name>_edge_empty():\\n    assert ...",
    "def test_<function_name>_raises():\\n    import pytest\\n    with pytest.raises(...):\\n        ..."
  ]
}}

User request:
{query}

Extracted parameters (use as concrete test inputs where applicable):
{params_json}

Retrieved functions to test:
{functions_json}

JSON output:
"""

_UNIT_TEST_PROMPT_GENERATED = """\
You are an expert Python test engineer.

You are given newly GENERATED code (one or more functions) for a TSN system,
along with any helper functions that were used during generation.
Your job is to write pytest-style unit tests that verify the generated code.

Rules:
- Return ONLY valid JSON. No markdown fences. No preamble.
- The JSON must have a single key "unit_tests" whose value is a list of strings.
- Each string is a complete, self-contained Python test function (def test_...).
- Each test function must:
    * import everything it needs at the top of the function body
    * define or inline any stubs/mocks for external dependencies
      (networkx, numpy, etc.)
    * contain at least one assertion
- Cover: normal cases, edge cases (empty input, zero, None), and expected exceptions.
- Use extracted parameter values as concrete test inputs wherever available.
- Do NOT rely on external files or network access.
- If helper functions are needed, define minimal stubs for them inline.

Schema:
{{
  "unit_tests": [
    "def test_normal_case():\\n    ...\\n    assert ...",
    "def test_edge_case_empty():\\n    assert ...",
    "def test_raises_on_invalid():\\n    import pytest\\n    with pytest.raises(ValueError):\\n        ..."
  ]
}}

User request:
{query}

Extracted parameters:
{params_json}

Helper functions available (stubs may be needed):
{helpers_json}

Generated code under test:
{code}

JSON output:
"""


# ── Core function ─────────────────────────────────────────────────────────────


def generate_unit_tests_for_functions(
    query: str,
    functions: list[dict],
    extracted_params: Optional[dict] = None,
    source: str = "retrieved",          # "retrieved" | "generated"
    generated_code: Optional[str] = None,
    helper_functions: Optional[list[dict]] = None,
    max_code_chars: int = 3000,
    max_context_chars: int = 8000,
) -> list[str]:
    """
    Generates pytest-style unit tests for either retrieved or generated functions.

    Parameters
    ----------
    query : str
        The original user query (provides intent context to the LLM).
    functions : list[dict]
        For source="retrieved"  → the retrieved function dicts (must have "code").
        For source="generated"  → the generated function dicts (must have "code").
    extracted_params : dict, optional
        Parameter name→value mapping from extract_parameters_for_subquery().
    source : str
        "retrieved" uses _UNIT_TEST_PROMPT_RETRIEVED.
        "generated" uses _UNIT_TEST_PROMPT_GENERATED.
    generated_code : str, optional
        Raw generated code string (used when source="generated" and the code
        is not yet split into individual function dicts).
    helper_functions : list[dict], optional
        Helper function dicts used during generation (source="generated" only).
    max_code_chars : int
        Maximum characters of code included per function in the prompt.
    max_context_chars : int
        Maximum total characters for the helper/function block in the prompt.

    Returns
    -------
    list[str]
        List of pytest test-function source strings.
        Returns an empty list on any failure.
    """
    if not functions and not generated_code:
        return []

    params_json = json.dumps(extracted_params or {}, indent=2)

    if source == "retrieved":
        functions_json = json.dumps(
            [
                {
                    "name": fn.get("name"),
                    "class": fn.get("class") or fn.get("class_name"),
                    "signature": fn.get("signature"),
                    "description": fn.get("description") or fn.get("purpose"),
                    "parameters": fn.get("parameters"),
                    "preconditions": fn.get("preconditions"),
                    "postconditions": fn.get("postconditions"),
                    "raises": fn.get("raises"),
                    "code": _extract_code_block(str(fn.get("code") or ""))[
                        :max_code_chars
                    ],
                }
                for fn in functions
            ],
            indent=2,
        )[:max_context_chars]

        prompt = _UNIT_TEST_PROMPT_RETRIEVED.format(
            query=query,
            params_json=params_json,
            functions_json=functions_json,
        )

    else:  # "generated"
        # Prefer explicit generated_code string; fall back to joining fn["code"]
        code_str = generated_code or "\n\n".join(
            _extract_code_block(str(fn.get("code") or ""))
            for fn in functions
            if fn.get("code")
        )

        helpers_json = json.dumps(
            [
                {
                    "name": fn.get("name"),
                    "signature": fn.get("signature"),
                    "code": _extract_code_block(str(fn.get("code") or ""))[
                        :max_code_chars
                    ],
                }
                for fn in (helper_functions or [])
            ],
            indent=2,
        )[:max_context_chars]

        prompt = _UNIT_TEST_PROMPT_GENERATED.format(
            query=query,
            params_json=params_json,
            helpers_json=helpers_json,
            code=_extract_code_block(code_str)[:max_context_chars],
        )

    raw = Settings.llm.complete(prompt).text.strip()

    try:
        parsed = _extract_json_object(raw)
        tests = parsed.get("unit_tests", [])
        return [t for t in tests if isinstance(t, str) and t.strip()]
    except Exception as e:
        print(
            f"[unit_test_generator] WARNING: failed to parse LLM response ({e}).\n"
            f"Raw output (first 1000 chars):\n{raw[:1000]}"
        )
        return []