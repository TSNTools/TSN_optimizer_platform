"""
exact_match_resolver.py
=======================
Turns an EXACT_MATCH retrieval into an actual ANSWER.

Problem this solves
-------------------
When the agent finds the right existing function (recommended_action =
RETURN_EXISTING_FUNCTIONS), it used to return only the function *definition*.
But an imperative query like "Generate a k=4 fat-tree topology" wants the
function *applied* to the user's inputs, not a printout of its source.

This module does three things, in one LLM call:
  1. PARAMETER BINDING  — map the user's concrete values (which may arrive in a
     different shape, e.g. graphConfig={'typ':'ft','k':4}) onto the function's
     real signature parameters (k, directed).
  2. CALL-SITE          — emit a runnable snippet that imports and invokes the
     function with those bound arguments.
  3. ANSWER SYNTHESIS   — a short, direct natural-language answer to the query.

Decoupled from file1.py; depends only on the shared Settings.llm singleton.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from llama_index.core import Settings


# ── JSON helper (self-contained) ──────────────────────────────────────────────

def _extract_json_object(text: str) -> dict:
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


# ── Prompt ────────────────────────────────────────────────────────────────────

_RESOLVE_PROMPT = """\
You are a Python assistant. An existing function has been retrieved that fully
satisfies the user's request. Your job is to APPLY it to the user's request —
not to redefine or modify it.

Produce three things:
  1. A correct CALL-SITE: runnable Python that imports the function and calls it
     with the user's concrete values bound to the REAL signature parameters.
  2. ANSWER: a short, direct answer to the user's request (2-4 sentences),
     referencing what the call returns.
  3. ASSUMPTIONS: any place where you had to infer how a user value maps to a
     parameter (e.g. the user gave a config dict but the signature takes scalars).

CRITICAL RULES:
- Use ONLY parameters that exist in the signature. Do NOT invent parameters.
- The user's values may be in a different shape than the signature (a dict, a
  phrase, etc.). Extract the right scalar(s) and bind them to the correct
  parameter names. If the user gave nothing for a parameter, use its default and
  note it under assumptions; if there is no default, state what is missing.
- For a classmethod/staticmethod, call it on the class: ClassName.method(...).
  For an instance method, instantiate the class first.
- Import from the module where the function lives.
- Do NOT include the function's own source code in the call-site.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
  "call_site": "from <module> import <Class>\\n<result> = <Class>.<fn>(<bound args>)",
  "answer": "Direct answer to the request, mentioning the return value.",
  "assumptions": ["..."]
}}

User request:
{query}

Retrieved function:
  name:        {name}
  class:       {cls}
  kind:        {kind}
  module:      {module}
  signature:   {signature}
  parameters:  {parameters}
  description: {description}
  returns:     {postconditions}

User-provided values (may need reshaping to fit the signature):
{params_json}

JSON output:
"""


# ── Core ──────────────────────────────────────────────────────────────────────

def _module_import_name(module: str) -> str:
    """usecase_generator.py -> usecase_generator"""
    m = (module or "").strip()
    return m[:-3] if m.endswith(".py") else m


def _pick_primary(functions: List[dict]) -> Optional[dict]:
    """Choose the best-match function to apply."""
    if not functions:
        return None
    for fn in functions:
        if str(fn.get("match_rank", "")).upper() == "BEST_MATCH":
            return fn
    return functions[0]


def resolve_exact_match(
    query: str,
    functions: List[dict],
    extracted_params: Optional[dict] = None,
) -> dict:
    """
    Bind the user's inputs to the retrieved function's signature, build a
    runnable call-site, and synthesize a direct answer.

    Returns a dict: {"call_site": str, "answer": str, "assumptions": list[str]}.
    On failure returns empty strings so the caller can fall back to the plain
    function listing.
    """
    
    functions = [f for f in (functions or []) if f.get("code")]

    fn = _pick_primary(functions)
    if not fn:
        return {"call_site": "", "answer": "", "assumptions": []}

    prompt = _RESOLVE_PROMPT.format(
        query=query,
        name=fn.get("name") or "<unknown>",
        cls=fn.get("class") or fn.get("class_name") or "(none — module-level function)",
        kind=fn.get("type") or "function",
        module=_module_import_name(fn.get("module") or ""),
        signature=fn.get("signature") or "",
        parameters=json.dumps(fn.get("parameters") or [], indent=2),
        description=fn.get("description") or fn.get("purpose") or "",
        postconditions=json.dumps(fn.get("postconditions") or [], indent=2),
        params_json=json.dumps(extracted_params or {}, indent=2),
    )

    raw = Settings.llm.complete(prompt).text.strip()
    try:
        parsed = _extract_json_object(raw)
        return {
            "call_site": str(parsed.get("call_site", "")).strip(),
            "answer": str(parsed.get("answer", "")).strip(),
            "assumptions": [a for a in parsed.get("assumptions", []) if isinstance(a, str)],
        }
    except Exception as e:  # noqa: BLE001
        print(f"[exact_match_resolver] parse failed ({e}). Raw:\n{raw[:600]}")
        return {"call_site": "", "answer": "", "assumptions": []}
