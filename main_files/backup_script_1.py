# =============================================================================
# ARCHITECTURE OVERVIEW — how this file implements the agent's use case
# =============================================================================
# This script is the deterministic RAG agent pipeline for the TSN codebase.
# For every user query it runs through, in order:
#   1. UNDERSTAND the query   -> classify_query()/get_or_classify_query() splits
#      it into one or more per-module SubQuery objects (SIMPLE = one module,
#      COMPLEX = several) and tags each with an intent (DO / EXPLAIN_CONCEPT /
#      EXPLAIN_CODE) and any extracted parameters. See "Classify query" section.
#   2. CHECK THE CODEBASE     -> _lookup_one() retrieves candidate functions
#      from the module's pre-built vector+BM25 index for that sub-query.
#   3. DECIDE EXACT / PARTIAL / NO MATCH -> assess_function_coverage() (an LLM
#      judge, since reranker similarity scores alone don't discriminate) picks
#      one of RETURN_EXISTING_FUNCTIONS / GENERATE_WITH_HELPERS /
#      GENERATE_STANDALONE. Driven from answer_query_deterministic()'s Pass 1/3.
#   4. RETURN or GENERATE      -> an exact match is explained via
#      _explain_code_section(); a partial/no match enters
#      generate_verified_implementation(), which SPECIFIES before it writes:
#        4a generate_function_spec() emits a JSON specification of the function
#           that has to exist (name/type/class/description/signature/parameters/
#           pre+postconditions/dependencies/raises) and NO code — retrieved
#           helpers are context for the spec, or the spec stands alone when
#           nothing is reusable. The spec is NOT shown to the user — it feeds
#           test generation directly.
#        4b generate_code_from_spec() implements the spec (minus its
#           "raises" list) — this is the only step that writes code.
#   5. VERIFY generated code   -> between 4a and 4b, _derive_contract_from_spec()
#      turns the spec into a behavioral contract and _spec_based_tests()
#      writes pytest tests from it, so the tests exist before the code does;
#      _approval_gate() lets the user review/edit/approve them. _run_and_repair()
#      then executes them (via _execute_spec_tests(), a real pytest subprocess)
#      and, on failure, triages and repairs the IMPLEMENTATION ONLY — approved
#      tests are never rewritten.
# Everything else in the file (config, index loading, retrievers, prompt
# templates) is plumbing that stage 2-5 depend on.
# =============================================================================

# from llama_index.llms.groq import Groq  # UNUSED: an earlier LLM provider; superseded by llm_config's env-driven setup below.
import llm_config # configures Settings.llm from env (local / tunnel / direct GH) — side-effect import, no names used directly
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    # VectorStoreIndex, SimpleDirectoryReader,  # UNUSED here: only needed to BUILD indexes from raw docs (that's build_indexes.py's job); this script only loads already-persisted indexes via load_index_from_storage.
    PromptTemplate, StorageContext, load_index_from_storage
)
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
# from llama_index.core.agent import ReActAgent   # UNUSED: leftover from an earlier ReAct-agent design; this pipeline is now the deterministic classify->lookup->coverage->generate flow in answer_query_deterministic(), not a tool-calling agent loop.
# from llama_index.core.tools import FunctionTool # UNUSED: only needed to wrap functions as tools for a ReActAgent (see above); no FunctionTool is constructed anywhere in this file.
import os
import json
import torch
import ast
import builtins
import textwrap
import warnings
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
# from deepmultilingualpunctuation import PunctuationModel  # REMOVED: punctuation restoration unused (only called at line ~611 which is commented out); frees VRAM
# lookup_validation.validate_function_list() is NO LONGER USED in the hot path.
# It rejected every candidate whose class_name was non-null (104/108 functions in this
# knowledge base: evaluator 21/21, usecase_generator 46/47, tsn_scheduling_problem
# 37/40), which starved assess_function_coverage() and forced generation. Almost
# certainly it still indexes the *_functions.json catalogs by the old "name" key,
# which was renamed to "function_name"; its class lookup then finds nothing, and only
# entries with class_name=None skip the check and survive. Replaced by
# _validate_against_catalog() below, which uses the already-correctly-loaded
# FUNCTION_CATALOG_BY_MODULE. The import is kept optional so the module can be
# deleted or fixed independently without breaking this file.
try:
    from lookup_validation import validate_function_list as _external_validate_function_list
except Exception:  # noqa: BLE001
    _external_validate_function_list = None
from llama_index.core.schema import QueryBundle
from difflib import SequenceMatcher
# from collections import defaultdict  # UNUSED: only referenced inside the now-disabled merge_results_by_module() below.
# from exact_match_resolver import resolve_exact_match  # UNUSED: exact matches are now explained via the EXPLAIN_CODE prompt (_explain_code_section) instead of this resolver — see the notes near answer_query_deterministic().
from utils import extract_code_block
import time
from llama_index.core import Settings
# from openinference.instrumentation.llama_index import LlamaIndexInstrumentor  # UNUSED: imported but never instantiated/instrument()-ed; tracing is not wired up.
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

# Initialize the debug handler — logs every LLM/retriever call LlamaIndex makes internally (separate from the custom [LLM RESPONSE] timing wrapper below).
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([llama_debug])

start_time = time.time()


_orig_complete = Settings.llm.complete          # the real BOUND method (reading is fine)
_log = []
_generation_time_log = []  # wall-clock durations of each generate_verified_implementation() call this query; cleared per-query in run_one()

def _timed(*args, **kw):
    t0 = time.time()
    r = _orig_complete(*args, **kw)              # exact normal call path
    _log.append(time.time() - t0)
    # print(f"[LLM] #{len(_log)} {_log[-1]:5.1f}s  out~{len(getattr(r, 'text', ''))} chars")
    print(f"\n================ LLM RESPONSE #{len(_log)} ({_log[-1]:.1f}s) out~{len(getattr(r, 'text', ''))} chars ================")
    print(getattr(r, 'text', str(r)))
    print("=========================================================\n")
    return r

object.__setattr__(Settings.llm, "complete", _timed)   # bypass pydantic's field guard

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
# This file lives in main_files/. Resolve every on-disk path from __file__ (not
# from the current working directory), so it works the same whether launched as
# `python3 backup_script_1.py` from inside main_files/, `python3
# main_files/backup_script_1.py` from the repo root, or from an IDE "run" button
# with an arbitrary CWD. Sibling data now lives in dedicated top-level folders:
#   knowledge_base/         -> the *_functions.json catalogs + parameter_explanations.json
#   original_tsn_lib_files/ -> the real usecase_generator.py / tsn_scheduling_problem.py / evaluator.py
#   storage_*/, parameters.md -> unchanged, still at the repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../main_files
REPO_ROOT = os.path.dirname(BASE_DIR)                    # repo root (one level up)
KB_DIR = os.path.join(REPO_ROOT, "knowledge_base")
LIB_DIR = os.path.join(REPO_ROOT, "original_tsn_lib_files")

# ── Config ───────────────────────────────────────────────────────────────────

# API             = "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"
PARAMETERS_FILE = os.path.join(REPO_ROOT, "parameters.md")

# Tier 2: precomputed parameter explanations (built ONCE offline by build_param_map.py).
# Shape: { "<param_name>": {"meaning": "...", "used_for": "..."} }
# Loaded at startup with zero LLM cost; used for all parameter explanations at runtime.
PARAM_EXPLANATIONS_FILE = os.path.join(KB_DIR, "parameter_explanations.json")

# Loads the offline-built parameter->explanation map used by explain_extracted_parameters()
# so parameter meanings can be answered without an LLM call.
def _load_param_map() -> Dict[str, dict]:
    try:
        with open(PARAM_EXPLANATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}
    except FileNotFoundError:
        print(
            f"[WARN] {PARAM_EXPLANATIONS_FILE} not found — run build_param_map.py once. "
            "Parameter explanations will be value-only until then."
        )
    except Exception as e:
        print(f"[WARN] Could not load {PARAM_EXPLANATIONS_FILE}: {e}")
    return {}

PARAM_MAP: Dict[str, dict] = _load_param_map()

# Source .py files the agent's three code modules live in — fetch_code_from_source()
# re-parses these with ast to pull verbatim code for a matched/generated function name.
PYTHON_SOURCE_FILES: Dict[str, str] = {
    "usecase_generator": os.path.join(LIB_DIR, "usecase_generator.py"),
    "tsn_scheduling_problem": os.path.join(LIB_DIR, "tsn_scheduling_problem.py"),
    "evaluator": os.path.join(LIB_DIR, "evaluator.py"),
    # Algorithm-pathway only; see ALGO_MODULE_PERSIST.
    "schedule": os.path.join(LIB_DIR, "schedule.py"),
}

# FUNCTION_CATALOG_FILE = "./function_catalog.json"  # UNUSED: assigned but never read; MODULE_FILES (below) is what _load_function_catalogs() actually loads from.

# Per-module function catalogs (name/signature/docstring/parameters), used both to build
# the retrieval indexes (build_indexes.py) and, here, to hydrate a retrieved/matched
# function name back into a full function dict (see _load_function_catalogs()).
MODULE_FILES: Dict[str, str] = {
    "usecase_generator":      os.path.join(KB_DIR, "usecase_generator_functions.json"),
    "tsn_scheduling_problem": os.path.join(KB_DIR, "tsn_scheduling_problem_functions.json"),
    "evaluator":              os.path.join(KB_DIR, "evaluator_functions.json"),
}
# On-disk locations of the persisted vector+BM25 indexes for each module (built offline
# by build_indexes.py, loaded read-only here via load_persisted_index()). These stay at
# the repo root — only the source .py files and *_functions.json catalogs moved.
MODULE_PERSIST: Dict[str, str] = {
    "usecase_generator":      os.path.join(REPO_ROOT, "storage_usecase"),
    "tsn_scheduling_problem": os.path.join(REPO_ROOT, "storage_tsn"),
    "evaluator":              os.path.join(REPO_ROOT, "storage_evaluator"),
}

PARAMS_PERSIST = os.path.join(REPO_ROOT, "storage_params")

# Modules visible ONLY to the GENERATE_ALGORITHM pathway.
#
# schedule.py is deliberately not in MODULE_FILES/MODULE_PERSIST above. The normal
# pathway answers questions about the three modules a user asks about, and widening its
# retrieval pool would change every one of those answers. The algorithm pathway is the
# only one that needs the compiled-schedule model, and it needs it badly: a measured
# retrieval on the ledger requirement "when a stream needs to change its path, support
# the described FRER migration strategy" returned 15 candidates from the three
# registered modules and NOT ONE of Slot.__lt__ (which defines hop order as
# `end_ns <= other.start_ns`), Schedule.slots_by_stream, or Schedule.update_streams
# (which already reconstructs a route from sorted slots). The two path-ish hits it did
# return, shortestPath and randomPath, GENERATE routes on usecase.graph keyed by port
# NAME — a different object from the schedule-derived, port-id path the requirement is
# about. With nothing correct in view, that run defined a "path" as ports sorted by
# ascending port id, which discards hop order entirely.
ALGO_MODULE_FILES: Dict[str, str] = {
    "schedule": os.path.join(KB_DIR, "schedule_functions.json"),
}
ALGO_MODULE_PERSIST: Dict[str, str] = {
    "schedule": os.path.join(REPO_ROOT, "storage_schedule"),
}
ALGO_MODULE_LABELS = list(ALGO_MODULE_FILES.keys())

# The three source modules — also used as fallback broadcast targets.
CODE_MODULE_LABELS = list(MODULE_FILES.keys())

# All four routable knowledge domains (the three code modules plus the docs-only
# "parameters" module used for configuration-parameter lookups) — the classifier's
# module routing (classify_query) must resolve to one of these.
ALL_MODULE_LABELS = [
    "usecase_generator",
    "tsn_scheduling_problem",
    "evaluator",
    "parameters",
]


MODULE_FILE_LABELS: Dict[str, str] = {
    "usecase_generator":      "usecase_generator_functions.json",
    "tsn_scheduling_problem": "tsn_scheduling_problem_functions.json",
    "evaluator":              "evaluator_functions.json",
    "parameters":             "parameters.md",
}

def augment_query(text: str, module: str) -> str:
    """
    Prepends the target filename to a sub-query before it is sent to the
    retriever or generator. This is contextual prefix injection — it gives
    the embedding model and BM25 an explicit file-scope signal so retrieval
    stays anchored to the correct knowledge domain.

    SIMPLE example:
        in  → "generate a fat-tree topology with k=4"
        out → "[File: usecase_generator_functions.md] generate a fat-tree topology with k=4"

    COMPLEX sub-query example:
        in  → "evaluate end-to-end latency of scheduled streams"
        out → "[File: evaluator_functions.md] evaluate end-to-end latency of scheduled streams"
    """
    label = MODULE_FILE_LABELS.get(module, module)
    return f"[File: {label}] {text}"

# ── No score thresholds anywhere in this pipeline ────────────────────────────
# There is exactly one relevance decision — assess_function_coverage() (stage 3),
# an LLM that reads each candidate's actual code. Reranker scores are retained on
# candidates as `reranker_score` for diagnostics/logging ONLY; nothing branches on
# them. The old CONFIDENCE_THRESHOLD / T_EXACT / MARGIN / T_HELPERS knobs and the
# HIGH_CONFIDENCE/LOW_CONFIDENCE label they produced were removed: nothing read the
# label, and the thresholds encoded a score scale that no longer holds (it was an
# artefact of feeding the reranker an augment_query() prefix shared by every node
# in a module, which flattened all scores to ~0.94; see the note in _lookup_one()).


# ── Function catalogs in memory (zero LLM) ────────────────────────────
# One entry per function, keyed by name, from the same *_functions.json files
# build_indexes.py used. Lets the pure-retrieval path rebuild a full function
# dict from a matched node's `function_name` without an LLM.
def _load_function_catalogs(module_files: Dict[str, str]) -> Dict[str, Dict[str, dict]]:
    catalogs: Dict[str, Dict[str, dict]] = {}
    for module, path in module_files.items():
        by_name: Dict[str, dict] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in (items if isinstance(items, list) else []):
                if isinstance(item, dict) and item.get("function_name"):
                    by_name[item["function_name"]] = item
        except FileNotFoundError:
            print(f"[WARN] catalog not found: {path} — run build_indexes.py.")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] could not load catalog {path}: {e}")
        catalogs[module] = by_name
    return catalogs


FUNCTION_CATALOG_BY_MODULE: Dict[str, Dict[str, dict]] = _load_function_catalogs(
    {**MODULE_FILES, **ALGO_MODULE_FILES}
)


def _load_function_catalogs_classed(module_files: Dict[str, str]) -> Dict[str, Dict[tuple, dict]]:
    """Same catalogs, keyed by (class_name, function_name) instead of bare name.

    The bare-name index silently drops one entry per module, because a name can be
    reused across classes (__init__ under UsecaseGenerator vs Usecase, InstanceContext
    vs TSNScheduling, TSNConstraintEvaluator vs its UnionFind sub-class). This index
    keeps every variant so a retrieved node carrying class_name metadata resolves to
    the RIGHT entry rather than whichever one happened to be loaded last.
    """
    classed: Dict[str, Dict[tuple, dict]] = {}
    for module, path in module_files.items():
        by_key: Dict[tuple, dict] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in (items if isinstance(items, list) else []):
                if not isinstance(item, dict):
                    continue
                name = item.get("function_name") or item.get("name")
                if name:
                    by_key[(item.get("class_name"), name)] = item
        except Exception:  # noqa: BLE001 — the bare-name loader already warns
            pass
        classed[module] = by_key
    return classed


FUNCTION_CATALOG_BY_MODULE_CLASSED: Dict[str, Dict[tuple, dict]] = _load_function_catalogs_classed(
    {**MODULE_FILES, **ALGO_MODULE_FILES}
)


def _validate_against_catalog(functions: List[dict], routed_module: str):
    """In-process replacement for lookup_validation.validate_function_list().

    Design rule: validation is a sanity check on retrieval, NOT a second retrieval
    stage. It may only discard a candidate that is useless anyway — one with no
    catalog entry AND no resolvable source code. It can never discard a candidate
    just because of a class/module label mismatch, which is what made the previous
    validator reject 96% of the knowledge base.

    Returns (valid, invalid) and annotates each fn with the same "catalog_validation"
    shape the old validator produced, so nothing downstream changes.
    """
    catalog = FUNCTION_CATALOG_BY_MODULE.get(routed_module, {})
    classed = FUNCTION_CATALOG_BY_MODULE_CLASSED.get(routed_module, {})

    valid: List[dict] = []
    invalid: List[dict] = []
    for fn in functions:
        name = fn.get("name")
        cls = fn.get("class_name")

        # Exact (class, name) first; fall back to bare name; class mismatch is NOT fatal.
        entry = classed.get((cls, name)) or catalog.get(name)

        if entry is not None:
            ok, reason = True, f"Found in the '{routed_module}' catalog."
        elif fn.get("code"):
            ok, reason = True, (
                f"'{name}' is not in the '{routed_module}' catalog, but its source "
                f"was resolved from the module file — accepted."
            )
        else:
            ok, reason = False, (
                f"'{name}' is not in the '{routed_module}' catalog and no source "
                f"code could be resolved — nothing usable to return."
            )

        fn["catalog_validation"] = {
            "valid": ok,
            "reason": reason,
            "matched_entry": (entry or {}).get("function_name") or (entry or {}).get("name"),
            "alternatives": [],
        }
        (valid if ok else invalid).append(fn)

    return valid, invalid

# ── Catalog health check ─────────────────────────────────────────────────────
# A catalog that silently loads as {} (renamed JSON key, moved file, bad path) is
# indistinguishable downstream from "the codebase has no such function": every
# lookup misses, coverage returns NO_MATCH, and the pipeline generates from
# scratch. Print the counts at startup so that failure mode can never be silent.
for _m, _cat in FUNCTION_CATALOG_BY_MODULE.items():
    _n_raw = 0
    try:
        with open(MODULE_FILES[_m], "r", encoding="utf-8") as _f:
            _n_raw = len(json.load(_f))
    except Exception:  # noqa: BLE001
        pass
    if not _cat:
        print(f"[catalog] *** {_m}: 0 functions loaded from {MODULE_FILES[_m]} "
              f"({_n_raw} raw entries) — check the JSON key names.")
    else:
        _lost = _n_raw - len(_cat)
        print(f"[catalog] {_m}: {len(_cat)} functions"
              + (f"  (!! {_lost} entry/entries dropped to name collisions — "
                 f"same function name under two classes)" if _lost > 0 else ""))


def _catalog_entry_to_function(entry: dict, module: str) -> dict:
    """
    Build the function dict downstream consumers expect, from a catalog entry.
    NOTE: reads "function_name" (the *_functions.json on-disk key) but re-keys it to
    "name" below — "name" is this pipeline's own internal fn-dict contract, used by
    dozens of downstream consumers (inject_code, format_subquery_answer, validate_
    function_list, etc.), and is intentionally kept distinct from the JSON schema.
    """
    name = entry.get("function_name")
    code = fetch_code_from_source(name, module) or entry.get("code", "")
    return {
        "name": name,
        "class_name": entry.get("class_name"),
        "sub_class_name": entry.get("sub_class_name"),
        "method_type": entry.get("method_type"),
        "signature": entry.get("signature"),
        "docstring": entry.get("docstring"),
        "nl_summary": entry.get("nl_summary", ""),
        "description": entry.get("description") or entry.get("purpose") or "",
        "parameters": entry.get("parameters") or [],
        "postconditions": entry.get("postconditions") or [],
        "modules": entry.get("module") or module,
        "module": entry.get("module") or module,
        "code": code,
        "routed_module": module,
    }

# VALIDATION AND REPAIR FOR CLASSIFIER OUPUT
VALID_MODULES = set(ALL_MODULE_LABELS)

VALID_INTENTS = {"DO", "EXPLAIN_CONCEPT", "EXPLAIN_CODE", "GENERATE_ALGORITHM"}

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

# ── Punctuation restoration ───────────────────────────────────────────────────
# REMOVED: the punctuation model (oliverguhr/fullstop-punctuation-multilang-large)
# is never used in this script — restore_punctuation() is only referenced at the
# commented-out line ~611. Kept out to free VRAM. Restore this block (and the
# import above) if you re-enable query punctuation restoration.
#
# _punct_model = None
#
#
# def get_punctuation_model() -> PunctuationModel:
#     global _punct_model
#
#     if _punct_model is None:
#         _punct_model = PunctuationModel(
#             model="oliverguhr/fullstop-punctuation-multilang-large"
#         )
#
#     return _punct_model
#
#
# def restore_punctuation(text: str) -> str:
#     model = get_punctuation_model()
#     restored = model.restore_punctuation(text)
#     print(f"[Punctuation] '{text}'\n           -> '{restored}'")
#     return restored

# ── Classify query (pipeline stage 1: understand the user's query) ───────────
# The enums/dataclasses below are the typed output of classify_query(): a
# ClassifiedQuery holding one SubQuery per (module, action) the request touches.

class QueryType(str, Enum):
    SIMPLE = "SIMPLE"    # request maps to exactly one module
    COMPLEX = "COMPLEX"  # request spans two or more modules


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Intent(str, Enum):
    DO = "DO"                            # build / generate / return / run / compute
    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"  # conceptual TSN question, not tied to one function
    EXPLAIN_CODE = "EXPLAIN_CODE"        # explain a specific existing function/method/class
    GENERATE_ALGORITHM = "GENERATE_ALGORITHM"  # design + implement a multi-part algorithm/
                                                # strategy/heuristic, not a single function

# One self-contained, single-module question, e.g. "generate a fat-tree topology" ->
# module="usecase_generator". A COMPLEX user query is split into several of these.
@dataclass
class SubQuery:
    text: str
    module: str
    intent: str = "unknown"
    action: Intent = Intent.DO
    parameters: dict = field(default_factory=dict)


# Full result of understanding one user query: how many modules it touches, the
# per-module SubQuery breakdown, and diagnostics (confidence/warnings) for debugging.
@dataclass
class ClassifiedQuery:
    original_query: str
    normalized_query: str
    query_type: QueryType
    sub_queries: List[SubQuery]
    classification_confidence: Confidence = Confidence.MEDIUM
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    raw_classifier_output: Optional[str] = None

# ── Coverage assessment (pipeline stage 3: exact / partial / no match) ───────
# Output type of assess_function_coverage(): does the retrieved code fully satisfy
# the sub-query, partially help, or not at all?
class MatchType(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"      # existing function(s) fully satisfy the request as-is
    PARTIAL_MATCH = "PARTIAL_MATCH"  # existing function(s) are useful helpers, but glue/new logic is still needed
    NO_MATCH = "NO_MATCH"            # nothing retrieved is useful; must generate standalone


class RecommendedAction(str, Enum):
    RETURN_EXISTING_FUNCTIONS = "RETURN_EXISTING_FUNCTIONS"  # -> explain/return the matched function(s) as-is
    GENERATE_WITH_HELPERS = "GENERATE_WITH_HELPERS"          # -> generate code, using retrieved functions as helpers
    GENERATE_STANDALONE = "GENERATE_STANDALONE"              # -> generate code from scratch, no usable helpers found


# The verdict assess_function_coverage() returns for one sub-query: which MatchType
# applies, why, what's missing, and the RecommendedAction to take next.
@dataclass
class CoverageAssessment:
    match_type: MatchType
    reason: str
    missing_requirements: List[str]
    recommended_action: RecommendedAction

# ── Stage 0: pre-classification algorithm-generation gate ────────────────────────
# Runs on the RAW user query, BEFORE classify_query()/CLASSIFICATION_PROMPT ever sees
# it. YES routes the whole query to run_algorithm_generation_pathway() instead of the
# normal classify -> sub-query -> per-sub-query pipeline; NO leaves the existing
# pathway (CLASSIFICATION_PROMPT and everything downstream) completely untouched.
IS_ALGORITHM_GENERATION_PROMPT = """\
You are a routing gate for a TSN (Time-Sensitive Networking) coding assistant.

Decide ONE thing: does the user's query ask for a NEW ALGORITHM, STRATEGY, or
HEURISTIC to be designed and implemented — a multi-part orchestration with its own
design decisions (candidate generation, evaluation, selection, a search/optimization
loop, coordination of multiple steps, ...) — as opposed to an ordinary request to
write, fix, use, retrieve, or explain a single, straightforward function, or a
conceptual/explanatory question?

Criteria:
- The query must be phrased as an INSTRUCTION, not a question (e.g. "design a
  simulated-annealing scheduler for X", "implement a heuristic that does X", "write a
  strategy to solve X", "build an algorithm that computes Y").
- Answer YES only when the query EXPLICITLY asks for an algorithm, heuristic,
  strategy, or procedure to be designed, implemented, created, or built. A plain
  "write a function that does X" stays NO even when the function itself is internally
  complex — the request must name the design of a multi-step algorithm, not just a
  function.
- MANDATORY: if the query contains "generate"/"implement"/"create"/"build"/"design"/
  "write" together with the word "algorithm" anywhere in it (any tense or phrasing —
  "generate an algorithm", "I want to build an algorithm that...", "implement an
  algorithm for X", etc.), the answer is ALWAYS YES, with no exceptions.
- Answer NO for everything else, including requests to write/fix/explain a specific
  function, retrieve existing functionality, or explain a concept.

User query:
{query}

Output rules:
- Return ONLY a valid JSON object, no preamble, no markdown fences, no extra text.
- Use exactly this shape:
{{
"is_algorithm_generation": true or false,
"reason": "one short sentence"
}}

JSON:"""

# Stage 0b prompt: turns an algorithm-generation query into an atomic, numbered
# requirements ledger BEFORE any design/implementation work happens. Every later stage
# of the algorithm pathway (not yet built) is expected to carry this ledger forward and
# have each artifact it produces cite the requirement id(s) it satisfies — the ledger is
# the single source of truth for "did we build what was actually asked for", the exact
# gap the earlier GENERATE_ALGORITHM sub-query pathway had no mechanism to check.
GENERATE_ALGORITHM_REQUIREMENTS_PROMPT = """\
You are a requirements analyst for a TSN (Time-Sensitive Networking) coding assistant.

The user's query asks for an algorithm to be designed and implemented. Before any
design or implementation happens, extract the query into a ledger of atomic, numbered
requirements — R1, R2, R3, and so on — each anchored to the user's own words, so that
every later stage can be checked against exactly what was asked for.

For every requirement you also write its meaning as TRIPLES — subject, predicate,
object — over the closed vocabulary given below. The sentence records what was asked;
the triples record what it is ABOUT, as named things and named relations between them.
Both are required.

THIS LEDGER RECORDS WHAT THE USER ASKED — NOTHING MORE. It is not the design. Where the
query relies on something it does not define, you do
not define it: you leave it undefined (triple rule 2 says when it still becomes a
resource), and a later stage, the blueprint, defines it with knowledge of the library.
Your job is to make sure that nothing the user asked is lost, merged with something
else, or quietly decided.

Rules:
- ATOMIC: one independently-checkable requirement per entry. Split a compound sentence
  into as many requirements as it actually states — "compute X and determine Y" is two
  requirements, not one. A numbered or bulleted list is one requirement per item, plus
  one for the sentence that introduces it.
  SPLITTING MUST NOT CHANGE THE MEANING. Where a sentence applies one condition,
  criterion or qualifier to several things — "do X and Y based on Z", "produce X and Y
  that satisfy Z" — Z governs every one of them, so every requirement you split out of
  that sentence carries Z. A split that leaves the criterion on one half and strips it
  from the other has changed what the query asked for, and is worse than not splitting
  at all: the stripped half now reads as an open choice the query never left open, and
  no later stage can tell that anything is missing. If you cannot see which parts a
  qualifier governs, keep the sentence as one requirement - unless the qualifier is
  itself a demand ("... while minimizing Z", "... ending at Y", "... in order"): a demand
  is always its own requirement, and its text keeps what it qualifies ("X ... while
  minimizing Z"), so nothing is stripped from either.
- ANCHORED: quote or closely paraphrase the exact phrase from the query that each
  requirement comes from, so it can be traced back to the original text. Do not invent
  a requirement the query does not state or clearly imply, and do not soften, drop, or
  generalize a requirement's wording.
- COMPLETE: cover the entire query — every input, every output, every constraint, every
  step of any described procedure or strategy, every ordering or selection criterion,
  and every implementation directive. Nothing the
  query states may be left unlisted — including the sentence that says what the
  algorithm is to compute, which usually comes first and is easy to mistake for a title.
- HOW-TO-BUILD DIRECTIVES ARE REQUIREMENTS TOO, and each one gets its own entry. A
  sentence about how the algorithm is to be CONSTRUCTED counts exactly as much as one
  about what it computes, and these are the entries most often dropped — they tend to
  sit in a closing sentence, after the interesting technical content, and read like
  preamble. They are not preamble: a sentence listing several directives is one
  requirement per directive.
- Number requirements in the order they appear in the query.
- Do not merge distinct requirements to shorten the list, and do not split one
  requirement into near-duplicates to inflate it. Two requirements that share a
  criterion are NOT near-duplicates: "choose A based on Z" and "choose B based on Z"
  are two different obligations and the criterion belongs in both.

======================= WHAT THE CODEBASE ALREADY HAS =======================

This algorithm will be built out of an existing TSN codebase, and most of the query's
nouns already exist in it. Before you write a single triple, work out which. The
catalogue below is read out of that codebase's own source: every name in it is real.

{library_resources}

============================== DECLARE THE RESOURCES ==============================

Declare every distinct thing your triples mention, once each, in a "resources" list.
That list is the CLOSED set of names the triples may use — a triple may not mention
anything it does not declare. Write it AFTER the requirements, by going through every
triple you wrote and declaring each name in it: the list is then exactly the names the
triples use, no more and no fewer.

The algorithm the query asks for is one of these names. It is the process whose inputs
and outputs the query lists (hasInput and hasOutput need a process as subject), and the
catalogue has no such thing, so its corresponds_to is null. null is a binding, never a
name: it does not appear in a triple.

Each resource has three fields:

  name            The CamelCase identifier the triples use. One thing, one name.
  corresponds_to  What TYPE this resource is in the catalogue: a schedule form (its id,
                  or its code as the catalogue lists it, which a short explanation
                  may follow), a class or nested class, one stream's part of a schedule form
                  (candidate_array[:, i]), a library dict named by its code
                  (Usecase.streams[i]), one key of it written as code
                  (Usecase.streams[i]['pathLists']) — or
                  a LIST of those when several are each correct and a later stage picks
                  between them — or null only when the catalogue holds no such type at
                  all. Two resources that are the same type give the same
                  corresponds_to.
  note            One line: what it is; WHERE THIS VALUE COMES FROM whenever another
                  resource is bound to the same library type — that is what tells two values of one type apart, and two
                  resources bound to null do not share a type; and why that binding, or
                  why null.

HOW TO BIND. These six rules decide whether this whole exercise helps or hurts.

1. corresponds_to IS THE TYPE, NEVER THE VALUE. corresponds_to says what
   kind of library thing a value IS. It does not mean "relates to it", "is built out of
   it" or "is eventually turned into it". If you would have to explain the connection in
   more than a naming, it is null. A wrong binding is worse than no binding: a later stage
   will import the thing you named and operate on it directly, and it will fail on the
   type you told it to expect. It never says WHICH value it is: two resources of one type
   with different origins are different things, and the note saying where each comes
   from is how every later stage tells them apart.

2. null IS A REAL ANSWER, AND OFTEN THE IMPORTANT ONE. A query asks for things a
   codebase has no counterpart for — a migration plan, a migration strategy, FRER
   itself, a score, an ordering. None of those is in the catalogue, and null is the
   correct and useful answer for each. It is what tells the later stages which things
   must be BUILT rather than found. An invented binding does not merely mislead: it hides
   work that nobody will then do. Like every binding it is about the TYPE (rule 1): a
   value the algorithm computes is bound to its type exactly like a value it is handed,
   whenever the catalogue has that type. Computing a value is not building a type.

3. null MEANS THE LIBRARY HAS NOTHING — NOT "I CANNOT DECIDE WHICH". These are two
   completely different answers and collapsing them throws away the useful one. When
   several catalogue entries would each be a correct binding and the query does not
   single one out, GIVE THEM ALL, as a list:

       "corresponds_to": ["candidate_array", "compiled_schedule"]

   A list says "it is one of these, and a later stage picks" — which is true, useful,
   and the right answer for every schedule in a query that does not name a form. null
   says "the catalogue contains no such thing". A schedule is NOT null: the codebase
   is full of schedules. Choosing BETWEEN the listed forms is the configuration
   stage's decision and not yours.

4. A DICT KEY NAMES A FIELD, NOT A VALUE. The library keeps streams and ports as plain
   dicts, so some things are one key of such a dict rather than the whole dict; bind
   those as the code that reaches the key, with the dict named exactly as the catalogue
   names it, e.g. "Usecase.streams[i]['pathLists']". A key holds ONE value per dict, for
   the use case's own current state; a value that belongs to one particular schedule is
   part of THAT schedule, bound to that schedule's part as the catalogue lists it, not to
   the key.

5. ONE THING, ONE NAME — BUT A SHARED TYPE IS NOT A SHARED IDENTITY. Two of the query's
   words are one resource only when they name the SAME value: the same
   corresponds_to AND the same origin. Two words for the same value are one resource;
   declare it once and let the note record the other wording. Two values of one type with
   different origins are two resources.

6. EVERY RESOURCE YOU DECLARE MUST APPEAR IN AT LEAST ONE TRIPLE. If none does, you
   have declared something the query never relates to anything, and a later stage will
   carry it as a thing to build.

============================= THE TRIPLE VOCABULARY =============================

A triple is {{"s": Subject, "p": predicate, "o": Object}}.

THE PREDICATE LIST IS CLOSED. These eighteen are the only predicates that exist.

  WHAT A THING IS
    isA             S is a kind, or an instance, of O.
    partOf          S is a component of O.
    represents      S stands for O in the real network.

  DATA FLOW
    hasInput        S is given O.
    hasOutput       S produces O.

  BEHAVIOUR OVER TIME
    transformsFrom  S starts from O.
    transformsInto  S arrives at O.
    hasStep         O is one step of the procedure S.
    precedes        step S happens before step O.
    mustEndAt       S must terminate at O.

  CONDITIONS
    appliesWhen     S is used only in situation O.
    precondition    O must hold before S may happen.
    mustSatisfy     S must have quality O.

  DECISIONS AND OBJECTIVES
    determines      S decides O.
    criterion       S's choice is governed by O.
    minimizes       S is to be made as small as possible, measured by O.

  HOW IT IS TO BE BUILT
    builtBy         S is to be constructed by means of O.
    constrainedBy   S is limited by O.

If nothing on this list fits what a requirement states, emit no triple for that
requirement. Do not invent a predicate, do not stretch one past its meaning above, and
do not fall back on isA to mean "has something to do with".

EVERY PREDICATE CONSTRAINS WHAT MAY SIT ON EACH END. These are not style notes: a
triple that breaks one of them states something the query did not say.

  hasInput / hasOutput   S must be a PROCESS — the algorithm, a procedure, a step —
                         never a value. Where one value sits inside another, the
                         predicate is partOf. Where the query describes ONE value twice,
                         it is one resource (binding rule 5), not two joined by an edge.

  represents             S and O must be DIFFERENT resources, and no two resources may
                         represent the same O.

  minimizes              S must be the same resource the algorithm's hasOutput points
                         at. The algorithm itself does not vary; what it produces does,
                         and an objective ranks what varies.

  partOf                 S is a PIECE of O: every S sits inside one O. Read the triple
                         back as "an S is a piece of an O" and drop it if that is false.
                         A QUALITY is not a
                         piece of what it qualifies, and an ACTION is not a piece of the
                         value it acts on. A step is not partOf its procedure: hasStep
                         already says so, and one link is never stated twice.

RULES FOR THE TRIPLES. These are what make the graph usable, so follow them exactly.

1. USE ONLY DECLARED RESOURCES. Every "s" and every "o" must be the exact "name" of a
   resource you declared above — never a bare noun, never a variant spelling, never
   something that only appears in a triple. Every name a triple needs is declared in the
   resources list that follows the requirements.

2. AN UNDEFINED QUALITY IS A RESOURCE ONLY IF SOMETHING MUST COMPUTE OR CHECK IT. When
   the query names a quality, a measure or a condition without defining it, the word
   names something the query leaves undefined. If the algorithm will have to compute or check it — an objective, a
   ranking criterion, a precondition, a validity test — declare it as a resource
   (corresponds_to null, its note saying the query leaves it undefined) so an edge can
   point at it:
       {{"s": "<a step>",              "p": "precondition", "o": "<the condition it waits for>"}}
       {{"s": "<the result>",          "p": "mustSatisfy",  "o": "<the property it must have>"}}
   An adjective whose meaning the same sentence already supplies is NOT a thing of its own.
   A word made into a resource is turned into a
   data structure and given a formula downstream, so make one only where something
   really has to be computed.

3. A QUALIFIER THAT GOVERNS SEVERAL THINGS GETS ONE TRIPLE PER THING. "choose A and B
   based on Z" is two triples — (A, criterion, Z) and (B, criterion, Z) — never one
   triple about A-and-B together, and never one that mentions only A.

4. DO NOT BUILD AN ONTOLOGY. State only relations the query itself states. Do not
   classify what it leaves unclassified, do not decompose what it leaves whole, and do
   not add background facts about TSN, FRER, scheduling or networking, however true
   they are — the later stages bring the domain knowledge, and this ledger is the record
   of what the user asked. A phrase that says what something IS describes that resource: it goes in its note,
   not into a second resource joined to it by represents.

5. EVERY REQUIREMENT GETS AT LEAST ONE TRIPLE unless it genuinely asserts no relation
   between two things. A requirement's triples must say what its text says — no more,
   and no less.

6. WHEN THE QUERY NAMES ONE MEMBER OF A KIND, THE KIND IS A RESOURCE TOO. "one possible
   X is Y", "a strategy such as Y", "for example Y" — declare both X and Y, emit
   Y isA X, and attach every criterion, every objective and every choice to X, never to
   Y. The query is telling you a choice EXISTS. The ACTIONS the query lists for that one
   member, on the other hand, are that member's own: (Y, hasStep, ...) — not steps of the
   general kind, and not steps of the whole sequence the algorithm produces. A listed item
   that says something stays or holds is not an action (rule 7, example A).

7. A STEP CHANGES SOMETHING; A RULE THAT SOMETHING STAYS IS NOT A STEP. The object of
   hasStep is something the algorithm DOES to the data, named as its verb and object.
   A numbered item that says something must STAY as it is or must HOLD is a
   condition, not an action: write it with mustSatisfy when it holds throughout, or with
   precondition when it must hold before one particular step — never with hasStep. A
   step that is really a state has nothing for anyone to implement and becomes a
   function that changes nothing.
   A step's verb has an object in the query's own words — what it activates, removes or
   replicates — and often something it is done over or with. Each is a resource and the
   step's input, (<the step>, hasInput, <the thing it acts on>). A step whose
   triples never say what it acts on leaves a later stage to guess.

8. A NUMBERED LIST IS AN ORDER, AND THE NUMBERS ARE INFORMATION. The steps that remain
   after rule 7 get one precedes edge between each consecutive pair, in the query's
   order. Each edge goes in the requirement of the LATER step of the pair: its number is
   part of what that item says.

9. A DIRECTION ABOUT HOW TO BUILD IS NOT A RELATION BETWEEN TWO THINGS. A sentence
   about the MANNER of construction names no object the algorithm stands in a relation
   to. Such a requirement keeps its text — it is still a requirement, and later stages
   are still checked against it — and emits no triple. The exception is something the
   query asks you to CREATE, which earns its place by having something SAID about it:
   give it an edge of its own, such as what it produces.

10. A CLAUSE THAT SAYS WHEN SOMETHING APPLIES IS PART OF WHAT IS SAID. "When X", "if X",
   "only for X" limit where the rest of the sentence holds. Carry the limit as
   (S, appliesWhen, X), or as a precondition when it limits one step, with X a resource
   under rule 2. Dropping it turns a conditional obligation into an unconditional one.

This ledger is carried through every subsequent stage of building the algorithm, and
every artifact produced afterward must cite the requirement id(s) it satisfies —
precision here is what makes everything downstream checkable against what the user
actually asked for.

User query:
{query}

Output rules:
- Return ONLY a valid JSON object, no preamble, no markdown fences, no extra text.
- Use exactly this shape:
{{
"requirements": [
  {{"id": "R1",
    "text": "<ONE demand, in the user's own words with every qualifier the sentence gives it>",
    "triples": [
      {{"s": "<a declared resource name>", "p": "<one predicate from the closed list>",
        "o": "<a declared resource name>"}}
    ]}},
  {{"id": "R2", "text": "...", "triples": []}}
],
"resources": [
  {{"name": "<an input the request names>",
    "corresponds_to": <a catalogue name as a string, a JSON list of them, or null - never a list or null inside quotes>,
    "note": "<one line: what it is, where it comes from, and why that binding or why null>"}},
  {{"name": "<an item the request acts on that the catalogue holds as an element (e.g. one stream)>",
    "corresponds_to": "<its code in the catalogue, e.g. Usecase.streams[i]>",
    "note": "<what it is, and why that binding>"}},
  {{"name": "<a quality or measure the request names without defining it>",
    "corresponds_to": null,
    "note": "<the query leaves it undefined; a later stage defines it>"}},
  {{"name": "<a result the catalogue does not hold>", "corresponds_to": null,
    "note": "<null: the catalogue holds no such thing, so this must be built>"}}
]
}}

JSON:"""

# Stage 0b2 prompt: CONFIGURATION. Runs between the requirements ledger and the
# blueprint, and settles the ENVIRONMENT the algorithm will run inside — which library
# objects exist, how each one is obtained, and above all what a non-zero schedule cell
# MEANS — before any algorithm is designed on top of it.
#
# Split out because that decision kept being made implicitly, late, and wrongly. Across
# six runs the encoding was never chosen anywhere: designs read cell values only as
# `> 0` presence flags, took the pre-built `problem.evaluator` (always PRIORITY, since
# initialize_problem never passes scheduling_type), and scored start-time data with a
# constraint family that cannot read it. Every call site was individually correct; the
# environment was never settled, so there was nothing for them to be consistent WITH.
# Making it a separate artifact means the choice is written down once, checked once, and
# handed to the blueprint as a fact rather than left to emerge from the algorithm.
ALGORITHM_CONFIGURATION_PROMPT = """\
You are configuring the environment for a TSN (Time-Sensitive Networking) algorithm,
before the algorithm itself is designed.

You are given a requirements ledger (R1..Rn). Your job is to build the ENVIRONMENT that
those requirements need in order to be implementable — the objects the algorithm will be
handed, how each is obtained, and what the data flowing between them means. The next
stage, the blueprint, treats everything you write as settled fact and uses it exactly as
written: every object, type and form you name here is what it builds on, and it may not
add, rename or retype any of them. So what you write must be complete and correct for
THIS request — a gap you leave cannot be filled later.

You build it FROM THE MENU BELOW, with ONE exception: the objects the algorithm is HANDED
(step 4). Everything else in the environment is a menu entry. The menu is the complete
set of configuration this library has: every option it exposes, every schedule form it
fixes, and for each one the functions that act upon it.

If something a requirement needs is neither a menu option nor an object the algorithm is
handed, it is not configuration — it is algorithm logic, and it belongs to the blueprint.
Deriving anything from an input, ordering, choosing among options, and defining a
quality or measure the request leaves undefined are all algorithm work. Putting any of it here does not
make it settled; it only hides it from the stage that would have done it properly. An
environment step that is neither a menu entry nor a handed-in object is wrong by
construction.

================================ THE MENU ================================

PART 1 — OPTIONS. Under each file is an option, and under each option are the functions
that option acts upon and what each of them does. Those functions are what your choice
SWITCHES ON. They are not functions your design calls, and no environment step calls them.
{config_option_menu}

PART 2 — SCHEDULE FORMS. What a schedule IS, fixed by the codebase, with the conventions
and limits below read straight out of its source. You do not restate these and you do not
invent a parallel version of one. You CHOOSE among them.
{fixed_representations}

============================== HOW TO BUILD IT ==============================

1. Read the ledger and decide which requirements need something from the environment at
   all. Most will not: a requirement about what the algorithm computes, orders, chooses
   or checks is discharged by the algorithm, not by its setup.

   THE LEDGER COMES WITH A GRAPH, and two parts of it are addressed to you.

   Its RESOURCES each say what they already ARE in this codebase, and each carries a
   NOTE: one line from the requirements stage on what the resource is, where its value
   comes from (which input, which schedule it is read off), why it is bound as it is, and,
   for a word the query leaves undefined, that it is undefined. Use the note to tell apart
   resources with the same "is" line and, for a
   resource on a hasInput edge of the algorithm, to obtain it the way the note says it
   arrives. A note is context, not a decision: it never overrides the "is" line.
   A resource's NAME is the requirements stage's name for it: where an environment step
   or a representation choice serves a resource, name that resource in its reason.
   - A resource whose "is" line names ONE thing is settled AS A TYPE — do not re-decide
     it. A binding says what a value IS, never where it comes from: two resources with
     the same "is" are two VALUES
     of one type, each obtained separately.
   - A resource whose "is" line names SEVERAL schedule forms (by PART 2 id, or by the
     code PART 2 lists for each) is the requirements stage
     telling you exactly which choice it left open and left to you: that is the choice
     step 3 asks you to make, already narrowed to the forms that would be correct. A
     resource bound to ONE STREAM'S PART of those forms (candidate_array[:, i] or
     Schedule.slots_by_stream(i)) follows the same choice: it is the part of the form
     `schedule` names, and reading it off a schedule is algorithm work.
   - A resource whose "is" line says it has to be BUILT is algorithm work, and no
     environment step settles it.
   - A resource whose "is" line is a KEY of a library dict, written as the code that
     reaches it (such as Usecase.streams[i]['pathLists']), names a field of one use
     case's data. No environment step obtains it, and which schedule its value is read
     off is algorithm work.

   Its TRIPLES are subject-predicate-object edges under each requirement. The ones that
   concern you are hasInput and hasOutput whose SUBJECT is the algorithm being built:
   they say what the algorithm is handed and what it must produce, so each resource on
   such a hasInput edge must be obtained by an environment step (step 4). A hasInput
   edge whose subject is one of the algorithm's own steps is
   data the algorithm passes to itself, not environment. The rest are not yours:
   hasStep, precedes, criterion, minimizes, mustSatisfy and precondition all describe
   what the algorithm DOES, and an environment step claiming to settle one of those is
   the mistake this stage exists to avoid.

   Words the request uses without defining them are answered by the blueprint. Do not pick an option, a weight or a
   metric as if it were their meaning.

2. For every option in PART 1 marked YOURS TO DECIDE, settle it and give the one-line
   reason the requirements force that answer. For every option NOT marked that way, do
   not create an environment step for it — the library derives it, or the existing file
   supplies it — with ONE exception, which is about the evaluator:
   problem.evaluator is built by initialize_problem WITHOUT scheduling_type, so it is
   always PRIORITY. When the encoding you settle is START_TIME, the evaluator must be
   CONSTRUCTED, and a constructed evaluator takes check_routing and check_task_placement
   from its own defaults unless they are passed. So whenever you construct it, write the
   whole call in its obtained_by, as the library's constructor takes it:
       TSNConstraintEvaluator(problem, check_routing=..., check_task_placement=...,
                              scheduling_type=ScheduleEncodingType.START_TIME)
   (example A below). Give the source of each flag:
   - True for both, whenever the algorithm installs a route or writes a column the use
     case does not fix. check_routing guards that each column is
     one connected talker-to-listener path; check_task_placement that each column has
     exactly one talker port and one listener-adjacent port. A route copied from an input
     schedule into a configuration the algorithm writes is still a route it installs.
     Name the requirement.
   - Derived as initialize_problem derives them — fr, ftp = problem.get_task();
     check_routing = not (fr and ftp); check_task_placement = not ftp, read inside the
     activation block — ONLY when every configuration the algorithm judges keeps the use
     case's own routes and placements (it changes start times alone). The derivation
     describes the ONE route and placement the use case fixes per stream and switches a
     check off exactly where it fixes them, so for any other configuration it leaves the
     algorithm's own routes unchecked.
   Both checks assume ONE LISTENER PER STREAM: flow conservation charges
   |in-degree - out-degree| at every switch, so a multicast tree is rejected where it
   branches. Whenever either is True, the use case step states the PRECONDITION
   len(usecase.streams[i]['lstnsrPssblPlcmnt']) == 1 for every i.
   weights stays None unless a requirement names a trade-off between penalties.

3. Fill the `representation` section of your answer. It is a section of YOUR ANSWER, not
   a library object, and none of its field names is a library member: it records how
   schedules and configurations are held as data in this design.

   `schedule` is the PART 2 form id that is canonical for this design — the one the
   algorithm's inputs are in. Read "WHAT IT CANNOT DO" before choosing: a requirement
   needing something a form cannot express is a reason to pick another form, or to carry
   two. If you carry two, put the other ids in `also_used` - every form `schedule_reason`
   names, stated as used, not "if needed". `schedule_reason` is the ONE
   place that says how the forms are used: why the requirements need the canonical form;
   which PART 2 form each library function this design relies on is handed; which
   direction each conversion runs; every value a conversion needs from the
   environment (Schedule.from_compact_encoding needs the evaluator's windLengths and
   stream_intervals, and cycle_ns = problem.macro_cycle — the evaluator's own macro_cycle
   is always 0). Carrying two is a normal answer; leaving
   it implicit is not.

   `configuration_type` is the exact Python type of ONE CONFIGURATION of the algorithm —
   the value its own functions pass between each other, and the value each element of
   any sequence it produces holds — written so the blueprint can copy it verbatim. When
   every state the requirements imply fits the canonical form, write that form's id, the
   same string as `schedule`. When the requirements imply a state NO single form can hold
   (read each form's WHAT IT CANNOT DO), write a COMPOSITION of PART 2 forms that holds
   it, never a new form, as its Python type followed by what it holds. `configuration_type_reason` says only why: "fits the
   canonical form", or the requirement that forces the composition and the WHAT IT
   CANNOT DO limit that rules out the single form. How the composition is turned into the
   forms library functions take is already in `schedule_reason`; do not repeat it. This
   is the one place the type of a configuration is decided: the blueprint may not choose
   another.

4. List the environment objects in the order they must be obtained, each either a menu
   entry or handed in. Cover the use case, the scheduling problem, the activation of one
   on the other, and the constraint evaluator — in that order, because each needs the
   one before it — and then the algorithm's inputs. Write every library call a step
   makes as the call itself (problem.use_instance(usecase), the evaluator constructor):
   the library members your environment uses are read off those calls, so a call
   described in words is a member the next stage never hears of.
   - The algorithm's INPUTS (every resource on a hasInput edge whose subject is the
     algorithm) are obtained "from the caller", unmodified, in the form step 3 chose.
   - The USE CASE those inputs are defined against is obtained "from the caller" too: a
     schedule's rows and columns are the port and stream ids of ONE use case, and
     generate_scheduling_usecase / instantiate build a NEW random use case whose ids do
     not match. Use them only when the request has no such inputs.
   - The PROBLEM is obtained "from the caller": a TSNScheduling the caller builds (menu:
     problem_definition) and hands in NOT activated. It holds no use case, and with the
     use case handed in its problem_definition path is never read — only
     generate_scheduling_usecase / instantiate read it — so nothing in the design may
     depend on that file's values. A condition the design relies on (one listener per
     stream, say) is a PRECONDITION on the handed-in use case: state it in that step
     exactly, written as code, so the blueprint designs to comply with it — never as a
     value quoted from a scenario file.
   - The ACTIVATION is `with problem.use_instance(usecase):`, opened ONCE by the
     algorithm's top-level function around all its work; use_instance is not reentrant.
   - The EVALUATOR is constructed inside that activation as step 2 settled, or is
     problem.evaluator only when the encoding is PRIORITY.
   In each step, "type" is the object's real library class (Usecase, TSNScheduling,
   InstanceContext — what use_instance returns — TSNConstraintEvaluator) or, for a schedule, the PART 2 form id it is in — for a
   handed-in schedule, the same string as `representation.schedule`: the blueprint
   matches its structures to these strings exactly, so write them exactly.

5. Claim a requirement id in `requirement_coverage` only when the environment genuinely
   settles it: an input obtained from the caller settles the requirement that names it as
   an input, and nothing else.

MISTAKES REAL ANSWERS TO THIS PROMPT MADE, AND THEIR CORRECTION. Each is a pattern, shown
on a request other than the one below: check your answer against every one before you
return it.

A. The evaluator built the wrong way.
   WRONG  "TSNConstraintEvaluator(usecase, scheduling_type=...)"           - the use case
          "TSNConstraintEvaluator(scheduling_type=..., check_routing=True)" - no problem
          "problem.initialize_problem(scheduling_type=...)"                  - takes no argument
   RIGHT  "TSNConstraintEvaluator(problem, check_routing=True, check_task_placement=True,
          scheduling_type=ScheduleEncodingType.START_TIME)"

B. A state the requirements imply, missing from the configuration type.
   Request: "... then add the new streams one at a time" (a stream that is in the finish
   schedule only, so it has no route until it is added).
   WRONG  "configuration_type": "candidate_array",
          "configuration_type_reason": "fits the canonical form"
          - written without checking each step's state: an all-zero column is read as a stream that exists and has no route, so the
          added stream's state before it is added cannot be held (WHAT IT CANNOT DO).
   RIGHT  a type that holds the state - a composition of PART 2 forms, or the one form
          that can hold it - with a reason that cites the id of the requirement in YOUR
          ledger that implies the state, and the limit: "<that requirement's id> adds
          streams that are in only one schedule; candidate_array cannot mark a stream
          absent"
   The same holds for every other state a step implies: a step that sends one stream
   over two paths at once ("replicate", "keep the old path while the new one runs")
   puts it on two routes, and one candidate_array column holds one route.

Requirements ledger:
{requirements_ledger}

Library members available to this design (each carries "method_type" and "code" — the real
source; read it before deciding how an object is obtained):
{context_str}

Output rules:
- Return ONLY a valid JSON object, no preamble, no markdown fences, no extra text.
- Use exactly this shape:
{{
"environment": [
  {{"name": "usecase", "type": "Usecase", "obtained_by": "from the caller; then every PRECONDITION the design relies on, written as code", "reason": "..."}},
  {{"name": "problem", "type": "TSNScheduling", "obtained_by": "from the caller, not activated", "reason": "..."}},
  {{"name": "activation", "type": "InstanceContext", "obtained_by": "with problem.use_instance(usecase): opened once by the top-level function around all its work", "reason": "..."}},
  {{"name": "evaluator", "type": "TSNConstraintEvaluator", "obtained_by": "the whole constructor call as step 2 writes it, inside the activation", "reason": "the source of each flag, naming the requirement"}},
  {{
    "name": "one entry per input, named as the algorithm refers to it",
    "type": "the PART 2 form id it is in",
    "obtained_by": "concretely how it is obtained: 'from the caller, unmodified', or the menu entry's construction",
    "reason": "why it is obtained that way, naming the menu entry or the hasInput edge it comes from"
  }}
],
"representation": {{
  "schedule": "the PART 2 form id canonical for this design",
  "schedule_reason": "why the requirements need it; which form each library function this design relies on is handed; each conversion's direction and the environment values it needs",
  "also_used": ["the id of every other PART 2 form schedule_reason names - a form a library function is handed, or a conversion produces; [] only when schedule_reason names no other form"],
  "configuration_type": "the canonical form id, or a composition of PART 2 forms: its Python type, then what it holds",
  "configuration_type_reason": "each state the requirements imply for a configuration, with the requirement id that implies it and whether the canonical form can hold it (WHAT IT CANNOT DO); then 'fits the canonical form' only when it holds every one, else the composition's reason",
  "encoding": "START_TIME or PRIORITY",
  "encoding_reason": "the one-line reason the requirements force it"
}},
"requirement_coverage": [
  {{"requirement_id": "<the id of a requirement the environment settles>", "how": "one line on what the environment settles for it"}}
]
}}

JSON:"""

# Stage 0c prompt: takes the requirements ledger (R1..Rn) and produces ONE unified
# design for the WHOLE algorithm request — the piece that was missing when the earlier
# per-sub-query GENERATE_ALGORITHM pathway generated three independent, uncomposed
# functions (no shared data representation, the orchestrator never called its own
# sibling functions, and nothing checked the result against what was actually asked
# for). A single blueprint with a shared data model, an explicit call graph, and a
# requirement->function map is what makes those failure modes structurally impossible
# instead of just reviewed-for after the fact.
ALGORITHM_BLUEPRINT_PROMPT = """\
You are a software architect for a TSN (Time-Sensitive Networking) codebase.

You are given:
- a requirements ledger (R1..Rn) for an algorithm the user wants designed and implemented,
  with a graph of the resources the requirements are about;
- the configuration chosen by the previous stage (the environment the algorithm runs in);
- the library members available to this design (the candidates for reuse).

Produce ONE design for the whole request (not one design per requirement) that covers every
requirement in the ledger. The design has five sections: data structures, function
inventory, requirement map, reuse decisions and resource use.

1. DATA STRUCTURES

   List the structures that the functions of your design pass to each other. A structure
   says what a value IS. What the algorithm DOES with it belongs in a function's purpose,
   never in a structure.

   For each structure give:
   - "name": the name every function uses for it;
   - "type": a Python type concrete enough to write down (np.ndarray with shape and dtype,
     tuple[int, ...], dict[int, <a record of yours>], a record with named fields and their
     types, a Literal of exact strings);
   - "why": which requirements need it;
   - "used_for": what the functions do with it;
   - "corresponds_to": the library value or class this structure is, or null when the
     codebase has nothing like it.

   Rules for structures:
   - Define each structure once. Every function uses that same structure; no function
     has its own version of it (for example, one function treating a schedule as an array
     while another treats it as a dict of paths).
   - A "type" holds no rules. Words like "sorted by", "must equal", "only when" describe
     behaviour and go in the purpose of the function that does it.
   - null in "corresponds_to" means "the codebase has no such value". Use it only when that
     is true.
   - Every structure is read by at least one function, except the top-level function's
     return value and what it contains (the caller reads those).
   - Every structure is written by at least one function, except a structure that the
     configuration hands in from the caller (its "corresponds_to" names that library type).
   - When the request asks for a sequence the algorithm produces, each element of the
     sequence is one record holding both the value and what produced it. Do not use two
     parallel lists (values in one, labels in the other). The top-level function's return
     value must contain that sequence.
   - Every type used in any function signature is one of: a structure in this list, a
     library class named in this prompt, or a Python built-in. A name in a signature that
     is none of these does not exist when the code runs.

   These names are TAKEN by library classes: Schedule, Slot, Usecase, TSNScheduling,
   TSNConstraintEvaluator, UsecaseGenerator. Use one only when you mean that exact
   class. A structure of your own that is NOT that class gets a different name, however
   similar its role: a candidate schedule you define gets a name of its own, never
   `Schedule`, because the frozen interface imports the real `Schedule` and your field
   would then claim the library's list-of-Slot class while holding something else.

   Four structures are ALREADY FIXED by the codebase. Where your design needs one, name
   the existing type and point `corresponds_to` at it — never invent a parallel version:
   - a candidate schedule is a 2-D numpy int array indexed [port, stream], 0 meaning
     "this port is unused by this stream"; not a dict of paths. THREE LIMITS, and where a
     design needs past one, the configuration's configuration_type already says how: ONE value per (port, stream), so a stream
     cannot hold two windows on one port (to_compact_encoding raises rather than
     truncating), and the evaluator judges each column as one path (flow_conservation,
     looping); start times ONLY, no duration or period — hence from_compact_encoding
     taking windLengths and stream_intervals separately; and no way to mark a stream
     ABSENT — a stream that is not there is an all-zero column, which the
     evaluator reads as a stream with no route (talker_not_fixed, listener_not_fixed),
     so feasible_streams is False for it.
   - a compiled schedule is a schedule.Schedule, a list SUBCLASS holding schedule.Slot
     objects directly. Schedule.from_compact_encoding / to_compact_encoding convert
     between it and the array above.
   - a use case is a usecase_generator.Usecase; the problem state around it is a
     tsn_scheduling_problem.TSNScheduling.
   - a port is a 1-BASED integer id; id 0 is reserved and maps to the empty string.
   Usecase and TSNScheduling both expose "ports" and "streams", keyed DIFFERENTLY: on
   Usecase by port NAME, with pathLists of names; on TSNScheduling by port ID, with
   pathLists of ids. Say which one you mean wherever your structures say "ports" or "a
   stream path" — leaving it open produces functions that cannot be composed.

   What a schedule and a configuration are has already been decided by the configuration
   (given further below). Do not change it:
   - Its `representation.schedule` and `representation.also_used` name the schedule forms.
     Every schedule structure has one of those forms.
   - Its `representation.configuration_type` is the exact Python type of one
     configuration. The structure holding a configuration has that type, copied verbatim.
   - A structure that is one stream's part of a schedule (candidate_array[:, i] or
     Schedule.slots_by_stream(i)) corresponds to that part of the `schedule` form.
   - A stream passed around by position is the index i: type int, corresponds_to "the
     index i into Usecase.streams". It is the same i as candidate_array column i and
     Slot.stream_id.
   - For each object the configuration's environment steps obtain, the structure for it
     has "corresponds_to" equal to that step's "type", copied.
   - The structures you define yourself are the ones the configuration does not cover,
     written in terms of those forms and that configuration type.

   When a reused library function takes one of your structures as an argument (for
   example, TSNConstraintEvaluator built from a TSNScheduling), read that function's source
   in the candidates below. If it reads attributes off the argument, your structure must
   provide them.

2. FUNCTION INVENTORY

   List every function your design writes. A library member it uses is not listed here;
   it goes in the reuse decisions (section 4). For each function give:
   - "name";
   - "signature": parameter names and types, and the return type. Types follow the rule
     in section 1. Each parameter a function's callees need from it is one of its own
     parameters, or it builds it;
   - "purpose": what it does, specifically: each case it handles and what it does in that
     case, what it checks, and what it does when no case applies. The implementation stage
     gets only this, so a vague purpose gives a vague implementation;
   - "calls": the other functions of this inventory that it calls;
   - "reads": the structures (by their names in section 1) it takes as parameters, reaches
     through a parameter, or gets back from a call;
   - "writes": the structures it returns or builds, including one it gets from a library
     call or reads off an input. An empty "writes" means it produces nothing.

   Rules for functions:
   - Every function is reached from the top-level function through "calls". A function
     nothing calls is an error.
   - The top-level function's "calls" includes every function whose result it uses.
   - "reads" and "writes" use only names from section 1: never a parameter name, never an
     environment step's name.

   ONE SIGNATURE RULE, and it decides whether this design can run at all. A
   TSNScheduling is a VIEW over a use case, not a self-contained object: it holds no
   use case of its own, and every read-only view on it (n_ports, n_streams, ports,
   streams, macro_cycle, port_id_to_name, ...) raises ValueError until one is
   activated. So:
   - Any signature declaring a TSNScheduling must ALSO declare its Usecase. The problem
     object cannot yield one: no attribute holds the current use case, and regenerating
     from the config file gives a DIFFERENT random scenario — same topology, freshly
     randomised routes and talkers — whose port and stream ids do not match the
     schedules in hand. Pass only the problem and nothing on it can legally be read, so
     the implementation invents an attribute rather than admit that.
   - Exactly ONE function opens the activation block: the top-level orchestration, around
     the work it delegates. Functions called inside it read problem.n_ports and the rest
     DIRECTLY and open no block of their own — leaving a nested one resets the use case
     to None and the outer block's remaining work raises. Say so in the purpose of any
     helper that relies on it ("called with an active use case").

   Named states and derived values:
   - A field holding one of a fixed set of states has a Literal[...] type listing the exact
     strings the code will use. When the states describe the data, two states must leave
     the data different. If two would leave it identical, they are one state: drop one, or
     name the library field that holds the difference. Do not reuse an unrelated field to
     tell them apart.
   - A derived number or named condition is defined once, as "<Name> = <expression>", in
     the purpose of the one function that computes it. Every value in the expression is
     reachable from that function's parameters. Other functions use the value by its name
     and never restate or re-weight it.

3. REQUIREMENT MAP

   For each requirement id in the ledger, list the functions of the inventory that satisfy
   it. Leave out the ids the configuration already satisfies (listed below): they do not
   appear in the map at all, not even with an empty list. Every other id appears with at
   least one function. If an id has none, add a function or extend one.

4. REUSE DECISIONS

   For each codebase function or class used anywhere in your design, give:
   - "codebase_function": the candidate's "name", copied exactly;
   - "class_name": the candidate's "class_name" ("sub_class_name" when it is nested), or ""
     for a module-level function, copied exactly;
   - "used_by": the inventory function that uses it;
   - "reason": why this candidate does what that function needs.
   Copy names exactly as written in the candidate list: do not join them into a dotted
   name, do not fix spelling or casing, and do not name anything not in the list. Do not
   write signatures or call forms; the pipeline looks those up in the real source. If no
   candidate does what a function needs, the function is implemented new and has no reuse
   decision.

5. RESOURCE USE

   One entry per resource in the ledger, in the ledger's order, each exactly once:
   - "resource": its name, copied exactly;
   - "used_as": the structures, functions or values of your design that carry it. Write a
     field as Structure.field and an enum value as Type.VALUE (Type is the enum's owner,
     also when the enum is a field). A resource may be carried by several elements, and one
     element may carry several resources;
   - "reason": empty when "used_as" names something; otherwise one line on why the design
     needs nothing for it.

Library facts this design rests on (the pipeline looks every reused call's real signature up
after you answer; these are what the DESIGN must respect):
- evaluator.TSNConstraintEvaluator is built FROM a TSNScheduling, as the configuration below
  says; do not add a second one. A START_TIME cell of its array is a transmission start time
  PLUS ONE. evaluator.feasible_streams(array) returns one bool per stream column.
- schedule.Slot holds start_ns, end_ns, period_ns, stream_id, port_id, queue_id, index.
  Schedule.from_compact_encoding(array, windLengths, stream_intervals, cycle_ns=...,
  basetime=0) builds a Schedule from an array. validate_conflicts(touching_allowed=True)
  returns the list of overlapping window PAIRS per port, by time alone, whatever stream they
  belong to - so a window listed twice conflicts with itself; an empty list means none.
- The evaluator's macrocycle_overlap check counts two windows on one port as colliding when
  (phase1 - phase2) mod gcd(period1, period2) < window1 + window2, in either direction. This is
  stricter than Slot.conflicts_with and Schedule.validate_conflicts, which report only windows
  that actually overlap.

CONFIGURATION (already decided; treat it as fact)
The previous stage produced this from the same ledger, and it has been checked against the
real library. The objects it lists exist and are obtained the way it says, and the encoding
it names is what a schedule cell means everywhere in your design. So:
- The use case, the problem and every other object it obtains "from the caller" are
  parameters of the top-level function. The top-level function activates the use case and
  builds the evaluator exactly as the environment steps say. No other function sets up
  anything.
- Every function that touches a schedule uses this encoding.
- Every state named in its configuration_type_reason is a state your design produces.
{configuration_json}

Requirements already satisfied by the configuration, and so removed from the ledger below:
{configuration_satisfies}. Add no functions for them.

Requirements ledger:
{requirements_ledger}

HOW THE LEDGER'S GRAPH MAPS ONTO YOUR DESIGN. The ledger lists RESOURCES (the things the
requirements are about, each with an "is" line saying what it is in the codebase) and
TRIPLES (subject, predicate, object) saying how they relate. Each kind of line asks for
something in your design; ignoring one drops a requirement.

  a resource         Its "is" line says what it is in the codebase. If your design has a
                     structure for it, that structure's corresponds_to is the "is" line,
                     copied; a resource whose "is" says it must be BUILT has corresponds_to
                     null. Use its name as fits your design: as a structure, a function (in
                     snake_case), an enum value, merged into another element, or not at all.
                     Record the choice in RESOURCE USE.

  a resource's note  Says what the resource is and where its value comes from (which input,
                     which schedule). The structure's "why" and the function that writes it
                     must agree with that. If the note says the query leaves the word
                     undefined, your design defines it: a Literal type listing its cases, or
                     a "<Name> = ..." expression in the purpose of the function computing it.
                     The note never overrides the "is" line.

  hasInput           The object is a parameter of the function doing the subject's work, and
                     is in that function's "reads".
  hasOutput          The object is returned by that function, and is in its "writes".

  hasStep            Each object is an action, and each action is its own function (not a
                     comment inside a bigger one). The subject says whose steps they are:
                     a strategy's steps are called by that strategy's function, not by the
                     top-level function.

  precedes           The subject runs before the object. The design must not allow the other
                     order.

  precondition       The object must hold before the subject runs. There is a function that
                     checks it, and the subject does not run when the check fails.

  mustSatisfy        The subject must have the quality named by the object. When the object
  minimizes          is a resource to be built, a function of yours computes it and lists it
                     in "writes".

  criterion          The subject is chosen by ranking candidates on the object. A function
                     scores the candidates and orders them. Two criterion triples with
                     different subjects are two separate rankings.

  isA                The subject is one kind of the object. The design has a function that
                     chooses among the kinds, and handles the other kinds too.

  appliesWhen        The subject is used only in the situation the object names. The
                     function that chooses the subject tests that situation first.

  transformsFrom     The first element of what the subject's function produces is the object.
  transformsInto     The last element of what the subject's function produces is the object.
  mustEndAt          Same as transformsInto: the last element equals the object.

  partOf             The object's type contains values of the subject's type, directly or
                     inside a record it holds.

  determines         The object is written by the function that realises the subject, or by a
                     function it calls.

  builtBy            The subject is produced by the object, which is the top-level function.

Library members available to this design (the candidates for reuse). Each carries
"method_type" and "code" (the real source): read the code to confirm what a member does
before reusing it. Reuse a member only where it does what one of your functions needs, as it
is; where none fits, write the function new.
{context_str}

MISTAKES EARLIER ANSWERS TO THIS PROMPT MADE. Each is shown as a pattern with placeholders.
Check your design against every one before answering.

A. A name from the wrong list in "reads"/"writes".
   WRONG  "reads": ["<an environment step's name>"], while the structure for that input has
          another name
   RIGHT  "reads": ["<the structure's own name from section 1>"]

B. The configuration's calls paraphrased instead of copied.
   WRONG  top-level purpose: "... opens the activation block and constructs the evaluator"
   RIGHT  the top-level purpose quotes both calls: "<the activation step's obtained_by> ...
          <the evaluator step's obtained_by: the whole constructor call, copied>". The
          implementation builds exactly what this line says.

C. A configuration with several parts whose parts no function defines.
   WRONG  the top-level function starts with <a part> empty, while <a strategy function> leaves
          one entry in it when it finishes, so the last configuration never equals the finish
   RIGHT  when configuration_type has more than one part, one function builds a configuration
          from an input, and its purpose says what each part holds in every state, including
          when a part is empty: "<build function>(<input>)": "<part one> holds <...>; <part
          two> holds <...>, and is empty when <...>". Every other function that makes a
          configuration calls it, or fills each part the way that purpose says.

D. A step or a check that is not a function.
   WRONG  "function_inventory": [{{"name": "<the top-level function>"}}], with the hasStep
          objects written as enum values
   RIGHT  each hasStep object is its own function, and each precondition its own checking
          function, called by the function that carries out the strategy

E. A plan step that changes nothing, or two that make the same change.
   WRONG  <the step that keeps the old part> "emits the configuration unchanged", then <the
          activating step> and <the replicating step> each emit the same state
   RIGHT  a plan step is a change of the data. A hasStep that keeps something as it is, is a
          check. hasSteps that make the same change emit one step together. Every step
          differs from the one before it, and no closing step repeats the last change.

F. A strategy of several changes that returns only one.
   WRONG  <the function that carries out a strategy> returns one step, although the strategy
          makes two changes
   RIGHT  it returns all of them: its return type is a sequence of steps, one per change, in
          order

G. A result the caller uses but does not list.
   WRONG  <a checking function> "writes": ["<the condition it decides>"], and its caller's
          "reads" leaves that condition out
   RIGHT  the caller's "reads" includes "<the condition it decides>": a value a function gets
          back from a call is read by that function

H. An enum value written with its field.
   WRONG  "used_as": ["<Owner>.<field> <VALUE>"]
   RIGHT  "used_as": ["<Owner>.<VALUE>"]

I. A call that needs what its caller does not have.
   WRONG  <a function that applies a move>, taking only the configuration and the item,
          "calls" <a check that needs the evaluator>, so it cannot make that call
   RIGHT  every function receives every value its callees need, as a parameter or inside one
          context record (the evaluator, the sizes, the inputs). Compare each "calls" entry's
          signature with the caller's parameters.

J. A type in a signature that is not defined anywhere.
   WRONG  "signature": "def <f>(evaluator: <a requirements-stage name for the evaluator>, ...)"
          while no structure in section 1 has that name
   RIGHT  "signature": "def <f>(evaluator: TSNConstraintEvaluator, ...)", the library class,
          or a structure of section 1 whose corresponds_to names it

Output rules:
- Return ONLY a valid JSON object, no preamble, no markdown fences, no extra text.
- Use exactly this shape:
{{
"data_structures": [
  {{
    "name": "<one entry per schedule input the configuration obtains from the caller, named as its ledger resource>",
    "type": "<the exact Python type of that form, e.g. for candidate_array: np.ndarray, shape (n_ports, n_streams), dtype int>",
    "why": "the input it is, naming its hasInput requirement",
    "used_for": "what the functions read off it",
    "corresponds_to": "<the same string as that environment step's type>"
  }},
  {{
    "name": "<the structure holding ONE configuration>",
    "type": "<the configuration's representation.configuration_type, copied verbatim>",
    "why": "what one configuration holds",
    "used_for": "what the functions pass between each other, and each element of the sequence returned",
    "corresponds_to": "<the form id when configuration_type is one, else null>"
  }},
  {{
    "name": "<the structure the top-level function returns>",
    "type": "<a record holding what the request asks to be returned>",
    "why": "the result the request asks for",
    "used_for": "what the caller reads off it",
    "corresponds_to": null
  }},
  {{
    "name": "<every set of named states, condition or score the design uses, and every value a function writes>",
    "type": "<the value it holds: Literal[...] of the exact state strings, bool for a condition, float for a score>",
    "why": "the requirement it decides",
    "used_for": "who reads it and what they do with it",
    "corresponds_to": null
  }},
  {{
    "name": "<a structure of your own, named for what it holds>",
    "type": "<its exact Python type, e.g. np.ndarray, shape (n_ports, n_streams), dtype int>",
    "why": "why the requirements need it, coherently with the whole algorithm",
    "used_for": "what it will actually be used for",
    "corresponds_to": "the library value/type this IS, or null if genuinely new"
  }}
],
"function_inventory": [
  {{
    "name": "<the top-level function>",
    "signature": "def <name>(<every object the configuration obtains from the caller>) -> <what it returns>",
    "purpose": "opens `<the activation step's obtained_by, copied>`, inside it builds `<the evaluator step's obtained_by: the whole constructor call, copied>`, then <the work it delegates>",
    "calls": ["..."], "reads": ["..."], "writes": ["..."]
  }},
  {{
    "name": "function_name - one this design writes; a library member it calls is a reuse_decisions entry, never a function_inventory entry",
    "signature": "def function_name(arg: Type, ...) -> ReturnType - every type is a data_structures name, a library class or a built-in; every value the functions in its 'calls' list need from it is a parameter here (or is built here): compare each callee's signature with this one",
    "purpose": "what it does, specifically: each case it handles and what it does in it, what it checks, and what it does when no case applies - the implementation stage is given only this, so a generic line is implemented generically",
    "calls": ["other_inventory_function_name", "..."],
    "reads": ["data_structures names this function consumes - never a parameter's or an environment step's own name"],
    "writes": ["data_structures names this function produces - every structure it returns or builds, including one it gets from a library call or reads off an input (a stream's column, a metrics result)"]
  }}
],
"requirement_map": [
  {{"requirement_id": "R1", "functions": ["function_inventory names only - a structure or an enum value is never one"]}},
  {{"requirement_id": "<a ledger id on how the code is built>", "functions": ["the functions that carry out that directive"]}}
],
"reuse_decisions": [
  {{
    "codebase_function": "the candidate's own \\"name\\", copied exactly",
    "class_name": "the candidate's own \\"class_name\\", copied exactly, or \\"\\" if it is module-level",
    "used_by": "function_name from function_inventory",
    "reason": "why this candidate satisfies the need"
  }}
],
"resource_use": [
  {{"resource": "<a ledger resource>", "used_as": ["<the structure, function or value that carries it>"], "reason": ""}},
  {{"resource": "<a ledger resource>", "used_as": [], "reason": "<why the design needs nothing for it>"}}
]
}}

JSON:"""

# Stage 0d prompt: INTERFACE FREEZE. Takes the blueprint's data model and function
# inventory and emits them as REAL, COMPILED Python — types as dataclasses, functions
# as complete `def`s with exact signatures and a `raise NotImplementedError` body —
# before any implementation is written. The point: once this compiles, every later
# stage (implementation, tests) reads the literal frozen code instead of re-deriving
# types/signatures from the blueprint's prose independently, which is exactly how the
# earlier per-sub-query pathway ended up with three functions that silently disagreed
# on what a "schedule" was.
INTERFACE_FREEZE_PROMPT = """\
You are freezing an interface for a TSN (Time-Sensitive Networking) algorithm design,
before any function body is implemented.

You are given the approved DATA MODEL and FUNCTION INVENTORY from a design blueprint,
plus the REQUIREMENT MAP that ties requirement ids to the functions that satisfy them.
Emit ONE complete Python source file that:

1. Brings every data structure into the file exactly once, in the order listed below,
   by ONE of these three routes — never more than one, and never the wrong one. A
   structure whose corresponds_to names a library CLASS takes route (a); one whose
   corresponds_to names a library value that is not a class — a schedule form such as
   candidate_array, a library dict or one key of it written as code such as
   Usecase.streams[i]['pathLists'], or a composition of those — takes route (c); one
   with corresponds_to null takes route (b):

   a. THE TYPE ALREADY EXISTS IN THE CODEBASE -> IMPORT IT. These are real classes the
      library already defines, and a data model entry describing one of them is naming
      it, not asking for a copy of it:
          from schedule import Schedule, Slot
          from usecase_generator import Usecase, UsecaseGenerator
          from tsn_scheduling_problem import TSNScheduling
          from evaluator import TSNConstraintEvaluator, ScheduleEncodingType
      Never redefine one of these as a local `@dataclass`, never alias it to `Any` or
      `object`, and never give it a same-named stand-in "wrapper". A local definition
      SHADOWS the import that sits above it, so every real method and attribute of that
      class — the instance-activation context manager, the port counts, the evaluator's
      checks — silently becomes unreachable for the rest of the file, and the functions
      below end up operating on an empty look-alike instead of the real thing.

   b. THE TYPE IS GENUINELY NEW TO THIS DESIGN -> define it here, in whichever of these
      two forms the shape actually calls for. This is the route for types the design
      invents, and only those.
        - A RECORD, i.e. a shape with named parts: a `@dataclass` (using
          `from dataclasses import dataclass` and `from typing import ...` as needed)
          with one field per named part of that shape, using concrete type annotations.
        - NOT a record — a tuple of something, a fixed set of string states, an alias
          for a built-in: a type alias instead, which is the honest form for it.
          Do not wrap a non-record in a one-field dataclass to satisfy a rule:
              <APath>: TypeAlias = tuple[int, ...]
              <ASetOfStates>: TypeAlias = Literal["EXACT", "LITERAL", "VALUES"]
              <AnOrder>: TypeAlias = tuple[<ARecord>, ...]
          Where the data model states the allowed values of such a type, the
          `Literal[...]` must list exactly those, with no additions and no changes of
          casing.

   c. THE CODEBASE HAS THE VALUE BUT NO CLASS FOR IT -> a type alias of the data
      model's "type", with a comment naming what it corresponds to. There is nothing to
      import and nothing to invent: the alias writes the data model's own type down as
      Python (with `import numpy as np` and `from typing import TypeAlias`):
          <ASchedule>: TypeAlias = np.ndarray  # candidate_array, (n_ports, n_streams), int
          <AComposition>: TypeAlias = <its Python type>  # what it holds
      Never make one of these a `@dataclass`: the library hands such a value around as
      the plain array, dict or list it is, and a wrapper would not be that value.

   IF a type alias uses another type alias of this file, THEN the alias it uses is declared
   first, even when the list below names them in the other order, because a type alias runs
   when the file is loaded, and a name that is not defined yet stops the file from loading.
   WRONG  <a mapping alias>: TypeAlias = dict[int, <an alias declared further down>]
   RIGHT  <the alias it uses> declared above it, and every other structure in the list's order

2. Defines every function in the function inventory as a real, complete `def` — exact
   name, exact parameter names and order, with type annotations drawn from the data
   model types you just defined or imported (or a plain built-in type where the
   blueprint's own signature already uses one) and a return type annotation. Do NOT
   change, drop, add, or reorder any parameter relative to the blueprint's signature
   for that function.

3. Gives each function a docstring with these lines, in this order — the first four
   always, the fifth only where rule 5 calls for it:
   "Purpose: <the function's purpose from the inventory, copied IN FULL - every case, check,
   formula and fallback it states, never shortened or reworded; the implementation stage is
   given nothing else about what the body must do>"
   "Satisfies: <the ids in the function's own "satisfies" list, comma-separated, copied
   exactly, or 'none' if the list is empty>"
   "Calls: <comma-separated names of the other inventory functions this function
   calls, taken from its "calls" list, or 'none' if it calls none>"
   "Flow: <the TYPE of each parameter, in order, comma-separated> -> <the return type>"

   The Flow line names TYPES, never parameter names, and every name in it must be one
   of the data model types above, a type imported from the library, or a plain built-in.
   It must agree exactly with the signature written in 2 — the same types, in the same
   order. Restating it is the point: a signature and a flow line are written from
   different angles, so when they disagree it means the design of that function is not
   settled in your own head, and this is the one place that disagreement is visible
   instead of silent. Write the flow line from what the function DOES, then check it
   against the signature; if they differ, fix whichever one is wrong before continuing.

4. Gives every function EXACTLY ONE statement as its body: `raise NotImplementedError`.
   No other code, no partial logic, no explanatory comments, no `pass`, no return
   value — the interface is being frozen, not implemented.

5. Carries every entry of the CONSTANTS section into this file, by the route its kind
   calls for. These values exist because the implementation and its tests are generated
   SEPARATELY from this file: a literal or a definition that reaches neither is one both
   of them invent for themselves, and they will not invent the same one.

   a. kind "enum" -> the `Literal[...]` of the type it names, listing exactly its
      declared values in exactly their declared casing.
        - if its "name" is a data model TYPE, that type's alias IS the Literal:
              <ASetOfStates>: TypeAlias = Literal["<STATE_A>", "<STATE_B>"]
        - if its "name" is `Type.field`, do NOT create a new binding. Annotate that
          field with the alias that already carries those values:
              <field>: <TheAliasCarryingItsValues>        # not `<field>: str`

   b. kind "formula" -> a FIFTH docstring line on the function its "computed_by" names,
      verbatim, as the last line of that docstring:
              Computes: <name> = <definition>
      A formula is not a value yet — no function body exists in this file — so it has no
      module-level form. It belongs to the function that will produce it, where the
      implementation reads it while writing that body and the tests read it while
      deciding what to assert.

   HARD CONSTRAINT on both routes: a constant's name is written for a reader
   (`<Record>.<field>`, `<a quality in words>`) and is usually NOT a valid Python
   identifier. Never force one into a binding. In particular never write
   `globals()["some name"] = ...`, and never assign a definition as a STRING — a
   formula stored as prose in a variable is still prose, and it pollutes the module
   namespace besides. If a constant fits neither route above, leave it out of the code
   and say nothing about it.

Rules:
- Every structure in the data structures and every function in the function inventory
  must appear in the output. Nothing may be silently dropped.
- Do not invent a type, field, or function that is not present in the blueprint below.
- Every type alias whose corresponds_to is not null carries it as a comment on the same
  line, copied (`<AnIndex>: TypeAlias = int  # the index i into Usecase.streams`): the
  Python type alone does not say which value it holds.
- Where the data model's "type" has a description after its Python type ("<python type>: <what
  it holds>"), the alias writes the Python type and keeps that description, complete, as its
  comment - also when corresponds_to is null: the description is the only place that says what
  the value holds (which keys, what each one maps to, which states it covers).
- The file must be complete, self-contained, and syntactically valid Python — every
  import it needs must be present at the top.

Data structures (each carries name, type, why, used_for, corresponds_to, written_by,
read_by — a corresponds_to naming a library class means IMPORT it, route (a); one naming
a library value that is not a class means a type alias of its type, route (c); null means
define it, route (b)). Each function carries its own reads and
writes over those structures:
{data_model_json}

Constants:
{constants_json}

Function inventory:
{function_inventory_json}

Requirement map:
{requirement_map_json}

Output rules:
- Return ONLY the Python source. No JSON, no markdown fences, no prose, no
  restatement of the blueprint.

Frozen interface:
"""

CLASSIFICATION_PROMPT = """\
You are a query planner for a TSN (Time-Sensitive Networking) coding assistant.

Route each query to exactly one knowledge module. These modules form a PIPELINE and
deliberately share vocabulary (topology, ports, streams, latency, macro-cycle, fat-tree,
JSON). Do NOT route on the noun. Route on the LIFECYCLE STAGE / verb:

    CREATE the problem      -> usecase_generator
    SET UP / represent it    -> tsn_scheduling_problem
    SCORE a candidate soln   -> evaluator
    define a parameter       -> parameters

------------------------------------------------------------------------------
1. usecase_generator   (CREATE the problem instance from a config)
   Owns building a use case from scratch: topology, streams, and routing. Nothing is
   being solved or scored here yet.
   - Topology GENERATION: makeTopology, k_ary_fat_tree, makeRandomGraph
     (Waxman / Barabasi-Albert / Erdos-Renyi), changeDegree, set_link_data,
     generate_fattree_regions.
   - Stream GENERATION: streamGenerator, generateStream, add_stream, remove_stream
     (create streams with interval T, maxFramesPerInterval, frame size, maxLatency).
   - Talker/listener placement sampling: talkerEmbedding, listenerEmbedding,
     fixTalkers, fixListeners.
   - Routing / path GENERATION at build time: findAllPaths, shortestPath, randomPath,
     addRouting, removeRouting, pathConvenient (is a path feasible vs maxLatency at
     generation time), periodConvenient.
   - The Usecase object and its GRAPH-LEVEL views/details/distances: hosts, switches, links, ports (as
     graph attributes keyed by port NAME), window_length, distance_to_hosts.
   - RAW (de)serialization of a Usecase: to_file, to_json, from_json, topology2json,
     guess_config, get_config.
   - Drawing the BASE use-case topology: drawTopology, drawFattree.
   - This module also writes/saves/exports to JSON
    Route here when the user wants to build, generate, randomize, sample, or serialize a
    use case / topology / streams / routes. Never claim a function does not exist until a
    routed lookup and a cross-module fallback lookup both fail.
   NOT here:  per-stream pruned subgraphs, delay budgets, macro-cycle value -> tsn_scheduling_problem. 
   Checking a candidate schedule -> evaluator.

------------------------------------------------------------------------------
2. tsn_scheduling_problem   (SET UP and REPRESENT the problem for a solver)
   Owns the orchestration layer that turns a generated Usecase into a solvable problem:
   the active-instance lifecycle, solver-facing id mappings, per-stream port
   configuration / custom subgraphs, and problem-level delay/timing quantities. It is
   the bridge between usecase_generator and evaluator. It does NOT score solutions.
   - Instance lifecycle & metadata: TSNScheduling, instantiate, use_instance /
     InstanceContext, import_scheduling_usecase, generate_scheduling_usecase,
     generate_incremental_usecases, build_metadata, initialize_problem (constructs the
     evaluator).
   - Solver-facing INTEGER-ID mappings (distinct from usecase_generator's name-keyed
     views): port_id_to_name, port_name_to_id, port_id_to_node, node_to_port_id
     (egress_map / ingress_map / adjacent_egress_map), nodes, n_ports, n_streams,
     n_nodes, next_prev_hops.
   - Per-stream PORT CONFIGURATION and custom/pruned SUBGRAPH construction:
     configure_stream_ports (force-in / exclude ports), make_stream_config,
     make_default_stream_config, get_concurrent_ports, network_borders.
   - Problem-level DELAY-BUDGET and hop-cost computation on the problem (NOT on a
     candidate schedule): bfs_forward, bfs_backward, hops_costs, hop_cost_eg2eg.
   - macro_cycle as a VALUE (LCM of stream intervals); streams/links/ports enriched
     with port ids; _frame_size.
   - Drawing PER-STREAM custom/pruned graphs: plot_all_stream_custom_graphs,
     drawFattree (for stream subgraphs).
   Route here when the user asks about setting up / instantiating the problem, mapping
   ports or nodes to ids, restricting or pruning which ports a stream may use, building
   a stream's custom graph, computing delay budgets / hop costs, or the macro-cycle value.
   NOT here: generating the topology/streams/routes in the first place ->
   usecase_generator. Counting violations, overlap collisions, or metrics of a
   candidate chromosome -> evaluator.

------------------------------------------------------------------------------
3. evaluator   (SCORE a candidate solution / chromosome)
   Owns everything that takes an already-set-up problem plus a CANDIDATE solution
   (a chromosome matrix, priority- or start-time-encoded) and measures how good or
   feasible it is. If the query mentions a chromosome, a candidate/proposed schedule,
   violations, penalties, feasibility, or repair, it belongs here.
   - Encoding types: ScheduleEncodingType (PRIORITY vs START_TIME).
   - Per-stream CONSTRAINT VIOLATION checks on a chromosome:
     _per_stream_host_role_violations (task placement: talker/listener fixed, overlap,
     strays), _per_stream_flow_continuity_violations,
     _per_stream_flow_conservation_violations (backwarding / looping / flow
     conservation), _per_stream_priority_conflicts,
     _per_stream_latency_violations_priority / _start_time (cumulative path delay vs
     maxLatency OF A CANDIDATE), _per_stream_talker_offset_violations,
     _per_stream_macrocycle_overlap_violations (periodic collision test WITHIN the
     macro-cycle), _per_stream_window_sequence_violations.
   - Connectivity grouping helpers: group_connected_ports, talker_listener_groups,
     group_penalty, find / union (union-find).
   - Aggregation & feasibility: per_stream_all_violations, feasible_streams, evaluate.
   - Quality METRICS of a schedule: compute_metrics (congestion, load imbalance,
     utilization, average path length).
   - REPAIR guidance: suggest_corrections (removal / addition pressure, value-correction
     bias to fix conflicts).
   Route here for anything about evaluating, validating, scoring, penalizing, checking
   feasibility of, detecting collisions in, or repairing a candidate schedule.
   NOT here: producing the macro-cycle value or delay budgets of the problem itself ->
   tsn_scheduling_problem. Generating streams/topology -> usecase_generator.

------------------------------------------------------------------------------
FILE I/O — read this before routing ANY query that mentions a file, importing,
exporting, saving, or loading a use case. "Touches a file" is NOT a module
signal by itself: nearly every generation function reads a config file
somewhere in its call chain. What matters is WHICH KIND of file and WHICH
DIRECTION:

A) GENERATING a brand-new use case FROM a .ini-style CONFIG file (topology_config
   / streams_config settings: numhosts, portspeed, numofstreamsperinstance, ...).
   This is CREATION, not loading something that already exists.
   - usecase_generator owns the actual generation: generateUC, ucGenerator,
     empty_uc (all take a configFile and build a NEW Usecase from it).
   - tsn_scheduling_problem's instantiate / generate_scheduling_usecase /
     generate_incremental_usecases are orchestration wrappers around the above
     — they store the config path and forward it, then keep the result in
     self.instances. Route here when the query frames it as "for the problem"
     / "instantiate" / "keep it as an instance", not just "generate a use case".

B) IMPORTING a previously EXPORTED use-case JSON file (shape: nodes/links/
   streams/index — NOT a .ini config) to RESTORE an existing use case.
   - The ONLY function that opens such a file is
     tsn_scheduling_problem's import_scheduling_usecase(file_path,
     remove_routing=False, fix_task_placement=False). Any query about
     "importing" / "loading" / "reading in" a use case FROM A FILE routes here.
   - usecase_generator's Usecase.from_json and Usecase.guess_config do NOT
     read a file — both take an ALREADY-PARSED dict as their argument, and
     are internal helpers import_scheduling_usecase calls after it has
     already opened and parsed the file itself. Do not offer either of them
     as the answer to a query about importing FROM A FILE — neither one
     touches a path or calls open()/json.load.

C) EXPORTING/saving a use case to a JSON file.
   - usecase_generator's Usecase.to_file is the only writer (to_json /
     topology2json build the in-memory dict it writes but do not touch disk).
   - Note the read/write split is DELIBERATELY asymmetric with (B): loading a
     saved use case back in is tsn_scheduling_problem's job; saving one out
     is usecase_generator's job. Do not assume symmetry.
------------------------------------------------------------------------------

4. parameters
   - The following parameter names are the ones currently known: numhosts, numswitchs, portspeed, procdelay, propdelay, linkBw, 
   queuecountperport, graphConfig, numofstreamsperinstance, maxstrpernode, period_range, numberframes_range, datasize_range, 
   maxlatency_range, numplacestalker, numplaceslistener, numlistenersperstr, routing, fixPath, shortestPath, generation, incrementation, 
   expiration, finite_duration, start, stop, typ, k, prob, numcoreswitches, Minnodedegreepersw, minnodedegreepersw, numEdgN
   - meanings of parameters
 
Your task:
Analyze the user query and produce a reliable execution plan.
 
Classification rules:
- SIMPLE means the query requires only ONE module.
- COMPLEX means the query requires TWO OR MORE modules.
- Split COMPLEX queries into the minimum number of sub-queries.
- Each sub-query must map to exactly one module.
- Do not split just because there are multiple sentences instead split the query when user asks for
 actions from different modules.
- If the user asks what a parameter means, route to parameters.
- IMPORTANT — distinguish a CONFIGURATION parameter from a FUNCTION ARGUMENT.
  The `parameters` module is ONLY for configuration parameters in the Known
  parameters list below and above (e.g. "what does linkBw mean", "default range of
  num_switches"). IF A PARAMETER IS NOT FROM THOSE LISTS THEN DO NOT CLASSIFY THE QUERY AS "parameter".
  A function/method ARGUMENT — a name in a function's signature,
  such as the `scheduling_type` argument of `TSNConstraintEvaluator.__init__` — is
  NOT a configuration parameter. Questions about a function's arguments, their
  default values, its signature, or its return value belong to the CODE module
  that DEFINES that function, with action=EXPLAIN_CODE.
- If the query names a specific function, method, and/or class (e.g.
  `TSNConstraintEvaluator.__init__`, `compute_metrics`, `k_ary_fat_tree`. etc.), route to
  the module that OWNS it — even if the query uses the words "parameter", "default",
  or "argument".
  THIS RULE FIXES THE MODULE ONLY. It does NOT fix the action. Naming a function makes
  the action EXPLAIN_CODE only when the user is ASKING ABOUT that function. If the user
  is instead INSTRUCTING you to produce something and merely mentions the function to
  say what the result must fit, the action is DO. See the "action" rules below.
- If the user asks about units, data types, output formats, field meanings, or properties
  of a specific module's outputs (e.g. "what units does UsecaseGenerator use for time and
  bandwidth", "what does generateStream return"), assign action=EXPLAIN_CONCEPT and route
  to the module that owns those outputs (usecase_generator, tsn_scheduling_problem, or
  evaluator). Only route to the parameters module when the user is asking about the
  meaning or name of a configuration parameter (e.g. "what does linkBw mean").
 
- For EACH sub-query also assign an "action", exactly one of:
    * DO              — user wants something built or generated or computed. An action needs to be done.
                        Written as an INSTRUCTION to you, not as a question: "Implement
                        X", "Write X", "Add X", "Give me the code for X". A function or
                        class named inside such an instruction does NOT turn it into
                        EXPLAIN_CODE — it only tells you the module.
    * EXPLAIN_CONCEPT — a conceptual TSN question NOT tied to one specific function,
                        including questions about units, output formats, data types, or
                        properties of a module's output ("what is a fat-tree", "why does
                        TSN need gate control lists", "what units does UsecaseGenerator
                        use for time and bandwidth").
    * EXPLAIN_CODE    — explain or walk through a specific existing function, method AND/OR
                        class ("what does compute_metrics do", "explain k_ary_fat_tree"),
                        or point the user at which function to call ("what function can I
                        use to attach a schedule"). Written as a QUESTION.
  When torn between EXPLAIN_CODE and EXPLAIN_CONCEPT, prefer EXPLAIN_CODE.
  When torn between EXPLAIN_CODE and DO, decide by the FORM of the original query, not by
  which verbs appear in it:
    * A QUESTION — "what/which/how ... ?", "what functions can I use to generate X?" —
      is EXPLAIN_CODE, even when it contains words like generate, create or build.
    * An INSTRUCTION — "Implement X.", "Write X." — is DO, even when it names a function.
- action is independent of module routing and of SIMPLE/COMPLEX. The SAME module may
  appear in two sub-queries when the user wants different actions on it (e.g. generate
  code in a module AND explain a function from that module).
 
- WRITING EACH SUB-QUERY'S "text" (IMPORTANT):
  * Each "text" must be a self-contained, standalone question that can be fully
    answered using ONLY the routed module's knowledge. Someone reading the "text"
    alone — without the original query — must understand exactly what is being asked.
  * Do NOT copy the original user query into "text". Rewrite it in your own words.
  * The "text" values of different sub-queries MUST differ from one another. Never
    emit identical or near-identical text for two sub-queries.
  * If a module can answer only PART of the query, its "text" must cover ONLY that
    part and must drop everything that belongs to other modules. Do not restate the
    whole cross-module question under each module. In particular, for a "how does X
    affect/influence Y" query where X and Y are owned by different modules, the module
    that owns X is asked only about X, and the module that owns Y is asked only about
    Y — neither "text" should mention the other module's half.
- Output ONLY valid JSON.
- Do not use markdown fences.
- Do not include comments.
 
- Required JSON schema:
{{
  "query_type": "SIMPLE" | "COMPLEX",
  "classification_confidence": "HIGH" | "MEDIUM" | "LOW",
  "sub_queries": [
    {{
      "text": "<standalone sub-query text>",
      "module": "usecase_generator" | "tsn_scheduling_problem" | "evaluator" | "parameters",
      "action": "DO" | "EXPLAIN_CONCEPT" | "EXPLAIN_CODE",
      "intent": "<short intent label such as topology_generation, scheduling, evaluation, parameter_lookup>",
      "parameters": {{ "<known_param_name>": <value or null> }}
    }}
  ]
}}
 
Parameter extraction (per sub-query):
- From the known parameter list below, include every parameter the sub-query explicitly or implicitly references (e.g. "8 switches" -> num_switches: 8).
- Use null when a matched parameter has no value. Never invent names not in the list.
 
Known parameters:
{param_definitions}
 
Examples of correct decomposition. Study how each sub-query "text" is rewritten to be
standalone and answerable from a SINGLE module, and how any cross-module "bridge" wording
is dropped. Never copy the original query verbatim into a sub-query.
 
Example 1 — two modules, explanatory. The query names no functions, so the sub-queries stay conceptual and use EXPLAIN_CONCEPT (do NOT invent function names that are not in the query):
Query: How is a stream's transmission interval turned into the macro-cycle that scheduling uses, and where does each step happen?
Output:
{{
  "query_type": "COMPLEX",
  "classification_confidence": "HIGH",
  "sub_queries": [
    {{
      "text": "How is a stream's transmission interval set?",
      "module": "usecase_generator",
      "action": "EXPLAIN_CONCEPT",
      "intent": "stream_generation",
      "parameters": {{}}
    }},
    {{
      "text": "How are all streams' interval values combined into a single macro-cycle?",
      "module": "tsn_scheduling_problem",
      "action": "EXPLAIN_CONCEPT",
      "intent": "scheduling",
      "parameters": {{}}
    }}
  ]
}}
 
Example 2 — three modules, explanatory. The query itself names functions, so each sub-query keeps ONLY the name(s) its own module owns and drops the rest; action is EXPLAIN_CODE:
Query: Trace a port's link speed from makeTopology, through TSNScheduling's ports property, to where TSNConstraintEvaluator stores it in port_speeds. What does each contribute?
Output:
{{
  "query_type": "COMPLEX",
  "classification_confidence": "HIGH",
  "sub_queries": [
    {{
      "text": "How does makeTopology assign a link/port speed to each port, and where does the speed value come from?",
      "module": "usecase_generator",
      "action": "EXPLAIN_CODE",
      "intent": "topology_generation",
      "parameters": {{}}
    }},
    {{
      "text": "How does TSNScheduling's ports property expose each port and its speed, and how are ports keyed?",
      "module": "tsn_scheduling_problem",
      "action": "EXPLAIN_CODE",
      "intent": "port_mapping",
      "parameters": {{}}
    }},
    {{
      "text": "How does TSNConstraintEvaluator store port speeds in its port_speeds array, what does it read, and how is it indexed?",
      "module": "evaluator",
      "action": "EXPLAIN_CODE",
      "intent": "evaluation",
      "parameters": {{}}
    }}
  ]
}}
 
Example 3 — three modules, DO (multi-action pipeline):
Query: Generate a fat-tree use case with k=4 and 16 streams, solve its TSN schedule, then evaluate the schedule for constraint violations.
Output:
{{
  "query_type": "COMPLEX",
  "classification_confidence": "HIGH",
  "sub_queries": [
    {{
      "text": "Generate a fat-tree use case with k=4 and 16 streams (topology and streams).",
      "module": "usecase_generator",
      "action": "DO",
      "intent": "usecase_generation",
      "parameters": {{"k": 4, "numofstreamsperinstance": 16}}
    }},
    {{
      "text": "Solve and optimize the TSN schedule for the generated use case.",
      "module": "tsn_scheduling_problem",
      "action": "DO",
      "intent": "scheduling",
      "parameters": {{}}
    }},
    {{
      "text": "Evaluate the resulting schedule for constraint violations.",
      "module": "evaluator",
      "action": "DO",
      "intent": "evaluation",
      "parameters": {{}}
    }}
  ]
}}

Example 4 — import vs. generate, and the file-vs-dict trap:
Query: I want to import a use case I previously saved to JSON, and separately,
generate a brand-new use case from my config file. What functions can I use?
Output:
{{
  "query_type": "COMPLEX",
  "classification_confidence": "HIGH",
  "sub_queries": [
    {{
      "text": "How can I import a use case that was previously saved to a JSON file?",
      "module": "tsn_scheduling_problem",
      "action": "EXPLAIN_CODE",
      "intent": "usecase_import",
      "parameters": {{}}
    }},
    {{
      "text": "How can I generate a brand-new use case from a config file?",
      "module": "usecase_generator",
      "action": "EXPLAIN_CODE",
      "intent": "usecase_generation",
      "parameters": {{}}
    }}
  ]
}}
Note: "import a use case from JSON" and "generate a use case from a config
file" sound similar (both take a file, both produce a Usecase) but are
OPPOSITE operations on OPPOSITE modules — restoring an existing one vs.
creating a new random one. Never answer the first with from_json or
guess_config; neither reads a file.

Example 5 — a "schedule" query that is NOT tsn_scheduling_problem. Both parts
are the same module and the same action, so they merge into ONE sub-query:
Query: I have a computed schedule and want to attach it to my use case, and later clear it. What functions can I use?
Output:
{{
  "query_type": "SIMPLE",
  "classification_confidence": "HIGH",
  "sub_queries": [
    {{
      "text": "How can a schedule be attached to a Usecase object and later removed from it?",
      "module": "usecase_generator",
      "action": "EXPLAIN_CODE",
      "intent": "schedule_attachment",
      "parameters": {{}}
    }}
  ]
}}
Note: tsn_scheduling_problem has no schedule-related function at all.
Usecase.addSchedule / Usecase.removeSchedule are the only ones.

Example 5b — SAME module and SAME intent as Example 5, but the OPPOSITE action,
because this query is an INSTRUCTION rather than a question:
Query: Implement the Schedule object that addSchedule expects, with update_graph and update_streams.
Output:
{{
  "query_type": "SIMPLE",
  "classification_confidence": "HIGH",
  "sub_queries": [
    {{
      "text": "Write a Schedule class, with update_graph and update_streams methods, that matches what Usecase.addSchedule expects to be given.",
      "module": "usecase_generator",
      "action": "DO",
      "intent": "schedule_attachment",
      "parameters": {{}}
    }}
  ]
}}
Note: Example 5 asks "What functions can I use?" — a question, so EXPLAIN_CODE.
This one says "Implement ..." — an instruction, so DO. Mentioning `addSchedule`
only routes the sub-query to usecase_generator; it does not make the action
EXPLAIN_CODE. Do not let the wording it shares with Example 5 pull it there.

Example 6 — a graph distance that is NOT a delay cost:
Query: I want the shortest-path host distances and then to detect flow-conservation violations in a candidate. What functions can I use?
Output:
{{
  "query_type": "COMPLEX",
  "classification_confidence": "HIGH",
  "sub_queries": [
    {{
      "text": "How can I get the shortest-path graph distance from every host to every other node of the use case topology?",
      "module": "usecase_generator",
      "action": "EXPLAIN_CODE",
      "intent": "topology_query",
      "parameters": {{}}
    }},
    {{
      "text": "How can I detect flow-conservation violations in a candidate chromosome?",
      "module": "evaluator",
      "action": "EXPLAIN_CODE",
      "intent": "evaluation",
      "parameters": {{}}
    }}
  ]
}}
Note: "shortest-path distances" here is unitless graph geometry
(Usecase.distance_to_hosts), NOT a nanosecond delay — so it is not
tsn_scheduling_problem's hops_costs, which needs a frame size.

Note: no evaluator function already returns this exact table — deadline, achieved
latency and pass/fail each come from a different existing piece, and slack has to be
computed from them. That is exactly what DO covers: the user wants the ASSEMBLED
RESULT handed to them, not a pointer to which function to call. Compare Example 6,
where the query says "What functions can I use?" — asking to be POINTED at code, which
is EXPLAIN_CODE. Same-sounding rewritten "text" phrasing, opposite actions, because the
ORIGINAL query asks for different things.


User query:
{query}
"""



# Collapses near-duplicate SubQuery objects the classifier LLM sometimes emits for the
# same (module, action) pair — keeps the first, drops later ones whose text is >60%
# similar (SequenceMatcher ratio). Called at the end of validate_classification().
def deduplicate_sub_queries(sub_queries):
    seen = []
    for sq in sub_queries:
        duplicate = any(
            sq.module == s.module and
            sq.action == s. action and
            SequenceMatcher(None, sq.text, s.text).ratio() > 0.6
            for s in seen
        )
        if not duplicate:
            seen.append(sq)
    return seen

# Query normalization
# UNUSED/DISABLED: would call restore_punctuation(), which depends on the punctuation
# model that was removed above (see "Punctuation restoration" block) to free VRAM.
# def normalize_query(query: str) -> str:
#     """
#     Normalizes a user query once.
#     Currently restores punctuation. Later you can add whitespace cleanup,
#     typo correction, etc.
#     """
#     query = " ".join(query.strip().split())

#     if not query:
#         return query

#     return restore_punctuation(query)

# JSON Extraction helpers
import re
def extract_json_object(text: str) -> dict:
    """
    Extracts the first JSON object from an LLM response.
    Handles raw JSON and JSON inside markdown fences.
    Raises json.JSONDecodeError if parsing fails.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    candidate = text[start : end + 1]
    return json.loads(candidate)

def extract_json_array(text: str) -> list:
    """
    Extracts the first JSON array from an LLM response.
    Handles raw JSON and JSON inside markdown fences.
    Raises json.JSONDecodeError if parsing fails.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip())

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON array found", text, 0)

    candidate = text[start : end + 1]
    parsed = json.loads(candidate)

    if not isinstance(parsed, list):
        raise json.JSONDecodeError("Extracted JSON is not an array", text, 0)

    return parsed


# ── Plain-text fallback for JSON-output prompts ───────────────────────────────
# Every prompt below asks for JSON first — it round-trips structure exactly and needs
# no bespoke parser, so it stays the primary path everywhere. This is only reached
# once json.loads()/extract_json_object() has ALREADY failed once: rather than give up
# (the previous behavior — print a [WARN] and return an empty/None result), re-ask the
# SAME request in plain, delimited text and hand the response to a caller-specific
# parser. This sidesteps the single largest real source of malformed JSON in this
# pipeline — an LLM failing to correctly escape quotes/newlines/backslashes inside a
# JSON string value — which is exactly why SPEC_BASED_TEST_PROMPT was designed around
# delimited text blocks instead of JSON from the very start (see _spec_based_tests()'s
# own docstring). The four parsers below reuse that same proven "### BLOCK / plain
# 'LABEL: value' lines" shape rather than inventing a new convention per call site.
def _retry_as_text(prompt: str, error: Exception, format_instructions: str,
                    max_tokens: Optional[int] = None) -> str:
    """Re-asks `prompt` for the same content in plain text instead of JSON.

    Returns the raw completion; parsing it into a usable structure is the caller's
    job, since every call site's expected shape is different.
    """
    retry_prompt = (
        f"{prompt}\n\n"
        f"YOUR PREVIOUS ANSWER WAS REJECTED: it was not valid JSON "
        f"({type(error).__name__}: {error}).\n"
        f"Do NOT use JSON this time. Instead, answer in PLAIN TEXT using EXACTLY this "
        f"format, with no extra commentary before or after it:\n\n"
        f"{format_instructions}"
    )
    if max_tokens is None:
        max_tokens = llm_config.stage_max_tokens(retry_prompt)
    return Settings.llm.complete(retry_prompt, max_tokens=max_tokens).text.strip()


# The closed predicate vocabulary GENERATE_ALGORITHM_REQUIREMENTS_PROMPT hands the
# model. Kept here, as data, so the report below can name a predicate that is not on the
# list without the list being restated in two places and drifting apart.
_REQUIREMENT_PREDICATES = (
    "isA", "partOf", "represents",
    "hasInput", "hasOutput",
    "transformsFrom", "transformsInto", "hasStep", "precedes", "mustEndAt",
    "appliesWhen", "precondition", "mustSatisfy",
    "determines", "criterion", "minimizes",
    "builtBy", "constrainedBy",
)

_LEDGER_TEXT_FORMAT = (
    "R1: <first requirement, anchored to the user's own words>\n"
    "  <Subject> | <predicate> | <Object>\n"
    "  <Subject> | <predicate> | <Object>\n"
    "R2: <second requirement>\n"
    "  <Subject> | <predicate> | <Object>\n"
    "... one 'R<n>: <text>' line per requirement, then its triples indented beneath it,"
    " one per line, fields separated by ' | '. A requirement with no triple gets no"
    " indented line. Predicates must come from the closed list given above. Then every"
    " name the triples use, one line each:\n"
    "RES | <ResourceName> | <catalogue name or null> | <one-line note>\n"
    "RES | <ResourceName> | <catalogue name or null> | <one-line note>\n"
)
_LEDGER_LINE_RE = re.compile(r"^\s*R(\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)
# An indented 'S | p | O' line belonging to the requirement above it. A trailing fourth
# field from an older format is tolerated and ignored.
# 'RES | name | corresponds_to | note', tried before the triple pattern because a RES line
# would otherwise match the triple pattern and be read as a triple whose subject is the
# literal string "RES".
_LEDGER_RESOURCE_RE = re.compile(
    r"^[ \t]*RES\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]*?)\s*(?:\|\s*(.*?))?\s*$"
)
_LEDGER_TRIPLE_RE = re.compile(
    r"^[ \t]+([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*(?:\|\s*(.*?))?\s*$"
)


def _parse_ledger_text(raw: str) -> list[dict]:
    """Plain-text fallback parser for build_algorithm_requirements_ledger(): one
    'R<n>: <text>' line per requirement, each optionally followed by indented
    'S | p | O' triple lines.

    Written line-by-line rather than as one regex because the triples belong to
    whichever requirement line precedes them, and that ownership is positional.
    """
    ledger: list[dict] = []
    _REQUIREMENT_RESOURCES.clear()
    for line in raw.splitlines():
        head = _LEDGER_LINE_RE.match(line)
        if head:
            ledger.append({"id": f"R{head.group(1)}", "text": head.group(2).strip(),
                           "triples": []})
            continue
        res = _LEDGER_RESOURCE_RE.match(line)
        if res:
            name, corresponds, note = ((g or "").strip() for g in res.groups())
            _REQUIREMENT_RESOURCES.append({
                "name": name,
                "corresponds_to": None if corresponds.lower() in ("", "null", "none")
                                  else tuple(c.strip() for c in corresponds.split(",")
                                             if c.strip()),
                "note": note,
            })
            continue
        trip = _LEDGER_TRIPLE_RE.match(line)
        if trip and ledger:
            s, p, o, _ = trip.groups()
            ledger[-1]["triples"].append({"s": s.strip(), "p": p.strip(), "o": o.strip()})
    return ledger


_TRIAGE_TEXT_FIELDS = [
    ("VERDICT", "verdict"),
    ("CONFIDENCE", "confidence"),
    ("EXPLANATION", "explanation"),
    ("EVIDENCE_FROM_CONTRACT", "evidence_from_contract"),
    ("REQUIRED_IMPL_CHANGE", "required_impl_change"),
    ("SUGGESTED_FIX_TARGET", "suggested_fix_target"),
]
_TRIAGE_TEXT_FORMAT = (
    "VERDICT: IMPL_INCORRECT or FLAKY_OR_ENV\n"
    "CONFIDENCE: HIGH, MEDIUM, or LOW\n"
    "EXPLANATION: why the implementation does not satisfy the contract\n"
    "EVIDENCE_FROM_CONTRACT: quote the relevant behavior\n"
    "REQUIRED_IMPL_CHANGE: concretely, what the implementation must do differently\n"
    "SUGGESTED_FIX_TARGET: implementation or none\n"
    "Each field's value may continue on following lines until the next field label."
)
_TRIAGE_TEXT_RE = re.compile(
    r"(" + "|".join(label for label, _ in _TRIAGE_TEXT_FIELDS) + r")\s*:\s*(.*?)"
    r"(?=\n(?:" + "|".join(label for label, _ in _TRIAGE_TEXT_FIELDS) + r")\s*:|\Z)",
    re.DOTALL,
)


def _parse_triage_text(raw: str) -> dict:
    """Plain-text fallback parser for _triage_failure(): one 'FIELD: value' line (or
    multi-line value) per field, in TRIAGE_PROMPT's own field order."""
    field_map = dict(_TRIAGE_TEXT_FIELDS)
    result = {}
    for label, value in _TRIAGE_TEXT_RE.findall(raw):
        result[field_map[label]] = value.strip()
    return result


_SPEC_TEXT_FLAT_FIELDS = {
    "NAME": "name", "TYPE": "type", "CLASS": "class",
    "DESCRIPTION": "description", "SIGNATURE": "signature", "CALLABLE": "callable",
}
_SPEC_TEXT_LIST_FIELDS = {
    "PARAMETERS": "parameters", "PRECONDITIONS": "preconditions",
    "POSTCONDITIONS": "postconditions", "DEPENDENCIES_INTERNAL": "__dep_internal",
    "DEPENDENCIES_EXTERNAL": "__dep_external", "RAISES": "raises",
}
_SPEC_TEXT_FORMAT = (
    "NAME: exact function name\n"
    "TYPE: function (or class, only if a whole class was explicitly requested)\n"
    "CLASS: none\n"
    "DESCRIPTION: detailed description of what the function does\n"
    "SIGNATURE: the full def line\n"
    "CALLABLE: true\n"
    "PARAMETERS:\n- param_name\n- another_param\n"
    "PRECONDITIONS:\n- condition that must hold before calling\n"
    "POSTCONDITIONS:\n- guarantee about the return value\n"
    "DEPENDENCIES_INTERNAL:\n- ProjectClassOrFunctionName\n"
    "DEPENDENCIES_EXTERNAL:\n- numpy\n"
    "RAISES:\n- ExceptionName\n"
    "Leave a list section header with no '-' lines under it if it is empty."
)


def _parse_spec_text(raw: str) -> dict:
    """Plain-text fallback parser for generate_function_spec(): flat 'FIELD: value'
    lines plus list sections ('PARAMETERS:', 'PRECONDITIONS:', ...) each followed by
    '- item' lines until the next header. _normalize_spec() fills in whatever this
    does not manage to extract, so this only needs to be a best effort, not exhaustive.
    """
    out: dict = {}
    dep_internal: list[str] = []
    dep_external: list[str] = []
    all_headers = list(_SPEC_TEXT_FLAT_FIELDS) + list(_SPEC_TEXT_LIST_FIELDS)
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        header = next((h for h in all_headers if line.upper().startswith(h + ":")), None)
        if header is None:
            i += 1
            continue
        if header in _SPEC_TEXT_FLAT_FIELDS:
            out[_SPEC_TEXT_FLAT_FIELDS[header]] = line[len(header) + 1:].strip()
            i += 1
            continue
        items = []
        i += 1
        while i < len(lines) and lines[i].strip().startswith("-"):
            item = lines[i].strip().lstrip("-").strip()
            if item:
                items.append(item)
            i += 1
        target = _SPEC_TEXT_LIST_FIELDS[header]
        if target == "__dep_internal":
            dep_internal = items
        elif target == "__dep_external":
            dep_external = items
        else:
            out[target] = items

    if "callable" in out:
        out["callable"] = out["callable"].strip().lower() in ("true", "yes", "1")
    if isinstance(out.get("parameters"), list):
        out["parameters"] = [{"name": p, "default": None} for p in out["parameters"]]
    out["dependencies"] = {"internal": dep_internal, "external": dep_external}
    return out


_BLUEPRINT_TEXT_FORMAT = (
    "One '### TYPE' block per data-model type:\n"
    "### TYPE\nNAME: Schedule\nSHAPE: concrete field-by-field description\n### END TYPE\n\n"
    "One '### FUNCTION' block per function in the inventory:\n"
    "### FUNCTION\nNAME: function_name\nSIGNATURE: def function_name(arg: Type) -> ReturnType\n"
    "PURPOSE: one-line purpose\nCALLS: other_function_a, other_function_b\n### END FUNCTION\n"
    "(CALLS may be empty after the colon if the function calls nothing.)\n\n"
    "One '### REQUIREMENT' block per ledger requirement:\n"
    "### REQUIREMENT\nID: R1\nFUNCTIONS: function_a, function_b\n### END REQUIREMENT\n\n"
    "One '### REUSE' block per codebase function/class reused:\n"
    "### REUSE\nCODEBASE_FUNCTION: ClassName.method_name\nUSED_BY: function_name\n"
    "CALL_FORM: exact call expression\nREASON: why this candidate satisfies the need\n"
    "### END REUSE"
)
_BLUEPRINT_TYPE_RE = re.compile(
    r"###\s*TYPE\s*\n\s*NAME\s*:\s*(?P<name>.+?)\n\s*SHAPE\s*:\s*(?P<shape>.*?)\n\s*###\s*END\s*TYPE",
    re.DOTALL,
)
_BLUEPRINT_FUNCTION_RE = re.compile(
    r"###\s*FUNCTION\s*\n\s*NAME\s*:\s*(?P<name>.+?)\n\s*SIGNATURE\s*:\s*(?P<signature>.+?)\n"
    r"\s*PURPOSE\s*:\s*(?P<purpose>.*?)\n\s*CALLS\s*:\s*(?P<calls>.*?)\n\s*###\s*END\s*FUNCTION",
    re.DOTALL,
)
_BLUEPRINT_REQUIREMENT_RE = re.compile(
    r"###\s*REQUIREMENT\s*\n\s*ID\s*:\s*(?P<id>.+?)\n\s*FUNCTIONS\s*:\s*(?P<functions>.*?)\n"
    r"\s*###\s*END\s*REQUIREMENT",
    re.DOTALL,
)
_BLUEPRINT_REUSE_RE = re.compile(
    r"###\s*REUSE\s*\n\s*CODEBASE_FUNCTION\s*:\s*(?P<codebase_function>.+?)\n"
    r"\s*USED_BY\s*:\s*(?P<used_by>.+?)\n\s*CALL_FORM\s*:\s*(?P<call_form>.*?)\n"
    r"\s*REASON\s*:\s*(?P<reason>.*?)\n\s*###\s*END\s*REUSE",
    re.DOTALL,
)


def _parse_blueprint_text(raw: str) -> dict:
    """Plain-text fallback parser for build_algorithm_blueprint(): the same
    '### BLOCK ... ### END BLOCK' delimited convention SPEC_BASED_TEST_PROMPT already
    uses successfully for lists of structured items, applied to each of the
    blueprint's four sections.
    """
    return {
        # The text fallback can only recover name+type; the other three fields have no
        # delimited form, and neither do a function's reads/writes. A run that lands here
        # loses why/used_for/corresponds_to and all flow direction, which is a degraded
        # answer, not a wrong one — the JSON path is the normal one.
        # _structure_flow_violations therefore says nothing about a fallback-parsed
        # blueprint rather than reporting every structure as unwritten and unread.
        "data_structures": [
            {"name": m.group("name").strip(), "type": m.group("shape").strip()}
            for m in _BLUEPRINT_TYPE_RE.finditer(raw)
        ],
        "function_inventory": [
            {
                "name": m.group("name").strip(),
                "signature": m.group("signature").strip(),
                "purpose": m.group("purpose").strip(),
                "calls": [c.strip() for c in m.group("calls").split(",") if c.strip()],
            }
            for m in _BLUEPRINT_FUNCTION_RE.finditer(raw)
        ],
        "requirement_map": [
            {
                "requirement_id": m.group("id").strip(),
                "functions": [f.strip() for f in m.group("functions").split(",") if f.strip()],
            }
            for m in _BLUEPRINT_REQUIREMENT_RE.finditer(raw)
        ],
        "reuse_decisions": [
            {
                "codebase_function": m.group("codebase_function").strip(),
                "used_by": m.group("used_by").strip(),
                "call_form": m.group("call_form").strip(),
                "reason": m.group("reason").strip(),
            }
            for m in _BLUEPRINT_REUSE_RE.finditer(raw)
        ],
    }


_CONTRACT_TEXT_FORMAT = (
    "One '### BEHAVIOR' block per behavior (at least one per postcondition, exactly "
    "one error_case per declared exception):\n"
    "### BEHAVIOR\nID: B1\nDESCRIPTION: observable behavior in plain English\n"
    "GIVEN: input condition / state\nWHEN: the function is called with ...\n"
    "THEN: the observable, assertable outcome\n"
    "CATEGORY: happy_path, edge_case, error_case, or invariant\n### END BEHAVIOR\n\n"
    "One '### DOMAIN' block per parameter:\n"
    "### DOMAIN\nPARAM: param_name\nVALID: comma-separated representative valid values\n"
    "BOUNDARY: comma-separated boundary values\nINVALID: comma-separated values that "
    "should trigger errors\n### END DOMAIN\n\n"
    "INVARIANTS:\n- property that must always hold across all inputs\n"
    "- another such property\n"
    "(leave the INVARIANTS list empty of '-' lines if there are none)"
)
_CONTRACT_BEHAVIOR_RE = re.compile(
    r"###\s*BEHAVIOR\s*\n\s*ID\s*:\s*(?P<id>.+?)\n\s*DESCRIPTION\s*:\s*(?P<description>.*?)\n"
    r"\s*GIVEN\s*:\s*(?P<given>.*?)\n\s*WHEN\s*:\s*(?P<when>.*?)\n\s*THEN\s*:\s*(?P<then>.*?)\n"
    r"\s*CATEGORY\s*:\s*(?P<category>.*?)\n\s*###\s*END\s*BEHAVIOR",
    re.DOTALL,
)
_CONTRACT_DOMAIN_RE = re.compile(
    r"###\s*DOMAIN\s*\n\s*PARAM\s*:\s*(?P<param>.+?)\n\s*VALID\s*:\s*(?P<valid>.*?)\n"
    r"\s*BOUNDARY\s*:\s*(?P<boundary>.*?)\n\s*INVALID\s*:\s*(?P<invalid>.*?)\n"
    r"\s*###\s*END\s*DOMAIN",
    re.DOTALL,
)
_CONTRACT_INVARIANT_LINE_RE = re.compile(r"^\s*-\s*(.+?)\s*$", re.MULTILINE)


def _parse_contract_text(raw: str) -> dict:
    """Plain-text fallback parser for _derive_contract_from_spec(): the same
    '### BLOCK' convention as the blueprint parser for behaviors/domains, plus a
    flat '- item' list for invariants.
    """
    invariants_section = raw.split("INVARIANTS:", 1)[1] if "INVARIANTS:" in raw else ""
    return {
        "behaviors": [
            {
                "id": m.group("id").strip(),
                "description": m.group("description").strip(),
                "given": m.group("given").strip(),
                "when": m.group("when").strip(),
                "then": m.group("then").strip(),
                "category": m.group("category").strip(),
            }
            for m in _CONTRACT_BEHAVIOR_RE.finditer(raw)
        ],
        "input_domains": {
            m.group("param").strip(): {
                "valid": [v.strip() for v in m.group("valid").split(",") if v.strip()],
                "boundary": [v.strip() for v in m.group("boundary").split(",") if v.strip()],
                "invalid": [v.strip() for v in m.group("invalid").split(",") if v.strip()],
            }
            for m in _CONTRACT_DOMAIN_RE.finditer(raw)
        },
        "invariants": [m.group(1) for m in _CONTRACT_INVARIANT_LINE_RE.finditer(invariants_section)],
    }


# def merge_results_by_module(results):  # UNUSED: an earlier, standalone version of the
#     merged = defaultdict(lambda: {"functions": [], "missing_requirements": []})  # same-module merge now done inline as Pass 2 of answer_query_deterministic()
#     for r in results:                                                            # (the merged_by_module dict built there covers this and also keys on action).
#         module = r["module"]
#         merged[module]["functions"].extend(r.get("functions", []))
#         merged[module]["missing_requirements"].extend(
#             r.get("coverage", {}).get("missing_requirements", [])
#         )
#     # Deduplicate missing requirements
#     for module in merged:
#         merged[module]["missing_requirements"] = list(
#             set(merged[module]["missing_requirements"])
#         )
#     return merged

# Splices real source code into every <CODE:function_name> placeholder found in `answer`,
# rather than having the LLM copy code inline — cheaper and avoids the LLM mistyping or
# truncating code it only needs to reference by name. Safe to call even when no
# placeholder is present (returns `answer` unchanged).
def inject_code(answer: str, all_functions: list[dict]) -> str:
    """
    Replaces every <CODE:function_name> placeholder in `answer` with
    the actual code taken from the matching function dict.
    Falls back gracefully if a function name is not found.
    """
    code_map = {fn["name"]: fn.get("code", "") for fn in all_functions if fn.get("name")}

    for fn_name, code in code_map.items():
        placeholder = f"<CODE:{fn_name}>"
        if placeholder in answer:
            formatted = f"```python\n{code}\n```" if code else "# code not available"
            answer = answer.replace(placeholder, formatted)

    return answer

# Module validation helpers — repair/guess raw classifier-LLM output so a sloppy
# module/action label never reaches VALID_MODULES/VALID_INTENTS as garbage. Used
# by validate_classification() below to sanitize each raw sub-query before it
# becomes a typed SubQuery.
def _normalize_module_label(module: str) -> str:
    """
    Normalizes common module label mistakes from the classifier.
    """
    if not module:
        return ""

    module = module.strip().lower()

    aliases = {
        "usecase": "usecase_generator",
        "use_case_generator": "usecase_generator",
        "usecase generation": "usecase_generator",
        "topology": "usecase_generator",
        "topology_generation": "usecase_generator",
        "scheduler": "tsn_scheduling_problem",
        "scheduling": "tsn_scheduling_problem",
        "optimization": "tsn_scheduling_problem",
        "optimizer": "tsn_scheduling_problem",
        "evaluation": "evaluator",
        "metrics": "evaluator",
        "parameter": "parameters",
        "params": "parameters",
    }

    return aliases.get(module, module)

def _normalize_intent_label(action: str) -> str:
    """Normalizes classifier action output; always returns a valid Intent string."""
    if not action:
        return "DO"
    a = action.strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "EXPLAIN": "EXPLAIN_CONCEPT",
        "CONCEPT": "EXPLAIN_CONCEPT",
        "EXPLAIN_CONCEPTS": "EXPLAIN_CONCEPT",
        "EXPLAIN_FUNCTION": "EXPLAIN_CODE",
        "EXPLAIN_THE_CODE": "EXPLAIN_CODE",
        "CODE_EXPLANATION": "EXPLAIN_CODE",
        "GENERATE": "DO", "RETRIEVE": "DO", "RUN": "DO", "DO_TASK": "DO",
    }
    a = aliases.get(a, a)
    return a if a in VALID_INTENTS else "DO"


def guess_intent_by_keywords(text: str) -> str:
    """Deterministic action fallback when the classifier omits/garbles 'action'."""
    q = text.lower()
    # Concept-specific phrasings first ("what is", "why ...").
    if any(t in q for t in ("what is", "what are", "why ", "concept",
                            "difference between", "intuition behind")):
        return "EXPLAIN_CONCEPT"
    # Then code-explanation phrasings.
    if any(t in q for t in ("explain", "walk me through", "walkthrough", "how does",
                            "what does", "break down", "this function", "this code",
                            "this method")):
        return "EXPLAIN_CODE"
    # Algorithm-synthesis phrasing: an INSTRUCTION to design/implement a multi-part
    # algorithm, strategy, or heuristic, not just a single function. Checked last so
    # it never overrides the EXPLAIN_CONCEPT/EXPLAIN_CODE checks above (e.g. "explain
    # this algorithm" already returned EXPLAIN_CODE before reaching here).
    if any(t in q for t in ("algorithm", "heuristic", "strategy for", "strategy to",
                            "approach to solve", "approach for solving")):
        return "GENERATE_ALGORITHM"
    return "DO"

def guess_module_by_keywords(text: str) -> str:
    """
    Deterministic fallback for obvious query routing.
    Used only when the LLM output is invalid or incomplete.
    """
    q = text.lower()

    parameter_terms = [
        "parameter",
        "default",
        "range",
        "meaning",
        "what does",
        "what is",
    ]

    usecase_terms = [
        "topology",
        "fat-tree",
        "fattree",
        "fat tree",
        "graph",
        "stream",
        "route",
        "path",
        "use case",
        "usecase",
        "generate network",
        "host",
        "switch",
        "generate",
        "empty use case",
        "generate stream",
        "stream generator"
    ]

    scheduling_terms = [
        "schedule",
        "scheduling",
        "optimize",
        "optimization",
        "gcl",
        "gate control",
        "constraint",
        "deadline",
        "offset",
        "queue",
        "solver",
        "fitness",
    ]

    evaluator_terms = [
        "evaluate",
        "evaluation",
        "latency",
        "metric",
        "validate",
        "check correctness",
        "deadline miss",
        "violation",
        "performance",
        "assess",
    ]

    scores = {
        "parameters": sum(term in q for term in parameter_terms),
        "usecase_generator": sum(term in q for term in usecase_terms),
        "tsn_scheduling_problem": sum(term in q for term in scheduling_terms),
        "evaluator": sum(term in q for term in evaluator_terms),
    }

    best_module = max(scores, key=scores.get)

    if scores[best_module] == 0:
        return "usecase_generator"

    return best_module

# Turns the classifier LLM's raw parsed JSON into a trustworthy ClassifiedQuery: repairs
# invalid query_type/confidence/module/action values, builds a fallback sub-query if the
# LLM returned none, merges sub-queries that share (module, action), and re-derives
# SIMPLE/COMPLEX from the final unique-module count (never trusts the LLM's own label).
# This is the last step of "understand the query" before lookup/generation can start.
def validate_classification(
    parsed: dict,
    original_query: str,
    normalized_query: str,
    raw_output: str,
) -> ClassifiedQuery:
    warnings_list = []

    raw_query_type = parsed.get("query_type", "SIMPLE")
    if raw_query_type not in {"SIMPLE", "COMPLEX"}:
        warnings_list.append(f"Invalid query_type: {raw_query_type}. Using SIMPLE.")
        raw_query_type = "SIMPLE"

    raw_confidence = parsed.get("classification_confidence", "MEDIUM")
    if raw_confidence not in {"HIGH", "MEDIUM", "LOW"}:
        warnings_list.append(
            f"Invalid classification_confidence: {raw_confidence}. Using MEDIUM."
        )
        raw_confidence = "MEDIUM"

    reason = str(parsed.get("reason", "")).strip()

    raw_sub_queries = parsed.get("sub_queries", [])
    if not isinstance(raw_sub_queries, list) or not raw_sub_queries:
        warnings_list.append("Missing or empty sub_queries. Creating fallback sub-query.")
        fallback_module = guess_module_by_keywords(normalized_query)
        raw_sub_queries = [
            {
                "text": normalized_query,
                "module": fallback_module,
                "intent": "unknown",
            }
        ]

    sub_queries = []

    for item in raw_sub_queries:
        if not isinstance(item, dict):
            warnings_list.append(f"Invalid sub-query item ignored: {item}")
            continue

        text = str(item.get("text", "")).strip() or normalized_query
        module = _normalize_module_label(str(item.get("module", "")).strip())
        intent = str(item.get("intent", "unknown")).strip() or "unknown"

        if module not in VALID_MODULES:
            guessed = guess_module_by_keywords(text)
            warnings_list.append(
                f"Invalid module '{module}' for sub-query '{text}'. Using '{guessed}'."
            )
            module = guessed

        raw_action = str(item.get("action", "")).strip()
        action = (
            _normalize_intent_label(raw_action)
            if raw_action else guess_intent_by_keywords(text)
        )
        if module == "parameters" and action == "DO":
            # TEMPORARILY DO THE FOLLOWING ONLY: instead of demoting this sub-query to
            # EXPLAIN_CONCEPT (its normal fate, since "parameters" is a docs-only index
            # with no generation engine and no real .py file for generated tests to import
            # from), reroute it to whichever real code module its own text best matches,
            # so it goes through actual code generation instead of doc lookup. Falls back
            # to "usecase_generator" if the keyword guesser itself lands back on
            # "parameters" for this text. This is a temporary override and will be
            # reverted later — do not treat it as a permanent change to how the
            # "parameters" module is routed.
            reroute_module = guess_module_by_keywords(text)
            if reroute_module == "parameters":
                reroute_module = "usecase_generator"
            warnings_list.append(
                f"TEMPORARY: Rerouted parameters/DO -> {reroute_module}/DO for generation "
                f"(docs module has no generation engine)."
            )
            module = reroute_module
        params = item.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        sub_queries.append(
            SubQuery(text=text, module=module, intent=intent,
                     action=Intent(action), parameters=params)   # NEW
        )

    if not sub_queries:
        fallback_module = guess_module_by_keywords(normalized_query)
        sub_queries = [
            SubQuery(
                text=normalized_query,
                module=fallback_module,
                intent="unknown",
                action=Intent(guess_intent_by_keywords(normalized_query)),
            )
        ]
        warnings_list.append("All sub-queries were invalid. Used fallback.")

    # Enforce SIMPLE/COMPLEX consistency.
    unique_modules = {sq.module for sq in sub_queries}

    # Merge sub-queries sharing BOTH module and action; never merge across
    # differing actions (DO vs EXPLAIN_* on the same module must stay separate).
    # GENERATE_ALGORITHM is deliberately excluded from this collapsing: "one sub-query
    # per functionality" (see CLASSIFICATION_PROMPT) means two functionality sub-queries
    # that happen to land in the same module are still DISTINCT pieces of the algorithm,
    # not accidental duplicates — so their key includes the sub-query's own text, which
    # makes each one unique unless the classifier genuinely repeated itself verbatim.
    grouped: Dict[tuple, SubQuery] = {}
    for sq in sub_queries:
        disambiguator = sq.text if sq.action == Intent.GENERATE_ALGORITHM else None
        key = (sq.module, sq.action, disambiguator)
        if key not in grouped:
            grouped[key] = sq
        else:
            prev = grouped[key]
            grouped[key] = SubQuery(
                text=f"{prev.text} {sq.text}".strip(),
                module=sq.module,
                intent=",".join(sorted({prev.intent, sq.intent})),
                action=sq.action,
                parameters={**prev.parameters, **sq.parameters},
            )
            warnings_list.append(
                f"Merged duplicate sub-queries for ({sq.module}, {sq.action.value})."
            )
    sub_queries = list(grouped.values())

    # SIMPLE/COMPLEX is purely about how many distinct MODULES are involved.
    unique_modules = {sq.module for sq in sub_queries}
    query_type = QueryType.SIMPLE if len(unique_modules) == 1 else QueryType.COMPLEX

    # If model said SIMPLE but there are multiple modules, override to COMPLEX.
    if query_type.value != raw_query_type:
        warnings_list.append(
            f"Classifier said {raw_query_type}, but validated query is {query_type.value}."
        )
    sub_queries = deduplicate_sub_queries(sub_queries)
    return ClassifiedQuery(
        original_query=original_query,
        normalized_query=normalized_query,
        query_type=query_type,
        sub_queries=sub_queries,
        classification_confidence=Confidence(raw_confidence),
        reason=reason,
        warnings=warnings_list,
        raw_classifier_output=raw_output,
    )

# Zero-LLM fallback classifier: only reached from classify_query() when the classifier
# LLM call itself throws/returns unparseable JSON. Scores keyword hits per module and
# routes to every module that scored >0 (so it can still produce a COMPLEX result),
# guaranteeing the pipeline always gets a usable ClassifiedQuery even if the LLM is down.
def heuristic_classify_query(
    original_query: str,
    normalized_query: str,
    raw_output: str = "",
    warning: str = "",
) -> ClassifiedQuery:
    q = normalized_query.lower()

    module_hits = {
        "usecase_generator": 0,
        "tsn_scheduling_problem": 0,
        "evaluator": 0,
        "parameters": 0,
    }

    keyword_map = {
        "usecase_generator": [
            "topology",
            "fat-tree",
            "fat tree",
            "fattree",
            "graph",
            "stream",
            "route",
            "path",
            "host",
            "switch",
            "use case",
            "usecase",
        ],
        "tsn_scheduling_problem": [
            "schedule",
            "scheduling",
            "optimize",
            "optimization",
            "gcl",
            "gate",
            "constraint",
            "deadline",
            "queue",
            "offset",
            "solver",
            "fitness",
        ],
        "evaluator": [
            "evaluate",
            "evaluation",
            "latency",
            "validate",
            "metric",
            "deadline miss",
            "violation",
            "correctness",
            "performance",
        ],
        "parameters": [
            "parameter",
            "meaning",
            "default",
            "range",
            "valid",
            "type",
            "what does",
            "what is",
        ],
    }

    for module, keywords in keyword_map.items():
        module_hits[module] = sum(keyword in q for keyword in keywords)

    active_modules = [
        module for module, score in module_hits.items()
        if score > 0
    ]

    warnings_list = []
    if warning:
        warnings_list.append(warning)

    if not active_modules:
        guessed = guess_module_by_keywords(normalized_query)
        active_modules = [guessed]
        warnings_list.append("No strong keyword hits. Used best guessed module.")

    fallback_action = Intent(guess_intent_by_keywords(normalized_query))

    sub_queries = [
        SubQuery(
            text=normalized_query,
            module=module,
            intent="heuristic_routing",
            action=fallback_action
        )
        for module in active_modules
    ]

    query_type = QueryType.SIMPLE if len(sub_queries) == 1 else QueryType.COMPLEX

    return ClassifiedQuery(
        original_query=original_query,
        normalized_query=normalized_query,
        query_type=query_type,
        sub_queries=sub_queries,
        classification_confidence=Confidence.LOW,
        reason="Heuristic fallback classification was used.",
        warnings=warnings_list,
        raw_classifier_output=raw_output,
    )

# Main classifier
def classify_query(
    query: str,
    normalized_query: Optional[str] = None,
) -> ClassifiedQuery:
    """
    Classifies a user query into SIMPLE or COMPLEX and creates validated
    module-specific sub-queries.

    This function is safe:
    - handles invalid JSON;
    - normalizes module labels;
    - validates SIMPLE/COMPLEX consistency;
    - falls back to heuristic routing;
    - never returns invalid modules.
    """
    original_query = query
    normalized_query = normalized_query or " ".join(query.split())

    prompt = CLASSIFICATION_PROMPT.format(query=normalized_query, param_definitions=_get_param_definitions(),)
    raw = Settings.llm.complete(prompt).text.strip()

    try:
        parsed = extract_json_object(raw)
        classification = validate_classification(
            parsed=parsed,
            original_query=original_query,
            normalized_query=normalized_query,
            raw_output=raw,
        )
    except Exception as e:
        print(f"[WARN] Classification failed: {e}")
        classification = heuristic_classify_query(
            original_query=original_query,
            normalized_query=normalized_query,
            raw_output=raw,
            warning=f"LLM classification failed: {e}",
        )

    print(
        "[Classification] "
        f"{classification.query_type.value} | "
        f"confidence={classification.classification_confidence.value} | "
        f"modules={[sq.module for sq in classification.sub_queries]}"
    )

    if classification.warnings:
        for warning in classification.warnings:
            print(f"[Classification warning] {warning}")

    return classification


# Loads one module's (or the parameters domain's) pre-built vector index + raw docstore
# nodes from disk. Read-only: this script never builds/embeds an index itself, only
# consumes what build_indexes.py already persisted to MODULE_PERSIST / PARAMS_PERSIST.
def load_persisted_index(persist_dir: str):
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        raise FileNotFoundError(
            f"Persisted index not found in '{persist_dir}'. "
            f"Run build_indexes.py first."
        )

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    nodes = list(storage_context.docstore.docs.values())
    return index, nodes


# ── Load persisted indexes only ──────────────────────────────────────────────

params_index, params_nodes = load_persisted_index(PARAMS_PERSIST)

module_indexes: Dict[str, any] = {}
module_nodes: Dict[str, list] = {}

for m in CODE_MODULE_LABELS:
    idx, nds = load_persisted_index(MODULE_PERSIST[m])
    module_indexes[m] = idx
    module_nodes[m] = nds

# The algorithm-only modules land in the SAME dicts — they have to, because candidate
# hydration and BM25 node lookup are keyed by module label — but CODE_MODULE_LABELS is
# left alone, so the normal pathway's retrievers below are built from three modules
# exactly as before. Only algo_combined_retriever sees the fourth.
for m in ALGO_MODULE_LABELS:
    idx, nds = load_persisted_index(ALGO_MODULE_PERSIST[m])
    module_indexes[m] = idx
    module_nodes[m] = nds


# ── Retrievers + reranker ─────────────────────────────────────────────────────

torch.manual_seed(42)

# ── Parameters hybrid retriever ───────────────────────────────────────────────
_params_semantic = VectorIndexRetriever(
    index=params_index,
    similarity_top_k=20,
    vector_store_query_mode=VectorStoreQueryMode.MMR,
    mmr_threshold=0.6,
)
_params_bm25 = BM25Retriever.from_defaults(
    nodes=params_nodes,
    similarity_top_k=20,
)
params_retriever = QueryFusionRetriever(
    retrievers=[_params_semantic, _params_bm25],
    similarity_top_k=20,
    mode="reciprocal_rerank",
    num_queries=1,
    use_async=False,
)

# ── Per-module lookup hybrid retrievers (MMR + BM25) ─────────────────────────
# Feeds stage 2 ("check the codebase", _lookup_one()): MMR trades some pure
# similarity for diversity, so the top-k candidates aren't near-duplicates of the
# same function — useful when deciding whether ANY existing function matches.
module_retrievers: Dict[str, QueryFusionRetriever] = {}
for m in CODE_MODULE_LABELS:
    sem = VectorIndexRetriever(
        index=module_indexes[m],
        similarity_top_k=47,
        vector_store_query_mode=VectorStoreQueryMode.MMR,
        mmr_threshold=0.6,
    )
    bm25 = BM25Retriever.from_defaults(
        nodes=module_nodes[m],
        similarity_top_k=47,
    )
    module_retrievers[m] = QueryFusionRetriever(
        retrievers=[sem, bm25],
        similarity_top_k=47,
        mode="reciprocal_rerank",
        num_queries=1,
        use_async=False,
    )

# ── Per-module generation hybrid retrievers (pure similarity + BM25, no MMR) ─
# Feeds stage 4's generation path (_build_generation_context()): once we already
# know we're generating, we want the MOST similar helper code, not a diverse
# spread — so plain similarity replaces MMR here.
module_gen_retrievers: Dict[str, QueryFusionRetriever] = {}
for m in CODE_MODULE_LABELS:
    sem = VectorIndexRetriever(
        index=module_indexes[m],
        similarity_top_k=47,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
    )
    bm25 = BM25Retriever.from_defaults(
        nodes=module_nodes[m],
        similarity_top_k=47,
    )
    module_gen_retrievers[m] = QueryFusionRetriever(
        retrievers=[sem, bm25],
        similarity_top_k=47,
        mode="reciprocal_rerank",
        num_queries=1,
        use_async=False,
    )

reranker_params = SentenceTransformerRerank(
    # model="BAAI/bge-reranker-large", top_n=5   # HEAVY (~2GB) — swapped for lighter base model
    model="BAAI/bge-reranker-base", top_n=5
)
# DISABLED to free ~2GB of VRAM: under UNIFIED_RETRIEVAL the retrieval/explain path
# reranks with combined_reranker instead, and _lookup_one()'s use of this reranker is
# unreachable (only sq.module == "parameters" still calls _lookup_one, and that branch
# returns its pseudo-result before reaching the rerank). The generation path prefers it
# but no longer requires it: _rerank_generation_nodes() falls back to combined_reranker
# (already resident for the grounding path) when it is None, so generation runs either way.
# READD FOR A DEDICATED GENERATION RERANKER.
# reranker_algo = SentenceTransformerRerank(
#     # top_n=10, not 15: 15 candidates overflowed the 16384-token context window on
#     # EXPLAIN_CODE queries routed to evaluator (two ~9.3k-char functions, __init__ and
#     # suggest_corrections, alone made up 47% of the prompt).
#     model="BAAI/bge-reranker-large", top_n=15   # HEAVY (~2GB) — can swap for lighter base model
#     # model="BAAI/bge-reranker-base", top_n=8
# )
reranker_algo = None   # readd for generation


# ── Combined cross-module retriever + reranker (grounding path only) ──────────
# Built ALONGSIDE the per-module retrievers/rerankers above; nothing existing is
# removed. Only the doc-context grounding path (_collect_doc_context, used by the
# EXPLAIN_CONCEPT and EXPLAIN_CODE-fallback sections) is switched onto these. The
# DO / exact-match lookup (module_retrievers) and the generation path
# (module_gen_retrievers) keep their per-module retrievers untouched.
#
# Effect: a single sub-query can now retrieve chunks from the WHOLE codebase
# instead of just its routed module, which is what a "follow X across modules"
# question needs. No extra LLM calls — QueryFusionRetriever(num_queries=1) does
# no query generation, exactly like the per-module fusion retrievers.
import re
from llama_index.core.schema import NodeWithScore

_combined_semantic = [
    VectorIndexRetriever(
        index=module_indexes[m],
        similarity_top_k=47,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
    )
    for m in CODE_MODULE_LABELS
]
_combined_bm25 = [
    BM25Retriever.from_defaults(nodes=module_nodes[m], similarity_top_k=47)
    for m in CODE_MODULE_LABELS
]
combined_retriever = QueryFusionRetriever(
    retrievers=[*_combined_semantic, *_combined_bm25],
    similarity_top_k=47,                 # fused candidates pooled across ALL modules
    mode="reciprocal_rerank",
    num_queries=1,                       # no LLM query expansion
    use_async=False,
)

# The algorithm pathway's retriever: the same three modules PLUS ALGO_MODULE_LABELS.
# Separate object rather than a wider combined_retriever, so nothing about the normal
# pathway's candidate pool changes. Same reranker is shared — it is stateless.
_algo_pool = CODE_MODULE_LABELS + ALGO_MODULE_LABELS
algo_combined_retriever = QueryFusionRetriever(
    retrievers=[
        *[VectorIndexRetriever(index=module_indexes[m], similarity_top_k=47)
          for m in _algo_pool],
        *[BM25Retriever.from_defaults(nodes=module_nodes[m], similarity_top_k=47)
          for m in _algo_pool],
    ],
    similarity_top_k=47,
    mode="reciprocal_rerank",
    num_queries=1,
    use_async=False,
)

# Cross-module reranker: keep the top 15 fused chunks (replaces the old k_nodes=6
# truncation — the reranked set itself is the context).
combined_reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-large", top_n=15   # HEAVY (~2GB) — can swap for lighter base model
    # model="BAAI/bge-reranker-base", top_n=10
)

# ── Dependency map: symbol name -> the node that DEFINES it (built once) ───────
# Lets us follow a reference inside a retrieved chunk (a call to set_link_data, or
# an attribute self.port_speeds) to the chunk that defines/populates it, and append
# that definition to the grounding context. Pure regex + dict lookup, one hop, no
# call graph. This is what closes the "names the function but says 'not shown'" gap:
# the reference is already in context, so we fetch the definition it points at.
_DEF_NAME_RE = re.compile(r"\bdef\s+(\w+)\s*\(")                  # def NAME(
_ATTR_DEF_RE = re.compile(r"\bself\.(\w+)\s*(?:\[|[-+*/]?=)")     # self.NAME = / self.NAME[...] =
_REF_CALL_RE = re.compile(r"\b(\w+)\s*\(")                        # NAME(...) call sites
_REF_ATTR_RE = re.compile(r"\.(\w+)\b")                          # .NAME attribute access

# Names that are noise if "expanded" (builtins / numpy-networkx / common methods).
# The symbol map is itself the main filter — only codebase-defined names resolve —
# but this skips obvious calls early.
_EXPAND_STOPWORDS = {
    "self", "cls", "np", "nx", "math", "json", "ast", "re", "range", "len", "sum",
    "min", "max", "abs", "int", "float", "str", "list", "dict", "set", "tuple",
    "bool", "zeros", "ones", "full", "array", "asarray", "arange", "append",
    "get", "keys", "values", "items", "update", "setdefault", "print", "enumerate",
    "zip", "sorted", "format", "split", "join", "copy", "sum", "mean", "std",
}

def _node_uid(n):
    """Stable id for dedup across NodeWithScore wrappers and bare nodes."""
    base = getattr(n, "node", n)
    return getattr(base, "node_id", None) or getattr(base, "id_", None) or id(base)

def _build_symbol_to_node() -> Dict[str, any]:
    """Map every defined function/method/attribute name -> its defining node.
    function_name metadata is authoritative; def/self-attribute names parsed from
    the chunk text fill in the rest (without overwriting an authoritative entry)."""
    sym2node: Dict[str, any] = {}
    for m in CODE_MODULE_LABELS:
        for node in module_nodes[m]:
            fn_name = node.metadata.get("function_name")
            if fn_name:
                sym2node.setdefault(fn_name, node)
            content = node.get_content()
            for name in _DEF_NAME_RE.findall(content):
                sym2node.setdefault(name, node)
            for name in _ATTR_DEF_RE.findall(content):
                sym2node.setdefault(name, node)
    return sym2node

SYMBOL_TO_NODE: Dict[str, any] = _build_symbol_to_node()
print(f"[dep-expand] symbol->node map: {len(SYMBOL_TO_NODE)} symbols "
      f"across {len(CODE_MODULE_LABELS)} modules")

def _expand_with_definitions(reranked_nodes, max_defs: int = 8):
    """Given the reranked chunks, find the symbols they REFERENCE and append the
    chunks that DEFINE those symbols (one hop, deduped, capped). Returns the
    original reranked nodes followed by any added definition nodes."""
    seen = {_node_uid(n) for n in reranked_nodes}
    referenced = []  # (name, owning_function_name)
    for n in reranked_nodes:
        text = n.node.get_content()
        own = n.node.metadata.get("function_name")
        for name in _REF_CALL_RE.findall(text):
            referenced.append((name, own))
        for name in _REF_ATTR_RE.findall(text):
            referenced.append((name, own))

    additions = []
    added_names = set()
    for name, own in referenced:
        if len(additions) >= max_defs:
            break
        if name in _EXPAND_STOPWORDS or name == own or name in added_names:
            continue
        def_node = SYMBOL_TO_NODE.get(name)
        if def_node is None:
            continue
        uid = _node_uid(def_node)
        if uid in seen:
            continue
        seen.add(uid)
        added_names.add(name)
        additions.append(def_node)

    if additions:
        print(f"[dep-expand] +{len(additions)} definition chunk(s): "
              f"{', '.join(sorted(added_names))}")
    wrapped = [NodeWithScore(node=d, score=0.0) for d in additions]
    return list(reranked_nodes) + wrapped


# ── Prompt templates ─────────────────────────────────────────────────────────

# Wired into params_engine's text_qa_template (constructed but never .query()'d — see
# the NOTE near "Query engines" below); the actual runtime parameter-explanation path
# is _explain_params_from_map(), not this prompt.
PARAMS_PROMPT = PromptTemplate("""
You are a TSN parameter reference assistant.

Your job is to explain what the parameters mean and how they are used in TSN optimization.

Rules:
- Only use information from the provided context.
- Explain each parameter
- Be precise and concise — one paragraph per parameter maximum.
- If a parameter is not in the context, say: PARAMETER_NOT_FOUND: <name>

Context:
{context_str}

Request:
{query_str}

Parameter definitions:
""")

# Wired into lookup_engines (also never .query()'d — see the NOTE near "Query engines"
# below). The real stage-2 lookup, _lookup_one(), is pure retrieval + reranking against
# FUNCTION_CATALOG_BY_MODULE with NO LLM call, so this prompt is not read at runtime;
# kept as the reference schema _parse_lookup_json()/_parse_lookup_payload() expect.
LOOKUP_PROMPT = PromptTemplate("""
You are a precise function-lookup assistant for a TSN codebase.

Your ONLY job is to find and select functions matching the request in the context and return them
as a JSON object.

Output rules:
- Return ONLY a valid JSON object with "functions" and "coverage", NO PREAMBLE, NO MARKDOWN FENCES, NO EXTRA TEXT.

"name"          : "exact function name as it appears in the context",
"type"          : "classmethod | staticmethod | instance_method | function | property",
"module"        : "source filename (e.g. \"usecase_generator_functions.md\")",
"class"         : "class the function belongs to, or null for module-level functions",
"description"   : "one sentence describing what the function does, inferred from its code/docstring",
"signature"     : "full def line verbatim from context (e.g. \"def k_ary_fat_tree(cls, k, directed=True):\")",
"parameters"    : [
                    {
                        "name": "parameter name",
                        "default": "default value or null if required"
                    }
                    ],
"preconditions" : ["conditions that must be true before calling the function"],
"postconditions": [
                    "guarantees about state changes after execution",
                    "includes return value semantics (e.g. what is returned and its structure)",
                    "includes side effects (e.g. mutations, I/O, logging, external updates)"
                    ],
"callable"        : true,
"dependencies"  :   {
                        "internal": [
                            "project-level functions, classes, or modules used by this function",
                            "may include calls like nx.shortest_path, np.zeros, etc."
                        ],
                        "external": [
                            "third-party libraries or external services not implemented in the codebase"
                        ]
                    },
"raises"       : ["exceptions explicitly raised or propagated by the function"],
"example_use"  : "short realistic example showing how the function is called",
"code_ref"     : "exact function name — code will be fetched directly from source file",
"unit_tests"   : ["Tests functions covering normal cases, edge cases, and errors (inputs, outputs, exceptions)."]

- If multiple functions match, include all of them ranked by relevance.
  Add a "match_rank" key to each: "BEST_MATCH", "SECONDARY_MATCH", "TERTIARY_MATCH".
  The context chunks are already ordered by reranker score (highest first) —
  use that ordering as your relevance signal.
- If no function exists in the context, return exactly this object:
    {
        "functions": [],
        "coverage": {
            "match_type": "NO_MATCH",
            "recommended_action": "GENERATE_STANDALONE",
            "missing_requirements": ["No existing callable candidate was found."]
        }
    }
                               
IMPORTANT NOTES:
- It can be the case that the user query cannot be answered using one function.
  In that case you MUST return multiple functions. 
- If some functions that partially answer the user query are found then include them,
  these functions will be "helper" functions which you can use to answer a query partially.

Context:
{context_str}

Request:
{query_str}

Return ONLY a valid JSON object, NO MARKDOWN, NO PREAMBLE:
{
  "functions": [ <function objects as specified above, or [] if none> ],
  "coverage": {
    "match_type": "EXACT_MATCH" | "PARTIAL_MATCH" | "NO_MATCH",
    "missing_requirements": ["<capability requested but NOT provided by the functions>"]
  }
}

Coverage rules:
- EXACT_MATCH: the returned function(s), when called directly and safely, fully satisfy the request; no glue code needed; missing_requirements = [].
- PARTIAL_MATCH: returned functions are useful helpers but new wrapper/orchestration/logic is still needed; list what's missing.
- NO_MATCH: nothing useful; functions = [].
- High relevance rank alone does NOT justify EXACT_MATCH.
- Do not use EXACT_MATCH unless the call form, required context, and output semantics are all directly supported by the retrieved code.
""")

# Stage 3's real decision-maker: called from assess_function_coverage() to judge, by
# reading each candidate's actual code, whether it's EXACT_MATCH / PARTIAL_MATCH /
# NO_MATCH for the sub-query (reranker similarity scores alone can't tell these apart).
COVERAGE_ASSESSMENT_PROMPT = """\
You are a strict TSN codebase coverage assessor.

Your job is to decide whether the retrieved function candidates fully satisfy ONE
module-specific sub-query.

You are given:
1. BACKGROUND ONLY — the user's original request. It has ALREADY been split
   across modules; other modules independently handle the other parts. Use it
   only to disambiguate wording in the sub-query. NEVER judge coverage against it.
2. THE THING YOU ARE JUDGING — one module-specific sub-query.
3. The module that was searched.
4. Retrieved function candidates from that module.

Your single question is: do these candidates satisfy the SUB-QUERY?
Coverage of the original request as a whole is not your concern and must not
influence your answer.

Decide one of:

EXACT_MATCH:
- Use this only if the retrieved function or combination of retrieved functions directly and completely satisfies the sub-query.
- The user could call the returned function(s) with appropriate parameters and achieve the requested functionality.
- No additional new function or glue code is needed.

PARTIAL_MATCH:
- Use this if retrieved functions are relevant helpers that the new code would actually
    CALL, but do not fully satisfy the sub-query on their own.
- Some new wrapper, orchestration, glue code, or missing logic is needed.
- The helper functions should be returned and used for generation.

NO_MATCH:
- Use this if no retrieved function is useful for the sub-query.
- Use this if candidates are unrelated, vague, or only match generic words.
- A retrieved candidate only counts as a "helper" if the missing functionality would
    actually CALL it. If satisfying the sub-query only requires standard-library or
    third-party imports (numpy, random, json, os, networkx, etc.) and none of the
    retrieved candidates would be invoked by the new code, that is NO_MATCH — needing
    an external library is not the same as needing a codebase helper, regardless of how
    topically related the candidates look.

Important:
- Judge each candidate by what its code and docstring actually DO versus the SUB-QUERY,
  not by retrieval order or how confident retrieval seemed.
- If functions are empty, the answer must be NO_MATCH.
- If the SUB-QUERY itself asks for multiple actions and the retrieved functions cover
    only one of them, use PARTIAL_MATCH. Actions that belong to the original request but
    NOT to this sub-query are irrelevant here — ignore them completely.
- Use EXACT_MATCH only when the retrieved function(s) can be called directly and safely
    to satisfy the sub-query without inventing wrapper logic, missing context, or an invalid
    call form.
- Prefer PARTIAL_MATCH whenever the retrieved function(s) are useful but the SUB-QUERY still
    needs orchestration, adapters, or any non-trivial glue code.

CRITICAL — what does NOT count as "glue code", "orchestration", or "missing context".
This codebase is object-oriented: 77% of its functions are methods that take `self`,
and in the evaluator module it is 100%. None of the following is a reason to downgrade
EXACT_MATCH to PARTIAL_MATCH:
- The function is a METHOD that takes `self`. Having or constructing the owning object is
    normal call mechanics, not glue code. `evaluator.some_method(chromosome)` IS calling
    the function directly.
- The function reads instance state (`self.foo`, `self.bar`). That is where its inputs
    live. It is not "missing context".
- The function's name starts with an underscore. Private or internal naming does not make
    it the wrong answer. If it performs the requested computation, it is an EXACT_MATCH.
- The function returns a different SHAPE or TYPE than the user's phrasing implies — an
    array of counts where the user said "check whether", a per-item result where the user
    said "any". Reading or aggregating a return value is the caller's business, not
    missing functionality.
- The function is one of several related methods and the user would still need to know
    which one to pick. Naming the right one IS the answer.
- The function is a context-manager protocol method (`__enter__`/`__exit__`), or a public
    method whose job is to RETURN a context manager for use in a `with` block. The user
    does not need to construct that context-manager object manually or invoke
    `__enter__`/`__exit__` themselves — `with obj.some_method(x):` (or equivalent) IS the
    direct call form. Do not downgrade just because the actual state change happens
    inside `__enter__` rather than in the public method's own body, or because it lives
    on a different (often internal) helper class than the method exposed to the user.

Downgrade to PARTIAL_MATCH only when the requested BEHAVIOUR is genuinely absent: a
computation, check, or transformation that no candidate performs and that would have to be
written from scratch.

Worked example of the most common mistake:
    Original request: "generate a fat-tree, solve its schedule, then evaluate it."
    Sub-query:        "Generate a fat-tree use case with k=4 and 16 streams."
    Candidates:       k_ary_fat_tree, streamGenerator.
    Correct answer:   EXACT_MATCH — the SUB-QUERY is fully covered.
    Answering PARTIAL_MATCH because the candidates do not also solve or evaluate is
    WRONG. Those parts belong to other sub-queries and other modules.

Second worked example — the object-oriented trap:
    Sub-query:  "How can I check whether any talker starts outside its stream period?"
    Candidate:  def _per_stream_talker_offset_violations(self, chromosome) — compares each
                talker's start offset against self.stream_intervals and returns the
                per-stream violation counts.
    Correct answer: EXACT_MATCH. It performs exactly the requested check.
    Answering PARTIAL_MATCH because it is private, because it needs `self`, or because it
    returns counts rather than a boolean is WRONG. The user asked which function to use;
    this is that function.

Third worked example — the context-manager trap:
    Sub-query:  a request to activate, attach, or enter some stateful context before
                doing further work.
    Candidates: a public method whose entire body is a one-line
                `return SomeContext(self, x)`, plus the `__enter__` method of that
                context-manager class, which is where the described state change
                (setting an attribute, building derived data, etc.) actually happens.
    Correct answer: EXACT_MATCH on the public method, called as
                `with obj.public_method(x): ...`.
    Answering NO_MATCH or PARTIAL_MATCH because the real work happens inside `__enter__`
    on a different class, or because the public method looks like "just" a thin wrapper,
    is WRONG — using it inside a `with` block is the normal, direct way this pattern is
    meant to be called.

Return ONLY valid JSON, no markdown fences.

Required JSON schema:
{{
  "match_type": "EXACT_MATCH" | "PARTIAL_MATCH" | "NO_MATCH",
  "recommended_action": "RETURN_EXISTING_FUNCTIONS" | "GENERATE_WITH_HELPERS" | "GENERATE_STANDALONE"
}}

BACKGROUND — the user's original request (context only; do NOT judge coverage against this):
{original_query}

SUB-QUERY — this is what you are judging coverage against:
{sub_query}

Searched module:
{module}

Retrieved function candidates:
{functions_json}

JSON output:
"""

# ── Stage 4a: SPECIFY (what to build) ────────────────────────────────────────
# Called manually (via Settings.llm.complete) from generate_function_spec() whenever
# coverage is PARTIAL_MATCH or NO_MATCH. This prompt does exactly ONE thing: describe
# the function that has to exist. It never writes code — spec extraction and code
# generation are deliberately separate LLM calls, so the specification is an
# independent artefact the user can review and amend BEFORE anything is implemented
# (and before the contract/tests are derived from it).
FUNCTION_SPEC_PROMPT = PromptTemplate("""
You are a software specification analyst for a TSN (Time-Sensitive Networking) codebase.

Your ONLY job is to specify the single function that must be built to satisfy the request.
You do NOT write an implementation. You do NOT return any code.

How to use the context:
- The context and the helper list show what this codebase ALREADY provides. Anything that
  fits the request must be named under dependencies.internal, using its exact name.
- Whatever the codebase does NOT already provide is what this specification has to cover.
- If a needed capability requires a third-party library (e.g. networkx, numpy, lxml/xml,
  ncclient), declare that library under dependencies.external.

WHAT SHAPE TO SPECIFY — read this before writing the signature:
- DEFAULT, and what you should specify almost every time: ONE PLAIN MODULE-LEVEL FUNCTION.
  Its signature must NOT begin with `self` or `cls`, and "type" must be "function" with
  "class": null. This holds even when the logic conceptually belongs to
  TSNConstraintEvaluator / UsecaseGenerator / TSNScheduling: the generated code is written
  to its own module and called directly, so a `self` parameter is a fiction that nothing
  can supply, and specifying one is the single largest cause of unusable output.
- Every piece of state the function needs — anything you would otherwise have written as
  `self.<attr>` — becomes an EXPLICIT NAMED PARAMETER instead, named exactly after the
  attribute (self.n_ports -> n_ports, self.all_hops_costs -> all_hops_costs), listed in
  "parameters", and given its concrete shape/dtype in "preconditions". Order them: the
  request's own inputs first (chromosome, stream, ...), then the state parameters.
- ONLY exception: if the request explicitly asks for a CLASS (a "class", a context
  manager, a stateful decorator, an object with several methods), specify that one
  self-contained class instead — set "type": "class", put the class name in "name", and
  make "signature" the `class <Name>:` line. Its methods may use `self` normally, because
  the class is generated whole and the caller instantiates it through its own __init__.
  Never specify a single method belonging to an existing project class.

Rules:
- Specify exactly ONE function (or, under the exception above, ONE class).
- "description" must be detailed enough that a developer who never sees this request could
  build the function from the specification alone: what it consumes, what it computes, what
  it returns, and how it behaves at the edges.
- "signature" must be a real, complete def line, and must agree with "parameters".
- Never invent project class names or helper names that do not appear in the context.
- preconditions / postconditions must be observable and checkable statements about the
  function's contract, not notes about how to implement it.
- If a frozen interface appears in the Request below, this function may already have a
  real, complete body there instead of just a signature. When it does, every claim you
  make about its return value's TYPE and STRUCTURE — dict vs tuple, which fields/keys
  it has, the exact casing of a status string — must come from literally reading that
  function's own return statement(s), never from reasoning about what a well-designed
  version of this function would return. Confirmed on a real run: asked to specify a
  function whose frozen body already returned a list of dicts with keys
  type/key/stream_id/start_value/target_value and lowercase added/removed/modified, the
  spec instead described an unrelated, invented shape — a sorted list of 5-element
  tuples with an uppercase ADDED/DELETED/UPDATED category — that has nothing to do with
  the code it was given. If the real shape looks inelegant, specify the inelegant thing;
  a description of a nicer function that isn't the one in front of you is not a
  correction, it is a fabrication the test-writer will build an entire wrong suite on.
- For each state parameter (the ones that would have been `self.<attr>`), give its
  concrete shape and dtype (or type) as a short precondition entry — e.g.
  "stream_intervals: 1D numpy array, shape (n_streams,), float" — not just
  "stream_intervals is precomputed". This is the ONLY description of that parameter's
  shape the code generator and the test writer will ever see, and they work from it
  independently; leave it vague and each will guess a shape, with nothing guaranteeing the
  guesses agree — the generated code can then fail on a plain shape-broadcast error
  unrelated to whether the logic itself is correct. If you are not fully certain of the
  real shape, state your single best concrete guess instead of leaving it unstated. Keep
  each entry to ONE short line: name, shape, dtype — no elaboration. If the function
  genuinely needs more than about 6-8 such parameters, take only the ones the
  description's postconditions actually depend on for their observable behavior (the ones
  a test would need to set up to exercise that behavior) — drop state that is merely read
  defensively or passed through. A complete, valid, slightly-less-exhaustive JSON object
  is far more useful than one that runs out of room mid-array and never closes.
- Write parameter names bare in preconditions and postconditions: `n_ports`, never
  `self.n_ports`. There is no `self` for a plain function, and a precondition that talks
  about one tells the test writer to build an object that the implementation never receives.
- "raises" lists the exceptions a caller can observe. List an exception only if the
  function itself is specified to raise it for a caller-reachable input; do not list
  errors that would only ever come from a broken environment or a malformed internal
  state, because every entry here becomes a test that demands that exception.
- dependencies.internal is for PROJECT symbols only — functions, classes and modules that
  live in this codebase (e.g. TSNConstraintEvaluator, ScheduleEncodingType, a helper
  function's exact name). Third-party and standard-library modules (numpy/np, scipy's
  csr_matrix, networkx, configparser, json, time, ...) belong in dependencies.external.
  A piece of state the function receives (n_ports, all_hops_costs, ...) is NOT a
  dependency at all: it is a parameter, so list it under "parameters", describe its shape
  in preconditions as required above, and leave it out of both dependency lists.

Output rules:
- Return ONLY a valid JSON object, no preamble, no markdown fences, no extra text.
- THE JSON MUST BE COMPLETE AND CLOSED. Every brace and bracket you open must be closed,
  every array must be terminated, and the object must end with its final closing brace.
  Never stop part-way through "parameters" or any other array. A specification that ends mid-array
  cannot be parsed at all and the entire request is thrown away — so before you emit a
  long "parameters" list, SHORTEN IT: a complete object describing 8 parameters is
  infinitely more useful than a truncated one that was going to describe 24. If you are
  running long, cut parameters, not the closing braces.
- NEVER LIST THE SAME PARAMETER TWICE. Each parameter name appears exactly once in
  "signature" and exactly once in "parameters", and the two must contain the same names in
  the same order. A repeated name (e.g. `..., listener_adj_mask, adjacency_mask,
  to_node_mask, listener_adj_mask, ...`) is not valid Python: the def line raises
  "duplicate argument in function definition" and nothing you specified can be built.
  Re-read your signature and check for repeats before you emit it.
- Use exactly these keys, and no others:

{
"name"          : "exact function name (or class name when type is \"class\")",
"type"          : "function   — use this unless the request explicitly asks for a class, in which case use \"class\"",
"class"         : null,
"description"   : "detailed description of what sort of function needs to be constructed",
"signature"     : "full def line, WITHOUT a leading self/cls parameter (or the `class <Name>:` line when type is \"class\")",
"parameters"    : [
                    {
                        "name": "parameter name",
                        "default": "default value or null if required"
                    }
                    ],
"preconditions" : [
                    "conditions that must be true before calling the function",
                    "for each internal self.* dependency: its concrete shape/dtype/type, e.g. 'self.stream_intervals is a 1D numpy array of shape (n_streams,)'"
                    ],
"postconditions": [
                    "guarantees about state changes after execution",
                    "includes return value semantics (e.g. what is returned and its structure)",
                    "includes side effects (e.g. mutations, I/O, logging, external updates)"
                    ],
"callable"        : true,
"dependencies"  :   {
                        "internal": [
                            "project-level functions, classes, or modules used by this function"
                        ],
                        "external": [
                            "third-party libraries or external services not implemented in the codebase"
                        ]
                    },
"raises"       : ["exceptions explicitly raised or propagated by the function"]
}

Context:
{context_str}

Request:
{query_str}
{revision_block}
JSON specification:
""")


# ── Stage 4b: IMPLEMENT (build what was specified) ───────────────────────────
# Called manually (via Settings.llm.complete, not through gen_engines) from
# generate_code_from_spec(), AFTER the user has approved the contract-derived tests.
# It is given the spec MINUS its "raises" key (see
# _spec_for_codegen()) and returns raw Python source — no JSON, no metadata. Every
# non-code field of the generated function dict comes from the spec, so the
# implementation can never quietly redefine its own contract.
GENERATOR_PROMPT = PromptTemplate("""
You are a TSN algorithm code generator. You are given an APPROVED SPECIFICATION for one
function, and you must implement exactly that function. You ALWAYS return code — never a
refusal, never a stub, never "NOT_IN_CONTEXT".

The specification is authoritative:
- Implement the function named in "name", using EXACTLY the "signature" given.
- EXCEPT that no parameter may appear twice in your def line. If the specification's
  "signature" happens to list the same name more than once, keep only its FIRST
  occurrence and drop the later ones — a duplicated parameter is a Python syntax error
  ("duplicate argument in function definition") and the whole file then fails to load.
  Everything else about the signature stays exactly as specified.
- Assume every "preconditions" entry holds; guarantee every "postconditions" entry.
- Call the helpers listed in dependencies.internal by their exact names.
- You may import anything listed in dependencies.external, plus the Python standard library.

Calling convention for helpers — each helper in "Available helper functions" below carries
a "method_type" field. Get this wrong and the call raises TypeError or silently does the
wrong thing. Use EXACTLY the form for that helper's method_type:
- "function": a plain module-level function. Call it directly: helper_name(args).
    Real example — `drawFattree` is a module-level function in usecase_generator.py:
        from usecase_generator import drawFattree
        drawFattree(usecase.graph)
- "instance_method": its first parameter is `self`, so it needs a real instance of its
    "class_name" ("sub_class_name" when the helper is nested) to be called on. What you
    are writing is NOT a method of that class (see Rules below: a "function" spec has no
    `self` at all, and a "class" spec defines a new class of its own), so such an instance
    can only reach you as one of your own declared parameters.
    You are writing code against a signature, not running it: you can see the parameter
    NAMES the signature declares (and whatever the preconditions say each one holds), and
    you never see their runtime VALUES. So decide whether an instance is available by
    reading the signature and preconditions, and then write the call against that
    parameter NAME — never by imagining, inspecting or assuming what the value will be:
        # signature: def summarise_feasibility(evaluator, chromosome):
        # preconditions say `evaluator` is a TSNConstraintEvaluator, so its methods are
        # reachable through that name, whatever it turns out to hold at run time
        violations_array = evaluator.per_stream_all_violations(chromosome)
    If the signature declares no parameter holding such an instance, do NOT call the
    helper and do NOT construct a project object just to reach it: work from the parameter
    names you do have, or inline the small piece of logic you need. Inside a "class" spec,
    `self.<...>` refers only to YOUR OWN class's methods and attributes, never to a
    project class's.
- "classmethod": its first parameter is `cls`, bound automatically by Python. Call it on
    the class itself: ClassName.helper_name(args) — never pass the class or `cls`
    explicitly as an argument.
    Real example — UsecaseGenerator.generateUC is a @classmethod:
        from usecase_generator import UsecaseGenerator
        usecase = UsecaseGenerator.generateUC(config_file)
- "staticmethod": takes neither `self` nor `cls`. Call it on the class exactly like a
    plain function attached to a namespace: ClassName.helper_name(args).
    Real example — Usecase.generate_fattree_regions is a @staticmethod:
        from usecase_generator import Usecase
        regions = Usecase.generate_fattree_regions(4)
- "property": NOT callable — it is a computed attribute. Access it as
    instance.helper_name (no parentheses). Calling it as instance.helper_name() will
    raise TypeError (or silently misbehave) because the result of the property, not
    the property itself, is what gets "called".
    Real example — Usecase.hosts is a plain @property:
        host_nodes = usecase.hosts          # correct
        host_nodes = usecase.hosts()        # WRONG: TypeError, list is not callable

How the three real modules are actually organised — this is what the helpers above belong
to, and the rules here override any guess you would otherwise make:

1. usecase_generator.UsecaseGenerator — EVERY public member is a @classmethod (generateUC,
   ucGenerator, empty_uc, streamGenerator, generateStream, talkerEmbedding,
   listenerEmbedding, findAllPaths, shortestPath, randomPath, pathConvenient,
   periodConvenient, makeTopology, set_link_data, k_ary_fat_tree, makeRandomGraph,
   changeDegree). Never instantiate it; call on the class, and pass the usecase in:
       stream = UsecaseGenerator.shortestPath(stream, usecase)
       graph  = UsecaseGenerator.k_ary_fat_tree(4)

2. usecase_generator.Usecase — a normal object you hold an instance of. Its properties
   (hosts, switches, links, ports, port_name_id_mapping, distance_to_hosts) are plain
   @property: read them as attributes. Its ordinary methods take real arguments:
       wl = usecase.window_length(stream_idx, port_name)   # instance method
       uc = Usecase.from_json(imported_dict)               # @classmethod
   Note `usecase.ports[port_name]['speed']` is in Mbps and the library converts with the
   module constant MBPS_2_BPNS (import it, do not hardcode 0.001).

3. evaluator.TSNConstraintEvaluator — a normal object; ALL of its methods are plain
   instance methods (no decorators anywhere in the class). It is constructed FROM a
   TSNScheduling problem, which is where its ~40 precomputed attributes come from:
       evaluator = TSNConstraintEvaluator(problem, check_routing=True)
   Construct it explicitly like that whenever the schedules being evaluated are based
   on their transmission START TIMES (START_TIME encoded) rather than on QUEUE
   PRIORITIES (PRIORITY encoded) — that is, whenever a non-zero cell means "this
   stream starts transmitting on this port at this time" and not "this stream sits
   in this queue". initialize_problem creates problem.evaluator without ever passing
   scheduling_type, so the pre-built instance is always PRIORITY and is only ever
   correct for priority-based schedules. Decide which of the two your cells hold
   before you evaluate anything: the evaluator reads them one way or the other
   whether or not your own logic ever looks at the values.
       violation_sums, metrics = evaluator.evaluate(chromosome)
   `self.scheduling_type` holds a ScheduleEncodingType ENUM member, never a string:
       from evaluator import ScheduleEncodingType
       if self.scheduling_type == ScheduleEncodingType.PRIORITY:   # correct
       if self.scheduling_type == "PRIORITY":                      # WRONG: always False
   Its per-chromosome caches are keyed by `chromosome.tobytes()`, never by the array
   itself (a numpy array is unhashable and using one as a dict key raises TypeError):
       key = chromosome.tobytes()
       cached = self._violation_evaluation_cache.get(key)
   UnionFind is a class NESTED inside TSNConstraintEvaluator, so it is reached through the
   owner — from inside the class `self.UnionFind(active_ports)` (what group_connected_ports
   does), from outside `TSNConstraintEvaluator.UnionFind(active_ports)`.

4. tsn_scheduling_problem.TSNScheduling — the one that needs real care. Most of its
   members carry the module's own @require_instance decorator, and its read-only views
   (n_ports, n_streams, ports, port_id_to_name, port_name_to_id, port_id_to_node, nodes,
   n_nodes, node_to_port_id, macro_cycle, streams, links) are @property STACKED ON TOP of
   @require_instance. require_instance checks `self.usecase` and, when it is unset, calls
   no_instance_assigned(), which RAISES ValueError("No current instance is set!..."). So
   these are not merely "call it on an instance" — the instance must additionally have an
   active usecase, and the supported way to establish that is the context manager:
       with problem.use_instance(instance):
           n = problem.n_ports                     # @property @require_instance
           cost = problem.hops_costs(frame_size)   # @require_instance method
   Entering that block also runs build_metadata() once per usecase. Outside such a block
   every one of those names raises ValueError. Since what you write is not a method of
   TSNScheduling, this only concerns you when your signature DECLARES a parameter that the
   preconditions describe as a TSNScheduling object — and then you must open the block
   yourself, written against that parameter name, before touching any of those names:
       # signature: def port_count_for(problem, instance):
       with problem.use_instance(instance):
           return problem.n_ports
   If instead the signature declares the derived values as plain parameters of their own
   (n_ports, ports, macro_cycle, ...), use those parameter names directly and do not reach
   for a TSNScheduling object at all.
   Exceptions in this class that do NOT need the with-block:
   _frame_size is a plain @classmethod (TSNScheduling._frame_size(stream)), and drawFattree
   and plot_all_stream_custom_graphs are plain @staticmethods.

Two same-name traps in this codebase — check "class_name" before you call, because the two
are not interchangeable:
- `_frame_size` is a @classmethod on TSNScheduling taking a stream DICT
  (TSNScheduling._frame_size(stream)), but an INSTANCE METHOD on Usecase taking a stream
  INDEX (usecase._frame_size(stream_idx)).
- `drawFattree` is a module-level function in usecase_generator (drawFattree(graph)) and
  also a @staticmethod on TSNScheduling (TSNScheduling.drawFattree(graph, ax=None)).

Rules:
 - Write ONE complete definition: the spec's def line, a docstring, and a real, runnable body.
 - When the spec's "type" is "function" (the normal case): define it at module level with
   EXACTLY the parameters the signature lists, and do NOT wrap it in a class. The signature
   has no `self`/`cls`, so never read state off an object — every value you need is either
   one of your own parameters or something you import. If you find yourself wanting
   `self.<attr>`, use the parameter of that name instead; if there is no such parameter,
   derive the value from the parameters you do have.
 - When the spec's "type" is "class": emit that complete class — the `class <Name>:` line
   and every method the specification describes, `self` included, plus an `__init__` that
   accepts what the preconditions say the caller supplies. The caller instantiates it
   normally, so it must be constructible and usable on its own.
 - Prefer runnable logic over stubs. Do not invent project class names, helper names, or
   signatures, and do not re-declare helpers that already exist in the context.
 - The real project modules ARE importable while this function runs, so import the real
   symbols you need instead of redefining, faking or re-implementing them — e.g.
   `from evaluator import ScheduleEncodingType, TSNConstraintEvaluator` or
   `from usecase_generator import UsecaseGenerator, Usecase, MBPS_2_BPNS, BYTE_2_BIT`.
   Compare enum-valued state against the real enum member (ScheduleEncodingType.PRIORITY),
   never against a bare string like "PRIORITY": the real attribute holds an enum, so a
   string comparison is silently always False and the branch never runs. Import only what
   you actually use, at the top of the file.
 - Use only the parameters the signature declares. If you need a value the signature does
   not provide, derive it from the parameters you were given rather than inventing a new
   one or reaching for object state that does not exist here.
 - Every `if`/`for`/`while`/`try`/`def` you open must be followed by a real indented block,
   and every string and docstring must be closed. The output is compiled before it runs;
   source that does not parse is rejected outright.
 - The docstring must end with an "Integration:" line stating, in one sentence, exactly how
   to fold this into the real codebase: that it can be pasted at module level in the target
   module as-is, and — if its parameters mirror attributes of an existing class — that it
   can alternatively become a method of that class by taking `self` and reading those
   values off it instead of receiving them as arguments.

Output rules:
- Return ONLY the Python source of that one function. No JSON, no markdown fences, no prose.

Approved specification:
{spec_json}

Context (existing codebase helpers):
{context_str}

Original request (background only — on any conflict the specification wins):
{query_str}

Implementation:
""")

# The GENERATE_ALGORITHM counterpart to GENERATOR_PROMPT. Used only for sub-queries whose
# action is Intent.GENERATE_ALGORITHM (a request to design/implement a multi-part
# algorithm/strategy/heuristic, not a single straightforward function) — see
# generate_algorithm_from_spec() and generate_verified_implementation()'s
# use_algorithm_prompt flag. Everything upstream (spec, contract, spec-based tests,
# approval gate) and downstream (execute/repair) of the implementation step is IDENTICAL
# to the normal generation path; only this one call is swapped in for GENERATOR_PROMPT.
GENERATE_ALGORITHM_PROMPT = PromptTemplate("""
You are an algorithm synthesis agent operating on an existing codebase.

Your goal is to turn an algorithm idea into a working implementation while maximizing
reuse of the existing codebase and minimizing unnecessary new code.

Do NOT immediately generate code from the user's initial request.

Follow this workflow:

1. UNDERSTAND & CLARIFY
   - Analyze the user's algorithm idea.
   - Identify ambiguities, missing assumptions, constraints, inputs, outputs, objectives, and evaluation criteria.
   - Ask targeted questions when clarification would materially affect the algorithm.
   - Do not ask unnecessary questions.
   - augment the user input with necessary details: search the internet for concepts that are not clarified and if not available ask the user to clarify them.

2. FORMALIZE
   - Convert the clarified idea into a precise algorithmic problem.
   - Define inputs, outputs, state, constraints, objective function(s), and validity conditions.
   - Identify what constitutes a successful solution.

3. DECOMPOSE
   - Break the problem into concrete subtasks/components.
   - Distinguish between:
     a. capabilities already likely available in the codebase,
     b. capabilities that can be composed from existing functions,
     c. genuinely missing capabilities that must be implemented.

4. CODEBASE ANALYSIS
   - Search the codebase for relevant functions, classes, algorithms, data structures, and utilities.
   - Prefer existing implementations over creating duplicates.
   - Understand function interfaces, assumptions, side effects, and dependencies before reusing them.
   - Identify candidate functions and explain why each is relevant.

5. ALGORITHM DESIGN
   - Design the algorithm as a high-level orchestration layer around reusable capabilities.
   - Explicitly define the data flow between components and their order of execution.
   - Decide how candidate solutions are generated, evaluated, selected, and validated.
   - When multiple approaches are possible, compare them and choose the most appropriate one based on the user's objective and the available code base.
   - Do not force every part of the solution into existing functions if they are insufficient.

6. GAP ANALYSIS & COMPOSITION
   - Utilize existing functions where possible.
   - If a required capability is missing, implement only that capability.
   - Keep new functionality modular and compatible with the existing architecture.
   - Avoid rewriting existing functionality unless necessary.

7. PLAN BEFORE IMPLEMENTATION
   Present a concise implementation plan containing:
   - proposed algorithm,
   - subtasks,
   - reused functions,
   - functions to compose,
   - new functions required,
   - top-level orchestration,
   - expected inputs/outputs,
   - validation strategy.

   Wait for user approval if the system is operating interactively and approval is required.

8. IMPLEMENT
   - Implement the smallest coherent set of changes needed.
   - Build the final algorithm as an umbrella/orchestration layer that calls the reusable components.
   - Preserve existing APIs and behavior whenever possible.

9. VALIDATE
   - Check correctness against the formalized requirements.
   - Test edge cases and invalid states.
   - Verify that intermediate states satisfy the relevant constraints.
   - Verify that the final state matches the requested target.
   - Evaluate the algorithm against its stated objective(s).

10. EXPLAIN
   - Explain the resulting architecture and algorithm.
   - Clearly distinguish reused code, composed functionality, and newly generated functionality.
   - Explain important design decisions and limitations.

GENERAL PRINCIPLES:
- Reuse before reinventing.
- Compose before rewriting.
- Implement gaps only when necessary.
- Separate problem understanding from implementation.
- Prefer modular algorithms over monolithic generated code.
- Make assumptions explicit.
- Treat constraints and validity conditions as first-class requirements.
- If the user's objective is ambiguous, clarify it before optimizing for it.
- Do not assume that the first plausible algorithm is the best one; consider alternatives when appropriate.
- The final implementation should be understandable, testable, and integrated with the existing codebase.

IMPORTANT — HOW THIS WORKFLOW RUNS HERE:
- You are being called as a single, non-interactive completion: there is no live
  back-and-forth with a user available to you right now, and you have no internet access.
  Where step 1 says to "ask targeted questions" or "search the internet", instead state
  your best-judgment assumption explicitly and proceed — never stall or refuse for
  missing clarification.
- Step 7's "wait for user approval" is already covered by this pipeline's own review
  step, which happens AFTER this call, over the spec-based tests derived from what you
  implement — so do not wait; go straight from your internal plan (step 7) to writing
  the implementation (step 8).
- Work through steps 1-7 in your own reasoning, but your FINAL output must still be
  nothing but the Python source described under "Output rules" below — only the code
  from steps 8-9 is kept; the plan and explanation are for your own reasoning, not
  the output.
- Steps 2-6 were done by the earlier stages: the design blueprint and the frozen interface
  below ARE the design. Implement them; do not redesign them.

Calling convention for helpers — each helper in the Context section below carries
a "method_type" field. Get this wrong and the call raises TypeError or silently does the
wrong thing. Use EXACTLY the form for that helper's method_type:
- "function": a plain module-level function. Call it directly: helper_name(args).
    Real example — `drawFattree` is a module-level function in usecase_generator.py:
        from usecase_generator import drawFattree
        drawFattree(usecase.graph)
- "instance_method": its first parameter is `self`, so it needs a real instance of its
    "class_name" ("sub_class_name" when the helper is nested) to be called on. What you
    are writing is NOT a method of that class, so such an instance can only reach you as
    one of your own declared parameters (or a parameter of a private helper you write).
    You are writing code against a signature, not running it: you can see the parameter
    NAMES the signature declares (and whatever the preconditions say each one holds), and
    you never see their runtime VALUES. So decide whether an instance is available by
    reading the signature and preconditions, and then write the call against that
    parameter NAME — never by imagining, inspecting or assuming what the value will be:
        # signature: <a function taking an evaluator and a chromosome>
        # preconditions say `evaluator` is a TSNConstraintEvaluator, so its methods are
        # reachable through that name, whatever it turns out to hold at run time
        violations_array = evaluator.per_stream_all_violations(chromosome)
    If the signature declares no parameter holding such an instance, do NOT call the
    helper and do NOT construct a project object just to reach it: work from the parameter
    names you do have, or inline the small piece of logic you need.
- "classmethod": its first parameter is `cls`, bound automatically by Python. Call it on
    the class itself: ClassName.helper_name(args) — never pass the class or `cls`
    explicitly as an argument.
    Real example — UsecaseGenerator.generateUC is a @classmethod:
        from usecase_generator import UsecaseGenerator
        usecase = UsecaseGenerator.generateUC(config_file)
- "staticmethod": takes neither `self` nor `cls`. Call it on the class exactly like a
    plain function attached to a namespace: ClassName.helper_name(args).
    Real example — Usecase.generate_fattree_regions is a @staticmethod:
        from usecase_generator import Usecase
        regions = Usecase.generate_fattree_regions(4)
- "property": NOT callable — it is a computed attribute. Access it as
    instance.helper_name (no parentheses). Calling it as instance.helper_name() will
    raise TypeError (or silently misbehave) because the result of the property, not
    the property itself, is what gets "called".
    Real example — Usecase.hosts is a plain @property:
        <the hosts> = usecase.hosts          # correct
        <the hosts> = usecase.hosts()        # WRONG: TypeError, list is not callable

How the three real modules are actually organised — this is what the helpers above belong
to, and the rules here override any guess you would otherwise make:

1. usecase_generator.UsecaseGenerator — EVERY public member is a @classmethod (generateUC,
   ucGenerator, empty_uc, streamGenerator, generateStream, talkerEmbedding,
   listenerEmbedding, findAllPaths, shortestPath, randomPath, pathConvenient,
   periodConvenient, makeTopology, set_link_data, k_ary_fat_tree, makeRandomGraph,
   changeDegree). Never instantiate it; call on the class, and pass the usecase in:
       stream = UsecaseGenerator.shortestPath(stream, usecase)
       graph  = UsecaseGenerator.k_ary_fat_tree(4)

2. usecase_generator.Usecase — a normal object you hold an instance of. Its properties
   (hosts, switches, links, ports, port_name_id_mapping, distance_to_hosts) are plain
   @property: read them as attributes. Its ordinary methods take real arguments:
       wl = usecase.window_length(stream_idx, port_name)   # instance method
       uc = Usecase.from_json(<a dict read from JSON>)               # @classmethod
   Note `usecase.ports[port_name]['speed']` is in Mbps and the library converts with the
   module constant MBPS_2_BPNS (import it, do not hardcode 0.001).

3. evaluator.TSNConstraintEvaluator — a normal object; ALL of its methods are plain
   instance methods (no decorators anywhere in the class). It is constructed FROM a
   TSNScheduling problem, which is where its ~40 precomputed attributes come from:
       evaluator = TSNConstraintEvaluator(problem,
                                          scheduling_type=ScheduleEncodingType.START_TIME)
   WHICH evaluator to build is NOT yours to decide: the encoding was settled before the
   blueprint was written, and the blueprint's reuse decision for TSNConstraintEvaluator
   carries the exact call form. Use that call form, inside the activation block, in the
   function the blueprint says builds the evaluator; the line above only shows the shape
   of a START_TIME construction (check_routing and check_task_placement default to True).
   Why it matters: initialize_problem creates problem.evaluator without ever passing
   scheduling_type, so the pre-built instance is always PRIORITY - it reads a non-zero
   cell as "this stream sits in this queue" - and a START_TIME schedule ("this stream
   starts transmitting on this port at this time") evaluated by it scores every timing
   violation as zero.
       <the violation sums>, <the metrics> = evaluator.evaluate(chromosome)
   evaluate() returns TWO NUMPY ARRAYS, never dicts — do not call .values() on them or
   index them by name. metrics always has length 6, ordered congestion_score, avg_util,
   std_util, load_imbalance, path_len_score, avg_path_len.
   The violation sums carry one entry per VIOLATION KIND, which is NOT the same as one per
   check: a single check function returns several rows (_per_stream_host_role_violations
   alone returns 5), so there are 11 kinds from 5 checks under PRIORITY and 14 kinds from
   7 checks under START_TIME. The row order, read out of the evaluator's own source:
{violation_row_order}
   Do NOT try to recover this mapping at run time from `evaluator.violations_keys`: that
   is `relevant_weight_keys`, a SET, so its iteration order has no relationship to these
   rows. The order above is the only mapping there is — it comes from the order checks are
   appended in __init__ plus each check's own vstack order.
   One caveat, real and in the library: _per_stream_window_sequence_violations returns 1
   row instead of 2 when a topology has no upstream ports, which shifts every row after
   it. So if you branch on a specific violation kind, prefer calling that check through
   per_stream_all_violations and summing, or slice defensively — do not hard-code an
   index and assume it holds on every topology.
   The chromosome it takes is a 2-D int array indexed [port, stream] where 0 means
   inactive; under PRIORITY a non-zero cell is a queue priority, under START_TIME it is
   a transmission start time PLUS ONE.
   The evaluator's macrocycle_overlap check counts two windows on one port as colliding when
   (phase1 - phase2) mod gcd(period1, period2) < window1 + window2, in either direction. This is
   stricter than Slot.conflicts_with and Schedule.validate_conflicts, which report only windows
   that actually overlap.
   `self.scheduling_type` holds a ScheduleEncodingType ENUM member, never a string:
       from evaluator import ScheduleEncodingType
       if self.scheduling_type == ScheduleEncodingType.PRIORITY:   # correct
       if self.scheduling_type == "PRIORITY":                      # WRONG: always False
   Its per-chromosome caches are keyed by `chromosome.tobytes()`, never by the array
   itself (a numpy array is unhashable and using one as a dict key raises TypeError):
       key = chromosome.tobytes()
       cached = self._violation_evaluation_cache.get(key)
   UnionFind is a class NESTED inside TSNConstraintEvaluator, so it is reached through the
   owner — from inside the class `self.UnionFind(active_ports)` (what group_connected_ports
   does), from outside `TSNConstraintEvaluator.UnionFind(active_ports)`.

4. tsn_scheduling_problem.TSNScheduling — the one that needs real care. Most of its
   members carry the module's own @require_instance decorator, and its read-only views
   (n_ports, n_streams, ports, port_id_to_name, port_name_to_id, port_id_to_node, nodes,
   n_nodes, node_to_port_id, macro_cycle, streams, links) are @property STACKED ON TOP of
   @require_instance. require_instance checks `self.usecase` and, when it is unset, calls
   no_instance_assigned(), which RAISES ValueError("No current instance is set!..."). So
   these are not merely "call it on an instance" — the instance must additionally have an
   active usecase, and the supported way to establish that is the context manager:
       with problem.use_instance(instance):
           n = problem.n_ports                     # @property @require_instance
           cost = problem.hops_costs(frame_size)   # @require_instance method
   Entering that block also runs build_metadata() once per usecase. Outside such a block
   every one of those names raises ValueError. Since what you write is not a method of
   TSNScheduling, this only concerns you when your signature DECLARES a parameter that the
   preconditions describe as a TSNScheduling object — and then you must open the block
   yourself, written against that parameter name, before touching any of those names:
       # signature: <a function taking the problem and an instance>
       with problem.use_instance(instance):
           return problem.n_ports
   If instead the signature declares the derived values as plain parameters of their own
   (n_ports, ports, macro_cycle, ...), use those parameter names directly and do not reach
   for a TSNScheduling object at all.
   Exceptions in this class that do NOT need the with-block:
   _frame_size is a plain @classmethod (TSNScheduling._frame_size(stream)), and drawFattree
   and plot_all_stream_custom_graphs are plain @staticmethods.

5. schedule.Slot and schedule.Schedule — the compiled-schedule model. Slot is a
   @dataclass (start_ns, end_ns, period_ns, stream_id, port_id, queue_id, index) whose
   only property is duration_ns. Schedule SUBCLASSES list, so a Schedule instance IS the
   container of its Slots — append to it, index it and iterate it directly; there is no
   .slots attribute to reach for:
       from schedule import Schedule, Slot
       sch = Schedule(cycle_ns=macro_cycle, basetime=0)
       sch.append(Slot(start_ns=..., end_ns=..., period_ns=..., stream_id=i, port_id=p))
   Two @classmethods build one: Schedule.from_compact_encoding(compact_encoding,
   windLengths, stream_intervals, cycle_ns=..., basetime=0, tt_queue_id=0) and
   Schedule.from_usecase(usecase, port_name_to_id). compact_encoding must be a 2-D array
   shaped (n_ports, n_streams), never one 1-D column: the column index IS the stream id
   (it picks stream_intervals[j] and windLengths[:, j]), so a stream's windows are read from
   column j of that array and nowhere else. Everything else is a plain instance
   method: to_compact_encoding(n_ports, n_streams), slots_by_port(port_id),
   slots_by_stream(stream_id), validate_conflicts(touching_allowed=True),
   update_graph(graph, port_id_to_name), update_streams(streams, ports, port_id_to_name),
   plot(...). TransmissionWindow is a @dataclass NESTED inside Schedule, reached as
   Schedule.TransmissionWindow(...).
   Port ids here are 1-BASED: from_compact_encoding stores port_idx + 1 and
   to_compact_encoding writes back to row port_id - 1. Never mix a 0-based row index
   with a port id.
Three same-name traps in this codebase — check "class_name" before you call, because the
members are not interchangeable:
- `_frame_size` is a @classmethod on TSNScheduling taking a stream DICT
  (TSNScheduling._frame_size(stream)), but an INSTANCE METHOD on Usecase taking a stream
  INDEX (usecase._frame_size(stream_idx)).
- `drawFattree` is a module-level function in usecase_generator (drawFattree(graph)) and
  also a @staticmethod on TSNScheduling (TSNScheduling.drawFattree(graph, ax=None)).
- `ports` and `streams` exist on BOTH Usecase and TSNScheduling and are keyed
  differently. usecase.ports is keyed by port NAME (a string like 'host-1-p0.Port1') and
  usecase.streams[i]["pathLists"][j] holds port NAMES, ending at the destination node;
  problem.ports is keyed by port ID (an int) and problem.streams[i]["pathLists"][j] holds
  port IDs. They are not the same object either. Reading one and indexing the other is
  the most common failure here and it surfaces late, as a KeyError on an integer — so
  decide from the preconditions which of the two a parameter holds, and stay on it.

Real config.ini shape — if any function you implement constructs a TSNScheduling or
Usecase from a config file, or reasons about what `usecase.config` actually contains,
this is the exact, complete set of keys usecase_generator.py and tsn_scheduling_problem.py
ever read for the 'ft' (fat-tree) topology path, confirmed by reading every config-reading
call site in both files directly, not an abbreviated or representative sample:
    [topology_config]
    numhosts = 8
    numswitchs = 4
    portspeed = 1000
    procdelay = 1
    propdelay = 100
    linkBw = 1000
    queuecountperport = 8
    graphConfig = {{'typ':'ft','k':4}}

    [streams_config]
    numofstreamsperinstance = 4
    period_range = {{'start':100000,'stop':200000,'step':100000}}
    numberframes_range = {{'start':1,'stop':2}}
    datasize_range = {{'start':64,'stop':128}}
    maxlatency_range = 1.0,2.0
    numplacestalker = 1
    numplaceslistener = 1
    numlistenersperstr = 1
    routing = {{'fixPath':True,'shortestPath':True}}
    expiration = {{'finite_duration':False,'start':1,'stop':2}}
    stride = {{'active':False}}
    staggered = {{'active':False}}
    hotspot = {{'active':False}}
This exact file already exists at this project's root as `config.ini` — reuse it by path
where a real file is genuinely needed rather than fabricating a different one or a
different shape. `expiration`, `stride`, `staggered`, and `hotspot` are each read as a
dict literal and indexed by key (e.g. `stride['active']`), so a bare number or an empty
`{{}}` fails with KeyError exactly like omitting the key; `maxlatency_range` is the one
exception, read as a comma-separated string of floats, never a dict. Do not add
`generation`, a standalone `fixPath` key, `maxstrpernode`, or `minnodedegreepersw` —
each looks plausible but is dead in the real library code (present only in a
commented-out line, or reachable only on a non-'ft' topology path).

Two implementation-bug patterns confirmed via real test execution, both worth checking
directly against whatever you write:
1. Validate every one of a function's own parameters (shape, type, range, dimension-
   matching) before using any parameter as a collaborator - before entering a
   `with x.use_instance(y):` block, calling a method on it, or passing it into another
   function. A test may deliberately give a minimal or fake object for an argument it is
   not testing; reaching for that argument before finishing validation of the argument
   actually under test means crashing on the fake object instead of raising the
   validation error the test expects - on a real run this was the single largest cause
   of test failures, well above every other cause combined.
2. If your design introduces a lightweight wrapper dataclass standing in for a real
   library type (for example a `class TSNScheduling: value: object` alongside the real
   tsn_scheduling_problem.TSNScheduling), never call a real-API-only method - such as
   `.use_instance(...)`, `.n_ports`, or `.group_connected_ports(...)` - directly on an
   instance of your own wrapper; it will not have that method. Either give the wrapper
   every method you actually call on it, or unwrap to the real object first and call
   real methods only on that unwrapped object.

Rules:
 - You are given a FROZEN INTERFACE for the WHOLE algorithm (below): every function's
   exact name, signature, and docstring (its Purpose, Satisfies, Calls and Flow lines) is
   already fixed. Implement the body of EVERY function in it, replacing its
   `raise NotImplementedError` with real logic. Do NOT rename, add, remove, or reorder
   any parameter, and do NOT alter any function's name or docstring except as noted
   below — the interface was frozen precisely so nothing downstream has to guess it.
 - You are also given the DESIGN BLUEPRINT that produced the frozen interface (its data
   model, function inventory with call graph, requirement map, and reuse decisions).
   Each function's "Calls" docstring line records which OTHER frozen functions it must
   call — implement it so it actually does so; a function that was designed to call
   another but does not is a design violation, not a valid implementation choice.
 - You may define small additional private helper functions in the SAME block of
   source, alongside the frozen ones, when your GAP ANALYSIS genuinely needs them for
   decomposition (e.g. `_generate_neighbor`, `_accept`) — but every function already
   named in the frozen interface must remain a top-level entry point; do not fold one
   into another or remove one.
 - No parameter of any frozen function's signature may appear twice in your def line —
   a duplicated parameter is a Python syntax error ("duplicate argument in function
   definition") and the whole file then fails to load.
 - Use the data model's type definitions from the frozen interface as-is; do not invent
   an alternate representation of a type the frozen interface already defines.
 - A numpy array, or a value that holds them (a list, dict or dataclass of arrays), has no
   truth value: `if`, `==` and `!=` work cell by cell and then raise. Test an array with
   `is None`, `.size` or `.any()`, and compare with `np.array_equal`, element by element.
 - A check that returns the PROBLEMS it found passes when it returns none:
   `Schedule.validate_conflicts()` returns the list of conflicting slot pairs, so a
   schedule is valid when that list is EMPTY - `valid = not schedule.validate_conflicts()`,
   never `if not schedule.validate_conflicts(): return False` or `return bool(...)` as
   validity. Read what a library call returns before testing its truth value.
 - `validate_conflicts` compares every pair of windows on a port by time alone, whatever
   stream they belong to, so the same window listed twice conflicts with itself: a
   schedule you check lists each window ONCE.
 - The evaluator's per-column checks (feasible_streams, per_stream_all_violations) judge an
   all-zero column as infeasible and report violations for it.
   IF a library function returns one result per item (a row or a column), THEN read it only for
   the items the input actually holds: `all(feasible[i] for i in present)`, never
   `np.all(feasible)`.
   IF you leave a stream's column at zero in an array (for example because its two routes
   are checked in arrays of their own), THEN that stream is not present in this array,
   and its verdict in this array is not read.
       WRONG  reading the verdict of every stream in the configuration, including the one whose
              column was left at zero. Its zero column is always judged infeasible, so every
              configuration that holds a stream on two routes is rejected, and FRER never starts.
       RIGHT  reading the verdict only for the streams whose column in THIS array holds a route
 - `evaluator.feasible_streams(array)` judges the streams of ONE array together: an overlap
   between two streams over the macro cycle (macrocycle_overlap) is found only when both are
   columns of the same array, and validate_conflicts does not repeat that check.
 - IF a purpose asks for a check at a step (e.g. "... (check validity)"), THEN
   the code calls the validity function on that step's configuration and does what the purpose
   says when it is False, because a step that is never checked is emitted even when it collides,
   and every fallback that waits for a refused move never runs.
       WRONG  making the change and appending the step, with no call of the validity function
       RIGHT  build the step's configuration on a copy, check it, append the step only if it is
              valid; otherwise return what the purpose says for a refused move (no steps), with
              the current configuration left unchanged
 - IF the code needs a value the library holds (a window length, a period, a size), THEN it reads
   it from there (evaluator.windLengths[port, i], evaluator.stream_intervals[i]), because an
   assumed number is wrong for every other input.
       WRONG  <a value> = 1000   # assumed
       RIGHT  <a value> = <read from the library attribute that holds it>
 - IF a purpose names several things a function does (for example "first tries X; if that fails,
   does Y", or "stops when Z"), THEN the code does every one of them, in the order the purpose
   gives.
       WRONG  the purpose says "first tries X; if that fails, does Y", and the code only does Y.
       RIGHT  the code tries X first, and does Y only when X fails; a loop the purpose says
              stops on some condition has that check in it.
 - IF a loop applies changes one after another, THEN after each change it sets its current
   state to the result of that change.
       WRONG  the loop records a change but keeps working from the state before it, so the next
              change is built on an old state and undoes the one before.
       RIGHT  record the change, then set <the current state> to <the result of that change>
 - IF the design defines how a value is computed (a function, or a formula in a purpose), THEN
   the value is always computed that way, never written as a fixed number.
       WRONG  <the value> = 1.0 everywhere, so anything ranked or minimised by it means nothing
       RIGHT  <the value> = <the design's function or formula>(<its inputs>)
 - IF a stream is on two routes and its old route is removed (the end of FRER), THEN the stream
   ends with exactly ONE stored route, the new one - whatever form a route has in your
   configuration.
       WRONG  writing the new route into every route the stream has stored: two identical
              copies remain, the stream never matches the finish, and the plan cannot end
       RIGHT  replacing the stream's whole list of routes with a list holding only the new route
 - IF a step's result equals the configuration before it (for example, replicating a stream
   after activation already put both of its routes in place), THEN no step is emitted and the
   strategy continues with its next step; an unchanged result is not a failure.
       WRONG  raising an error when replication changes nothing: every FRER attempt then fails,
              and a path change falls back to a one-step switch with no overlap
       RIGHT  emit no step for it and go on to the strategy's next step (here: removing the
              old route once the new one is safe)
 - A stream's talker start (its value at the talker port) must stay below the stream's own
   period: the evaluator rejects `talker start >= stream_intervals[i]`.
 - The evaluator judges each column as ONE path.
 - When a purpose says which options apply to which cases ("X only for A", "Y for every
   other case"), that IS the candidate list for each case: never also offer an option to a
   case the purpose did not give it, even when it would score better.
 - A value stored in a result you return must never change afterwards.
   `d.copy()` and `list(d)` copy only the outer container: the lists and arrays inside are
   still shared, so `config[i].append(col)` after `steps.append(config.copy())` changes the
   stored step too.
   IF a function stores several values one after the other, THEN each stored value is its own
   copy, made before it is changed.
       WRONG  storing a value, then changing the same object to build the next one: the stored
              value changes too
       RIGHT  new = copy.deepcopy(old); change new; store new
 - Never read an attribute of a library object through getattr with a default, or test for it with
   hasattr: if the class does not have the attribute, that hides the bug and the code fails later,
   somewhere else. Use only attributes the library code shown here has.
 - In a condition, write the guard BEFORE what it protects: `if <a list> and <x> == <a list>[0]`,
   never `if <x> == <a list>[0] and <a list>` - `and` evaluates left to right, so an index read
   first raises IndexError on an empty list before the guard is reached.
 - Reuse the codebase functions the blueprint's reuse decisions call for, from the Context
   section below; call them by their exact names, respecting their "method_type" exactly as
   described above. Implement directly, in a private helper,
   only the capabilities you determine are genuinely missing — do not re-declare a
   helper that already exists in the context.
 - You may import anything listed in dependencies.external, plus the Python standard
   library, plus the real project modules described above.
 - Every `if`/`for`/`while`/`try`/`def` you open must be followed by a real indented block,
   and every string and docstring must be closed. The output is compiled before it runs;
   source that does not parse is rejected outright.
 - On the function that no other frozen function calls (the top-level entry point),
   append one additional docstring line, "Integration: ...", stating in one sentence
   exactly how to fold this into the real codebase.
 - You ALWAYS return code — never a refusal, never a stub, never "NOT_IN_CONTEXT".

Output rules:
- Return ONLY the Python source (every frozen function, fully implemented, plus any
  private helpers they call). No JSON, no markdown fences, no prose, no restatement of
  your plan.

Design blueprint. Each function in it carries "reads" and "writes": the data structures
it consumes and the ones it produces. Use them as the body's contract — what a function
writes must come back out of it, and a structure it does not read is not its business.
{blueprint_json}

Frozen interface (implement every function's body — do not change any name, signature,
or existing docstring line):
{frozen_interface}

Context (the library members the blueprint's reuse decisions name, each with its real source and caveats):
{context_str}

Implementation:
""")

# UNUSED: superseded by the spec-first, code-blind test flow (SPEC_EXTRACTION_PROMPT ->
# SPEC_BASED_TEST_PROMPT, see generate_verified_implementation()). This prompt would write
# tests FROM the generated code itself, which can't catch an implementation bug the
# generator itself introduced; the spec-based flow derives tests from the request instead.
# UNIT_TEST_GENERATION_PROMPT = """\
# You are an expert Python test engineer.
#
# Given a user request, helper functions, and candidate code, write pytest-style unit tests.
#
# Rules:
# - Return ONLY valid JSON. No markdown fences. No preamble.
# - The JSON must have a single key "unit_tests" whose value is a list of strings.
# - Each string is a complete, self-contained Python test function (def test_...).
# - Each test function must import what it needs, define or inline any stubs/mocks for
#   external dependencies (networkx, numpy, etc.), and contain at least one assertion.
# - Cover: normal cases, edge cases (empty input, zero, None), and expected exceptions.
# - Use extracted parameter values as concrete test inputs wherever available.
# - Do NOT rely on external files or network access.
# - If the code under test depends on a class or helper, define a minimal stub inline.
#
# Schema:
# {{
#   "unit_tests": [
#     "def test_normal_case():\\n    # inline imports and stubs\\n    assert ...",
#     "def test_edge_case_empty():\\n    assert ...",
#     "def test_raises_on_invalid():\\n    import pytest\\n    with pytest.raises(ValueError):\\n        ..."
#   ]
# }}
#
# User request:
# {query}
#
# Extracted parameters:
# {params_json}
#
# Helper functions available:
# {helpers_json}
#
# Code under test:
# {code}
#
# JSON output:
# """


# ── Parameter extraction prompt ───────────────────────────────────────────────
# UNUSED: was used exclusively by extract_parameters() (now disabled above) — the
# classifier's CLASSIFICATION_PROMPT already extracts per-sub-query parameters in the
# same LLM call that does routing, making this separate extraction pass redundant.
# PARAM_EXTRACTION_PROMPT = """\
# You are a parameter extraction assistant for a TSN (Time-Sensitive Networking) system.
#
# You are given:
# 1. A user query in natural language.
# 2. A list of known parameter definitions from the TSN codebase (name + description).
#
# Your job:
# - Read the query carefully.
# - Identify every parameter from the known list that is explicitly or implicitly referenced.
# - For each matched parameter, extract the concrete value the user specified.
# - If no value is given for a matched parameter, use null.
# - If the query references a concept that matches a parameter but uses different wording
#   (e.g. "switches" → num_switches, "8 hosts" → num_hosts), still match it.
# - Only include parameters that appear in the known list. Never invent parameter names.
# - Output ONLY a valid JSON object, no preamble, no markdown fences.
#
# Known parameters:
# {param_definitions}
#
# Query:
# {query}
#
# JSON output:
# """

# UNUSED: superseded by _explain_params_from_map(), which builds the same
# {value, meaning, used_for} shape from the precomputed, zero-LLM-cost PARAM_MAP
# (built offline by build_param_map.py) instead of an LLM call per query.
# PARAM_EXPLANATION_PROMPT = """\
# You are a TSN parameter explanation assistant.
#
# Explain only the parameters listed below using the provided context.
#
# Return ONLY JSON.
#
# Schema:
# {{
#   "<parameter_name>": {{
#     "value": "<given value or null>",
#     "meaning": "<brief explanation>",
#     "used_for": "<brief usage explanation>"
#   }}
# }}
#
# Context:
# {context_str}
#
# Parameters:
# {params_json}
# """

# Stage 5 (verification), step 1: called from _derive_contract_from_spec() to turn the
# generated specification (signature/description/pre-post-conditions/raises) plus the
# user's request into a behavioral contract. It runs before the implementation is written
# at all, so the contract cannot mirror whatever the code happens to do (which would let a
# buggy implementation "pass" its own test) — it can only reflect the specified intent.
SPEC_EXTRACTION_PROMPT = """\
You are a software specification analyst. Produce a behavioral contract
for the function described below. You are NOT given the implementation,
and you must NOT guess at implementation details.

Derive the contract ONLY from the description, signature, preconditions,
postconditions, and declared exceptions.

COVERAGE — this contract is the ONLY definition of what will ever be tested, so anything
you leave out of it is never checked by anything, and the implementation passes by default:
- Emit AT LEAST ONE behavior per postcondition.
- Emit EXACTLY ONE error_case behavior per declared exception.
- NEVER merge two postconditions into one behavior.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
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


# Stage 5, step 2: called from _spec_based_tests() to turn the behavioral contract (not
# the implementation) into pytest tests — exactly three per contract behavior. This is the set of
# tests shown to the user for review/edit/approval before anything is executed.
SPEC_BASED_TEST_PROMPT = """\
You are a test engineer writing pytest tests from a BEHAVIORAL CONTRACT.

Exact function signature (authoritative — this is the real def line the implementation
will have; call the function EXACTLY like this in every test):
{signature}

Calling convention — the implementation is a plain module-level function, and the
signature above is complete and authoritative. Call it directly, passing EVERY parameter
the signature declares, in the exact order it declares them:
  {function_name}(...args in signature order...).
There is no `self` and no `cls`: the state the function needs arrives through ordinary
named parameters (n_ports, all_hops_costs, ...), so build real values for them —
real numpy arrays, real ints, real dicts, with the shapes and dtypes the contract's
preconditions state. Never invent an extra leading argument, never drop a declared one,
and never call it as instance.{function_name}(...) or ClassName.{function_name}(...) —
that attribute does not exist and the call raises AttributeError no matter how correct
your test logic is. Get the argument order wrong and EVERY test silently binds the wrong
value to the wrong parameter — check each call against the signature before writing it.
- If the signature above is a `class <Name>:` line instead of a def, the implementation is
  a complete class: instantiate it through its own __init__ with the arguments the
  preconditions describe, then exercise the methods the contract names. Do not define a
  stand-in for it — the real class is what you are testing.
- Never pass a bare object(): it has no attributes and no __dict__, so the first attribute
  access inside the implementation raises AttributeError and the test fails for a reason
  unrelated to the behavior it checks. Never pass a bare MagicMock() where the
  implementation does arithmetic or comparisons: a MagicMock attribute is a Mock, not a
  number or array, so it fails or silently misbehaves. Populate real values.

Calling convention for real library helpers you use in your OWN test setup code (building
a real Usecase, TSNScheduling, TSNConstraintEvaluator, or Schedule to pass into the
function under test) — get this wrong and the call raises TypeError or silently does the
wrong thing. Use EXACTLY the form for that helper's kind:
- a plain module-level function: call it directly, helper_name(args).
    Real example — `drawFattree` is a module-level function in usecase_generator.py:
        from usecase_generator import drawFattree
        drawFattree(usecase.graph)
- an instance method (its first parameter is `self`): it needs a real instance to be
  called on — build one first, then call the method on it:
        # evaluator is a real TSNConstraintEvaluator instance
        violations_array = evaluator.per_stream_all_violations(chromosome)
- a classmethod (its first parameter is `cls`, bound automatically): call it on the class
  itself — never pass the class or `cls` explicitly as an argument.
    Real example — UsecaseGenerator.generateUC is a @classmethod:
        from usecase_generator import UsecaseGenerator
        usecase = UsecaseGenerator.generateUC(config_file)
- a staticmethod (takes neither `self` nor `cls`): call it on the class exactly like a
  plain function attached to a namespace: ClassName.helper_name(args).
    Real example — Usecase.generate_fattree_regions is a @staticmethod:
        from usecase_generator import Usecase
        regions = Usecase.generate_fattree_regions(4)
- a property: NOT callable — it is a computed attribute. Access it as instance.name (no
  parentheses). Calling it as instance.name() raises TypeError (or silently misbehaves)
  because the result of the property, not the property itself, is what gets "called".
    Real example — Usecase.hosts is a plain @property:
        host_nodes = usecase.hosts          # correct
        host_nodes = usecase.hosts()        # WRONG: TypeError, list is not callable

How the three real modules are actually organised — this is what the helpers above belong
to, and the rules here override any guess you would otherwise make. The frozen interface
and any dataclasses the implementation defines for its OWN internal use (even ones that
happen to share a name like "TSNScheduling", "Usecase", or "Evaluator" with one of these
real classes) are NOT these classes — a same-named placeholder dataclass the implementation
defines for its own type hints has none of the real behavior described below, so when a
test needs the REAL behavior, import the real class from the module named here, not from
generated_impl. If this specific test needs real behavior from a name generated_impl ALSO
defines as its own wrapper, do not import the real one under its bare name — alias it
immediately (e.g. `from tsn_scheduling_problem import TSNScheduling as RealTSNScheduling`)
and use only that alias throughout this test, never the bare name. Some other function's
tests in this same concatenated file may import the bare name from generated_impl's wrapper
instead, and whichever import runs LAST across the WHOLE file at collection time decides
what the bare name resolves to for every test that references it when it actually runs, not
just the one that wrote it. Confirmed via real execution: a construction call that exactly
matched the real one-argument constructor still failed at runtime with `AttributeError: ...
has no attribute 'use_instance'`, because the object actually produced was the OTHER
(wrapper) class of the same name, not the real one this test's own import line asked for.

1. usecase_generator.UsecaseGenerator — EVERY public member is a @classmethod (generateUC,
   ucGenerator, empty_uc, streamGenerator, generateStream, talkerEmbedding,
   listenerEmbedding, findAllPaths, shortestPath, randomPath, pathConvenient,
   periodConvenient, makeTopology, set_link_data, k_ary_fat_tree, makeRandomGraph,
   changeDegree). Never instantiate it; call on the class, and pass the usecase in:
       stream = UsecaseGenerator.shortestPath(stream, usecase)
       graph  = UsecaseGenerator.k_ary_fat_tree(4)

2. usecase_generator.Usecase — a normal object you hold an instance of. Its properties
   (hosts, switches, links, ports, port_name_id_mapping, distance_to_hosts) are plain
   @property: read them as attributes. Its ordinary methods take real arguments:
       wl = usecase.window_length(stream_idx, port_name)   # instance method
       uc = Usecase.from_json(imported_dict)               # @classmethod
   Note `usecase.ports[port_name]['speed']` is in Mbps and the library converts with the
   module constant MBPS_2_BPNS (import it, do not hardcode 0.001).

3. evaluator.TSNConstraintEvaluator — a normal object; ALL of its methods are plain
   instance methods (no decorators anywhere in the class). It is constructed FROM a
   TSNScheduling problem, which is where its ~40 precomputed attributes come from:
       evaluator = TSNConstraintEvaluator(problem, check_routing=True)
   Construct it explicitly like that whenever the schedules being evaluated are based
   on their transmission START TIMES (START_TIME encoded) rather than on QUEUE
   PRIORITIES (PRIORITY encoded) — that is, whenever a non-zero cell means "this
   stream starts transmitting on this port at this time" and not "this stream sits
   in this queue". initialize_problem creates problem.evaluator without ever passing
   scheduling_type, so the pre-built instance is always PRIORITY and is only ever
   correct for priority-based schedules. Decide which of the two your cells hold
   before you evaluate anything: the evaluator reads them one way or the other
   whether or not your own logic ever looks at the values.
       violation_sums, metrics = evaluator.evaluate(chromosome)
   evaluate() returns TWO NUMPY ARRAYS, never dicts — do not call .values() on them or
   index them by name. metrics always has length 6, ordered congestion_score, avg_util,
   std_util, load_imbalance, path_len_score, avg_path_len.
   violation_sums carries one entry per VIOLATION KIND, which is NOT the same as one per
   check: a single check function returns several rows (_per_stream_host_role_violations
   alone returns 5), so there are 11 kinds from 5 checks under PRIORITY and 14 kinds from
   7 checks under START_TIME. The row order, read out of the evaluator's own source:
{violation_row_order}
   Do NOT try to recover this mapping at run time from `evaluator.violations_keys`: that
   is `relevant_weight_keys`, a SET, so its iteration order has no relationship to these
   rows. The order above is the only mapping there is — it comes from the order checks are
   appended in __init__ plus each check's own vstack order.
   One caveat, real and in the library: _per_stream_window_sequence_violations returns 1
   row instead of 2 when a topology has no upstream ports, which shifts every row after
   it. So if you branch on a specific violation kind, prefer calling that check through
   per_stream_all_violations and summing, or slice defensively — do not hard-code an
   index and assume it holds on every topology.
   The chromosome it takes is a 2-D int array indexed [port, stream] where 0 means
   inactive; under PRIORITY a non-zero cell is a queue priority, under START_TIME it is
   a transmission start time PLUS ONE. evaluator.feasible_streams(chromosome) returns a
   boolean array, one entry per stream.
   `self.scheduling_type` holds a ScheduleEncodingType ENUM member, never a string:
       from evaluator import ScheduleEncodingType
       if self.scheduling_type == ScheduleEncodingType.PRIORITY:   # correct
       if self.scheduling_type == "PRIORITY":                      # WRONG: always False
   Its per-chromosome caches are keyed by `chromosome.tobytes()`, never by the array
   itself (a numpy array is unhashable and using one as a dict key raises TypeError):
       key = chromosome.tobytes()
       cached = self._violation_evaluation_cache.get(key)
   UnionFind is a class NESTED inside TSNConstraintEvaluator, so it is reached through the
   owner — from inside the class `self.UnionFind(active_ports)` (what group_connected_ports
   does), from outside `TSNConstraintEvaluator.UnionFind(active_ports)`.

4. tsn_scheduling_problem.TSNScheduling — the one that needs real care. Most of its
   members carry the module's own @require_instance decorator, and its read-only views
   (n_ports, n_streams, ports, port_id_to_name, port_name_to_id, port_id_to_node, nodes,
   n_nodes, node_to_port_id, macro_cycle, streams, links) are @property STACKED ON TOP of
   @require_instance. require_instance checks `self.usecase` and, when it is unset, calls
   no_instance_assigned(), which RAISES ValueError("No current instance is set!..."). So
   these are not merely "call it on an instance" — the instance must additionally have an
   active usecase, and the supported way to establish that is the context manager:
       with problem.use_instance(instance):
           n = problem.n_ports                     # @property @require_instance
           cost = problem.hops_costs(frame_size)   # @require_instance method
   Entering that block also runs build_metadata() once per usecase. Outside such a block
   every one of those names raises ValueError. A REAL TSNScheduling is constructed from a
   config file path: `problem = TSNScheduling(str(config_path))` — that alone does NOT
   build a usable usecase; pair it with `usecase = UsecaseGenerator.generateUC(str(config_path))`
   (real topology and streams) before entering `with problem.use_instance(usecase):` —
   constructing a bare `Usecase(str(config_path))` directly leaves its graph as None and
   its streams as an empty list, since Usecase.__init__ only stores the config parser
   itself and does not run topology/stream generation.
   Exceptions in this class that do NOT need the with-block:
   _frame_size is a plain @classmethod (TSNScheduling._frame_size(stream)), and drawFattree
   and plot_all_stream_custom_graphs are plain @staticmethods.

5. schedule.Slot and schedule.Schedule — the compiled-schedule model. Slot is a
   @dataclass (start_ns, end_ns, period_ns, stream_id, port_id, queue_id, index) whose
   only property is duration_ns. Schedule SUBCLASSES list, so a Schedule instance IS the
   container of its Slots — append to it, index it and iterate it directly; there is no
   .slots attribute to reach for. Schedule's constructor REQUIRES cycle_ns as its first
   argument — `Schedule()` with no arguments raises TypeError, it is never a valid way to
   get an empty schedule:
       from schedule import Schedule, Slot
       sch = Schedule(cycle_ns=macro_cycle, basetime=0)
       sch.append(Slot(start_ns=..., end_ns=..., period_ns=..., stream_id=i, port_id=p))
   Two @classmethods build one: Schedule.from_compact_encoding(compact_encoding,
   windLengths, stream_intervals, cycle_ns=..., basetime=0, tt_queue_id=0) — compact_encoding
   must be a 2-D array shaped (n_ports, n_streams), never a 1-D array — and
   Schedule.from_usecase(usecase, port_name_to_id). Everything else is a plain instance
   method: to_compact_encoding(n_ports, n_streams), slots_by_port(port_id),
   slots_by_stream(stream_id), validate_conflicts(touching_allowed=True),
   update_graph(graph, port_id_to_name), update_streams(streams, ports, port_id_to_name),
   plot(...). TransmissionWindow is a @dataclass NESTED inside Schedule, reached as
   Schedule.TransmissionWindow(...).
   Port ids here are 1-BASED: from_compact_encoding stores port_idx + 1 and
   to_compact_encoding writes back to row port_id - 1. Never mix a 0-based row index
   with a port id.
   Three behaviours to respect, all of them real:
   - update_streams REQUIRES every stream to have at least two slots. A stream with no
     slots raises IndexError; a stream with exactly one raises UnboundLocalError. Build
     a slot for every hop of every stream before calling it — usecase.addSchedule calls
     update_streams for you, so the same precondition applies there.
   - to_compact_encoding raises ValueError on an empty Schedule, and raises ValueError
     again when two slots share one (port, stream) pair, since the array holds only one.
   - total_occupied_time and utilization are DEAD: both call self.compile(), which
     exists neither on Schedule nor on list, so reading either raises AttributeError.
     Never call them; compute occupancy from the slots yourself if you need it.

Three same-name traps in this codebase — check which class you actually have before you
call, because the members are not interchangeable:
- `_frame_size` is a @classmethod on TSNScheduling taking a stream DICT
  (TSNScheduling._frame_size(stream)), but an INSTANCE METHOD on Usecase taking a stream
  INDEX (usecase._frame_size(stream_idx)).
- `drawFattree` is a module-level function in usecase_generator (drawFattree(graph)) and
  also a @staticmethod on TSNScheduling (TSNScheduling.drawFattree(graph, ax=None)).
- `ports` and `streams` exist on BOTH Usecase and TSNScheduling and are keyed
  differently. usecase.ports is keyed by port NAME (a string like 'host-1-p0.Port1') and
  usecase.streams[i]["pathLists"][j] holds port NAMES, ending at the destination node;
  problem.ports is keyed by port ID (an int) and problem.streams[i]["pathLists"][j] holds
  port IDs. They are not the same object either. Reading one and indexing the other is
  the most common failure here and it surfaces late, as a KeyError on an integer — so
  decide which of the two you actually built and stay on it.

CRITICAL RULES:
- You are given a contract, NOT an implementation.
- Write AT LEAST THREE tests for every behavior in the contract. Never fewer than three
  for any behavior, and emit them as three separate blocks (see the output format below).
- The three must be MATERIALLY DIFFERENT ways of exercising that behavior, not the same
  assertion with the numbers changed: cover the ordinary case, a boundary or degenerate
  case (empty, zero, single element, minimum/maximum, duplicate or equal values), and the
  case most likely to break the implementation for that behavior — the error case where
  the contract declares one. Three near-duplicates verify no more than one does.
- DRAW THOSE THREE CASES FROM "input_domains" IN THE CONTRACT. It already lists, per
  parameter, the "valid" values, the "boundary" values worth testing, and the "invalid"
  values that should trigger errors — that analysis is done, so use it rather than
  inventing your own: "valid" feeds the ordinary case, "boundary" feeds the boundary case,
  "invalid" feeds the error case. Use a listed value verbatim where one applies. Only when
  "input_domains" says nothing about a parameter may you choose that parameter's value
  yourself. If "boundary" lists more distinct values than your three tests can cover, add a
  fourth and fifth test rather than dropping them.
- Each test name must reference the behavior id AND be unique across the whole set, since
  all tests are concatenated into one file: test_B1_returns_sorted_paths,
  test_B1_empty_graph, test_B1_raises_on_missing_node. Two tests sharing a name means the
  second silently replaces the first and the first never runs at all.
- Every single test MUST actually call {function_name} (imported via the import line
  below) and assert on ITS real return value or the exception IT actually raises.
  Never fabricate, simulate, or hand-compute what you expect the function to return and
  then assert your own object equals itself — a test that never calls the function under
  test verifies nothing and will be rejected outright, no matter how detailed its
  reasoning comments are.
- Where "input_domains" does not cover a parameter, you may freely CHOOSE the argument
  values your test passes in
  (n_ports = 2, a frame size of 1500.0, and so on) — that is ordinary test setup. What you
  must NOT do is ASSERT against a magic number that the contract never states and that
  your own setup does not determine. Concretely: if the contract does not pin down a
  conversion factor, a default window length or a macro-cycle, do not assert
  `result == 17.5`; instead assert a relationship the contract does state, or one you can
  compute from the values you yourself just set. An asserted constant you invented is
  very unlikely to match the implementation's real constant, and the test then fails for a
  reason that has nothing to do with correctness.
- The real project modules ARE importable from your tests, so when a precondition says a
  parameter holds a real project type, build it from the real symbol rather than a
  look-alike: `from evaluator import ScheduleEncodingType` then passing
  `scheduling_type=ScheduleEncodingType.PRIORITY`, not the string "PRIORITY".
  A string in place of an enum makes the implementation's own enum comparison silently
  False, so the branch under test never runs and the test proves nothing.
- Assert ONLY what the contract states. Do not invent behavior.
- EVERY ENTRY IN "invariants" MUST BE ASSERTED. An invariant is a property the contract
  says holds for ALL inputs, so it is not a separate test: fold it in as an extra assert
  inside the tests you are already writing, wherever the invariant applies to what that
  test just computed. Each invariant must end up asserted in at least one test, and a
  test that produces a result an invariant covers should check it alongside its own
  behavior. Skip an invariant only where it genuinely cannot apply to that test's inputs.
- Do NOT weaken assertions to make tests pass. The implementation does
  not exist yet from your perspective.
- For error_case behaviors, use pytest.raises with the declared exception — but only when
  the inputs you construct genuinely violate that behavior's stated "given". Do not expect
  an exception from a setup that satisfies the contract; "DID NOT RAISE" failures come
  from tests that demanded an error the contract never promised for those inputs.
- A test must exercise exactly the behavior it names. If a behavior's "given" describes a
  state you cannot construct from the contract alone (an internal cache, an "inactive
  port", a topology failure), set up the closest state the preconditions DO let you build
  and assert only the part of the "then" that state actually determines — never assert a
  specific outcome for a state you never managed to create.
- Include a docstring in each test quoting the behavior it verifies.
- Tests run standalone with NO conftest.py, so there are no project fixtures and nothing
  is shared across tests: construct setup data directly inside each test's own body, and
  never take a fixture parameter you would have to define yourself.
  pytest's OWN builtin fixtures ARE available and you SHOULD use them by declaring them as
  plain test parameters — `def test_B1_x(tmp_path):` for a real temporary directory,
  `def test_B2_y(monkeypatch):` for patching. That is the correct and only way to reach
  them. They are NOT attributes of the pytest module: `pytest.tmp_path`, `pytest.Path` and
  friends do not exist and raise AttributeError, so never write them. When a test needs a
  file on disk, take `tmp_path` and write a real one — never point at an invented path
  like 'path/to/config.ini' or 'fixtures/some_name.ini', which fails no matter what the
  implementation does, no matter how realistic the filename looks. If the function under
  test needs a TSNScheduling or Usecase config file, this exact project already ships one
  at its root, literally named `config.ini` — do NOT invent a different filename or a
  fixtures/ subdirectory, and do NOT pass a bare relative or absolute path to it directly
  (tests run in an isolated temp directory, not this project's own folder). Instead, write
  a file with the SAME two real section names into tmp_path, then point the code under
  test at that path, e.g.:
  `p = tmp_path / "config.ini"`
  `p.write_text("[topology_config]\\n" "numhosts = 8\\n" "numswitchs = 4\\n" "portspeed = 1000\\n" "procdelay = 1\\n" "propdelay = 100\\n" "linkBw = 1000\\n" "queuecountperport = 8\\n" "graphConfig = {{'typ':'ft','k':4}}\\n" "\\n" "[streams_config]\\n" "numofstreamsperinstance = 4\\n" "period_range = {{'start':100000,'stop':200000,'step':100000}}\\n" "numberframes_range = {{'start':1,'stop':2}}\\n" "datasize_range = {{'start':64,'stop':128}}\\n" "maxlatency_range = 1.0,2.0\\n" "numplacestalker = 1\\n" "numplaceslistener = 1\\n" "numlistenersperstr = 1\\n" "routing = {{'fixPath':True,'shortestPath':True}}\\n" "expiration = {{'finite_duration':False,'start':1,'stop':2}}\\n" "stride = {{'active':False}}\\n" "staggered = {{'active':False}}\\n" "hotspot = {{'active':False}}\\n")`
  `problem = TSNScheduling(str(p))`
  Every key above was confirmed by reading every config-reading call site in
  usecase_generator.py and tsn_scheduling_problem.py directly — this is the complete set
  those two files ever read for the 'ft' (fat-tree) topology path, not an abbreviated or
  representative sample. All fourteen keys are load-bearing: `expiration`, `stride`,
  `staggered`, and `hotspot` are each read as a dict literal via ast.literal_eval and
  indexed by key (`expiration['finite_duration']`, `stride['active']`, etc.), so writing
  them as a bare number or an empty `{{}}` fails with KeyError the moment that key is read,
  exactly like omitting the key entirely. `maxlatency_range` is the one exception — it is
  read as a plain comma-separated string of floats (`"1.0,2.0"`), never a dict. Do not add
  `generation`, `fixPath` as a standalone key, `maxstrpernode`, or `minnodedegreepersw` —
  each looks plausible but is dead/unused in the actual library code (only ever appears in
  a commented-out line, or on a code path that only runs for a non-'ft' topology type);
  including them is harmless but they are not required, so leave them out and keep this
  block exactly as short as the real requirement.
- Never compare numpy arrays (or containers holding them, including an object's __dict__)
  with `==`, `!=`, `in`, or a bare `assert x == y`: the result is an ARRAY, and pytest's
  truth test on it raises "ValueError: The truth value of an array with more than one
  element is ambiguous" — a crash, not a clean failure, whatever the implementation does.
  Use `np.array_equal(a, b)` (or `np.allclose(a, b)` for floats, or
  `np.testing.assert_array_equal`) for arrays, and compare array-valued dict entries
  key by key with those helpers rather than comparing the dicts directly.
- Every test's code must put its imports (pytest, the function under
  test, numpy, etc.) at MODULE level — before the "def test_..." line, not
  indented inside the function body. All tests are concatenated into one file,
  so a name a later test's body relies on (e.g. calling the function under
  test, or pytest.raises) must already be bound at module scope; an import
  nested inside one function is invisible to every other function.
- Make sure the code does not have syntax errors — it must be valid, parseable Python.
- Every block MUST start with the literal line "### TEST" and end with the literal line
  "### END TEST" — a block missing either marker is silently discarded in its entirety, so
  never omit one, no matter how many blocks you are writing.

Do NOT use JSON. Output each test as plain text in EXACTLY this block format,
one block per TEST — write exactly 3 blocks per behavior, i.e. exactly 3
consecutive blocks carrying the same BEHAVIOR_ID for each behavior in the
contract — with NOTHING else before, between, or after the blocks
(no markdown fences, no commentary, no numbering). Write the test's code
completely unescaped, exactly as it would appear in a real .py file — real
newlines, real quotes, real backslashes:

### TEST
BEHAVIOR_ID: B1
TEST_NAME: test_B1_returns_sorted_paths
CODE:
import pytest
from some_module import some_function

def test_B1_returns_sorted_paths():
    \"\"\"B1: ...\"\"\"
    ...
### END TEST

### TEST
BEHAVIOR_ID: B1
TEST_NAME: test_B1_empty_input_returns_empty
CODE:
def test_B1_empty_input_returns_empty():
    \"\"\"B1: same behavior, boundary case.\"\"\"
    ...
### END TEST

### TEST
BEHAVIOR_ID: B1
TEST_NAME: test_B1_single_node_graph
CODE:
def test_B1_single_node_graph():
    \"\"\"B1: same behavior, degenerate case.\"\"\"
    ...
### END TEST

Function to import: {import_hint}

the users (sub)query/specific request is: {query_str}

Behavioral contract:
{contract_json}

Design blueprint (the larger design this function is one part of — its data model,
full function inventory, requirement map, and reuse decisions; use it to understand
how this function fits into the whole, e.g. what type its arguments/return value
really are and which other functions produce/consume them):
{blueprint_json}

Frozen interface (the real signatures and docstrings, fixed before any implementation —
authoritative alongside the contract above for this function's exact signature):
{frozen_interface}
{revision_block}
Output (plain text, one "### TEST" ... "### END TEST" block per behavior):
"""

# Stage 5, step 4: called from _triage_failure() when an approved test fails against the
# generated implementation. It does NOT choose a side to blame any more — the tests and the
# contract they came from were both approved by the user, so they are frozen. The only
# repairable artefact is the implementation, and the only question left is WHAT the
# implementation has to do differently (or whether the failure is purely environmental and
# therefore not repairable at all). The old TEST_INCORRECT / SPEC_AMBIGUOUS verdicts and the
# "test"/"spec" fix targets were removed with the test-repair path in _run_and_repair().
TRIAGE_PROMPT = """\
You are a test-failure triage analyst.

A USER-APPROVED test derived from a behavioral contract failed against an implementation.
The contract and the tests are FIXED: the user reviewed and approved the tests, and they
are never rewritten or weakened. The implementation is the only thing that can change.

Your job is to explain WHY the implementation fails the contract and state concretely what
the implementation must do differently. Never propose editing, relaxing, or deleting a test.

Use "FLAKY_OR_ENV" (fix target "none") ONLY for failures outside the implementation's
control: a missing test runner, an unavailable third-party package, a harness timeout. A
missing or misnamed function, a wrong return type or shape, an unmet postcondition is IMPL_INCORRECT.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
  "verdict": "IMPL_INCORRECT" | "FLAKY_OR_ENV",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "explanation": "why the implementation does not satisfy the contract",
  "evidence_from_contract": "quote the relevant behavior",
  "required_impl_change": "concretely, what the implementation must do differently",
  "suggested_fix_target": "implementation" | "none"
}}

Behavioral contract:
{contract_json}

Test that failed:
{test_code}

Implementation under test:
{impl_code}

Failure output (stdout + stderr):
{failure_output}

JSON output:
"""


# ── Query engines — one set per module ───────────────────────────────────────
# params_engine   → single engine backed by the parameters index.
# lookup_engines  → one per module, MMR retriever + reranker, LOOKUP_PROMPT.
# gen_engines     → one per module, pure-similarity retriever + reranker, GENERATOR_PROMPT.
#                   Separate from lookup so generation gets depth, not diversity.
#
# NOTE: none of these RetrieverQueryEngines ever has .query() called on it — the
# actual pipeline (stages 2-5) calls each retriever's .retrieve() + the reranker's
# .postprocess_nodes() directly, then formats its own prompt and calls
# Settings.llm.complete() (see _lookup_one, _build_generation_context,
# _retrieve_param_context). That gives full control over prompt formatting and
# JSON parsing that RetrieverQueryEngine's built-in synthesizer doesn't offer.
# lookup_engines/params_engine are therefore built but effectively inert; only
# gen_engines is still read, purely as a "does this module support generation?"
# membership check in generate_verified_implementation(). Left in place (cheap to build,
# and gen_engines' membership check depends on this loop) rather than removed.

params_engine = RetrieverQueryEngine(
    retriever=params_retriever,
    node_postprocessors=[reranker_params]
)
params_engine.update_prompts({"response_synthesizer:text_qa_template": PARAMS_PROMPT})

lookup_engines: Dict[str, RetrieverQueryEngine] = {}
gen_engines:    Dict[str, RetrieverQueryEngine] = {}

for m in CODE_MODULE_LABELS:
    le = RetrieverQueryEngine(
        retriever=module_retrievers[m],
        # [] when reranker_algo is disabled — readd for generation
        node_postprocessors=[reranker_algo] if reranker_algo is not None else []
    )
    le.update_prompts({"response_synthesizer:text_qa_template": LOOKUP_PROMPT})
    lookup_engines[m] = le

    ge = RetrieverQueryEngine(
        retriever=module_gen_retrievers[m],
        # [] when reranker_algo is disabled — readd for generation
        node_postprocessors=[reranker_algo] if reranker_algo is not None else []
    )
    ge.update_prompts({"response_synthesizer:text_qa_template": GENERATOR_PROMPT})
    gen_engines[m] = ge


# ── Agent tool functions ──────────────────────────────────────────────────────

# UNUSED: neither helper is called anywhere — source citations and top-score reporting
# are done inline where needed (e.g. the "[gate] ... top=... 2nd=..." log line in
# assess_function_coverage computes its own top/second score locally).
# def _format_sources(source_nodes) -> str:
#     return "\n".join(
#         f"[Source: {n.node.metadata.get('file_name', '?')} | "
#         f"Section: {n.node.metadata.get('header_path', n.node.metadata.get('section', '?'))} | "
#         f"Score: {f'{n.score:.3f}' if n.score is not None else 'N/A'}]"
#         for n in source_nodes
#     )

# def _top_score(source_nodes) -> float:
#     return max((n.score or 0.0) for n in source_nodes) if source_nodes else 0.0


# Cache for parameter definitions text — built once on first extraction call.
# Avoids re-querying params_engine on every user query.
_param_definitions_cache: Dict[str, str] = {}

def _get_param_definitions() -> str:
    """
    Returns the full parameter definitions text from parameters.md.
    Read directly from disk (no LLM) and cached in memory for the session.
    """
    if "definitions" not in _param_definitions_cache:
        try:
            with open(PARAMETERS_FILE, "r", encoding="utf-8") as f:
                result = f.read().strip()
        except OSError as e:
            raise RuntimeError(
                f"[Startup] Could not read {PARAMETERS_FILE}: {e}"
            )
        if not result:
            raise RuntimeError(f"[Startup] {PARAMETERS_FILE} is empty.")
        _param_definitions_cache["definitions"] = result
    return _param_definitions_cache["definitions"]

def _retrieve_param_context(query_text: str) -> str:
    """
    Retrieval-only parameter context — NO LLM synthesis.
    Replaces params_engine.query() wherever only the relevant parameter TEXT is
    needed (not a generated answer). Uses the existing params_retriever.
    """
    try:
        nodes = params_retriever.retrieve(query_text)
    except Exception as e:
        print(f"[WARN] param retrieval failed: {e}")
        return ""
    return "\n\n".join(n.node.get_content() for n in nodes).strip()

def _explain_params_from_map(params: dict) -> dict:
    """
    Build {param: {value, meaning, used_for}} from the precomputed PARAM_MAP —
    NO LLM. Preserves the exact shape format_subquery_answer expects. Params not
    in the map fall back to value-only with an explanatory note.
    """
    out = {}
    for name, value in params.items():
        info = PARAM_MAP.get(name, {})
        out[name] = {
            "value": value,
            "meaning": info.get("meaning")
                       or "No definition found in parameters.md.",
            "used_for": info.get("used_for", ""),
        }
    return out

def explain_extracted_parameters(params: dict) -> dict:
    if not params:
        return {}
    return _explain_params_from_map(params)

# UNUSED: no caller builds parameter explanations from a matched function's own
# signature this way — format_subquery_answer only ever receives explanations derived
# from the sub-query's extracted_params (explain_extracted_parameters), not from the
# retrieved functions' own parameter lists.
# def explain_function_parameters(functions: list[dict]) -> tuple[dict, dict]:
#     """
#     Build ({param: value}, {param: {value, meaning, used_for}}) for the SIGNATURE
#     parameters of the returned function(s), explained from each function's own
#     docstring / signature / code. Same shape and Meaning/Used-for format as
#     explain_extracted_parameters, so format_subquery_answer renders it unchanged.
#     """
#     func_params: dict = {}
#     context_parts: list[str] = []
#
#     for fn in functions:
#         context_parts.append(
#             f"Function: {fn.get('name')}\n"
#             f"Signature: {fn.get('signature') or ''}\n"
#             f"Description: {fn.get('description') or ''}\n"
#             f"Code:\n{str(fn.get('code') or '')[:2000]}"
#         )
#         for p in (fn.get("parameters") or []):
#             name = p.get("name")
#             if not name or name == "self":          # skip the bound receiver
#                 continue
#             default = p.get("default")
#             func_params[name] = None if default in (None, "None", "null") else default
#
#     if not func_params:
#         return {}, {}
#
#     explanations = _explain_params_from_map(func_params)
#     return func_params, explanations

# Simple dict cache for ClassifiedQuery objects — avoids repeated LLM classification.
_classification_cache: Dict[str, ClassifiedQuery] = {}
def get_or_classify_query(query: str) -> ClassifiedQuery:
    """
    Normalizes and classifies a query, reusing the cached classification
    when the same normalized query was already processed.
    """
    normalized_query = " ".join(query.split())

    if normalized_query in _classification_cache:
        return _classification_cache[normalized_query]

    classification = classify_query(query, normalized_query=normalized_query)
    _classification_cache[classification.normalized_query] = classification
    return classification


# ── Stage 0 handlers: called from answer_query_deterministic() BEFORE classification ──
def is_algorithm_generation_query(query: str) -> bool:
    """
    Stage 0 pre-classification gate: decides whether the WHOLE user query is an
    algorithm-design/implementation request, before it ever reaches classify_query()/
    CLASSIFICATION_PROMPT. True routes the query to run_algorithm_generation_pathway()
    instead of the normal classify -> sub-query -> per-sub-query pipeline.

    Fails safe to False (the existing, proven classification pathway) on any parse
    error or malformed response — a broken gate must never block queries that the old
    pathway already handles correctly.
    """
    prompt = IS_ALGORITHM_GENERATION_PROMPT.format(query=query)
    raw = Settings.llm.complete(prompt).text.strip()

    try:
        parsed = extract_json_object(raw)
    except Exception as e:
        print(f"[WARN] IS_ALGORITHM_GENERATION gate failed to parse JSON ({e}); "
              f"defaulting to the normal classification pathway.")
        return False

    if not isinstance(parsed, dict):
        print(f"[WARN] IS_ALGORITHM_GENERATION gate returned non-object JSON "
              f"(got {type(parsed).__name__}); defaulting to the normal classification "
              f"pathway.")
        return False

    is_algo = bool(parsed.get("is_algorithm_generation", False))
    print(f"[IS_ALGORITHM_GENERATION] {is_algo} | reason={parsed.get('reason', '')}")
    return is_algo


# Stage 0b declares the resources its triples are built over, and the ledger itself is a
# list[dict] that six later stages already consume by that shape. Rather than change that
# contract — and every caller with it — the resource declarations are kept here, written
# once per run by the builder below, and read by the renderer and the triple report.
# Last-write-wins is correct: one run builds one ledger.
_REQUIREMENT_RESOURCES: list[dict] = []


def build_algorithm_requirements_ledger(query: str) -> list[dict]:
    """
    Calls GENERATE_ALGORITHM_REQUIREMENTS_PROMPT and returns the parsed requirements
    ledger as a list of {"id": "R<n>", "text": "..."} dicts, in the order the model
    emitted them. This is the artifact every later stage of the algorithm pathway is
    expected to carry forward and cite requirement ids against.

    Fails safe to an empty list (with a printed warning) on any parse error or
    malformed response, mirroring generate_function_spec()'s handling of the same
    failure class.
    """
    for note in _verify_library_resources(LIB_DIR):
        print(f"[WARN] {note}")
    prompt = GENERATE_ALGORITHM_REQUIREMENTS_PROMPT.format(
        query=query,
        library_resources=_render_library_resources(LIB_DIR),
    )
    raw = Settings.llm.complete(prompt, max_tokens=llm_config.stage_max_tokens(prompt)).text.strip()

    try:
        parsed = extract_json_object(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] GENERATE_ALGORITHM_REQUIREMENTS JSON parse failed ({e}). "
              f"Retrying in plain text.")
        text_raw = _retry_as_text(prompt, e, _LEDGER_TEXT_FORMAT)
        ledger = _parse_ledger_text(text_raw)
        if ledger:
            return ledger
        print(f"[WARN] GENERATE_ALGORITHM_REQUIREMENTS plain-text retry also failed. "
              f"Raw response:\n{text_raw[:2000]}")
        return []

    if not isinstance(parsed, dict) or not isinstance(parsed.get("requirements"), list):
        print(f"[WARN] GENERATE_ALGORITHM_REQUIREMENTS did not return the expected "
              f"{{'requirements': [...]}} shape (got {type(parsed).__name__}).")
        return []

    _REQUIREMENT_RESOURCES.clear()
    for entry in (parsed.get("resources") or []):
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            print(f"[WARN] GENERATE_ALGORITHM_REQUIREMENTS resource entry is malformed "
                  f"(missing 'name'); skipping it: {entry!r}")
            continue
        name = str(entry["name"]).strip()
        corresponds = entry.get("corresponds_to")
        # A list means "one of these, and a later stage picks" — kept as a tuple of
        # names rather than flattened to a string, so the checks below can test each
        # candidate against the catalogue and the configuration stage can see the set.
        if isinstance(corresponds, (list, tuple)):
            bound = tuple(str(c).strip() for c in corresponds
                          if str(c).strip() and str(c).strip().lower() != "null") or None
        elif corresponds in (None, "", "null", "None"):
            bound = None
        else:
            bound = (str(corresponds).strip(),)
        _REQUIREMENT_RESOURCES.append({
            "name": name,
            "corresponds_to": bound,
            "note": str(entry.get("note") or "").strip(),
        })

    ledger = []
    for i, entry in enumerate(parsed["requirements"], start=1):
        if not isinstance(entry, dict) or not str(entry.get("text") or "").strip():
            print(f"[WARN] GENERATE_ALGORITHM_REQUIREMENTS entry {i} is malformed "
                  f"(missing/empty 'text'); skipping it: {entry!r}")
            continue
        req_id = str(entry.get("id") or f"R{i}").strip()
        triples = []
        for t in (entry.get("triples") or []):
            if not isinstance(t, dict):
                continue
            s, p, o = (str(t.get(k) or "").strip() for k in ("s", "p", "o"))
            if not (s and p and o):
                print(f"[WARN] GENERATE_ALGORITHM_REQUIREMENTS {req_id} has a triple "
                      f"missing s/p/o; skipping it: {t!r}")
                continue
            triples.append({"s": s, "p": p, "o": o})
        ledger.append({"id": req_id, "text": str(entry["text"]).strip(),
                       "triples": triples})

    if not ledger:
        print("[WARN] GENERATE_ALGORITHM_REQUIREMENTS produced zero usable requirements.")

    # No deterministic gate acts on the ledger, so there is no "after checks" form:
    # printing one would imply a check ran. The triple report below is deliberately
    # printed under its own heading for the same reason — it changes nothing.
    _print_prompt_output("0b REQUIREMENTS LEDGER", ledger)

    for note in _drop_directive_triples(ledger, _REQUIREMENT_RESOURCES):
        print(f"[0b triples] {note}")

    findings = _requirement_triple_report(ledger, query)
    n_triples = sum(len(r.get("triples") or []) for r in ledger)
    n_bound = sum(1 for r in _REQUIREMENT_RESOURCES if r["corresponds_to"])
    print(f"\n[0b triples] {len(_REQUIREMENT_RESOURCES)} resource(s), {n_bound} bound to "
          f"the library and {len(_REQUIREMENT_RESOURCES) - n_bound} new; "
          f"{n_triples} triple(s) across {len(ledger)} requirement(s); "
          f"{len(findings)} finding(s) — report only, nothing was changed or blocked.")
    for finding in findings:
        print(f"    - {finding}")

    return ledger


def _render_requirements_ledger(ledger: list[dict]) -> str:
    """Human-readable rendering of the requirements ledger for the final answer."""
    if not ledger:
        return ("ALGORITHM REQUIREMENTS LEDGER\n" + "=" * 30 +
                "\n\n(no requirements could be extracted — see [WARN] logs)")
    lines = []
    if _REQUIREMENT_RESOURCES:
        lines.append("-- RESOURCES, and what each one IS in the codebase --")
        for resource in _REQUIREMENT_RESOURCES:
            candidates = resource["corresponds_to"]
            if not candidates:
                binding = "nothing in the library — must be built"
            elif len(candidates) == 1:
                binding = candidates[0]
            else:
                binding = (" or ".join(candidates)
                           + "  (the configuration stage picks)")
            lines.append(resource['name'])
            lines.append(f"    is  : {binding}")
            if resource["note"]:
                lines.append(f"    note: {resource['note']}")
        lines.append("")
        lines.append("-- REQUIREMENTS --")
    for req in ledger:
        lines.append(req["id"] + ": " + req["text"])
        for t in (req.get("triples") or []):
            lines.append(f"    {t['s']} --{t['p']}--> {t['o']}")
    return (
        "ALGORITHM REQUIREMENTS LEDGER\n" + "=" * 30 + "\n\n"
        + "\n".join(lines)
        + "\n\n[Stage 0b] This ledger is carried through every subsequent stage of the "
          "algorithm pathway, and every artifact produced from here on must cite the "
          "requirement id(s) it satisfies."
    )


# The two predicates that carry a how-to-build direction. A manner of construction is
# not a thing the algorithm stands in a relation to, so an edge to one is an edge to
# nothing — but these same predicates also carry real relations
# ("ReplicateStreamOverBothPaths builtBy FRER"), which is why the test below is a
# signature and not a list of these predicates.
_DIRECTIVE_PREDICATES = ("builtBy", "constrainedBy")


def _drop_directive_triples(ledger: list[dict], resources: list[dict]) -> list[str]:
    """Delete triples whose object is a MANNER of construction rather than a thing.

    Unlike every other check at this stage, this one acts: the triple is removed from the
    ledger and the resource declaration it orphaned is removed with it. The REQUIREMENT
    TEXT is never touched — "reuse existing functions" stays a numbered requirement and
    later stages are still checked against it, which is the whole reason stage 0b was
    told to extract how-to-build directives in the first place. What goes is only a fake
    edge.

    THE SIGNATURE, rather than a list of names, because the names drift: across six runs
    one concept appeared as ExistingFunctions, ExistingCodebaseFunctions and
    ComposedExistingFunctions. An object is a manner, not a thing, when all four hold:

        * it is the object of builtBy or constrainedBy;
        * it appears in exactly ONE triple in the whole graph;
        * it is never a subject — nothing about it is ever asserted;
        * its corresponds_to is null, so the library has no such thing either.

    What that buys over a name list is visible in the runs it was derived from. It keeps
    "ReplicateStreamOverBothPaths builtBy FRER", a real relation on the same predicate.
    And it keeps "MigrationAlgorithm builtBy TopLevelOrchestration" in the one run where
    the model also wrote "TopLevelOrchestration hasOutput MigrationPlan" — a top-level
    orchestration IS a thing that will exist, and it survives exactly when the design
    says something about it, rather than by being exempted by name.

    Returns one note per dropped edge, printed rather than swallowed: this pipeline
    compares models across runs, and "how many directive edges did it invent" is signal.
    """
    if not ledger:
        return []

    degree: dict = {}
    subjects: set = set()
    for req in ledger:
        for triple in (req.get("triples") or []):
            subjects.add(triple["s"])
            for end in ("s", "o"):
                degree[triple[end]] = degree.get(triple[end], 0) + 1

    bound = {r["name"]: r.get("corresponds_to") for r in resources}

    notes: list[str] = []
    dropped_objects: set = set()
    for req in ledger:
        kept = []
        for triple in (req.get("triples") or []):
            target = triple["o"]
            is_manner = (triple["p"] in _DIRECTIVE_PREDICATES
                         and degree.get(target) == 1
                         and target not in subjects
                         and not bound.get(target))
            if is_manner:
                notes.append(f"{req['id']}: dropped "
                             f"{triple['s']} --{triple['p']}--> {target}; nothing in "
                             f"the graph says anything about '{target}' and the library "
                             f"has no such thing, so the edge reaches nothing — the "
                             f"requirement text keeps it")
                dropped_objects.add(target)
            else:
                kept.append(triple)
        req["triples"] = kept

    # A resource that existed only to be the object of a dropped edge goes with it.
    still_used = {end for req in ledger for t in (req.get("triples") or [])
                  for end in (t["s"], t["o"])}
    for name in sorted(dropped_objects - still_used):
        for index, resource in enumerate(resources):
            if resource["name"] == name:
                del resources[index]
                notes.append(f"    and dropped the resource '{name}', which no "
                             f"remaining triple uses")
                break

    return notes


def _requirement_triple_report(ledger: list[dict], query: str) -> list[str]:
    """What the requirement triples look like as a graph, as printable findings.

    This is a REPORT, not a gate: nothing downstream is blocked or rewritten by it, and
    the ledger is handed on exactly as the model emitted it. It exists because the four
    things that silently ruin a triple graph are all invisible when you read the triples
    one at a time:

      * a predicate outside the closed vocabulary — the prompt gives eighteen, and an
        invented nineteenth cannot be checked by anything later;
      * two names for one thing (StartSchedule / start_schedule / TheStartSchedule) —
        the graph silently fails to join up, and every later traversal is wrong;
      * a resource that appears exactly once in the whole graph — either the alias case
        above, or a quality the query never defined and nothing reaches, which is the
        case worth finding, because that is what a dropped obligation looks like.

    Returns one string per finding, most structural first; an empty list means the graph
    is internally consistent (which is not the same as correct).
    """
    findings: list[str] = []
    if not ledger:
        return findings


    # ---- the resource declarations, which the triples are supposed to be built over ----
    declared = {r["name"]: r for r in _REQUIREMENT_RESOURCES}
    catalog = _library_resource_catalog(LIB_DIR)
    catalog_names = ({f["id"] for f in catalog["forms"]}
                     | {c["name"] for c in catalog["classes"]}
                     | {c["name"].split(".")[-1] for c in catalog["classes"]}
                     | {r["name"] for r in catalog["records"]})

    record_keys = {f"{r['name']}['{key}']" for r in catalog["records"] for key in r["keys"]}
    catalog_names |= record_keys
    # A schedule form may be named by its code instead of its id, and the code may be
    # followed by a short explanation ("schedule.Schedule  (a list of schedule.Slot ...)"),
    # so a form binding is matched on its leading dotted name.
    form_codes = {code for f in catalog["forms"] for code in f.get("codes", ())}
    catalog_names |= form_codes
    stream_parts = {f["stream_part"] for f in catalog["forms"] if f.get("stream_part")}
    catalog_names |= stream_parts

    def _known_binding(candidate: str) -> bool:
        if candidate in catalog_names:
            return True
        if any(candidate.startswith(part + " ") for part in stream_parts):
            return True
        lead = re.match(r"[A-Za-z_][\w.]*", candidate)
        return bool(lead) and lead.group(0) in form_codes

    for resource in _REQUIREMENT_RESOURCES:
        for candidate in (resource["corresponds_to"] or ()):
            if not _known_binding(candidate):
                findings.append(
                    f"resource {resource['name']} binds to '{candidate}', which is not a "
                    f"name in the library catalogue")

    used = {t[k] for req in ledger for t in (req.get("triples") or []) for k in ("s", "o")}
    for name in sorted(used - set(declared)):
        findings.append(f"triples use '{name}', which was never declared as a resource")
    for name in sorted(set(declared) - used):
        findings.append(f"resource '{name}' was declared but no triple uses it")

    known = set(_REQUIREMENT_PREDICATES)
    degree: dict = {}
    forms: dict = {}
    untripled: list[str] = []

    for req in ledger:
        triples = req.get("triples") or []
        if not triples:
            untripled.append(req["id"])
        for t in triples:
            if t["p"] not in known:
                findings.append(
                    f"{req['id']}: predicate '{t['p']}' is not in the closed vocabulary "
                    f"({t['s']} --{t['p']}--> {t['o']})")
            for name in (t["s"], t["o"]):
                degree[name] = degree.get(name, 0) + 1
                forms.setdefault("".join(ch for ch in name.lower() if ch.isalnum()),
                                 set()).add(name)

    for _, variants in sorted(forms.items()):
        if len(variants) > 1:
            findings.append("same resource under several names: "
                            + ", ".join(sorted(variants)))

    singletons = sorted(name for name, d in degree.items() if d == 1)
    if singletons:
        findings.append("named once in the whole graph, so nothing else reaches it: "
                        + ", ".join(singletons))

    if untripled:
        findings.append("requirement(s) with no triple at all: " + ", ".join(untripled))

    return findings


ALGORITHM_CONTEXT_CAP = 40  # merged-candidate ceiling; see _retrieve_algorithm_context()
# Per-requirement ceiling for the BLUEPRINT's context. Deliberately far below the
# reranker's top_n=15: on a 22-requirement ledger, 15 each would be 330 slots and the
# prompt would be mostly candidate listings. 5 is enough to show a requirement what it
# could reuse without burying the design work.
ALGORITHM_CONTEXT_PER_REQUIREMENT = 3   # v91: 5 -> 3; across runs 88a/90a/91a the members a design used ranked 1-3 in 86% of cases


def _retrieve_algorithm_context(query: str, ledger: Optional[list[dict]] = None) -> str:
    """
    Cross-module candidate retrieval for the algorithm pathway. Stage 0 bypasses
    classify_query() entirely, so there is no per-sub-query module to scope retrieval to
    the way _build_generation_context() does for the normal pipeline.

    Retrieving ONCE on the whole raw query is not enough: measured directly against a
    real multi-part algorithm query, the cross-encoder reranker scored every single
    usecase_generator candidate (14 of them were in the pre-rerank pool) below the
    top_n=15 cutoff, because the query's dominant vocabulary (impact, evaluate,
    migration, priority) matches evaluator/tsn_scheduling_problem far more strongly —
    an entire relevant module was squeezed out even though it had real candidates. That
    is exactly how the earlier run ended up hallucinating `from your_module import
    UsecaseGenerator`: nothing correct was ever shown to reuse instead.

    So this retrieves+reranks PER REQUIREMENT from the ledger (each one focused enough
    that a module's own vocabulary isn't drowned out by unrelated parts of the query),
    plus once on the raw query for anything the requirement split lost, then merges by
    (module, class_name, name) keeping the best score seen for each, sorted best-first
    and capped so the prompt stays a bounded size regardless of ledger length.
    """
    merged, _groups = _retrieve_algorithm_candidates(query, ledger)
    return _render_helpers_for_generation(merged)


# v98: when TSN_LIBRARY_MEMBERS names a file (one "Class.member" or module-level function name per line,
# "#" starts a comment), the configuration and blueprint stages get exactly those library members, taken
# from the library catalogue, instead of what retrieval ranks. The list is an input of the request, like
# the query. Unset -> retrieval as before.
ALGORITHM_LIBRARY_MEMBERS_FILE = os.environ.get("TSN_LIBRARY_MEMBERS") or None


def _fixed_library_members(path: str) -> list[dict]:
    """The function dicts for the members listed in `path`, in the file's order."""
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cls, _, name = line.rpartition(".")
        for module, entries in FUNCTION_CATALOG_BY_MODULE_CLASSED.items():
            entry = entries.get((cls or None, name)) or entries.get((cls, name))
            if entry is not None:
                out.append(_catalog_entry_to_function(entry, module))
                break
        else:
            print(f"[algogen] library member {line} is not in the library catalogue; left out")
    return out


def _retrieve_algorithm_candidates(
    query: str, ledger: Optional[list[dict]] = None
) -> tuple[list[dict], list[tuple[str, str, list[dict]]]]:
    """Retrieve+rerank once, return BOTH shapes the pathway needs.

    Returns (merged, groups):
      merged  — deduplicated best-first, capped at ALGORITHM_CONTEXT_CAP. What the
                configuration stage gets: it reasons about environment setup, which is
                not per-requirement work.
      groups  — [(requirement_id, requirement_text, top-N candidates)], in ledger order.
                What the blueprint gets, because its job IS per-requirement: it has to
                map every id to functions, and a flat pool makes that a search problem.

    Retrieval runs ONCE. Reranking 22 probes through a cross-encoder is the expensive
    part of this stage, so the two renderings share it rather than each paying for it.
    """
    if ALGORITHM_LIBRARY_MEMBERS_FILE:
        members = _fixed_library_members(ALGORITHM_LIBRARY_MEMBERS_FILE)
        print(f"[algogen] {len(members)} library members from {ALGORITHM_LIBRARY_MEMBERS_FILE} (no retrieval)")
        return members, []
    probes: list[tuple[str, str]] = [
        (str(r.get("id") or f"R?{i}"), r["text"])
        for i, r in enumerate(ledger or [])
    ] + [("(whole query)", query)]

    best: dict[tuple, dict] = {}
    groups: list[tuple[str, str, list[dict]]] = []
    for rid, text in probes:
        probe_sq = SubQuery(text=text, module="")
        functions, _, _ = _lookup_cross_module(probe_sq, unified=True, algo=True)
        for fn in functions[:ALGORITHM_CONTEXT_PER_REQUIREMENT]:   # v91: the merged pool is built from each probe's top 3 only
            key = (fn.get("routed_module"), fn.get("class_name"), fn.get("name"))
            score = fn.get("reranker_score") or 0.0
            if key not in best or score > (best[key].get("reranker_score") or 0.0):
                best[key] = fn
        groups.append((rid, text, functions[:ALGORITHM_CONTEXT_PER_REQUIREMENT]))

    merged = sorted(best.values(), key=lambda f: f.get("reranker_score") or 0.0, reverse=True)
    merged = merged[:ALGORITHM_CONTEXT_CAP]

    by_module: dict[str, int] = {}
    for fn in merged:
        m = fn.get("routed_module") or "?"
        by_module[m] = by_module.get(m, 0) + 1
    print(f"[algo-context] {len(merged)} candidate(s) merged from {len(probes)} probe(s), "
          f"by module: {by_module}; top {ALGORITHM_CONTEXT_PER_REQUIREMENT} kept per "
          f"requirement for the blueprint")

    return merged, groups


def _render_candidates_by_requirement(
    groups: list[tuple[str, str, list[dict]]]
) -> str:
    """The per-requirement candidate sets as prompt text.

    A candidate's FULL entry is written once, the first time any requirement reaches it;
    every later requirement that also reaches it gets a one-line back-reference. Without
    that, the top-5-per-requirement sets overlap heavily — the evaluator's members match
    most requirements — and the same code body would be pasted a dozen times. The
    reference keeps the per-requirement grouping, which is the point, at the cost of one
    line instead of one function body.
    """
    if not groups:
        return "No helpers found in context."

    seen: dict[tuple, str] = {}
    blocks: list[str] = []
    for rid, text, functions in groups:
        header = f"### {rid}: {text}"
        if not functions:
            blocks.append(f"{header}\n(nothing in the codebase ranked for this one)")
            continue
        lines = [header]
        fresh: list[dict] = []
        for fn in functions:
            key = (fn.get("routed_module"), fn.get("class_name"), fn.get("name"))
            label = (f"{fn.get('class_name')}.{fn.get('name')}"
                     if fn.get("class_name") else str(fn.get("name")))
            if key in seen:
                continue   # v91: no duplicates - a member is shown once, under the first requirement it ranked for
            seen[key] = rid
            fresh.append(fn)
            lines.append(f"  - {label}")
        if not fresh:
            continue       # every member it ranked was already shown under an earlier requirement
        lines.append(_render_helpers_for_generation(fresh))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _form_id_of(binding: str) -> Optional[str]:
    """The PART 2 id of a schedule-form binding written as its id or as its code
    (e.g. "schedule.Schedule  (a list of ...)" -> "compiled_schedule"); None otherwise."""
    head = str(binding).strip()
    for entry in _LIBRARY_FIXED_REPRESENTATIONS:
        if entry.get("stream_part") and head.startswith(entry["stream_part"]):
            return None              # one stream's part of a form is not the form
    for entry in _LIBRARY_FIXED_REPRESENTATIONS:
        for name in (entry["id"], *entry.get("codes", ())):
            if head == name or any(head.startswith(name + c) for c in " ([,"):
                return entry["id"]
    return None


def _render_ledger_for_prompt(ledger: list[dict], form_ids: bool = False) -> str:
    """The R1..Rn ledger as every downstream prompt is given it — sentences AND the graph.

    Stage 0b produces three things: the requirement sentences, the RESOURCES they are
    about with what each one is in the library, and the TRIPLES saying how those
    resources relate. Until this function was changed it passed on only the sentences,
    so the resources and the triples existed, were checked, were printed in the run log
    — and reached no later stage at all. Every binding 0b had worked out, every
    obligation it had made explicit, was re-derived from prose by the configuration and
    the blueprint, or not derived at all.

    The resources come first because they are what the later stages must not re-decide:
    a resource already bound to a library form is settled, and a resource bound to
    nothing is work. The triples follow each requirement they belong to, so a stage
    reading R9 sees R9's edges rather than a separate graph it has to join up itself.
    """
    if not ledger:
        return "(no requirements extracted)"

    lines: list[str] = []
    if _REQUIREMENT_RESOURCES:
        lines.append("THE RESOURCES THESE REQUIREMENTS ARE ABOUT, and what each one IS "
                     "in the codebase:")
        for resource in _REQUIREMENT_RESOURCES:
            candidates = resource.get("corresponds_to")
            if not candidates:
                binding = "nothing in the library — this has to be built"
            else:
                # form_ids: the configuration stage writes a schedule's type as its PART 2
                # id, so a binding the ledger wrote as code is shown with that id first.
                shown = [(f"{_form_id_of(c)} — written as {c}"
                          if form_ids and _form_id_of(c) and _form_id_of(c) != c.strip() else c)
                         for c in candidates]
                binding = (shown[0] if len(shown) == 1 else
                           " or ".join(shown) + "  (one of these; still to be chosen)")
            lines.append(f"  {resource['name']}")
            lines.append(f"      is   : {binding}")
            if resource.get("note"):
                lines.append(f"      note : {resource['note']}")
        lines.append("")
        lines.append("THE REQUIREMENTS, each followed by its triples:")

    for requirement in ledger:
        lines.append(f"{requirement['id']}: {requirement['text']}")
        for triple in (requirement.get("triples") or []):
            lines.append(f"      {triple['s']} --{triple['p']}--> {triple['o']}")
    return "\n".join(lines)


_UNICAST_PRECONDITION = (
    "PRECONDITION: every stream is unicast, len(usecase.streams[i]['lstnsrPssblPlcmnt']) "
    "== 1 for every i - the evaluator's routing and placement checks charge "
    "|in-degree - out-degree| at every switch, so a multicast tree is rejected where it "
    "branches")
_START_TIME_EVALUATOR_CALL = (
    "TSNConstraintEvaluator(problem, check_routing=True, check_task_placement=True, "
    "scheduling_type=ScheduleEncodingType.START_TIME), constructed inside the activation "
    "block")


def _routes_change(configuration: dict) -> bool:
    """Does this design install routes the use case does not fix? True when the ledger
    binds a resource to one stream's part of a schedule (a route that differs between
    the schedules handed in) or the configuration type is a composition rather than the
    canonical form (a state no single form holds: two routes, or an absent stream)."""
    parts = ("candidate_array[:, i]", "Schedule.slots_by_stream(i)")
    if any(c in parts for r in _REQUIREMENT_RESOURCES for c in (r.get("corresponds_to") or ())):
        return True
    rep = configuration.get("representation") or {}
    return (str(rep.get("configuration_type") or "").strip()
            not in ("", str(rep.get("schedule") or "").strip()))


def _normalize_form_ids(configuration: dict) -> list[str]:
    """An environment step's schedule type written as the form's code (schedule.Schedule,
    numpy.ndarray) is that form; the blueprint matches the PART 2 id exactly, so the id is
    written in its place."""
    notes: list[str] = []
    for step in configuration.get("environment") or []:
        if isinstance(step, dict):
            raw = str(step.get("type") or "").strip()
            form = _form_id_of(raw)
            if form and form != raw:
                step["type"] = form
                notes.append(f"step `{step.get('name')}`: type '{raw}' written as its form id '{form}'")
    return notes


def _enforce_evaluator_facts(configuration: dict) -> list[str]:
    """Two library facts no wording of the prompt makes every model follow, so they are
    applied here and reported, the way a call form is replaced by the real signature.

    1. A START_TIME evaluator for a design that installs routes the use case does not fix
       checks routes and placements: the derivation via get_task() switches the checks off
       exactly where the use case fixes them, which is false for the algorithm's own
       configurations - the new path would never be checked for connectivity.
    2. Those checks assume one listener per stream, so the use case step carries that
       precondition, written as code, for the blueprint to design against.
    """
    notes: list[str] = []
    rep = configuration.get("representation") or {}
    if str(rep.get("encoding") or "").strip().upper() != "START_TIME":
        return notes
    env = [e for e in configuration.get("environment") or [] if isinstance(e, dict)]
    evaluators = [e for e in env if str(e.get("type") or "") == "TSNConstraintEvaluator"]
    if _routes_change(configuration):
        for step in evaluators:
            obtained = str(step.get("obtained_by") or "")
            constructor = re.search(r"TSNConstraintEvaluator\(\s*problem\b", obtained)
            if not (constructor
                    and "check_routing=True" in obtained.replace(" ", "")
                    and "check_task_placement=True" in obtained.replace(" ", "")):
                step["obtained_by"] = _START_TIME_EVALUATOR_CALL
                step["reason"] = (str(step.get("reason") or "").rstrip(". ") + ". Set to the "
                                  "constructor with both flags True: the algorithm installs "
                                  "routes the use case does not fix, which the get_task() "
                                  "derivation would leave unchecked.")
                notes.append(f"evaluator step `{step.get('name')}`: was "
                             f"'{obtained[:120]}', set to the constructor call with "
                             f"problem, check_routing=True and check_task_placement=True")
    flags_on = any("=True" in str(e.get("obtained_by") or "").replace(" ", "") for e in evaluators)
    if flags_on:
        for step in env:
            if str(step.get("type") or "") == "Usecase" and \
                    "lstnsrPssblPlcmnt" not in str(step.get("obtained_by") or ""):
                step["obtained_by"] = (str(step.get("obtained_by") or "").rstrip(". ")
                                       + ". " + _UNICAST_PRECONDITION)
                notes.append(f"use case step `{step.get('name')}`: added the unicast "
                             f"precondition the evaluator's checks assume")
    return notes


def _reuse_from_environment(configuration: dict, lib_dir: str) -> list[dict]:
    """The configuration's reuse decisions, read off the calls its environment steps write.

    The model used to list them as a separate section, but that list can only ever be what
    the steps call, and restating it was the one place models went wrong: naming the
    algorithm's own conversions and checks, or the constructor of an object the caller
    hands in. Here a step's `Class(` is that class's constructor and `name.method(` is a
    method of the class of the step called `name`; a step obtained "from the caller"
    constructs nothing. Each entry then goes through the same resolver as before.
    """
    index = _library_member_index(lib_dir)
    classes = {cls: True for module in index.values() for cls in module["classes"]}
    methods = {(cls, m) for module in index.values()
               for cls, entry in module["classes"].items() for m in entry["methods"]}
    env = [e for e in configuration.get("environment") or [] if isinstance(e, dict)]
    type_of = {str(e.get("name") or ""): str(e.get("type") or "") for e in env}
    decisions: list[dict] = []
    seen: set = set()
    for step in env:
        call = str(step.get("obtained_by") or "")
        if call.strip().lower().startswith("from the caller"):
            continue
        found = [(cls, "__init__") for cls in re.findall(r"\b([A-Z]\w*)\s*\(", call)
                 if cls in classes]
        for var, method in re.findall(r"\b([a-z_]\w*)\.(\w+)\s*\(", call):
            owner = type_of.get(var)
            if owner and (owner, method) in methods:
                found.append((owner, method))
        for cls, member in found:
            key = (cls, member, step.get("name"))
            if key in seen:
                continue
            seen.add(key)
            decisions.append({"codebase_function": member, "class_name": cls,
                              "used_by": step.get("name"),
                              "reason": f"called by environment step `{step.get('name')}`"})
    return decisions


def _reuse_named_in_purposes(blueprint: dict, lib_dir: str) -> list[str]:
    """Reuse decisions for library members a function's purpose already names, in place.

    A purpose that says "validates each view with evaluator.feasible_streams" has decided
    to reuse feasible_streams; the separate reuse list is a second copy of that decision
    and models leave it out (Luna, 2026-10-24: both checks named in the purpose, neither
    in reuse_decisions). Only a dotted name counts - `Class.member`, or `x.member` where
    exactly one library class has that member - so a word in prose is never read as a
    call. Returns one line per decision added."""
    index = _library_member_index(lib_dir)
    owners: dict = {}
    for module in index.values():
        for cls, entry in module["classes"].items():
            for m in entry["methods"]:
                owners.setdefault(m, set()).add(cls)
    decisions = blueprint.setdefault("reuse_decisions", [])
    if not isinstance(decisions, list):
        return []
    have = {(str(d.get("class_name")), str(d.get("codebase_function")))
            for d in decisions if isinstance(d, dict)}
    added: list[str] = []
    for fn in blueprint.get("function_inventory") or []:
        if not isinstance(fn, dict):
            continue
        for owner, member in re.findall(r"\b([A-Za-z_]\w*)\.(\w+)", str(fn.get("purpose") or "")):
            classes = owners.get(member) or set()
            cls = owner if owner in classes else (next(iter(classes)) if len(classes) == 1 else None)
            if cls is None or (cls, member) in have:
                continue
            have.add((cls, member))
            decisions.append({"codebase_function": member, "class_name": cls,
                              "used_by": fn.get("name"),
                              "reason": f"named in the purpose of {fn.get('name')}"})
            added.append(f"{cls}.{member} (named by {fn.get('name')})")
    return added


def _configuration_violations(configuration: dict, ledger: list, lib_dir: str) -> list:
    """Deterministic checks on the CONFIGURATION artifact, run before it is allowed to
    reach the blueprint.

    This is where the encoding question is settled, so it is where the pairing that broke
    six runs is now caught — at design time, on a small JSON document, rather than in
    generated code where it has to be inferred from which library functions happen to be
    called. The rule the pipeline previously applied to an implementation
    (`_encoding_pairing_violations`) has an exact statement here: if the design says
    START_TIME, the evaluator may not be the pre-built `problem.evaluator`, because
    initialize_problem builds that one without scheduling_type and it is always PRIORITY.
    """
    problems: list[str] = []
    members = set(_library_enum_members(lib_dir))

    representation = configuration.get("representation")
    if not isinstance(representation, dict):
        return ["the configuration has no `representation` section, so the encoding was "
                "never decided — every later stage would be guessing at what a non-zero "
                "schedule cell means"]

    # The representation is a CHOICE among library-fixed forms now, not free prose, so it
    # can be checked instead of read. Three things are verifiable here and none of them
    # were before: that the chosen id is real, that every id the design carries is real,
    # and that the encoding is one the chosen representation can actually hold.
    known_reps = {entry["id"]: entry for entry in _LIBRARY_FIXED_REPRESENTATIONS}
    chosen = str(representation.get("schedule") or "").strip()
    also = [str(i).strip() for i in (representation.get("also_used") or [])
            if str(i).strip()]
    if chosen not in known_reps:
        problems.append(
            f"`representation.schedule` is '{chosen}', which is not one of the "
            f"representations the codebase fixes ({', '.join(known_reps)}). It must be "
            f"one of those ids, copied exactly — a representation described in prose "
            f"instead of chosen is how an off-by-one in the row/port mapping reached "
            f"every later stage as settled fact"
        )
    for extra in also:
        if extra not in known_reps:
            problems.append(
                f"`representation.also_used` names '{extra}', which is not one of "
                f"{', '.join(known_reps)}"
            )
        elif extra == chosen:
            problems.append(
                f"`representation.also_used` repeats the canonical representation "
                f"'{chosen}'; list only the OTHER forms this design carries"
            )

    encoding = str(representation.get("encoding") or "").strip().upper()
    if not encoding:
        problems.append("`representation.encoding` is empty: the design must say whether "
                        "a non-zero cell is a transmission start time (START_TIME) or a "
                        "queue priority (PRIORITY)")
    elif members and encoding not in members:
        problems.append(
            f"`representation.encoding` is '{encoding}', which is not a member of the "
            f"library's ScheduleEncodingType ({', '.join(sorted(members))})"
        )

    # A design carrying both forms depends on converting between them, and the ONLY
    # conversion the library has is START_TIME in both directions — a non-zero array cell
    # is start_ns + 1. So "both forms, PRIORITY" names a conversion that does not exist.
    # Stated as the conversion rather than as a property of either representation,
    # because that is what the source actually shows.
    carried = {chosen, *also}
    if encoding == "PRIORITY" and {"candidate_array", "compiled_schedule"} <= carried:
        problems.append(
            "the design carries both candidate_array and compiled_schedule but declares "
            "PRIORITY. The only conversion between those two forms is "
            "Schedule.from_compact_encoding / Schedule.to_compact_encoding, and BOTH are "
            "START_TIME in both directions (a non-zero cell is start_ns + 1). There is no "
            "PRIORITY converter, so either the encoding is wrong or this design needs "
            "only one of the two forms"
        )

    if not str(representation.get("configuration_type") or "").strip():
        problems.append(
            "`representation.configuration_type` is empty: the blueprint copies the type "
            "of one configuration from it verbatim, so it must be the canonical form id "
            "or a composition of the schedule forms"
        )

    environment = configuration.get("environment")
    if not isinstance(environment, list) or not environment:
        problems.append("the configuration has no `environment` section, so nothing says "
                        "how the problem, use case or evaluator are obtained")
        environment = []
    # A handed-in schedule is in the canonical form, and the blueprint matches its
    # structures to the step's "type" string, so the two must be the same string.
    for step in environment:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type") or "").strip()
        if step_type in known_reps and step_type != chosen:
            problems.append(
                f"environment step `{step.get('name', '?')}` is a schedule of form "
                f"'{step_type}', but `representation.schedule` is '{chosen}': the "
                f"algorithm's inputs are in the canonical form, and the blueprint matches "
                f"its structures to both strings"
            )

    evaluator_steps = [e for e in environment if isinstance(e, dict)
                       and "evaluator" in str(e.get("type", "")).lower() + str(e.get("name", "")).lower()]
    if not evaluator_steps:
        problems.append("no environment step obtains a constraint evaluator, yet every "
                        "design in this pathway scores candidate schedules with one")
    elif encoding == "START_TIME":
        for step in evaluator_steps:
            obtained = str(step.get("obtained_by") or "")
            prebuilt = re.search(r"problem\s*\.\s*evaluator|getattr\s*\(\s*problem",
                                 obtained) or "pre-built" in obtained.lower()
            explicit = "scheduling_type" in obtained or "START_TIME" in obtained.upper()
            if prebuilt and not explicit:
                problems.append(
                    f"the encoding is START_TIME but `{step.get('name', 'the evaluator')}` "
                    f"is obtained as '{obtained.strip()}'. initialize_problem builds "
                    f"problem.evaluator without ever passing scheduling_type, so that "
                    f"instance is PRIORITY: it runs priority-conflict checks, none of the "
                    f"four START_TIME timing checks, and reads start times as queue "
                    f"priorities — every timing violation scores zero. Construct it "
                    f"explicitly with scheduling_type=ScheduleEncodingType.START_TIME "
                    f"inside the activation block"
                )
            elif not prebuilt and not explicit:
                problems.append(
                    f"the encoding is START_TIME but `{step.get('name', 'the evaluator')}` "
                    f"does not say which scheduling_type it is built with; state it, "
                    f"because the default is PRIORITY"
                )

    # A reuse decision naming something the activation lifecycle already performs. The
    # configuration that produced this check said, in the same document, that entering
    # `use_instance` "runs build_metadata once" AND that build_metadata is a function to
    # call — two instructions that cannot both be followed. Caught here rather than in
    # the generated code because here it is one JSON entry to drop, and because the
    # blueprint reads this document as settled fact.
    lifecycle = _lifecycle_performed_members(lib_dir)
    for entry in configuration.get("reuse_decisions") or []:
        if not isinstance(entry, dict):
            continue
        note = lifecycle.get((str(entry.get("class_name") or ""),
                              str(entry.get("codebase_function") or "")))
        if note:
            problems.append(
                f"`{entry.get('class_name')}.{entry.get('codebase_function')}` is listed "
                f"as a reuse decision for step `{entry.get('used_by')}`, but {note}. "
                f"Calling it explicitly repeats work the library caches on purpose, so "
                f"drop this reuse decision and let the activation block do it"
            )

    known_ids = {str(r.get("id")) for r in ledger if isinstance(r, dict)}
    for entry in configuration.get("requirement_coverage") or []:
        if not isinstance(entry, dict):
            continue
        rid = str(entry.get("requirement_id") or "")
        if known_ids and rid not in known_ids:
            problems.append(
                f"the configuration claims to satisfy '{rid}', which is not a requirement "
                f"id in the ledger"
            )
    return problems


_FIXED_CONFIGURATION_TYPE = (
    "dict[int, list[np.ndarray]]: every stream index i maps to its active route columns, each a "
    "candidate_array column of shape (n_ports,), dtype int, START_TIME encoding; no entry while "
    "the stream is absent, one column for one active route, two columns while it is on two "
    "routes at once"
)
_FIXED_INPUT_OBTAINED_BY = (
    "from the caller, unmodified: a START_TIME candidate array, numpy.ndarray of int, shape "
    "(n_ports, n_streams); every non-zero column is one route of its stream; an all-zero column "
    "means the stream is absent from that schedule and gets no route in a configuration"
)
_SCHEDULE_FORM_IDS = ("candidate_array", "compiled_schedule")
# Off from v84: the fixed representation is this algorithm's design (two routes per stream, absent
# streams), so the configuration stage decides it again from the requirements.
_FIX_SCHEDULE_REPRESENTATION_ENABLED = False


def _fix_schedule_representation(configuration: dict) -> list[str]:
    """Fix how schedules are held, for every algorithm whose inputs are schedules.

    Fresh runs of the configuration stage split between two designs - routes kept as
    candidate_array columns, and routes kept as compiled Schedule objects - and every later
    stage then had to fit whichever was chosen; generation did markedly worse on the second
    (complete runs lg31 56, lg32 54, lg35 47 of 89). The choice is not the requirements' to
    make: the library's evaluator takes candidate_array views, and a compiled_schedule is
    needed only for the all-windows conflict check. So it is set here, once, as the design
    the rest of the pipeline is written against: candidate_array canonical, compiled_schedule
    also used, one configuration = stream index -> its active route columns. Returns a note
    for every value it overrides, so a run log shows what the model had chosen.
    """
    notes = []
    env = configuration.get("environment") or []
    inputs = [e for e in env if isinstance(e, dict)
              and str(e.get("type", "")).strip() in _SCHEDULE_FORM_IDS
              and "caller" in str(e.get("obtained_by", "")).lower()]
    if not inputs:
        return notes
    rep = configuration.setdefault("representation", {})
    if rep.get("schedule") != "candidate_array":
        notes.append(f"fixed representation: schedule {rep.get('schedule')!r} -> 'candidate_array'")
        rep["schedule"] = "candidate_array"
    also = [x for x in (rep.get("also_used") or []) if x != "candidate_array"]
    if "compiled_schedule" not in also:
        also.append("compiled_schedule")
        notes.append("fixed representation: also_used gains 'compiled_schedule'")
    rep["also_used"] = also
    if rep.get("configuration_type") != _FIXED_CONFIGURATION_TYPE:
        notes.append(f"fixed representation: configuration_type {str(rep.get('configuration_type'))[:80]!r} -> the fixed dict of route columns")
        rep["configuration_type"] = _FIXED_CONFIGURATION_TYPE
    for e in inputs:
        if e.get("type") != "candidate_array" or e.get("obtained_by") != _FIXED_INPUT_OBTAINED_BY:
            notes.append(f"fixed representation: input {e.get('name')!r} {e.get('type')!r} -> 'candidate_array'")
            e["type"] = "candidate_array"
            e["obtained_by"] = _FIXED_INPUT_OBTAINED_BY
    return notes


def build_algorithm_configuration(
    query: str,
    ledger: list[dict],
    context_str: Optional[str] = None,
) -> Optional[dict]:
    """Stage 0b2: settle the ENVIRONMENT before any algorithm is designed on it.

    Produces the objects the algorithm will be handed, how each is obtained, and — the
    reason this stage exists — what a non-zero schedule cell MEANS, decided once and
    written down. Runs only in the algorithm-generation pathway, immediately after the
    requirements ledger, and its output is injected into the blueprint as settled fact.

    Structured to match `build_algorithm_blueprint` deliberately: the same ledger and the
    same candidate set go in, the same "identify the member, we resolve the call form"
    convention governs its reuse decisions, and the identical resolver runs over them. Two
    stages that disagree about how to name a codebase member would reintroduce exactly the
    drift this pathway exists to remove.

    Fails safe to None with a printed warning, matching the other stage builders.
    """
    if context_str is None:
        context_str = _retrieve_algorithm_context(query, ledger)
    ledger_text = _render_ledger_for_prompt(ledger, form_ids=True)

    # The fixed representations go in as text we generate from a verified constant, not
    # as a field the model writes. Anything stale is reported rather than shipped quietly.
    for note in _verify_fixed_representations(LIB_DIR):
        print(f"[WARN] {note}", file=sys.stderr)

    for note in _verify_config_option_menu(LIB_DIR):
        print(f"[WARN] {note}", file=sys.stderr)

    prompt = ALGORITHM_CONFIGURATION_PROMPT.format(
        requirements_ledger=ledger_text,
        context_str=context_str,
        query_str=query,
        fixed_representations=_render_fixed_representations(),
        config_option_menu=_render_config_option_menu(),
    )
    raw = Settings.llm.complete(prompt, max_tokens=llm_config.stage_max_tokens(prompt)).text.strip()

    try:
        configuration = extract_json_object(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] ALGORITHM_CONFIGURATION JSON parse failed ({e}). Raw response:\n"
              f"{raw[:1500]}")
        return None
    if not isinstance(configuration, dict):
        print(f"[WARN] ALGORITHM_CONFIGURATION did not return a JSON object "
              f"(got {type(configuration).__name__}).")
        return None

    # reuse_decisions is not asked of the model: it is derived from the environment's
    # calls below, by _reuse_from_environment.
    for key, empty in (("environment", []), ("requirement_coverage", [])):
        if not isinstance(configuration.get(key), list):
            print(f"[WARN] ALGORITHM_CONFIGURATION is missing or malformed section "
                  f"'{key}' — defaulting it to an empty list.")
            configuration[key] = empty
    if not isinstance(configuration.get("representation"), dict):
        print("[WARN] ALGORITHM_CONFIGURATION is missing the 'representation' section.")
        configuration["representation"] = {}

    # The same resolver the blueprint's reuse decisions go through, for the same reason:
    # a call form recalled from memory can only agree with the real one or silently
    # contradict it, and the library is the only authority on which.
    import copy as _copy

    _print_prompt_output("0b2 CONFIGURATION", configuration)
    for note in (_normalize_form_ids(configuration) + _enforce_evaluator_facts(configuration)
                 + (_fix_schedule_representation(configuration)
                    if _FIX_SCHEDULE_REPRESENTATION_ENABLED else [])):
        print(f"[config] {note}")
    configuration["reuse_decisions"] = _reuse_from_environment(configuration, LIB_DIR)
    configuration["_reuse_decisions_as_produced"] = _copy.deepcopy(
        configuration["reuse_decisions"]
    )
    decided_encoding = str(
        (configuration.get("representation") or {}).get("encoding") or ""
    )
    corrections, unresolved = _resolve_blueprint_reuse_decisions(
        configuration, LIB_DIR, encoding=decided_encoding
    )
    if corrections:
        print(f"[config] replaced {len(corrections)} proposed call form(s) with the real "
              f"signature from {os.path.basename(LIB_DIR)}/:")
        for line in corrections:
            print(f"    {line}")
    if unresolved:
        print(f"[WARN] ALGORITHM_CONFIGURATION named {len(unresolved)} reuse decision(s) "
              f"that could not be resolved against the real library:")
        for line in unresolved:
            print(f"    {line}")

    violations = _configuration_violations(configuration, ledger, LIB_DIR)
    if violations:
        print(f"[WARN] the configuration has {len(violations)} unresolved problem(s) — "
              f"the blueprint is about to build on it anyway:")
        for line in violations:
            print(f"    {line}")
        configuration["_violations"] = violations

    _print_prompt_output("0b2 CONFIGURATION", _blueprint_for_prompt(configuration),
                         after_checks=True)

    encoding = (configuration.get("representation") or {}).get("encoding")
    covered = [str(e.get("requirement_id")) for e in configuration["requirement_coverage"]
               if isinstance(e, dict)]
    print(f"[config] encoding={encoding or 'UNSET'}; "
          f"{len(configuration['environment'])} environment step(s); "
          f"covers {covered or 'no requirements'}")
    return configuration


# Words that mean a requirement describes the ALGORITHM'S BEHAVIOUR OR OUTPUT, not the
# ground it stands on. A requirement whose text matches any of these is never withheld
# from the blueprint, however confidently the configuration claims it.
#
# This guard is the whole reason the filter below is safe. A real run's configuration
# claimed R1 "compute a migration plan between two TSN schedules", R14 "ensure the final
# result is a valid sequence of intermediate configurations ending at the target", and
# R18 "create a top-level orchestration algorithm" — on the strength of having built an
# evaluator and a problem view. Filtering on the claim alone would have hidden the three
# most important requirements in the ledger from the stage that has to design for them.
#
# The reuse mandates are in here deliberately too. "Reuse existing functions", "compose
# when possible" and "implement only genuinely missing functionality" are satisfied
# PARTLY by the environment and PARTLY by the design; withholding them would take the
# reuse pressure off the stage that writes the function inventory.
_BEHAVIOURAL_REQUIREMENT_MARKERS = (
    "migration plan", "sequence of intermediate", "intermediate configuration",
    "order in which", "stream ordering", "ordering", "minimiz", "minimis",
    "disturbance", "orchestrat", "choose", "chooses", "strategy", "strategies",
    "valid sequence", "ends at", "ending at", "final result", "frer", "replicat",
    "old path", "new path", "next stream", "transforms", "transform the",
    "reusing existing", "reuse existing", "compose", "composing",
    "genuinely missing", "coordinates",
)


def _requirements_settled_by_configuration(
    configuration: Optional[dict], ledger: list[dict]
) -> tuple[list[str], list[dict], list[str]]:
    """Which ledger requirements the configuration FULLY settles, and the ledger without
    them.

    Returns (withheld_ids, ledger_for_blueprint, notes).

    A requirement is withheld from the blueprint only when all three hold:
      1. the configuration claims it in requirement_coverage,
      2. the id exists in the ledger, and
      3. its ledger TEXT carries no behavioural marker — nothing about producing,
         ordering, choosing, validating or minimising anything.

    Rule 3 is what keeps this from being destructive. Claims are self-reported by a stage
    that has overclaimed in practice, so the ledger's own wording is the arbiter, not the
    claim. In exchange the withheld set is small and genuinely finished: typically only
    the plain input-acceptance requirements, where once the environment supplies the
    object there is nothing left for a design to do.
    """
    claimed = [str(entry.get("requirement_id"))
               for entry in ((configuration or {}).get("requirement_coverage") or [])
               if isinstance(entry, dict) and entry.get("requirement_id")]
    by_id = {str(r.get("id")): str(r.get("text") or "")
             for r in ledger if isinstance(r, dict) and r.get("id")}

    withheld: list[str] = []
    notes: list[str] = []
    for rid in claimed:
        if rid not in by_id:
            notes.append(f"{rid} is claimed by the configuration but is not a ledger id")
            continue
        text = by_id[rid].lower()
        hit = next((m for m in _BEHAVIOURAL_REQUIREMENT_MARKERS if m in text), None)
        if hit:
            notes.append(
                f"{rid} stays in the blueprint's ledger despite being claimed: its text "
                f"says \"{hit}\", so it describes what the algorithm does, not how the "
                f"environment is set up"
            )
            continue
        withheld.append(rid)

    remaining = [r for r in ledger
                 if not (isinstance(r, dict) and str(r.get("id")) in withheld)]
    return withheld, remaining, notes


def build_algorithm_blueprint(
    query: str,
    ledger: list[dict],
    context_str: Optional[str] = None,
    configuration: Optional[dict] = None,
) -> Optional[dict]:
    """
    Stage 0c: calls ALGORITHM_BLUEPRINT_PROMPT with the requirements ledger and
    retrieved cross-codebase context, producing ONE unified design for the whole
    request: a shared data model, a function inventory with an explicit call graph, a
    requirement -> function map, and reuse decisions with exact call forms.

    `context_str` is passed in by run_algorithm_generation_pathway() so the blueprint,
    the implementation call and the per-function specs all see the IDENTICAL candidate
    set — a blueprint reuse decision naming a function the implementation prompt was
    never shown is exactly the kind of drift this pathway exists to remove. Retrieves
    on its own only when called standalone.

    Fails safe to None (with a printed warning) on any parse error or malformed
    response, matching build_algorithm_requirements_ledger()'s error handling.
    """
    if context_str is None:
        _cands, _grps = _retrieve_algorithm_candidates(query, ledger)
        context_str = (_render_candidates_by_requirement(_grps) if _grps
                       else _render_helpers_for_generation(_cands))
    # Requirements the configuration FULLY settled are removed from the ledger this stage
    # sees, rather than shown and annotated. Shown-but-marked was the earlier design and
    # it produced requirement_map entries with empty function lists — an id carried
    # forward meaning nothing. A requirement that is genuinely finished is better absent.
    # The guard inside the filter is what makes removal safe; see its docstring.
    withheld, ledger_for_blueprint, notes = _requirements_settled_by_configuration(
        configuration, ledger
    )
    for note in notes:
        print(f"[algogen] {note}", file=sys.stderr)
    if withheld:
        print(f"[algogen] withholding {withheld} from the blueprint: the configuration "
              f"settles them and their text carries no behavioural clause.",
              file=sys.stderr)

    # The shared renderer, not a second copy of it: this was an inline duplicate that
    # carried only the sentences, so the blueprint silently kept receiving a thinner
    # ledger than the configuration even after the configuration's was widened.
    ledger_text = _render_ledger_for_prompt(ledger_for_blueprint)

    # The configuration reaches the blueprint stripped of this pipeline's own
    # bookkeeping, the same way the blueprint reaches later stages: handing the model
    # both a verified call form and the unverified one it replaced gives it two sources
    # of truth and an even chance of copying the wrong one.
    covered = withheld
    # Derived from the evaluator's source, not written down: see _violation_row_order.
    for note in _verify_violation_row_order(LIB_DIR):
        print(f"[WARN] {note}", file=sys.stderr)

    prompt = ALGORITHM_BLUEPRINT_PROMPT.format(
        violation_row_order=_render_violation_row_order(LIB_DIR),
        requirements_ledger=ledger_text,
        configuration_json=json.dumps(_blueprint_for_prompt(configuration), indent=2)
        if configuration else "(no configuration was produced — design the environment "
                              "yourself, and say so in your reuse decisions)",
        configuration_satisfies=", ".join(covered) or "(none)",
        context_str=context_str,
        query_str=query,
    )
    raw = Settings.llm.complete(prompt, max_tokens=llm_config.stage_max_tokens(prompt)).text.strip()
    # Zero-width characters inside a name ("F\u200b\u200bRER", seen from Luna) make it
    # match nothing, and nobody can see why: they are never meant, so they are dropped.
    raw = re.sub(r"[\u200b-\u200d\u2060\ufeff]", "", raw)

    try:
        blueprint = extract_json_object(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] ALGORITHM_BLUEPRINT JSON parse failed ({e}). "
              f"Retrying in plain text.")
        text_raw = _retry_as_text(prompt, e, _BLUEPRINT_TEXT_FORMAT, max_tokens=llm_config.stage_max_tokens(prompt))
        blueprint = _parse_blueprint_text(text_raw)
        if not any(blueprint.values()):
            print(f"[WARN] ALGORITHM_BLUEPRINT plain-text retry also failed. "
                  f"Raw response:\n{text_raw[:2000]}")
            return None

    if not isinstance(blueprint, dict):
        print(f"[WARN] ALGORITHM_BLUEPRINT did not return a JSON object "
              f"(got {type(blueprint).__name__}).")
        return None

    # "constants" is no longer asked for; _enums_from_literal_types fills it from the types.
    if not isinstance(blueprint.get("constants"), list):
        blueprint["constants"] = []
    for key in ("data_structures", "function_inventory", "requirement_map",
                "reuse_decisions"):
        if not isinstance(blueprint.get(key), list):
            print(f"[WARN] ALGORITHM_BLUEPRINT is missing or malformed section "
                  f"'{key}' — defaulting it to an empty list.")
            blueprint[key] = []
    # v96: every requirement-map entry carries the requirement's TEXT next to its id. The interface and
    # generation stages see only this map (and "Satisfies: R6" lines), never the ledger, so without the text
    # the stage that writes a function cannot see what the requirement it satisfies says.
    _req_text = {str(r.get("id")): str(r.get("text") or "") for r in (ledger or []) if isinstance(r, dict)}
    for _entry in blueprint.get("requirement_map") or []:
        if isinstance(_entry, dict) and str(_entry.get("requirement_id")) in _req_text:
            _entry["requirement_text"] = _req_text[str(_entry.get("requirement_id"))]

    # Requirement map, checked against the division of labour the prompt now states.
    #
    # The earlier version of this warned about ledger ids missing from the map, which was
    # both the wrong question and a check that could not fire: each stage settles only a
    # subset of the ledger, and it tested whether an id APPEARED rather than whether
    # anything was mapped to it — so the five ids the configuration had already settled
    # appeared with empty `functions` lists and passed in silence while the renderer
    # printed "(unmapped)" five times.
    #
    # That empty-list third state is what section 3 no longer permits. An id is either
    # mapped to real functions or absent, and absent means the configuration owns it, so
    # both halves are now answerable:
    #   - an id the configuration settled must NOT be in the map (its coverage, with a
    #     line on how, lives in the configuration's own requirement_coverage),
    #   - every other ledger id must be present AND carry at least one function name.
    # Reported, not fatal: a blueprint with a thin map is still worth resolving and
    # reading, and this stage has no repair loop to hand it to.
    for note in _form_ids_as_types(blueprint):
        print(f"[blueprint] {note}")
    _enums_from_literal_types(blueprint)
    _derive_structure_flow(blueprint)
    for note in _carry_evaluator_call(blueprint, configuration):
        print(f"[blueprint] {note}")

    function_names = {str(fn.get("name")) for fn in blueprint["function_inventory"]
                      if isinstance(fn, dict) and fn.get("name")}
    mapped: dict = {}
    for entry in blueprint["requirement_map"]:
        if isinstance(entry, dict) and entry.get("requirement_id"):
            mapped[str(entry["requirement_id"])] = [
                str(f) for f in (entry.get("functions") or []) if str(f).strip()
            ]

    settled = {str(entry.get("requirement_id"))
               for entry in ((configuration or {}).get("requirement_coverage") or [])
               if isinstance(entry, dict) and entry.get("requirement_id")}

    claimed_already_settled = sorted(settled & set(mapped))
    if claimed_already_settled:
        print(f"[WARN] ALGORITHM_BLUEPRINT maps {claimed_already_settled}, which the "
              f"configuration already settled — those ids belong only to the "
              f"configuration's requirement coverage and should be omitted here.")

    empty_or_missing = [
        r["id"] for r in ledger
        if r["id"] not in settled and not mapped.get(r["id"])
    ]
    if empty_or_missing:
        print(f"[WARN] ALGORITHM_BLUEPRINT leaves {len(empty_or_missing)} requirement(s) "
              f"with no function against them: {empty_or_missing}. Each needs a real "
              f"function name, or belongs to the configuration instead.")

    unknown_targets = sorted(
        {name for rid, names in mapped.items() for name in names
         if name not in function_names}
    )
    if unknown_targets:
        print(f"[WARN] ALGORITHM_BLUEPRINT maps requirements to {unknown_targets}, which "
              f"are not in its own function inventory.")

    # CONSTANTS, checked here so the interface freeze is handed something it can carry.
    # A formula with no owning function has nowhere to live in a file of empty bodies,
    # which is how the first attempt at carrying these ended up inventing
    # `globals()["ImpactScore.impact"] = "<prose>"`.
    data_model_names = {str(t.get("name")) for t in blueprint["data_structures"]
                        if isinstance(t, dict) and t.get("name")}
    for entry in blueprint["constants"]:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        name = str(entry["name"])
        owner = name.partition(".")[0]
        if data_model_names and owner not in data_model_names:
            print(f"[WARN] ALGORITHM_BLUEPRINT constant '{name}' is not named after a "
                  f"data structure (expected a structure name, or Name.field), so no "
                  f"declaration in the frozen interface corresponds to it.")
        definition = entry.get("definition") or entry.get("formula")
        if definition and not entry.get("computed_by"):
            print(f"[WARN] ALGORITHM_BLUEPRINT constant '{name}' gives a definition but "
                  f"names no `computed_by` function, so nothing in the design is "
                  f"responsible for producing it.")
        elif definition and str(entry["computed_by"]) not in function_names:
            print(f"[WARN] ALGORITHM_BLUEPRINT constant '{name}' names "
                  f"`computed_by: {entry['computed_by']}`, which is not in its own "
                  f"function inventory.")

    # Which structures nothing produces, and which nothing consumes. Reported like the
    # checks above and not fatal: a mis-wired structure is a design the later stages can
    # still be run against, and this pipeline exists to compare models, so "how often was
    # a structure left unowned" is signal that a silent repair would destroy.
    for note in _structure_flow_violations(blueprint, configuration):
        print(f"[WARN] ALGORITHM_BLUEPRINT {note}")
    for note in _resource_use_violations(blueprint):
        print(f"[WARN] ALGORITHM_BLUEPRINT {note}")

    # Fill each reuse decision's call form in from the real library source. The prompt
    # asks the model to IDENTIFY the member and not to write the call form, so ordinarily
    # there is nothing here to correct — but a model that answers with one anyway gets it
    # replaced rather than believed, and a decision naming something the library does not
    # have is reported now instead of becoming a runtime crash several stages later.
    import copy as _copy

    # Kept so the rendered answer can show both versions — what the design asked for and
    # what the library actually requires. Stored under a leading underscore so
    # _blueprint_for_prompt drops it before any of this reaches a prompt.
    _print_prompt_output("0c BLUEPRINT", blueprint)
    blueprint["_reuse_decisions_as_produced"] = _copy.deepcopy(
        blueprint.get("reuse_decisions", [])
    )
    for line in _reuse_named_in_purposes(blueprint, LIB_DIR):
        print(f"[blueprint] reuse decision added from a purpose: {line}")
    # The encoding the CONFIGURATION stage already decided. Without it the
    # evaluator redirect falls back to its default answer, problem.evaluator, and
    # hands a START_TIME design the PRIORITY object — contradicting the reason the
    # model gave for the very same reuse decision. Confirmed on a real run.
    corrections, unresolved = _resolve_blueprint_reuse_decisions(
        blueprint, LIB_DIR,
        encoding=str(((configuration or {}).get("representation") or {})
                     .get("encoding") or ""),
    )
    if corrections:
        print(f"[blueprint] replaced {len(corrections)} proposed call form(s) with the "
              f"real signature from {os.path.basename(LIB_DIR)}/:")
        for line in corrections:
            print(f"    {line}")
    if unresolved:
        print(f"[WARN] ALGORITHM_BLUEPRINT named {len(unresolved)} reuse decision(s) that "
              f"could not be resolved against the real library — each is a capability the "
              f"design believes it can reuse and cannot:")
        for line in unresolved:
            print(f"    {line}")

    _print_prompt_output("0c BLUEPRINT", _blueprint_for_prompt(blueprint),
                         after_checks=True)
    return blueprint


def _print_prompt_output(stage: str, payload, after_checks: bool = False) -> None:
    """Print what one stage produced, so a run can be read prompt by prompt.

    Two forms, and the pairing is the point. "PROMPT OUTPUT" is what the model actually
    returned, before this pipeline touched it. "PROMPT OUTPUT AFTER CHECKS" is the same
    artifact once the deterministic gates have had their say — imports rewritten, call
    forms resolved against the real library, misplaced __future__ imports moved. Only
    stages that HAVE deterministic checks print the second form; for the rest the first
    is the whole story, and printing an unchanged "after" would imply a check ran.

    Diffing the pair is the fastest way to see which stage a defect entered at, and
    whether a gate repaired it or merely reported it.
    """
    heading = "PROMPT OUTPUT AFTER CHECKS" if after_checks else "PROMPT OUTPUT"
    title = f" {heading} — {stage} "
    if isinstance(payload, (dict, list)):
        try:
            body = json.dumps(payload, indent=2, default=str)
        except (TypeError, ValueError):
            body = repr(payload)
    else:
        body = "" if payload is None else str(payload)
    print(f"\n{title:=^100}")
    print(body)
    print("=" * 100, flush=True)


def _render_algorithm_configuration(configuration: Optional[dict]) -> str:
    """Human-readable rendering of the configuration for the console and the final
    answer, in the same shape as the blueprint's so a run reads as one document."""
    if not configuration:
        return ("ALGORITHM CONFIGURATION\n" + "=" * 23 +
                "\n\n(no configuration could be produced — see [WARN] logs)")

    lines = ["ALGORITHM CONFIGURATION", "=" * 23, ""]

    representation = configuration.get("representation") or {}
    lines.append("-- REPRESENTATION --")
    # The chosen id is expanded back into the fact it stands for, so a run stays readable
    # without the reader having to hold the catalogue in their head.
    chosen = str(representation.get("schedule") or "")
    known = {entry["id"]: entry for entry in _LIBRARY_FIXED_REPRESENTATIONS}
    lines.append(f"schedule : {chosen or '(unstated)'}")
    if chosen in known:
        lines.append(f"   i.e.   : {known[chosen]['what']}")
    if representation.get("schedule_reason"):
        lines.append(f"   because: {representation['schedule_reason']}")
    also = [str(i) for i in (representation.get("also_used") or []) if str(i)]
    if also:
        lines.append(f"also used: {', '.join(also)}")
    if representation.get("configuration_type"):
        lines.append(f"configuration type: {representation['configuration_type']}")
    if representation.get("configuration_type_reason"):
        lines.append(f"   because: {representation['configuration_type_reason']}")
    lines.append(f"ENCODING : {representation.get('encoding', 'UNSET')}")
    if representation.get("encoding_reason"):
        lines.append(f"   because: {representation['encoding_reason']}")
    lines.append("")

    lines.append("-- ENVIRONMENT, in the order it is obtained --")
    for index, step in enumerate(configuration.get("environment", []), start=1):
        if not isinstance(step, dict):
            continue
        lines.append(f"[{index}] {step.get('name', '?')}: {step.get('type', '?')}")
        lines.append(f"    obtained by: {step.get('obtained_by', '(unstated)')}")
        if step.get("reason"):
            lines.append(f"    because    : {step['reason']}")
    lines.append("")

    lines.append("-- REQUIREMENTS THIS SETTLES --")
    for entry in configuration.get("requirement_coverage", []):
        if isinstance(entry, dict):
            lines.append(f"{entry.get('requirement_id', '?')}: {entry.get('how', '')}")
    lines.append("")

    if configuration.get("_violations"):
        lines.append("-- UNRESOLVED PROBLEMS IN THIS CONFIGURATION --")
        for problem in configuration["_violations"]:
            lines.append(f"  ! {problem}")
        lines.append("")

    lines.append("-- REUSE DECISIONS, RESOLVED AGAINST THE REAL LIBRARY --")
    for entry in configuration.get("reuse_decisions", []):
        if not isinstance(entry, dict):
            continue
        owner = entry.get("class_name") or ""
        name = entry.get("codebase_function", "?")
        label = name if not owner or owner == name else f"{owner}.{name}"
        lines.append(f"{label}  used by {entry.get('used_by', '?')}")
        if entry.get("unresolved"):
            lines.append(f"    UNRESOLVED: {entry['unresolved']}")
        else:
            first, *rest = str(entry.get("call_form", "")).splitlines() or [""]
            lines.append(f"    call: {first}")
            for extra in rest:
                lines.append(f"          {extra}")
    return "\n".join(lines)


def _render_algorithm_blueprint(blueprint: Optional[dict]) -> str:
    """Human-readable rendering of the algorithm blueprint for the final answer."""
    if not blueprint:
        return ("ALGORITHM BLUEPRINT\n" + "=" * 20 +
                "\n\n(no blueprint could be produced — see [WARN] logs)")

    lines = ["ALGORITHM BLUEPRINT", "=" * 20, ""]

    lines.append("-- DATA STRUCTURES --")
    for t in blueprint.get("data_structures", []):
        lines.append(f"{t.get('name', '?')}: {t.get('type', t.get('shape', ''))}")
        if t.get("why"):
            lines.append(f"    why     : {t['why']}")
        if t.get("used_for"):
            lines.append(f"    used for: {t['used_for']}")
        corresponds = t.get("corresponds_to")
        lines.append(f"    is       : {corresponds}" if corresponds
                     else "    is       : new to this design (no library equivalent)")
        written_by = [str(u) for u in (t.get("written_by") or [])]
        read_by = [str(u) for u in (t.get("read_by") or [])]
        lines.append(f"    written by: {', '.join(written_by) or '(nobody)'}")
        lines.append(f"    read by   : {', '.join(read_by) or '(nobody)'}")
    lines.append("")

    # Printed even when empty: an absent CONSTANTS section is itself the finding, since
    # it means the strategy literals and the score formula were never pinned down and the
    # implementation and the tests are each free to invent their own.
    enums, formulas = _blueprint_constants(blueprint)
    lines.append("-- CONSTANTS --")
    if not enums and not formulas:
        lines.append("(none declared — any enum literal or derived score in the data "
                     "model above is unpinned)")
    for name, values in enums.items():
        lines.append(f"{name}: one of {sorted(values)}")
    for name, formula in formulas.items():
        lines.append(f"{name} = {formula or '(no definition given)'}")
    lines.append("")

    lines.append("-- FUNCTION INVENTORY --")
    for fn in blueprint.get("function_inventory", []):
        calls = ", ".join(fn.get("calls", []) or []) or "(none)"
        lines.append(f"{fn.get('name', '?')}  |  {fn.get('signature', '')}")
        lines.append(f"    purpose: {fn.get('purpose', '')}")
        lines.append(f"    calls:   {calls}")
        lines.append(f"    reads:   "
                     f"{', '.join(str(x) for x in (fn.get('reads') or [])) or '(none)'}")
        lines.append(f"    writes:  "
                     f"{', '.join(str(x) for x in (fn.get('writes') or [])) or '(none)'}")
    lines.append("")

    lines.append("-- REQUIREMENT MAP --")
    for entry in blueprint.get("requirement_map", []):
        funcs = ", ".join(entry.get("functions", []) or []) or "(unmapped)"
        lines.append(f"{entry.get('requirement_id', '?')} -> {funcs}")
    lines.append("")

    # Two views of the same section, deliberately kept apart rather than merged: the
    # resolved one is what every later stage actually received, and the produced one is
    # the record of what this model asked for. Reading a run means being able to tell
    # which is which at a glance, so neither is presented as the other.
    produced = blueprint.get("_reuse_decisions_as_produced") or []

    def label_of(entry: dict) -> str:
        owner = entry.get("class_name") or ""
        name = entry.get("codebase_function", "?")
        if not owner or owner == name:
            return name          # a constructor owns itself; don't print Foo.Foo
        return f"{owner}.{name}"

    lines.append("-- REUSE DECISIONS, RESOLVED AGAINST THE REAL LIBRARY --")
    lines.append("   (authoritative: every call form here was read out of the library")
    lines.append("    source, and this is the version the later stages were given)")
    lines.append("")
    for index, r in enumerate(blueprint.get("reuse_decisions", []), start=1):
        original = produced[index - 1] if index - 1 < len(produced) else {}
        proposed = str(original.get("call_form") or "").strip()
        kind = f"  [{r['kind']}]" if r.get("kind") else ""
        lines.append(f"[{index}] {label_of(r)}  used by {r.get('used_by', '?')}{kind}")

        if r.get("unresolved"):
            lines.append(f"    STATUS  : UNRESOLVED — {r['unresolved']}")
            lines.append("    (nothing in the library matches this name, so the design "
                         "is counting on")
            lines.append("     a capability that does not exist and cannot be reused "
                         "as written)")
        else:
            # Multi-line for an instance-gated member, whose activation block is part of
            # the call form rather than an aside about it.
            first, *rest = str(r.get("call_form", "")).splitlines() or [""]
            lines.append(f"    CALL    : {first}")
            for extra in rest:
                lines.append(f"              {extra}")
            if r.get("signature"):
                lines.append(f"    REAL SIG: {r.get('codebase_function', '?')}"
                             f"{r['signature']}")
            if r.get("import"):
                lines.append(f"    IMPORT  : {r['import']}")
            elif r.get("reached_via"):
                # Two different kinds of content share this field, so the label has
                # to say which. For an instance member it is a requirement on the
                # design's own signature (NEEDS); for a constructor or a pre-built
                # object it is library guidance (NOTE). Many lines either way, so the
                # continuations are indented under the field.
                reached = str(r["reached_via"]).splitlines() or [""]
                field = ("NOTE" if r.get("kind") in ("constructor", "provided_instance")
                         else "NEEDS")
                lines.append(f"    {field:<8}: {reached[0]}")
                for extra in reached[1:]:
                    lines.append(f"              {extra}")
            if r.get("precondition"):
                # A gated member whose user is NOT the function that owns the activation
                # block. The block belongs to the orchestrator, so what this call site
                # needs recorded is the state it runs in, not a block of its own.
                pre = str(r["precondition"]).splitlines() or [""]
                lines.append(f"    ASSUMES : {pre[0]}")
                for extra in pre[1:]:
                    lines.append(f"              {extra}")
            if r.get("_call_form_as_proposed"):
                lines.append(f"    STATUS  : CORRECTED — the design proposed "
                             f"`{r['_call_form_as_proposed']}`,")
                lines.append(
                    "              which builds a second one by hand instead of using "
                    "the existing object"
                    if r.get("kind") == "provided_instance" else
                    "              which cannot work against the real signature above"
                )
            elif proposed:
                lines.append(f"    STATUS  : verified — the design proposed "
                             f"`{proposed}`, which the")
                lines.append("              real signature accepts")
            else:
                lines.append("    STATUS  : resolved from the library (the design was "
                             "asked to identify")
                lines.append("              the member, not to write its call form)")
        if r.get("reason"):
            lines.append(f"    REASON  : {r['reason']}")
    lines.append("")

    if produced:
        lines.append("-- THE SAME REUSE DECISIONS, EXACTLY AS THE DESIGN PRODUCED THEM --")
        lines.append("   (record only: this version was NOT sent to any later stage, and")
        lines.append("    is kept so a run shows what the model asked for before lookup)")
        lines.append("")
        for index, original in enumerate(produced, start=1):
            proposed = str(original.get("call_form") or "").strip()
            lines.append(f"[{index}] {label_of(original)}  "
                         f"used by {original.get('used_by', '?')}")
            lines.append(f"    call form as written: "
                         f"{proposed or '(none — none was asked for)'}")

    return "\n".join(lines)


def _interface_constant_violations(code: str, blueprint: dict) -> dict:
    """Did each declared constant actually ARRIVE in the frozen interface?

    Two routes, because a constant has two honest forms in a file with no function
    bodies yet: an enum is the `Literal[...]` of a type, and a formula is a docstring
    line on the function that computes it. Checked separately, and a third list reports
    the route the first attempt at this took — a `globals()[...]` binding or a
    definition assigned as a string — which is now forbidden by name in the prompt and
    is worth catching rather than trusting the prompt about.

    The formula check is a PRESENCE check: it confirms the definition reached the file,
    not that the eventual body computes it. That stronger question belongs to
    `_uncomputed_constant_formulas` on the implementation, and even there it can only
    tell that something non-constant is computed.
    """
    result = {"enums": [], "formulas": [], "stringified": []}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result        # the syntax gate owns this; don't double-report.

    enums, formulas = _blueprint_constants(blueprint)
    computed_by: dict = {}
    for entry in blueprint.get("constants") or []:
        if isinstance(entry, dict) and entry.get("name") and entry.get("computed_by"):
            computed_by[str(entry["name"])] = str(entry["computed_by"])

    # The forbidden route first, because a file that took it will also fail the two
    # route checks below and the cause is worth naming once rather than three times.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Call)
                        and isinstance(target.value.func, ast.Name)
                        and target.value.func.id in ("globals", "vars")):
                    key = (target.slice.value
                           if isinstance(target.slice, ast.Constant) else "?")
                    result["stringified"].append(
                        f"line {node.lineno}: `globals()[{key!r}] = ...` — a constant "
                        f"whose name is not an identifier must not be forced into a "
                        f"binding at all"
                    )
                elif (isinstance(node.value, ast.Constant)
                      and isinstance(node.value.value, str)
                      and isinstance(target, ast.Name)
                      and target.id in formulas):
                    result["stringified"].append(
                        f"line {node.lineno}: `{target.id}` is assigned its definition "
                        f"as a STRING — a formula stored as prose is still prose"
                    )

    # Enum route: the Literal of the type it names must carry exactly its values.
    literals: dict = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            target, value = node.targets[0].id, node.value
        if target is None or value is None:
            continue
        if (isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name) and value.value.id == "Literal"):
            members = (value.slice.elts if isinstance(value.slice, ast.Tuple)
                       else [value.slice])
            literals[target] = {m.value for m in members
                                if isinstance(m, ast.Constant)
                                and isinstance(m.value, str)}

    # A field-scoped enum (Type.field) may carry its values on the field itself, as the
    # Literal annotating that field in the class Type - which is what the freeze prompt
    # asks for when no alias for those values exists. A field annotated with the NAME of
    # a module-level Literal alias carries that alias's values.
    field_literals: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)):
                continue
            ann = item.annotation
            if (isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name)
                    and ann.value.id == "Literal"):
                members = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
                field_literals[f"{node.name}.{item.target.id}"] = {
                    m.value for m in members
                    if isinstance(m, ast.Constant) and isinstance(m.value, str)}
            elif isinstance(ann, ast.Name) and ann.id in literals:
                field_literals[f"{node.name}.{item.target.id}"] = literals[ann.id]

    for name, values in enums.items():
        # A field-scoped name (Type.field) is satisfied by the Literal on that field, or
        # failing that by an alias of the field's own name.
        alias = name.rpartition(".")[2] if "." in name else name
        found = field_literals.get(name) if "." in name else None
        if found is None:
            found = literals.get(alias)
        if found is None:
            candidates = [k for k, v in literals.items() if v == values]
            if candidates:
                continue     # same value set under a different alias name: fine.
            result["enums"].append(
                f"`{name}`: no `Literal[...]` in the file carries its declared values "
                f"{sorted(values)}, so nothing fixes the spelling the implementation "
                f"and the tests must agree on"
            )
        elif found != values:
            missing = sorted(values - found)
            extra = sorted(found - values)
            parts = []
            if missing:
                parts.append(f"omits {missing}")
            if extra:
                parts.append(f"adds undeclared {extra}")
            result["enums"].append(f"`{name}`: its Literal " + ", and ".join(parts))

    # Formula route: a docstring line on the function named in computed_by.
    docstrings = {node.name: (ast.get_docstring(node) or "")
                  for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    for name, definition in formulas.items():
        owner = computed_by.get(name)
        if owner is None:
            result["formulas"].append(
                f"`{name}`: the blueprint names no `computed_by` function for it, so "
                f"there is nowhere in this file for its definition to live"
            )
            continue
        if owner not in docstrings:
            result["formulas"].append(
                f"`{name}`: its computed_by function `{owner}` is not in this file"
            )
            continue
        carried = any(line.strip().lower().startswith("computes:")
                      and name.lower() in line.lower()
                      for line in docstrings[owner].splitlines())
        if not carried:
            result["formulas"].append(
                f"`{name}`: `{owner}` carries no `Computes: {name} = ...` line, so its "
                f"definition ({definition[:60]}...) reaches neither the implementation "
                f"nor the tests"
            )
    return result


def _carry_formula_lines(code: str, blueprint: dict) -> tuple[str, list[str]]:
    """Each blueprint formula's `Computes: <name> = <definition>` docstring line, written into
    its computed_by function where the model left it out. The line is a copy of the
    blueprint's own definition, so asking for it only gave the model a chance to drop it
    (Qwen, 2026-10-26: every formula line missing); it goes after the function's Satisfies
    line, or at the end of its docstring. Returns the new code and one note per line added."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []
    _, formulas = _blueprint_constants(blueprint)
    computed_by = {str(e["name"]): str(e["computed_by"]) for e in blueprint.get("constants") or []
                   if isinstance(e, dict) and e.get("name") and e.get("computed_by")}
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    lines = code.split("\n")
    inserts: list[tuple[int, str]] = []
    notes: list[str] = []
    for name, definition in formulas.items():
        fn = funcs.get(computed_by.get(name, ""))
        doc = ast.get_docstring(fn) if fn else None
        if fn is None or doc is None:
            continue
        if any(l.strip().lower().startswith("computes:") and name.lower() in l.lower() for l in doc.splitlines()):
            continue
        node = fn.body[0]
        start, end = node.lineno - 1, node.end_lineno - 1
        if start == end:
            continue            # a one-line docstring has no line to add after
        indent = re.match(r"\s*", lines[start]).group(0)
        at = next((k + 1 for k in range(start, end) if lines[k].strip().startswith("Satisfies:")), end)
        inserts.append((at, f"{indent}Computes: {name} = {' '.join(str(definition).split())}"))
        notes.append(f"{fn.name}: added `Computes: {name} = ...` from the blueprint")
    for at, text in sorted(inserts, reverse=True):
        lines.insert(at, text)
    new = "\n".join(lines)
    if _syntax_error_of(new, "<interface_freeze>") is not None:
        return code, []
    return new, notes


def _interface_blueprint_violations(code: str, blueprint: dict) -> dict:
    """Does the frozen interface actually say what the blueprint said?

    The interface-freeze prompt states four rules — nothing dropped, nothing invented,
    no parameter changed/dropped/added/reordered, and the Satisfies line taken from the
    requirement map — and until now none of them was checked, although all four are a
    plain comparison against the blueprint JSON the same call just passed in. This is
    the one stage where that matters most: the frozen interface is handed to the
    implementation AND to the test generator as fixed, so a function that quietly
    changed shape here is a function the tests will be written against in its wrong
    shape, and no later stage is allowed to correct it.

    Returns one list per rule, so each gets its own message rather than being merged
    into an undifferentiated pile — "a parameter moved" and "a function is missing" need
    different fixes.
    """
    result = {"missing": [], "invented": [], "signatures": [], "satisfies": []}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result        # the syntax gate owns this; don't double-report.

    inventory = {}
    for fn in blueprint.get("function_inventory") or []:
        if isinstance(fn, dict) and fn.get("name"):
            inventory[str(fn["name"])] = fn

    emitted = {node.name: node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    result["missing"] = sorted(set(inventory) - set(emitted))
    result["invented"] = sorted(set(emitted) - set(inventory))

    def declared_parameters(signature: str) -> Optional[list]:
        """Parameter names, in order, out of the blueprint's own signature string."""
        text = (signature or "").strip()
        if not text:
            return None
        try:
            parsed = ast.parse(text if text.endswith(":") else text + ":\n    pass")
        except SyntaxError:
            # A signature the blueprint wrote as prose rather than as Python is not a
            # mismatch to report here — there is nothing to compare against.
            return None
        for node in ast.walk(parsed):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                return [a.arg for a in
                        list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
        return None

    for name, fn in inventory.items():
        node = emitted.get(name)
        if node is None:
            continue
        want = declared_parameters(str(fn.get("signature") or ""))
        if want is None:
            continue
        args = node.args
        got = [a.arg for a in
               list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
        if got != want:
            # Said precisely, because the three cases have different fixes: a renamed
            # parameter breaks every keyword call site, a reordered one breaks every
            # positional call site silently, and a dropped one breaks the caller that
            # was passing it.
            if sorted(got) == sorted(want):
                detail = f"reordered: expected {want}, got {got}"
            elif set(want) - set(got):
                detail = f"dropped {sorted(set(want) - set(got))}: expected {want}"
            elif set(got) - set(want):
                detail = f"added {sorted(set(got) - set(want))}: expected {want}"
            else:
                detail = f"expected {want}, got {got}"
            result["signatures"].append(f"{name}: {detail}")

    # Satisfies lines, against the requirement map that produced them.
    expected_ids: dict = {}
    for entry in blueprint.get("requirement_map") or []:
        if not isinstance(entry, dict) or not entry.get("requirement_id"):
            continue
        for target in entry.get("functions") or []:
            expected_ids.setdefault(str(target), set()).add(str(entry["requirement_id"]))

    for name, node in emitted.items():
        docstring = ast.get_docstring(node) or ""
        stated: set = set()
        for line in docstring.splitlines():
            if line.strip().lower().startswith("satisfies:"):
                body = line.split(":", 1)[1].strip()
                if body.lower() not in ("", "none"):
                    stated = {part.strip() for part in body.split(",") if part.strip()}
        want_ids = expected_ids.get(name, set())
        if stated != want_ids:
            missing = sorted(want_ids - stated)
            extra = sorted(stated - want_ids)
            parts = []
            if missing:
                parts.append(f"omits {missing}")
            if extra:
                parts.append(f"claims {extra} that the map does not assign to it")
            result["satisfies"].append(f"{name}: " + ", and ".join(parts))
    return result


def _with_satisfies(blueprint: dict) -> list:
    """The function inventory as the freeze prompt shows it: each function also carries
    "satisfies", the requirement ids whose requirement_map entry names it, in map order.
    The Satisfies docstring line is that list; inverting the map by hand is where every
    model slipped (an id dropped or added on one or two functions per run)."""
    ids: dict = {}
    for entry in blueprint.get("requirement_map") or []:
        if isinstance(entry, dict):
            for fn in entry.get("functions") or []:
                ids.setdefault(str(fn), []).append(str(entry.get("requirement_id")))
    out = []
    for fn in blueprint.get("function_inventory") or []:
        if isinstance(fn, dict):
            fn = dict(fn)
            fn["satisfies"] = ids.get(str(fn.get("name")), [])
        out.append(fn)
    return out


def build_interface_freeze(blueprint: dict) -> str:
    """
    Stage 0d: INTERFACE FREEZE. Emits the blueprint's data model and function
    inventory as real, compiled Python — dataclasses for the types, complete `def`s
    with exact signatures and a `raise NotImplementedError` body for every function —
    before any implementation is written. Every later stage is meant to read this
    literal file instead of re-deriving types/signatures from the blueprint's prose
    independently, which is what let the earlier per-sub-query pathway's generated
    functions silently disagree on what a "schedule" was.

    Same one-retry syntax gate as generate_code_from_spec()/generate_algorithm_from_spec():
    a malformed file is far cheaper to catch here, with a targeted retry quoting the
    exact SyntaxError, than to discover once something downstream tries to import it.
    """
    # The blueprint's CONSTANTS reach this stage, but by a route chosen to make the
    # failure that got the first attempt reverted unrepresentable.
    #
    # That attempt asked for each constant as a module-level binding "with the EXACT
    # name the section gives it", on two assumptions a real run falsified: that a
    # constant's name is a valid Python identifier (they arrive as
    # `ImpactScore.disturbance`, `schedule encoding`, `FRER safety` — dots and spaces),
    # and that a "formula" carries numeric constants to extract (a plain sum of named
    # terms carries none). Obeyed literally, that produced
    # `globals()["ImpactScore.impact"] = "disturbance + feasibility_penalty + ..."`:
    # the formula still prose, now executable prose in the module namespace, carried
    # verbatim into the generated implementation.
    #
    # So the shape of the carrier is no longer the model's problem to solve. An enum
    # becomes the Literal of a type that already exists in the data model, or an
    # annotation on a field — never a new binding. A formula becomes a docstring line on
    # the function that computes it, which is where the implementation and the test
    # generator both read it and which needs no identifier at all. The prompt states the
    # `globals()[...]`-and-stringified-definition route as forbidden by name, and
    # _interface_constant_violations reports a constant that failed to arrive.
    prompt = INTERFACE_FREEZE_PROMPT.format(
        data_model_json=json.dumps(blueprint.get("data_structures", []), indent=2),
        constants_json=json.dumps(blueprint.get("constants", []), indent=2),
        function_inventory_json=json.dumps(_with_satisfies(blueprint), indent=2),
        requirement_map_json=json.dumps(blueprint.get("requirement_map", []), indent=2),
    )
    raw = Settings.llm.complete(prompt, max_tokens=llm_config.stage_max_tokens(prompt)).text.strip()
    code = (extract_code_block(raw) or raw).strip()

    err = _syntax_error_of(code, "<interface_freeze>")
    if err is not None:
        print(f"[interface-freeze] generated source is not valid Python ({err}); "
              f"retrying once.", file=sys.stderr)
        retry_prompt = (
            f"{prompt}\n\n"
            f"YOUR PREVIOUS ANSWER WAS REJECTED: it was not valid Python.\n"
            f"{type(err).__name__}: {err}\n"
            f"Offending line {err.lineno}: {(err.text or '').rstrip()}\n"
            "Return the corrected, complete file. Every `def`/`class` header must be "
            "followed by a real indented block, and every string/docstring must be "
            "closed."
        )
        raw = Settings.llm.complete(retry_prompt).text.strip()
        retry_code = (extract_code_block(raw) or raw).strip()
        retry_err = _syntax_error_of(retry_code, "<interface_freeze>")
        if retry_err is None:
            code = retry_code
        else:
            print(f"[interface-freeze] retry still not valid Python ({retry_err}); "
                  f"handing it downstream anyway.", file=sys.stderr)
            code = retry_code or code

    # Deterministic KB-reuse gate. The prompt above asks for real library types to be
    # imported rather than redefined, but a redefinition is the single failure this whole
    # pathway cannot recover from later: the frozen interface is restated as FIXED to the
    # implementation and repair stages, and the approved tests are written against it, so
    # a stand-in `class Schedule` emitted here is still a stand-in at the end of the run —
    # every later stage is explicitly forbidden from replacing it. Confirmed across three
    # separate model runs (llama/qwen/luna) against an independent 89-test benchmark: each
    # one redefined the library's own Schedule/Usecase/TSNScheduling locally (or aliased
    # the evaluator to Any/object), and each needed a hand-written adapter afterwards to
    # translate between the real classes and its private look-alikes. Fixed here, at the
    # one point where it is still cheap, and deterministically rather than by asking the
    # model again — which module defines a class is a fact, not a judgement call.
    _print_prompt_output("0d INTERFACE FREEZE", code)
    redefined = _redefined_kb_classes(code, LIB_DIR)
    if redefined:
        print(f"[interface-freeze] the frozen interface redefines "
              f"{len(redefined)} real library class(es) instead of importing them: " +
              ", ".join(f"'{name}' (really {module}.{name})" for name, module in redefined) +
              "; replacing each with an import of the real class.", file=sys.stderr)
        fixed = _autofix_redefined_kb_classes(code, LIB_DIR)
        # Same fix-or-reject guardrail every other deterministic rewrite in this file
        # uses: a rewrite that cannot be parsed is worse than the problem it set out to
        # solve, because nothing downstream re-checks this file before importing it.
        if _syntax_error_of(fixed, "<interface_freeze>") is None:
            code = fixed
        else:
            print("[interface-freeze] the import rewrite produced invalid Python; "
                  "keeping the interface as generated.", file=sys.stderr)

    # Does this file say what the blueprint said? Four rules the prompt states and
    # nothing used to check, each reported on its own because each needs a different
    # fix. Reported rather than auto-corrected: which of the two artifacts is right when
    # a signature differs is a judgement (the blueprint may have written an impossible
    # signature), and rewriting a frozen interface by guess is how a stage that was
    # meant to settle things starts producing a third version.
    code, formula_notes = _carry_formula_lines(code, blueprint)
    for note in formula_notes:
        print(f"[interface-freeze] {note}", file=sys.stderr)
    conformance = _interface_blueprint_violations(code, blueprint)
    if conformance["missing"]:
        print(f"[interface-freeze] DROPPED {len(conformance['missing'])} function(s) the "
              f"blueprint requires: {conformance['missing']}. Every later stage reads "
              f"this file as the complete interface, so a dropped function is a "
              f"requirement with no code behind it and no test written for it.",
              file=sys.stderr)
    if conformance["invented"]:
        print(f"[interface-freeze] INVENTED {len(conformance['invented'])} function(s) "
              f"not in the blueprint: {conformance['invented']}. These have no purpose, "
              f"no requirement and no place in the call graph, and the implementation "
              f"will be asked to fill them in anyway.", file=sys.stderr)
    if conformance["signatures"]:
        print(f"[interface-freeze] CHANGED the parameters of "
              f"{len(conformance['signatures'])} function(s) relative to the blueprint: "
              + "; ".join(conformance["signatures"])
              + ". A rename breaks every keyword call site, a reorder breaks every "
                "positional one silently, and a drop breaks whichever caller was "
                "passing it.", file=sys.stderr)
    if conformance["satisfies"]:
        print(f"[interface-freeze] the Satisfies docstring line disagrees with the "
              f"requirement map on {len(conformance['satisfies'])} function(s): "
              + "; ".join(conformance["satisfies"])
              + ". This line is the only requirement trace that survives into the "
                "implementation and the tests.", file=sys.stderr)

    # Did the design's own constants survive into the file both later stages read? The
    # forbidden route is reported first and on its own, because a file that took it will
    # also fail the two route checks and the cause is worth naming once.
    constants = _interface_constant_violations(code, blueprint)
    if constants["stringified"]:
        print(f"[interface-freeze] forced {len(constants['stringified'])} constant(s) "
              f"into a binding they have no valid name for: "
              + "; ".join(constants["stringified"])
              + ". An enum belongs in the Literal of its type and a formula in a "
                "`Computes:` docstring line; a constant that fits neither is left out.",
              file=sys.stderr)
    if constants["enums"]:
        print(f"[interface-freeze] {len(constants['enums'])} declared value set(s) did "
              f"not reach this file: " + "; ".join(constants["enums"])
              + ". The implementation and the separately-generated tests will each "
                "choose their own spelling.", file=sys.stderr)
    if constants["formulas"]:
        print(f"[interface-freeze] {len(constants['formulas'])} declared formula(s) did "
              f"not reach this file: " + "; ".join(constants["formulas"])
              + ". These are the design's answers to the query's undefined terms — its "
                "impact, its disturbance, its safety test — and nothing downstream has "
                "them.", file=sys.stderr)

    # Run here as well as at the implementation stage, because this is where a bare
    # `object`/`Any` annotation on a library type is INTRODUCED — rule 1a forbids it
    # explicitly — and an erased annotation emitted here is the shape every later stage
    # treats as fixed. Catching it two stages downstream is catching it after the tests
    # have been written against it.
    erased = _erased_library_annotations(code, LIB_DIR)
    if erased:
        print(f"[interface-freeze] hides library types behind bare object/Any "
              f"annotations, which stops anything from checking calls made through "
              f"them: " + "; ".join(erased), file=sys.stderr)

    _print_prompt_output("0d INTERFACE FREEZE", code, after_checks=True)
    return code


def _frozen_functions(frozen_interface: str) -> list[dict]:
    """Parse the frozen interface and return one entry per top-level function.

    Deterministic, no LLM: the signature handed to every later stage is READ OUT of the
    frozen source with ast rather than re-transcribed by a model. That is the whole
    point of freezing it — a spec or test that re-states the signature in its own words
    is free to drift from it, and signature drift (a function that returned a tuple
    while its tests expected a list) is what made the previous algorithm run
    unsalvageable no matter how good the individual artifacts were.
    """
    try:
        tree = ast.parse(frozen_interface)
    except SyntaxError as e:
        print(f"[WARN] frozen interface does not parse ({e}); no signatures could be "
              f"extracted from it.")
        return []

    entries = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        signature = f"def {node.name}({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        signature += ":"
        entries.append({
            "name": node.name,
            "signature": signature,
            "docstring": ast.get_docstring(node) or "",
        })
    return entries


def _frozen_type_names(frozen_interface: str) -> list[str]:
    """Top-level class names defined in the frozen interface (the shared data model)."""
    try:
        tree = ast.parse(frozen_interface)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, ast.ClassDef)]


def _frozen_interface_for_tests(frozen_interface: str) -> str:
    """The frozen interface, prefixed with what a test author needs to know about it.

    Without this the test writer sees only "Function to import: from generated_impl
    import <one function>" and has no way to know the data-model types are importable
    too — so it either invents a look-alike or passes a dict where a dataclass is
    required, and the test fails for a reason that has nothing to do with the
    implementation being wrong. The whole algorithm is generated into ONE module, so
    every name in the frozen interface really is importable from it.
    """
    type_names = _frozen_type_names(frozen_interface)
    if type_names:
        importable = ", ".join(type_names)
        header = (
            f"# The WHOLE algorithm below is generated into a single module named "
            f"'{GENERATED_MODULE_NAME}'.\n"
            f"# Any name in it is importable in a test, not just the function under test:\n"
            f"#     from {GENERATED_MODULE_NAME} import {importable}\n"
            f"# Build REAL instances of these dataclasses for arguments (never a bare "
            f"object() or MagicMock).\n\n"
        )
    else:
        header = (
            f"# The WHOLE algorithm below is generated into a single module named "
            f"'{GENERATED_MODULE_NAME}'; any name in it is importable in a test.\n\n"
        )
    return header + frozen_interface


def _algorithm_request_block(
    query: str,
    ledger: list[dict],
    frozen_interface: str,
    fn_entry: Optional[dict] = None,
) -> str:
    """The request block handed to the spec/implementation prompts on this pathway.

    Mirrors _build_generation_context()'s request_block for the normal pipeline, but
    carries the requirements ledger instead of extracted parameters — on this pathway
    the ledger, not the raw query, is what every artifact is answerable to. When
    `fn_entry` is given the block is scoped to that one frozen function.
    """
    block = (
        f"Original request:\n{query}\n\n"
        f"Requirements ledger (every artifact must be traceable to these):\n"
        f"{_render_ledger_for_prompt(ledger)}\n"
    )
    if fn_entry is not None:
        block += (
            f"\nYou are working on ONE function of the frozen interface. Its frozen "
            f"signature is AUTHORITATIVE — do not restate it differently, rename a "
            f"parameter, or change the order:\n"
            f"{fn_entry['signature']}\n\n"
            f"Its frozen docstring (purpose, the requirement ids it satisfies, and the "
            f"other functions it calls):\n{fn_entry['docstring'] or '(none)'}\n\n"
            f"The complete frozen interface it lives in (all sibling functions and the "
            f"shared data model — the types in these signatures are the real ones):\n"
            f"{frozen_interface}\n"
        )
    return block


def build_algorithm_tests(
    query: str,
    ledger: list[dict],
    blueprint: dict,
    frozen_interface: str,
    context_str: str,
    interactive: bool,
) -> list[dict]:
    """
    Stage 0f: per frozen function, spec -> contract -> spec-based tests (-> approval).

    The FUNCTION_SPEC_PROMPT step is deliberately KEPT here (unlike on the
    implementation side, where the frozen interface replaced it): a contract's
    "behaviors" / "input_domains" / "invariants" have no equivalent anywhere in the
    blueprint, and those are what make a test suite check behavior instead of shape.

    What is NOT left to the model: the function's name and signature are overwritten
    with the ones parsed out of the frozen interface before the contract is derived, so
    a spec that drifted cannot carry that drift into the tests. Every function's tests
    import from the same generated module, because the whole algorithm is generated as
    ONE file — which also means a function that fails to call the sibling its design
    says it calls now shows up as a test failure instead of passing unnoticed.

    Approval happens ONCE, after every function's tests are generated — see
    _approval_gate_algorithm() — not per function as this loop runs. A reviewer looking
    at how two functions compose (e.g. "compute_migration_plan's B2 assumes X, but
    determine_stream_order's B1 returns Y") cannot make that comment while only one
    function's tests are ever on screen at a time, and _run_and_repair_algorithm() below
    already executes every function's tests together in one pytest run — so gating them
    one at a time bought no earlier feedback, only more prompts.

    Returns one entry per function: {name, spec, contract, import_hint, tests, query_str}.
    """
    results = []
    frozen_for_tests = _frozen_interface_for_tests(frozen_interface)
    # TEMPORARILY COMMENTED OUT
    # for fn_entry in _frozen_functions(frozen_interface):
    #     name = fn_entry["name"]
    #     request_block = _algorithm_request_block(query, ledger, frozen_interface, fn_entry)
    #
    #     spec = generate_function_spec(context_str, request_block)
    #     if spec is None:
    #         print(f"[spec] no usable specification for '{name}'; skipping its tests.",
    #               file=sys.stderr)
    #         continue
    #
    #     # The frozen interface wins over whatever the spec called it.
    #     if spec.get("name") != name or spec.get("signature") != fn_entry["signature"]:
    #         print(f"[spec] overriding spec name/signature for '{name}' with the frozen "
    #               f"interface's own.", file=sys.stderr)
    #     spec["name"] = name
    #     spec["signature"] = fn_entry["signature"]
    #
    #     try:
    #         contract = _derive_contract_from_spec(spec, query)
    #         contract["signature"] = fn_entry["signature"]
    #     except Exception as e:  # noqa: BLE001
    #         print(f"[tests] contract derivation failed for '{name}': {e}", file=sys.stderr)
    #         continue
    #
    #     import_hint = f"from {GENERATED_MODULE_NAME} import {name}"
    #     try:
    #         tests = _spec_based_tests(
    #             contract, import_hint, query_str=request_block,
    #             blueprint=blueprint, frozen_interface=frozen_for_tests,
    #         )
    #     except Exception as e:  # noqa: BLE001
    #         print(f"[tests] spec-based generation failed for '{name}': {e}", file=sys.stderr)
    #         continue
    #
    #     # Deterministic pre-flight (no LLM to ask twice): a test can correctly import a
    #     # real KB class and still call it with the wrong number of arguments for its
    #     # actual signature (see _bad_kb_call_arity's docstring) — the single largest
    #     # failure category traced on one real run, 78 of 199 failing tests, every one of
    #     # them written AFTER its import had already been fixed. Caught here, before a
    #     # single test ever runs, by reusing the same revision path the interactive review
    #     # loop itself uses below (_spec_based_tests(..., revision=..., previous_tests=)),
    #     # rather than waiting for a confusing TypeError to surface deep in a pytest run.
    #     bad_arity = _bad_kb_call_arity_in_tests(tests, LIB_DIR)
    #     if bad_arity:
    #         print(f"[tests] {name}: {len(bad_arity)} real-KB-call arity problem(s); "
    #               f"revising once.", file=sys.stderr)
    #         try:
    #             revised = _spec_based_tests(
    #                 contract, import_hint,
    #                 revision=(
    #                     "The following test(s) call a real, correctly-imported library "
    #                     "class/function with the wrong number of arguments for its "
    #                     "actual signature. Fix ONLY the call site(s) named below so each "
    #                     "matches the real signature shown for it; change nothing else "
    #                     "about any test:\n" + "\n".join(bad_arity)
    #                 ),
    #                 previous_tests=tests, query_str=request_block,
    #                 blueprint=blueprint, frozen_interface=frozen_for_tests,
    #             )
    #         except Exception as e:  # noqa: BLE001
    #             print(f"[tests] {name}: arity revision failed ({e}); keeping the "
    #                   f"original tests.", file=sys.stderr)
    #             revised = None
    #         if revised and not _bad_kb_call_arity_in_tests(revised, LIB_DIR):
    #             tests = revised
    #         else:
    #             print(f"[tests] {name}: revision did not resolve the arity problem(s); "
    #                   f"keeping the original tests.", file=sys.stderr)
    #
    #     # Deterministic pre-flight, same spirit as the arity check above: a test can
    #     # correctly import every real name it uses and still assert something about
    #     # this function's return shape (or reach for a name via reflection) that the
    #     # frozen body it was supposedly derived from directly contradicts. Fails
    #     # closed — a flagged test is dropped here rather than approved and left for
    #     # the repair loop to somehow reconcile later. See
    #     # _tests_contradict_frozen_interface's docstring for the real run this is
    #     # built from.
    #     tests = _drop_tests_contradicting_frozen_interface(
    #         tests, name, frozen_for_tests, stage="initial generation"
    #     )
    #
    #     print(f"[tests] {name}: {len(tests)} test(s) from "
    #           f"{len(contract.get('behaviors', []))} behavior(s).")
    #
    #     results.append({
    #         "name": name,
    #         "spec": spec,
    #         "contract": contract,
    #         "import_hint": import_hint,
    #         "tests": tests,
    #         "query_str": request_block,
    #     })
    for fn_entry in _frozen_functions(frozen_interface):
        name = fn_entry["name"]
        request_block = _algorithm_request_block(query, ledger, frozen_interface, fn_entry)
        results.append({
            "name": name,
            "spec": None,
            "contract": None,
            "import_hint": f"from {GENERATED_MODULE_NAME} import {name}",
            "tests": [],
            "query_str": request_block,
        })

    if interactive and any(r["tests"] for r in results):
        note = (
            f"NOTE: algorithm pathway — tests for all {len(results)} function(s) of the "
            f"frozen interface, shown together because they are generated and executed "
            f"as ONE file. Signatures are frozen; every test below was written from its "
            f"function's contract, not from the code."
        )
        results = _approval_gate_algorithm(note, results, blueprint, frozen_for_tests)

    # Re-applied on whatever the interactive loop above settled on, no matter how many
    # review rounds ran: a "describe changes to remake" regeneration can reintroduce
    # exactly this class of contradiction as easily as the first pass did, and this is
    # the one point every path through this function funnels through before its
    # result is treated as approved.
    def _tests_digest(entries):
        return [{"function": e["name"],
                 "test_count": len(e.get("tests") or []),
                 "tests": [t.get("code", "") for t in (e.get("tests") or [])]}
                for e in entries]

    _print_prompt_output("0f TESTS", _tests_digest(results))

    for entry in results:
        entry["tests"] = _drop_tests_contradicting_frozen_interface(
            entry.get("tests", []), entry["name"], frozen_for_tests, stage="post-review"
        )

    _print_prompt_output("0f TESTS", _tests_digest(results), after_checks=True)
    return results


def _repair_algorithm_impl(
    impl_code: str,
    blueprint: dict,
    frozen_interface: str,
    context_str: str,
    request_block: str,
    failure_output: str,
    guidance: str = "",
) -> str:
    """Whole-file repair for the algorithm pathway.

    _repair_impl() cannot be reused here: it asks for "a single module-level def" and is
    shaped around one spec + one contract, whereas this pathway generates every frozen
    function in ONE file. The frozen interface is restated as fixed so a repair can
    never "fix" a failure by changing a signature the tests were written against.

    `context_str` (the full retrieved-candidate dump, often 10k+ tokens with 40 full
    function bodies) is deliberately NOT included in the prompt below, even though it
    is still accepted as a parameter for call-site compatibility. Repair fixes bugs in
    an implementation that already made its reuse decisions during initial generation —
    those decisions are already recorded in `blueprint`'s reuse_decisions, which is a
    small fraction of the size. Resending the full candidate dump on every repair round
    was the single largest contributor to a real run overflowing the local model's
    32768-token context window at this exact call.
    """
    guidance_block = (
        f"Triage — what must change in the implementation:\n{guidance}\n\n" if guidance else ""
    )
    prompt = (
        "You are fixing a Python implementation that failed contract-derived tests.\n"
        "The frozen interface, the contracts and the tests are all FIXED — only the\n"
        "implementation may change. Never weaken a test, and never change a function's\n"
        "name, parameters, or return type to make one pass. Reuse decisions were already\n"
        "made during initial generation — consult the blueprint's reuse_decisions below\n"
        "for what to call and how; do not re-derive them from scratch.\n"
        "This restriction extends to every dataclass the frozen interface DEFINES for\n"
        "this design — a migration step, an impact score, a stream order: never add,\n"
        "remove, rename, or change the default of a field on one of those, even when a\n"
        "failure seems to call for it. Every already-approved test constructs them\n"
        "against their CURRENT field list; changing that list does not fix the failure,\n"
        "it breaks every one of those tests at construction time, before the function\n"
        "under test is even reached, which is worse than the original failure. If a\n"
        "failure genuinely cannot be fixed without a field one of those dataclasses does\n"
        "not have, do NOT add it: leave it exactly as given, do the closest correct thing\n"
        "you can within its shape for that one function, and say so in a one-line\n"
        "comment. That is a design gap in the frozen interface itself, not something a\n"
        "repair pass can fix.\n"
        "The ONE exception, and it goes the other way: a type the frozen interface\n"
        "IMPORTS from the library (schedule.Schedule, schedule.Slot,\n"
        "usecase_generator.Usecase, tsn_scheduling_problem.TSNScheduling,\n"
        "evaluator.TSNConstraintEvaluator) is the real class, with the real API and the\n"
        "real fields — use them. Do not shadow one with a local class or dataclass of the\n"
        "same name, do not alias one to Any or object, and if you find such a stand-in\n"
        "already in the implementation you are fixing, DELETE it and use the imported\n"
        "class instead: a same-named local definition silently hides the import above it\n"
        "and makes every real method of that class unreachable for the whole file. Fixing\n"
        "that is always in scope for a repair, and never counts as changing a frozen type.\n"
        "The same restriction applies to a function's actual logic, not just its frozen\n"
        "shape: never satisfy a failing test by making a function ignore its real input\n"
        "and return a constant, empty, or trivial result instead of genuinely computing\n"
        "from what it was given - check a fix against the requirements given below before\n"
        "deciding it is acceptable. If a failure genuinely cannot be fixed without\n"
        "violating what those requirements call for, do the closest correct thing within\n"
        "them for that one function, and say so in a one-line comment, rather than\n"
        "hollowing the function out just to stop the error.\n"
        "Real config.ini shape - if the fix constructs a TSNScheduling or Usecase from a\n"
        "config file, this is the exact, complete set of keys usecase_generator.py and\n"
        "tsn_scheduling_problem.py ever read for the 'ft' (fat-tree) topology path,\n"
        "confirmed by reading every config-reading call site in both files directly:\n"
        "    [topology_config]\n"
        "    numhosts = 8\n"
        "    numswitchs = 4\n"
        "    portspeed = 1000\n"
        "    procdelay = 1\n"
        "    propdelay = 100\n"
        "    linkBw = 1000\n"
        "    queuecountperport = 8\n"
        "    graphConfig = {'typ':'ft','k':4}\n"
        "\n"
        "    [streams_config]\n"
        "    numofstreamsperinstance = 4\n"
        "    period_range = {'start':100000,'stop':200000,'step':100000}\n"
        "    numberframes_range = {'start':1,'stop':2}\n"
        "    datasize_range = {'start':64,'stop':128}\n"
        "    maxlatency_range = 1.0,2.0\n"
        "    numplacestalker = 1\n"
        "    numplaceslistener = 1\n"
        "    numlistenersperstr = 1\n"
        "    routing = {'fixPath':True,'shortestPath':True}\n"
        "    expiration = {'finite_duration':False,'start':1,'stop':2}\n"
        "    stride = {'active':False}\n"
        "    staggered = {'active':False}\n"
        "    hotspot = {'active':False}\n"
        "This exact file already exists at this project's root as `config.ini` - reuse it\n"
        "by path where a real file is genuinely needed rather than fabricating a different\n"
        "one or a different shape. `expiration`, `stride`, `staggered`, and `hotspot` are\n"
        "each read as a dict literal and indexed by key (e.g. `stride['active']`), so a\n"
        "bare number or an empty `{}` fails with KeyError exactly like omitting the key;\n"
        "`maxlatency_range` is the one exception, read as a comma-separated string of\n"
        "floats, never a dict. Do not add `generation`, a standalone `fixPath` key,\n"
        "`maxstrpernode`, or `minnodedegreepersw` - each looks plausible but is dead in\n"
        "the real library code (only in a commented-out line, or reachable only on a\n"
        "non-'ft' topology path).\n"
        "Two implementation-bug patterns confirmed via real test execution, both worth\n"
        "checking directly against this fix:\n"
        "1. Validate every one of a function's own parameters (shape, type, range,\n"
        "   dimension-matching) before using any parameter as a collaborator - before\n"
        "   entering a `with x.use_instance(y):` block, calling a method on it, or\n"
        "   passing it into another function. A test may deliberately give a minimal or\n"
        "   fake object for an argument it is not testing; reaching for that argument\n"
        "   before finishing validation of the argument actually under test means\n"
        "   crashing on the fake object instead of raising the validation error the test\n"
        "   expects - on a real run this was the single largest cause of test failures.\n"
        "2. A lightweight stand-in class carrying a real library type's NAME (for example\n"
        "   a local `class TSNScheduling` with a single value field, sitting alongside the\n"
        "   real tsn_scheduling_problem.TSNScheduling) is itself the bug, not a constraint\n"
        "   to write code around. Such a definition shadows the real class for the whole\n"
        "   file, so .use_instance(...), .n_ports, .group_connected_ports(...) and every\n"
        "   other real member simply stop existing, and each call to one raises\n"
        "   AttributeError. Do NOT respond by giving the stand-in the methods it is\n"
        "   missing, and do NOT route around it: delete it, import the real class from its\n"
        "   own module, and call the real API on the real object.\n"
        "Return ONLY the corrected, complete Python file (every frozen function, fully\n"
        "implemented, plus any private helpers), no fences, no prose.\n\n"
        f"Here is the requirements the generated code needs to fulfill:\n{request_block}\n\n"
        f"Design blueprint:\n{json.dumps(_blueprint_for_prompt(blueprint), indent=2)}\n\n"
        f"Frozen interface (signatures are fixed):\n{frozen_interface}\n\n"
        f"{guidance_block}"
        f"Current implementation:\n{impl_code}\n\n"
        f"Test failure output:\n{failure_output}\n\n"
        "Corrected implementation:"
    )
    # Same max_tokens rationale as generate_algorithm_from_spec(): this returns a
    # complete re-implementation of every frozen function in one file, not a diff, so
    # it needs the same headroom above the 3000-token default.
    fixed = Settings.llm.complete(prompt, max_tokens=llm_config.stage_max_tokens(prompt)).text.strip()
    repaired = (extract_code_block(fixed) or fixed).strip()

    err = _syntax_error_of(repaired)
    if err is None:
        return repaired

    print(f"[algo-repair] repaired source is not valid Python ({err}); retrying once.",
          file=sys.stderr)
    retry = Settings.llm.complete(
        f"{prompt}\n\n"
        f"YOUR PREVIOUS ANSWER WAS REJECTED: it was not valid Python.\n"
        f"{type(err).__name__}: {err}\n"
        f"Offending line {err.lineno}: {(err.text or '').rstrip()}\n"
        "Return the corrected, complete file. Every `if`/`for`/`while`/`try`/`def` header "
        "must be followed by a real indented block, and every string/docstring must be closed.",
        max_tokens=llm_config.stage_max_tokens(prompt),
    ).text.strip()
    retry_code = (extract_code_block(retry) or retry).strip()
    if _syntax_error_of(retry_code) is None:
        return retry_code

    print("[algo-repair] retry still not valid Python; keeping the previous implementation.",
          file=sys.stderr)
    return impl_code


_FAILED_TEST_NAME_RE = re.compile(
    r"^(?:FAILED|ERROR)\s+test_generated\.py::(\w+)", re.MULTILINE
)


def _failing_test_names(failure_output: str) -> set[str]:
    """Test function names pytest's own summary reports as FAILED or ERROR.

    Used to shrink what _triage_failure() is shown: with many frozen functions each
    carrying many tests, concatenating every approved test's source for every triage
    call scales with the WHOLE suite's size regardless of how much of it actually
    failed — confirmed to overflow the local model's context window on a real run (7
    functions, 94 tests). Parsed from pytest's own "short test summary info" lines,
    which are already present in every failure_output this function receives.
    """
    return set(_FAILED_TEST_NAME_RE.findall(failure_output))


def _namespace_function_helpers(name: str, code: str) -> str:
    """Rename every private helper (`def _foo(...):`) and every module-level class this
    ONE function's own tests define to `_foo__<name>_<n>` / `Foo__<name>_<n>`, and
    rewrite this same code's own references to match — so two unrelated functions that
    each independently pick the identical name can never collide once concatenated into
    one file, AND so two of this SAME function's own tests that independently pick the
    identical name (see below) don't collide with each other either.

    Each function's tests are generated by its own isolated LLM call with zero
    visibility into any other function's tests (see feedback_cross_function_name_
    collision), so two functions picking the same private helper name is not a rare
    mistake, it is an expected consequence of that isolation — confirmed at least four
    times in one session (`_contexts`, `_make_evaluator`, a `TSNScheduling`/`Usecase`
    import collision, `_make_case`), each under a different name, each only noticed via
    a full review round. Since Python executes every module-level `def`/`class`
    top-to-bottom once at collection time, whichever definition is textually LAST in
    the concatenated file silently wins for literally every test in the file, not just
    its own section — confirmed with a real, reproduced failure: two functions' own
    `_make_case` helpers, one 7-parameter and one 3-parameter, concatenated without
    this fix raise `TypeError: _make_case() takes 3 positional arguments but 7 were
    given` on the earlier function's own, otherwise-correct tests.

    Classes are namespaced unconditionally (no leading-underscore requirement the way
    helper functions get one): a throwaway test-support stand-in — `class Value:`, `class
    FakeEvaluator:` — routinely does not follow that convention in practice, confirmed on
    a real run where 21 of one function's own 33 tests each defined their own differently
    -bodied `class Value:` as a fake evaluator, all sharing the one bare name.

    Each occurrence of a given name is numbered independently (`_1`, `_2`, ...) rather
    than collapsed onto one shared replacement, and each reference to that name is bound
    to whichever same-named definition most recently precedes it in top-level order —
    NOT to a single, shared rename for the whole file. This matters even within a SINGLE
    function's own tests: the `class Value:` case above is not a cross-function
    collision at all, it is 21 independent per-test definitions of the identical name in
    one function's own test batch, generated in one shot. A single shared rename would
    "fix" the concatenation-wide collision while recreating an identical collision
    between those 21 definitions, since they would then all share ONE new name instead
    of the original one — confirmed the naive single-rename version of this function
    collapses exactly that way, and 17 of those 21 tests kept failing afterward with the
    same `AttributeError` for a different-but-equally-wrong reason.

    This is orthogonal to `_bad_kb_call_arity`/`_wrong_module_imports`: those check a
    call against a REAL library object; this only ever touches names this function's own
    tests defined themselves, so renaming one is always semantics-preserving — there is
    no external meaning to accidentally change, unlike renaming a KB import would be.

    Edits are applied as exact text-span replacements at each AST node's own
    lineno/col_offset (not by re-serializing the whole file via ast.unparse), so a name
    that happens to also appear inside an unrelated string literal, comment, or
    docstring is left untouched — only a real `def`/`class` and its real reference sites
    are ever positions this walks. Returns `code` unchanged if it does not parse (a
    syntax gate handles that case elsewhere), if it defines no top-level private helper
    or class, or if applying the renames would itself produce invalid Python (never risk
    shipping a rewrite that is worse than doing nothing).

    Known, accepted limitation: a name that a specific test also re-defines LOCALLY
    inside its own body (shadowing the module-level one just for that one test) is not
    resolved via real Python scoping here — the nearest-preceding-top-level-definition
    heuristic above only reasons about module level, so that one test's shadowed
    reference would incorrectly redirect to the renamed OUTER definition instead of its
    own local one. Not fixed: this pattern has not been observed in any real generated
    test this session, and correctly resolving it needs real scope analysis (e.g. the
    `symtable` module), which is not worth the added complexity for a pattern that has
    never actually occurred.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code           # a syntax gate handles this case; don't double-report.

    # Only TOP-LEVEL (module-scope) definitions collide across concatenation — one
    # nested inside a test function's own body is already scoped to that test and
    # cannot leak into, or be leaked into by, any other definition of the same name.
    def _is_renamable(node) -> bool:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name.startswith("_") and not node.name.startswith("__")
        return isinstance(node, ast.ClassDef)

    base_names = {node.name for node in tree.body if _is_renamable(node)}
    if not base_names:
        return code

    def _def_span(node):
        # None of FunctionDef/AsyncFunctionDef/ClassDef has a separate node for just
        # the name token, so its own header line is searched directly for where the
        # name text starts.
        line = lines[node.lineno - 1]
        if isinstance(node, ast.AsyncFunctionDef):
            marker, skip = f"async def {node.name}(", len("async def ")
        elif isinstance(node, ast.FunctionDef):
            marker, skip = f"def {node.name}(", len("def ")
        else:
            marker, skip = f"class {node.name}", len("class ")
        idx = line.find(marker)
        if idx == -1:
            return None
        idx += skip
        return (node.lineno, idx, node.lineno, idx + len(node.name))

    lines = code.splitlines(keepends=True)
    spans: list[tuple[int, int, int, int, str]] = []
    occurrence_counts: dict[str, int] = {}
    current_target: dict[str, str] = {}    # base name -> the rename currently "live"

    for node in tree.body:
        if _is_renamable(node) and node.name in base_names:
            occurrence_counts[node.name] = occurrence_counts.get(node.name, 0) + 1
            new_name = f"{node.name}__{name}_{occurrence_counts[node.name]}"
            current_target[node.name] = new_name
            span = _def_span(node)
            if span:
                spans.append((*span, new_name))
        # Every Name reference inside this top-level statement binds to whichever
        # same-named definition is CURRENTLY live — i.e. the nearest one at or above
        # this point in top-level order, matching what each isolated pre-concatenation
        # test block meant, not a single global rename shared by every occurrence.
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in current_target:
                spans.append((sub.lineno, sub.col_offset, sub.end_lineno,
                              sub.end_col_offset, current_target[sub.id]))

    if not spans:
        return code

    # Applied back-to-front so an earlier span's offset is never shifted by a later
    # edit on the same or an earlier line.
    spans.sort(key=lambda s: (s[0], s[1]), reverse=True)
    for lineno, col, end_lineno, end_col, new_text in spans:
        if lineno != end_lineno:
            continue          # a Name node is always single-line; defensive only.
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:col] + new_text + line[end_col:]

    new_code = "".join(lines)
    if _syntax_error_of(new_code) is not None:
        return code           # the rename somehow broke parsing; never ship it.
    return new_code


def _run_and_repair_algorithm(
    impl_code: str,
    blueprint: dict,
    frozen_interface: str,
    context_str: str,
    request_block: str,
    per_function: list[dict],
    query: str,
    max_rounds: int = 3,
) -> tuple[Optional[dict], str]:
    """
    Stage 0g: run EVERY function's approved tests against the ONE generated file, and
    repair the file (never the tests) on failure — the algorithm-pathway counterpart of
    _run_and_repair().

    Running all functions' tests together against a single implementation is the point:
    it is what actually exercises the blueprint's call graph, so an orchestration
    function that quietly fails to call the sibling it was designed around fails a test
    here instead of shipping unnoticed.
    """
    # Namespaced PER FUNCTION before flattening: each function's tests are generated by
    # an isolated LLM call blind to every other function's tests (see
    # _namespace_function_helpers's docstring), so two functions independently picking
    # the same private helper name is expected, not rare — and left alone, whichever
    # definition lands last in this concatenation silently wins for the WHOLE file.
    test_codes = []
    for entry in per_function:
        fn_codes = [t.get("code", "") for t in entry.get("tests", []) if t.get("code")]
        if fn_codes:
            test_codes.append(
                _namespace_function_helpers(entry.get("name", ""), "\n\n".join(fn_codes))
            )
    if not test_codes:
        print("[exec] no approved tests to run; skipping execution.", file=sys.stderr)
        return None, impl_code

    def _accept_repair(repaired: str, previous: str) -> str:
        """Never let a repair round hand back code that cannot even be imported.
        Nothing downstream of `_repair_algorithm_impl` re-checked its output before
        this existed — confirmed on a real run: a repair correctly fixed the failing
        test's actual cause but also emitted `from __future__ import annotations`
        mid-file, a hard SyntaxError, and that broken version went straight to the
        next execution round (and, on the last round, to the user) instead of being
        caught here. Applies the one known, deterministic, mechanical fix first, then
        falls back to keeping `previous` rather than ever making things worse.

        Also reapplies _autofix_wrong_module_imports here, every round, not just
        pre-flight: a repair regenerates the WHOLE file from scratch, so it can
        reintroduce a wrong-module import (e.g. ScheduleEncodingType from
        tsn_scheduling_problem instead of evaluator) that the pre-flight check
        already fixed once — confirmed on a real run, where this exact import
        survived every repair round and the collection error it caused was still
        there at the very end. This function is the one choke point every repair's
        output passes through before becoming the next round's `impl_code`, so
        fixing it here, rather than only pre-flight, means even the repair that
        feeds the LAST round is covered, not just the first."""
        fixed = _fix_future_import_position(repaired)
        if _syntax_error_of(fixed) is not None:
            print("[algogen] repair produced invalid Python; keeping the previous "
                  "implementation for this round.", file=sys.stderr)
            return previous
        deimported = _autofix_wrong_module_imports(fixed, LIB_DIR)
        if _syntax_error_of(deimported) is not None:
            print("[algogen] wrong-module-import autofix produced invalid Python; "
                  "keeping the pre-autofix repair for this round.", file=sys.stderr)
            deimported = fixed

        # A repair regenerates the WHOLE file from scratch every round, so fixing one
        # function's failure can silently cost another its definition entirely —
        # confirmed on a real run: asked to fix get_migration_sequence, a repair
        # rewrote its logic inline and stopped calling classifying_change, then simply
        # left classifying_change out of the file it returned. Nothing above this
        # point would catch that: the code is valid Python, no import points at the
        # wrong module. The only way to know is comparing what the file defined before
        # against what it defines now, against what the approved tests still need.
        vanished = _vanished_frozen_names(previous, deimported, test_codes)
        if not vanished:
            return deimported
        restored = _restore_vanished_frozen_names(previous, deimported, vanished)
        if restored is not None and _syntax_error_of(restored) is None:
            print(f"[algogen] repair dropped {sorted(vanished)}, still imported by "
                  f"approved tests; spliced back in from the previous version.",
                  file=sys.stderr)
            return restored
        print(f"[algogen] repair dropped {sorted(vanished)}, still imported by "
              f"approved tests, and it could not be mechanically restored; keeping "
              f"the previous implementation for this round.", file=sys.stderr)
        return previous

    # Triage reads one contract; give it the union of the per-function contracts so it
    # can reason about whichever function actually failed.
    merged_contract = {
        "behaviors": [
            b for entry in per_function
            for b in (entry.get("contract") or {}).get("behaviors", [])
        ],
        "invariants": [
            i for entry in per_function
            for i in (entry.get("contract") or {}).get("invariants", [])
        ],
        "signatures": [entry.get("contract", {}).get("signature") for entry in per_function],
    }

    result = None
    for rnd in range(max_rounds + 1):
        result = _execute_spec_tests(impl_code, test_codes, module="")
        if result["success"]:
            print(f"[exec] round {rnd}: all tests passed.", file=sys.stderr)
            return result, impl_code
        print(f"[exec] round {rnd}: tests failed (rc={result['returncode']}).", file=sys.stderr)
        if rnd == max_rounds:
            print("[exec] max repair rounds reached; returning last result.", file=sys.stderr)
            return result, impl_code

        failure_output = (result["stdout"] + "\n" + result["stderr"]).strip()

        # Checked HERE rather than pre-flight: this repair loop is the one call site
        # that already has real failing-test output and test code to work from, which
        # is what a hallucinated numpy call actually needs to be fixed correctly rather
        # than just silenced — see _hallucinated_numpy_method_calls's own docstring.
        numpy_hallucinations = _hallucinated_numpy_method_calls(impl_code)
        numpy_guidance = (
            " Also, these call(s) use a numpy method/attribute that does not exist in "
            "the real, installed numpy and will always raise AttributeError — replace "
            "each with the correct real numpy operation based on what the failing "
            "test(s) actually assert, not a similarly-named guess: "
            + "; ".join(numpy_hallucinations)
            if numpy_hallucinations else ""
        )

        # Shrink triage's own prompt to just what actually failed: with 7 functions'
        # worth of tests concatenated, sending all of them to every triage call scales
        # with the whole suite regardless of how much of it failed — confirmed to
        # overflow the local model's context window on a real run. Falls back to the
        # full set if the failure output's own summary lines don't parse (format
        # changed, or a collection-level error with no per-test summary at all), so
        # triage never gets strictly less information than before, only ever the same
        # or a well-targeted subset.
        failing_names = _failing_test_names(failure_output)
        failing_function_names = {
            entry["name"] for entry in per_function
            if any(t.get("test_name") in failing_names for t in entry.get("tests", []))
        } if failing_names else set()

        if failing_function_names:
            triage_contract = {
                "behaviors": [
                    b for entry in per_function if entry["name"] in failing_function_names
                    for b in (entry.get("contract") or {}).get("behaviors", [])
                ],
                "invariants": [
                    i for entry in per_function if entry["name"] in failing_function_names
                    for i in (entry.get("contract") or {}).get("invariants", [])
                ],
                "signatures": [
                    entry.get("contract", {}).get("signature") for entry in per_function
                    if entry["name"] in failing_function_names
                ],
            }
            # Namespaced the same way test_codes was above: triage reads this
            # alongside the real pytest failure_output, whose traceback references
            # whatever name actually executed. Showing it the pre-namespacing source
            # here would have it look at a `_make_case(...)` call while the traceback
            # below it says `_make_case__<function>(...)`, for no reason.
            triage_test_codes = [
                _namespace_function_helpers(
                    entry["name"],
                    "\n\n".join(
                        t.get("code", "") for t in entry.get("tests", [])
                        if t.get("test_name") in failing_names and t.get("code")
                    ),
                )
                for entry in per_function
                if entry["name"] in failing_function_names
            ] or test_codes
            print(f"[triage] scoping to {len(failing_function_names)} function(s) with "
                  f"failing tests: {sorted(failing_function_names)}", file=sys.stderr)
        else:
            triage_contract = merged_contract
            triage_test_codes = test_codes

        try:
            triage = _triage_failure(
                triage_contract, "\n\n".join(triage_test_codes), impl_code, failure_output
            )
        except Exception as e:  # noqa: BLE001
            print(f"[triage] parse failed ({e}); repairing the implementation anyway.",
                  file=sys.stderr)
            impl_code = _accept_repair(
                _repair_algorithm_impl(
                    impl_code, blueprint, frozen_interface, context_str, request_block,
                    failure_output,
                    guidance=numpy_guidance,
                ),
                impl_code,
            )
            continue

        target = triage.get("suggested_fix_target")
        print(f"[triage] verdict={triage.get('verdict')} -> fix {target}", file=sys.stderr)
        if target == "none":
            print(f"[triage] {triage.get('verdict')} — outside the implementation; stopping.",
                  file=sys.stderr)
            return result, impl_code

        impl_code = _accept_repair(
            _repair_algorithm_impl(
                impl_code, blueprint, frozen_interface, context_str, request_block,
                failure_output,
                guidance=(triage.get("required_impl_change") or triage.get("explanation", ""))
                + numpy_guidance,
            ),
            impl_code,
        )

    return result, impl_code


def _render_algorithm_tests(per_function: list[dict]) -> str:
    """Human-readable rendering of the per-function approved tests."""
    if not per_function:
        return "UNIT TESTS\n" + "=" * 20 + "\n\n(no tests were produced — see [WARN] logs)"
    lines = ["UNIT TESTS", "=" * 20, ""]
    for entry in per_function:
        lines.append(f"-- {entry['name']} ({len(entry.get('tests', []))} test(s)) --")
        for t in entry.get("tests", []):
            lines.append(f"\n# {t.get('behavior_id', '?')}: {t.get('test_name', '')}")
            lines.append("```python")
            lines.append(t.get("code", ""))
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _render_algorithm_execution(result: Optional[dict]) -> str:
    """Human-readable rendering of the pytest run against the generated file."""
    if result is None:
        return "EXECUTION\n" + "=" * 20 + "\n\n(not executed)"
    passed = "PASSED" if result.get("success") else "FAILED"
    return (
        "EXECUTION\n" + "=" * 20 + "\n\n"
        f"Result: {passed} (return code {result.get('returncode')})\n\n"
        f"{(result.get('stdout') or '').strip()}\n"
        f"{(result.get('stderr') or '').strip()}".rstrip()
    )


def run_algorithm_generation_pathway(
    query: str,
    run_generated_code: bool = True,
    interactive: bool = False,
) -> str:
    """
    Stage 0 orchestrator for the algorithm-generation pathway, taken when
    is_algorithm_generation_query() returns True:
      0b. build_algorithm_requirements_ledger() -> R1..Rn requirements ledger
      0b2 build_algorithm_configuration()       -> the settled environment + encoding
      0c. build_algorithm_blueprint()           -> one unified design over the ledger
      0d. build_interface_freeze()              -> shared types + signatures as real code
      0e. generate_algorithm_from_spec()        -> every frozen body, in ONE file
      0f. build_algorithm_tests()               -> per-function spec -> contract -> tests
      0g. _run_and_repair_algorithm()           -> execute all tests, repair the file

    One retrieval feeds every stage (blueprint, implementation, per-function specs) so
    they cannot disagree about what the codebase offers. Each stage degrades gracefully:
    a failed blueprint stops before the interface freeze, a failed freeze stops before
    implementation, and missing tests skip execution rather than aborting the answer.
    """
    ledger = build_algorithm_requirements_ledger(query)

    # Retrieved and reranked ONCE, rendered two ways. Both views come from the same
    # candidate sets, so the stages cannot disagree about what the codebase offers.
    #   configuration -> the merged flat pool; its work is environment setup, which is
    #                    not per-requirement.
    #   blueprint     -> the same candidates grouped BY REQUIREMENT, top
    #                    ALGORITHM_CONTEXT_PER_REQUIREMENT each, because mapping every
    #                    id to functions is exactly its job and a flat pool turns that
    #                    into a search.
    _candidates, _groups = _retrieve_algorithm_candidates(query, ledger)
    context_str = _render_helpers_for_generation(_candidates)
    blueprint_context_str = (_render_candidates_by_requirement(_groups) if _groups
                             else _render_helpers_for_generation(_candidates))

    # 0b2: settle the environment BEFORE any algorithm is designed on it. Split out
    # because the encoding decision kept being made implicitly and wrongly: six runs read
    # cell values only as presence flags, took the always-PRIORITY problem.evaluator, and
    # scored start-time data with a constraint family that cannot read it. Deciding it
    # here makes it one small artifact that is checked once and handed on as fact.
    configuration = build_algorithm_configuration(query, ledger, context_str=context_str)
    print(f"[algogen] CONFIGURATION:\n{_render_algorithm_configuration(configuration)}",
          flush=True)

    blueprint = build_algorithm_blueprint(query, ledger,
                                          context_str=blueprint_context_str,
                                          configuration=configuration)
    print(f"[algogen] BLUEPRINT:\n{_render_algorithm_blueprint(blueprint)}", flush=True)
    sections = [_render_requirements_ledger(ledger),
                _render_algorithm_configuration(configuration),
                _render_algorithm_blueprint(blueprint)]
    if not blueprint:
        return "\n\n".join(sections)

    frozen_interface = build_interface_freeze(blueprint)
    if not frozen_interface:
        return "\n\n".join(sections)
    sections.append(
        "FROZEN INTERFACE\n" + "=" * 20 + "\n\n```python\n" + frozen_interface + "\n```"
    )

    request_block = _algorithm_request_block(query, ledger, frozen_interface)

    # 0e: implement every frozen body in one file.
    impl_code = generate_algorithm_from_spec(
        blueprint, frozen_interface, _generation_context(blueprint, _candidates), request_block
    )
    _print_prompt_output("0e IMPLEMENTATION", impl_code)

    # The inverse of every check below, and the one none of them can make: those all ask
    # whether this file's library calls are CORRECT, so a file that reaches for the
    # library NOWHERE passes all three in silence. That is the worst outcome this pathway
    # can produce — the query asks for an algorithm built primarily by reusing the
    # codebase — and it is confirmed real: one benchmarked run never called the evaluator
    # anywhere and nothing in the pipeline objected. Deliberately a warning, not a repair
    # trigger: "reuse more of the codebase" is a design-level judgement a repair round
    # cannot make from a traceback, and a false alarm here must never cost a rewrite.
    if not _library_modules_used(impl_code, LIB_DIR):
        print(
            "\n" + "!" * 78 + "\n"
            "[algogen] WARNING: the generated implementation imports NOTHING from the "
            "library.\n"
            f"          No module under {os.path.basename(LIB_DIR)}/ is imported, and "
            "none is loaded\n"
            "          dynamically either — so every capability the codebase already "
            "provides\n"
            "          (the evaluator, the scheduling problem, the use-case generator, "
            "the\n"
            "          Schedule/Slot model) was re-invented or skipped rather than "
            "reused.\n"
            "          Check this run's blueprint reuse decisions before trusting the "
            "output.\n"
            + "!" * 78 + "\n",
            file=sys.stderr,
        )

    # Deterministic pre-flight (no LLM), same as the normal generation path gets: a name
    # used but never imported otherwise surfaces as a NameError inside pytest, which
    # costs a whole repair round to diagnose from a traceback instead of being fixed
    # here from a plain list of the missing names. Checked alongside a second, distinct
    # failure `_undefined_names` cannot see at all: a name imported `from` the WRONG real
    # module (e.g. `from tsn_scheduling_problem import ScheduleEncodingType`, which is a
    # real name but only lives in evaluator.py) — the name IS bound, so nothing looks
    # missing, right up until that import statement itself raises ImportError and aborts
    # collection of the whole generated module before a single test can run.
    missing = _undefined_names(impl_code, provided=LIB_PROVIDED_NAMES)
    wrong_module = _wrong_module_imports(impl_code, LIB_DIR)
    # A third, distinct failure neither check above can see: a name imported from the
    # RIGHT module but called with the wrong number of arguments for its real signature
    # (e.g. TSNScheduling(a, b, c, ...11 args) after the import itself was already fixed
    # to point at the real class). See _bad_kb_call_arity's docstring — one real run had
    # 78 of 199 failing tests trace back to exactly this pattern.
    bad_arity = _bad_kb_call_arity(impl_code, LIB_DIR)
    # A fourth failure none of the three above can see, because it is neither an import
    # nor a call: an attribute READ on a parameter the signature annotates as a library
    # class, where the real class has no such attribute. `problem.instance` on a
    # TSNScheduling is the confirmed case — it sat on the first line of the generated
    # file's main path, passed every gate, and raised AttributeError before any test ran.
    unknown_attrs = _unknown_kb_attributes(impl_code, LIB_DIR)
    # A fifth: a call between the file's OWN functions with the wrong argument count.
    # `_bad_kb_call_arity` judges only library names, so most of an algorithm file's
    # actual composition — generated functions calling each other — is checked by
    # nothing until pytest raises TypeError and spends a repair round on it.
    internal_arity = _bad_internal_call_arity(impl_code)
    # And the flow declaration. Unlike the four above this is NOT proof of a defect: a
    # function whose Flow line disagrees with its signature may simply have a stale
    # docstring, which breaks nothing at runtime. It is reported and repaired because
    # the disagreement means the function's role was unsettled while it was written, and
    # that correlates with the body being wrong — but it must never gate acceptance,
    # since rejecting a working repair over a docstring would trade a real fix for a
    # cosmetic one.
    flow_mismatch = _flow_contract_violations(impl_code, frozen_interface)
    # Calls to a library member in ATTRIBUTE form — Class.member(...) and
    # parameter.member(...). `_bad_kb_call_arity` resolves only bare-name targets, so on
    # a real run both library calls in the file were attribute-form, it inspected zero of
    # them, and a fatal `macro_cycle=` keyword reached execution.
    member_calls = _bad_kb_member_call(impl_code, LIB_DIR)
    # Instance-gated members used outside an activation block. The only check here that
    # looks for a required thing ABSENT rather than a wrong thing present.
    activation = _activation_context_violations(impl_code, LIB_DIR)
    # Argument TYPES, via mypy against main_files/kb_stubs/. Silent when mypy is not
    # installed. Reported and repaired but deliberately NOT gated yet: the stubs are
    # hand-written and unproven, and a wrong stub would reject good repairs. Promote it
    # into the gate below once a few runs show it clean on correct code.
    type_errors = _type_check_violations(impl_code)
    # Library objects whose type has been erased to object/Any. Not a runtime defect —
    # the code runs — but it disables every check that resolves a parameter through its
    # annotation, so the model would otherwise control whether it gets checked at all.
    # Reported and repaired, not gated: an annotation is not grounds for throwing away a
    # repair that fixed something real.
    erased_types = _erased_library_annotations(impl_code, LIB_DIR)
    # A parameter annotated with a real library class that the body treats as a numpy
    # array. A guaranteed failure IN THIS PIPELINE, not merely a mislabel: the tests are
    # generated from the frozen interface, so they construct what the annotation
    # declares, and the function then raises on its own declared type. Gated.
    array_misuse = _library_value_used_as_array(impl_code, LIB_DIR)
    # Chromosomes converted with the library's START_TIME converters while being scored
    # by a PRIORITY evaluator. Replaces an earlier check that asked whether a decision
    # had been RECORDED — the wrong question, since the defect is a mismatch between two
    # things the file uses, set up by the library shipping a PRIORITY default for the
    # only chromosome format it can produce.
    undecided_mode = _encoding_pairing_violations(impl_code, LIB_DIR)
    # Arguments whose KIND differs from what the library itself always passes in that
    # position — the only evidence anywhere about what a parameter is meant to hold,
    # since no signature, annotation or docstring records it. Reported, not gated: a
    # convention is strong evidence of intent, not proof of error.
    unconventional = _unconventional_library_arguments(impl_code, LIB_DIR)
    # A parameter selecting a library MODE that carries a default. Gated: a defaulted
    # mode makes the branch guarded by the other value unreachable, so the design's own
    # correct code never runs — and unlike a convention mismatch there is nothing to
    # weigh, the parameter either forces the choice or it does not.
    defaulted_modes = _defaulted_mode_parameters(impl_code, LIB_DIR)
    # Explicit calls to something entering the activation block already does. Reported,
    # not gated: the file still runs, it just repeats work the library caches. The
    # configuration gate is where this is actually prevented — this is the backstop for
    # a call that appears during implementation without ever being a reuse decision.
    lifecycle_dupes = _lifecycle_duplicated_calls(impl_code, LIB_DIR)
    # The design disagreeing with its own code: a literal outside the enum the blueprint
    # declared, or a declared formula never computed. GATED, unlike every other check
    # here, for a reason specific to this pipeline — the tests are generated from the SAME
    # blueprint, independently, so a divergence here is not a risk of failure but a
    # guarantee of it, and the fix is mechanical because the right value is written down.
    bad_literals = _undeclared_constant_literals(impl_code, blueprint)
    missing_formulas = _uncomputed_constant_formulas(impl_code, blueprint)
    if missing or wrong_module or bad_arity or unknown_attrs or internal_arity \
            or flow_mismatch or member_calls or activation or type_errors \
            or erased_types or array_misuse or undecided_mode or unconventional \
            or defaulted_modes or lifecycle_dupes or bad_literals or missing_formulas:
        problems = []
        if missing:
            problems.append(f"uses these names without importing or defining them: "
                             f"{sorted(missing)}")
        if wrong_module:
            problems.append(
                "imports these names from the wrong module (the name is real, but not "
                "part of that module's public interface): " +
                ", ".join(f"'{name}' from '{mod}'" for name, mod in wrong_module)
            )
        if bad_arity:
            problems.append(
                "calls a real, correctly-imported library class/function with the wrong "
                "number of arguments for its actual signature: " + "; ".join(bad_arity)
            )
        if unknown_attrs:
            problems.append(
                "reads attributes that do not exist on the real library class a "
                "parameter is annotated with: " + "; ".join(unknown_attrs)
            )
        if internal_arity:
            problems.append(
                "calls one of its own functions with the wrong number of arguments: "
                + "; ".join(internal_arity)
            )
        if flow_mismatch:
            problems.append(
                "has functions whose declared Flow line contradicts their own "
                "signature: " + "; ".join(flow_mismatch)
            )
        if member_calls:
            problems.append(
                "calls a library class member with arguments that cannot bind to its "
                "real signature: " + "; ".join(member_calls)
            )
        if activation:
            problems.append(
                "uses members that require an active use case without ever entering "
                "`with problem.use_instance(usecase):`: " + "; ".join(activation)
            )
        if type_errors:
            problems.append(
                "passes arguments whose types do not match what the callee accepts: "
                + "; ".join(type_errors)
            )
        if erased_types:
            problems.append(
                "hides library objects behind bare object/Any annotations, which stops "
                "anything from checking the calls made through them: "
                + "; ".join(erased_types)
            )
        if array_misuse:
            problems.append(
                "declares parameters as library classes but operates on them as numpy "
                "arrays, so they fail on the type they declare: "
                + "; ".join(array_misuse)
            )
        if undecided_mode:
            problems.append(
                "pairs the library's START_TIME chromosome converters with a PRIORITY "
                "evaluator, which cannot read them: " + "; ".join(undecided_mode)
            )
        if unconventional:
            problems.append(
                "passes a kind of value the library never passes in that argument "
                "position: " + "; ".join(unconventional)
            )
        if defaulted_modes:
            problems.append(
                "gives a library-mode parameter a default, so the branch guarded by "
                "the other mode can never run: " + "; ".join(defaulted_modes)
            )
        if lifecycle_dupes:
            problems.append(
                "calls a library member that entering the activation block already "
                "performs, repeating work the library caches: "
                + "; ".join(lifecycle_dupes)
            )
        if bad_literals:
            problems.append(
                "uses string values the design's own CONSTANTS section does not declare, "
                "which the separately-generated tests will therefore never produce: "
                + "; ".join(bad_literals)
            )
        if missing_formulas:
            problems.append(
                "leaves a value the design defined by formula uncomputed: "
                + "; ".join(missing_formulas)
            )
        print(f"[algogen] implementation {'; and '.join(problems)}; repairing once "
              f"before running tests.", file=sys.stderr)
        arity_guidance = (
            " Fix any call listed above as having the wrong argument count/shape so it "
            "matches the real signature shown for it — check the library source under "
            "original_tsn_lib_files/ for that class/function's actual __init__ or "
            "definition rather than guessing what it takes."
            if bad_arity else ""
        )
        attr_guidance = (
            " For any attribute listed above as not existing on its class, do NOT invent "
            "a different attribute name: read that class in original_tsn_lib_files/ and "
            "use what it really exposes, or take the value as a parameter instead. An "
            "attribute reached for because the signature has no other way to supply the "
            "value — a use case, for example — is a signature problem, not a naming one: "
            "say so in a one-line comment rather than guessing an attribute."
            if unknown_attrs else ""
        )
        # Every problem above names something WRONG. Without this, the cheapest way to
        # satisfy each one is to delete it, and on a real run that is exactly what
        # happened three times in a single repair: a keyword the signature did not
        # accept was removed instead of renamed, the parameter carrying its value was
        # dropped from three signatures, and a use needing an active use case was
        # wrapped in a block that activated a different, throwaway object. All three
        # silenced the check and left the program broken.
        no_deletion_guidance = (
            " HOW TO FIX THESE. Removing something is only a fix when the thing was "
            "genuinely unnecessary. Deleting an argument, dropping a parameter that "
            "call sites are passing, or wrapping a statement in the smallest block that "
            "quiets the message are NOT fixes — they silence the report while leaving "
            "the defect, and usually make behaviour worse by discarding a value that was "
            "being passed for a reason. For each problem, work out what the code was "
            "TRYING to do and make it do that: a keyword the signature does not accept "
            "usually belongs in the parameter that does accept it; a value that no "
            "parameter accepts means the call site or the signature is wrong, not the "
            "value; and an object that needs an active use case must be THE SAME object "
            "the enclosing `with ....use_instance(...)` block activated, not a second "
            "one built alongside it."
            + (
                " The ONE exception is the repeated-lifecycle-call problem listed above: "
                "there the library genuinely performs the call for you when the block is "
                "entered, nothing else supplies that value, and deleting the explicit "
                "call is the whole fix. Delete only the calls named there."
                if lifecycle_dupes else ""
            )
        )
        flow_guidance = (
            " Where a Flow line and a signature disagree, decide which one is right "
            "from what the function must actually do, and fix that one. Do not delete a "
            "parameter merely to make the signature match the Flow line — if callers "
            "pass that value, the Flow line is what is wrong."
            if flow_mismatch else ""
        )
        repaired = _repair_algorithm_impl(
            impl_code, blueprint, frozen_interface, context_str, request_block,
            failure_output="The implementation " + "; and ".join(problems) + ".",
            guidance=(
                "Add the missing imports at the top of the file (e.g. `import numpy as "
                "np`, `from evaluator import ScheduleEncodingType`), or define the name, "
                "and move any misplaced import to the real module that actually defines "
                "that name — check the library source under original_tsn_lib_files/ rather "
                "than guessing which module a real name lives in. "
                "Import only what is actually used. Change nothing else."
                + arity_guidance + attr_guidance + flow_guidance
                + no_deletion_guidance
            ),
        )
        # _undefined_names/_wrong_module_imports both return "no problem found" on a
        # SyntaxError (each docstring says "a syntax gate handles this case" — but no
        # other syntax gate runs on `repaired` anywhere in this branch). Confirmed on a
        # real run: a repair correctly added the missing import while also emitting
        # `from __future__ import annotations` mid-file, and since neither check below
        # can even parse that far, both silently reported zero problems and the broken
        # code became the implementation every subsequent round executed against —
        # collection failed before a single test ran, on round 0, not because of any
        # later repair. Apply the same fix-or-reject guardrail the repair loop itself
        # uses before either check ever sees `repaired`.
        repaired = _fix_future_import_position(repaired)
        if _syntax_error_of(repaired) is not None:
            print("[algogen] import-fix repair produced invalid Python; keeping the "
                  "original implementation.", file=sys.stderr)
            repaired = impl_code
        # The retry above doesn't reliably fix a wrong-module import — it has reproduced
        # the identical mistake, or moved it to a different equally-wrong module. That
        # class of problem doesn't need an LLM at all: _wrong_module_imports already knows
        # exactly which name is misplaced, and the correct module is just whichever real
        # library file actually defines it, so fix it directly rather than hoping a second
        # LLM call complies where the first one didn't.
        candidate = repaired if not _undefined_names(repaired, provided=LIB_PROVIDED_NAMES) \
            else impl_code
        candidate = _autofix_wrong_module_imports(candidate, LIB_DIR)
        # No deterministic autofix exists for a bad-arity call the way there is for a
        # wrong-module import (there is no single correct rewrite — which of 11 stray
        # positional values, if any, was meant to be the real argument is a semantic
        # question, not a mechanical one), and the same is true of a nonexistent
        # attribute: which real member was meant, or whether the value should have been a
        # parameter all along, is a judgement. So this only re-checks the repair's work.
        # `flow_mismatch` and `type_errors` are deliberately absent from this gate — see
        # where each is computed. A docstring contradicting its signature is a signal,
        # not a runtime defect; and the type check rests on hand-written stubs that have
        # not yet been proven against enough correct code to be worth rejecting a repair
        # over. Both are still reported and still repaired.
        if (_syntax_error_of(candidate) is None
                and not _undefined_names(candidate, provided=LIB_PROVIDED_NAMES)
                and not _wrong_module_imports(candidate, LIB_DIR)
                and not _bad_kb_call_arity(candidate, LIB_DIR)
                and not _unknown_kb_attributes(candidate, LIB_DIR)
                and not _bad_internal_call_arity(candidate)
                and not _bad_kb_member_call(candidate, LIB_DIR)
                and not _activation_context_violations(candidate, LIB_DIR)
                and not _library_value_used_as_array(candidate, LIB_DIR)
                and not _defaulted_mode_parameters(candidate, LIB_DIR)
                # Gated here too: an undeclared literal or an uncomputed formula is a
                # guaranteed test failure, and the repair has the declared value to
                # hand, so accepting a repair that still diverges helps nothing.
                and not _undeclared_constant_literals(candidate, blueprint)
                and not _uncomputed_constant_formulas(candidate, blueprint)):
            impl_code = candidate
        else:
            print("[algogen] repair did not resolve the import/arity/attribute "
                  "problem(s); keeping the original implementation.", file=sys.stderr)

    _print_prompt_output("0e IMPLEMENTATION", impl_code, after_checks=True)

    # 0f: tests per frozen function (spec -> contract -> tests -> optional approval).
    per_function = build_algorithm_tests(
        query, ledger, blueprint, frozen_interface, context_str, interactive
    )

    # 0g: one pytest run over the whole file, repairing the file (never the tests).
    execution_result = None
    if run_generated_code:
        execution_result, impl_code = _run_and_repair_algorithm(
            impl_code, blueprint, frozen_interface, context_str, request_block,
            per_function, query,
        )

    sections.append(
        "GENERATED CODE\n" + "=" * 20 + "\n\n```python\n" + impl_code + "\n```"
    )
    sections.append(_render_algorithm_tests(per_function))
    if run_generated_code:
        sections.append(_render_algorithm_execution(execution_result))

    return "\n\n".join(sections)

# UNUSED: manual debug helper, not called anywhere in the deterministic pipeline. Left
# commented rather than deleted since it's handy to re-enable while debugging classify_query().
# def debug_classification(query: str) -> None:
#     classification = get_or_classify_query(query)
#
#     print("\n--- QUERY CLASSIFICATION DEBUG ---")
#     print(f"Original:   {classification.original_query}")
#     print(f"Normalized: {classification.normalized_query}")
#     print(f"Type:       {classification.query_type.value}")
#     print(f"Confidence: {classification.classification_confidence.value}")
#     print(f"Reason:     {classification.reason}")
#
#     print("\nSub-queries:")
#     for i, sq in enumerate(classification.sub_queries, 1):
#         print(f"  {i}. Module: {sq.module}")
#         print(f"     Intent:  {sq.intent}")
#         print(f"     Text:    {sq.text}")
#
#     if classification.warnings:
#         print("\nWarnings:")
#         for warning in classification.warnings:
#             print(f"  - {warning}")
#
#     print("----------------------------------\n")

# UNUSED: leftover from an earlier agent-tool design (its own docstring references calling
# it "before lookup_and_understand", a tool name that no longer exists in this file). In the
# current pipeline, per-sub-query parameters come directly from classify_query()'s single
# classification LLM call (CLASSIFICATION_PROMPT already asks for a "parameters" object per
# sub-query — see SubQuery.parameters / sq.parameters in answer_query_deterministic()), so a
# second, separate extraction pass is no longer needed.
# def extract_parameters(query: str) -> str:
#     """
#     Classifies the query (SIMPLE or COMPLEX), then for each sub-query extracts
#     any parameter names and values explicitly or implicitly mentioned by the user,
#     matched against the known parameters from parameters.md.
#
#     Returns a JSON string per sub-query. For SIMPLE queries a single JSON object
#     is returned. For COMPLEX queries one JSON object per sub-query is returned,
#     labelled by module.
#
#     This tool should be called FIRST — before lookup_and_understand — so that
#     the extracted parameter context can sharpen the subsequent function lookup.
#
#     Example:
#         Input : "Test TSN with 8 switches and 8 hosts"
#         Output: {"num_switches": 8, "num_hosts": 8}
#     """
#     classification = get_or_classify_query(query)
#
#     param_defs = _get_param_definitions()
#
#     def _extract_one(text: str, module: str) -> dict:
#         augmented_text = augment_query(text, module)
#         prompt = PARAM_EXTRACTION_PROMPT.format(
#             param_definitions=param_defs,
#             query=augmented_text,
#         )
#         raw = Settings.llm.complete(prompt).text.strip()
#
#         try:
#             parsed = extract_json_object(raw)
#             return parsed if isinstance(parsed, dict) else {}
#         except json.JSONDecodeError as e:
#             print(f"[WARN] Parameter extraction parse failed ({e}). Returning empty.")
#             return {}
#
#     if classification.query_type == QueryType.SIMPLE:
#         sq     = classification.sub_queries[0]
#         result = _extract_one(sq.text, sq.module)
#         print(f"[Params] SIMPLE [{sq.module}] → {result}")
#         return json.dumps(result, indent=2)
#
#     # COMPLEX: extract per sub-query and combine under module keys
#     combined = {}
#     for sq in classification.sub_queries:
#         result = _extract_one(sq.text, sq.module)
#         print(f"[Params] COMPLEX [{sq.module}] → {result}")
#         if result:
#             combined[sq.module] = result
#
#     return json.dumps(combined, indent=2)


# UNUSED as of the spec-first generation path: GENERATOR_PROMPT no longer returns a JSON
# array of function objects — it returns bare Python source, and every non-code field comes
# from the approved specification (parsed with extract_json_object in
# generate_function_spec()). Kept because it is the generic bare-array parser for this
# file's LOOKUP_PROMPT-shaped schema; nothing calls it today.
def _parse_lookup_json(raw: str) -> list[dict]:
    """
    Parses a bare JSON array of function objects from an LLM response.
    Returns an empty list on any parse failure.
    """
    try:
        raw = raw.replace("\\'", "'")
        result = extract_json_array(raw)
        return [item for item in result if isinstance(item, dict)]
    except json.JSONDecodeError as e:
        print(f"[WARN] Lookup JSON parse failed ({e}). Raw response:\n{raw[:2000]}")
        return []

# UNUSED: would parse the {"functions": [...], "coverage": {...}} shape LOOKUP_PROMPT
# defines, but nothing in the active pipeline calls the LLM with LOOKUP_PROMPT anymore
# (see the NOTE on LOOKUP_PROMPT/lookup_engines above) — _lookup_one() builds that same
# (functions, coverage) shape itself from pure retrieval, with no JSON to parse.
# def _parse_lookup_payload(raw: str) -> tuple[list[dict], Optional[dict]]:
#     raw = raw.replace("\\'", "'")
#     try:
#         obj = extract_json_object(raw)
#         if isinstance(obj, dict) and "functions" in obj:
#             funcs = [f for f in obj.get("functions", []) if isinstance(f, dict)]
#             cov = obj.get("coverage") if isinstance(obj.get("coverage"), dict) else None
#             return funcs, cov
#     except json.JSONDecodeError:
#         pass
#     return _parse_lookup_json(raw), None   # legacy bare-array fallback

# Pulls a function's exact source code (verbatim, via ast — not regenerated/paraphrased)
# out of its owning module's .py file by name. Used both to hydrate catalog entries
# (_catalog_entry_to_function) and to backfill code onto anything the lookup/generation
# path names but doesn't already carry the body for.
def fetch_code_from_source(function_name: str, module: str) -> str:
    source_file = PYTHON_SOURCE_FILES.get(module)
    if not source_file or not os.path.exists(source_file):
        print(f"[WARN] Source file not found for module: {module}")
        return ""

    with open(source_file, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"[WARN] Could not parse source file: {source_file}")
        return ""

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                lines = source.splitlines()
                start = node.decorator_list[0].lineno - 1 if node.decorator_list else node.lineno - 1
                end = node.end_lineno
                return "\n".join(lines[start:end])

    print(f"[WARN] Function '{function_name}' not found in {source_file}")
    return ""


def _render_candidates_for_coverage(functions: List[Dict]) -> str:
    """
    Behavioural evidence ONLY for the coverage LLM — name, signature, docstring,
    nl_summary, code. Deliberately omits match_rank and reranker_score: match_rank is a
    derived relevance label and the reranker score is saturated (~0.95+ for anything
    on-topic), so neither is trustworthy evidence. The assessor judges purely on what
    each function does.
    """
    rendered = []
    for fn in functions:
        rendered.append({
            "name": fn.get("name"),
            "signature": fn.get("signature"),
            "docstring": fn.get("docstring"),
            "nl_summary": fn.get("nl_summary") or "",
            "code": (fn.get("code") or "")[:4000],   # code for every candidate, not just a presumed "best" #originally was [:1500]
        })
    return json.dumps(rendered, indent=2)


def assess_function_coverage(
    *,
    query: str,
    sq: SubQuery,
    functions: List[Dict],
    top_score: float,
    second_score: float,
    module_label: Optional[str] = None,
) -> CoverageAssessment:
    """
    Decide DO-exact vs DO-gen-with-helpers vs DO-gen-standalone.

    This is the ONLY relevance decision in the pipeline. It is made entirely by an
    LLM reading each candidate's actual code — no score threshold participates. The
    score+margin gate that used to sit here has been removed; top_score/second_score
    survive purely so the [gate] log line below can record calibration data.

    `module_label` overrides what the prompt is told was searched. The default
    (sq.module) is right for the routed-module pass; the cross-module retry passes
    the real span of modules its candidates came from, so the assessor is not told
    the candidates all belong to a module they may not be in.
    """
    real = [f for f in functions if f.get("code")]
    margin = top_score - second_score

    # One log line per sub-query — this is your calibration data. Harvest it.
    print(f"[gate] module={sq.module} top={top_score:.3f} "
          f"2nd={second_score:.3f} margin={margin:.3f} "
          f"candidates={len(functions)} with_code={len(real)}")

    if sq.module == "parameters" or not real:
        return CoverageAssessment(
            match_type=MatchType.NO_MATCH,
            reason="No callable with source code retrieved; routing to generation.",
            missing_requirements=["No existing callable candidate was found."],
            recommended_action=RecommendedAction.GENERATE_STANDALONE,
        )

    # ── LLM coverage assessment (evidence-only: name / signature / docstring / code) ──
    prompt = COVERAGE_ASSESSMENT_PROMPT.format(
        original_query=query,
        sub_query=sq.text,
        module=module_label or sq.module,
        functions_json=_render_candidates_for_coverage(real),
    )
    raw = ""
    try:
        raw = Settings.llm.complete(prompt).text.strip()
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        match_type = MatchType(data["match_type"])
    except Exception as e:  # noqa: BLE001 — fail toward generation, never a false exact-match
        # This handler is a silent-failure trap: EVERY exception below turns into
        # PARTIAL_MATCH -> GENERATE_WITH_HELPERS, i.e. "the codebase can't do this, write
        # it from scratch", which is indistinguishable in the output from the assessor
        # having genuinely judged coverage. Concretely it can swallow:
        #   * LLM backend failures from Settings.llm.complete() — timeout, dropped
        #     connection, CUDA OOM, context-length overflow (large candidate blocks make
        #     this reachable), or a None/absent .text attribute -> AttributeError.
        #   * A response containing no "{" at all (the model answered in prose, or refused,
        #     or emitted markdown fences only). find/rfind then return -1, the slice is
        #     empty, and json.loads("") raises JSONDecodeError.
        #   * Malformed or truncated JSON -> json.JSONDecodeError.
        #   * A valid JSON object missing the "match_type" key -> KeyError.
        #   * A match_type string that is not a MatchType member — "EXACT", lowercase
        #     "exact_match", "FULL_MATCH" -> ValueError from the enum coercion.
        # Log the type, the message, and the head of the raw response so these are
        # distinguishable in the transcript instead of all looking like a coverage verdict.
        preview = (raw[:200] + "…") if len(raw) > 200 else raw
        print(
            f"[gate] *** COVERAGE ASSESSMENT FAILED — this is an ERROR, not a verdict.\n"
            f"       module={sq.module} candidates={len(real)}\n"
            f"       {type(e).__name__}: {e}\n"
            f"       raw response ({len(raw)} chars): {preview!r}\n"
            f"       -> defaulting to PARTIAL_MATCH / GENERATE_WITH_HELPERS."
        )
        return CoverageAssessment(
            match_type=MatchType.PARTIAL_MATCH,
            reason=(
                f"Coverage assessment FAILED ({type(e).__name__}: {e}); "
                f"defaulted to generating with retrieved helpers. This is not a judgement "
                f"that the codebase lacks the function."
            ),
            missing_requirements=["Coverage uncertain; generate to be safe."],
            recommended_action=RecommendedAction.GENERATE_WITH_HELPERS,
        )

    # Derive the action from match_type so an inconsistent LLM pairing can't leak through.
    action = {
        MatchType.EXACT_MATCH:   RecommendedAction.RETURN_EXISTING_FUNCTIONS,
        MatchType.PARTIAL_MATCH: RecommendedAction.GENERATE_WITH_HELPERS,
        MatchType.NO_MATCH:      RecommendedAction.GENERATE_STANDALONE,
    }[match_type]
    print(f"[gate] LLM -> match={match_type.value} action={action.value}")
    return CoverageAssessment(
        match_type=match_type,
        reason=f"LLM coverage assessor: {match_type.value}.",
        missing_requirements=[],
        recommended_action=action,
    )


# UNUSED: CoverageAssessment objects are consumed directly (as dataclasses) throughout
# answer_query_deterministic() — nothing ever needs the plain-dict serialization this
# would produce.
# def coverage_to_dict(coverage: CoverageAssessment) -> Dict:
#     return {
#         "match_type": coverage.match_type.value,
#         "reason": coverage.reason,
#         "missing_requirements": coverage.missing_requirements,
#         "recommended_action": coverage.recommended_action.value,
#     }

# UNUSED: operated on a dict-shaped "lookup_data" payload (query_type/sub_queries/coverage
# keys) from an earlier, non-dataclass version of the pipeline. The current pipeline checks
# `coverage.recommended_action in {GENERATE_WITH_HELPERS, GENERATE_STANDALONE}` inline, per
# sub-query, directly on the CoverageAssessment dataclass (see the `needs_generation =` line
# in answer_query_deterministic()'s Pass 3).
# def lookup_result_needs_generation(lookup_data: Dict) -> bool:
#     """
#     Returns True if any part of lookup result requires generation.
#     """
#     if lookup_data.get("query_type") == "SIMPLE":
#         coverage = lookup_data.get("coverage", {})
#         return coverage.get("recommended_action") in {
#             "GENERATE_WITH_HELPERS",
#             "GENERATE_STANDALONE",
#         }
#
#     for sq in lookup_data.get("sub_queries", []):
#         coverage = sq.get("coverage", {})
#         if coverage.get("recommended_action") in {
#             "GENERATE_WITH_HELPERS",
#             "GENERATE_STANDALONE",
#         }:
#             return True
#
#     return False

# Stage 2 ("check the codebase"): pure retrieval + rerank for one sub-query, no LLM call.
def _lookup_one(sq: SubQuery) -> tuple[list[dict], float, float]:
    """
    Pure-retrieval candidate selection (NO LLM call).

    Retrieves + reranks candidates straight from the index. The top reranked node
    is, by definition, the best match — node metadata carries `function_name`, and
    FUNCTION_CATALOG_BY_MODULE supplies the remaining fields. Returns
    (functions, top_score, second_score); the scores feed the coverage gate.
    """
    augmented_text = augment_query(sq.text, sq.module)

    # parameters module keeps its LLM-free pseudo-result (unchanged behaviour).
    if sq.module == "parameters":
        nodes = params_retriever.retrieve(augmented_text)
        score = max((n.score or 0.0) for n in nodes) if nodes else 0.0
        purpose = "\n\n".join(n.node.get_content() for n in nodes).strip()
        return [
            {
                "name": "name of the parameter",
                "class_name": None,
                "purpose": purpose,
                "library calls used": [],
                "preconditions": [],
                "postconditions": [],
                "signature": None,
                "docstring": None,
                "parameters": [],
                "modules": "parameters.md",
                "module": "parameters",
                "code": None,
                "match_rank": "BEST_MATCH",
                "reranker_score": round(score, 3),   # diagnostics only
                "routed_module": "parameters",
                "catalog_validation": {
                    "valid": True,
                    "reason": "Parameter pseudo-result bypasses function catalog validation.",
                    "matched_entry": None,
                    "alternatives": [],
                },
            }
        ], score, 0.0

    if sq.module not in module_retrievers:
        print(f"[WARN] No retriever for module: {sq.module}")
        return [], 0.0, 0.0

    nodes = module_retrievers[sq.module].retrieve(augmented_text)
    # RAW sub-query for the reranker, augmented text for the retriever only. The
    # "[File: ...]" prefix helps BM25/embedding recall, but every node in a module
    # already carries that filename in its text, so the cross-encoder matches the
    # prefix against ALL candidates equally: scores collapse to ~0.94 for everything
    # and the real signal is drowned. Measured on "three-tier switch hierarchy that
    # scales out from a single branching factor": k_ary_fat_tree ranks 1/47 on the
    # raw query, 22/47 (outside top_n=15) on the augmented one.
    # Unreachable under UNIFIED_RETRIEVAL (parameters returns earlier) — readd for generation
    nodes = reranker_algo.postprocess_nodes(
        nodes, query_bundle=QueryBundle(sq.text)
    )
    if not nodes:
        return [], 0.0, 0.0

    # Diagnostics only — these are logged by assess_function_coverage()'s [gate]
    # line as calibration data. No branch anywhere reads them.
    top_score = nodes[0].score or 0.0
    second_score = (nodes[1].score or 0.0) if len(nodes) > 1 else 0.0

    catalog = FUNCTION_CATALOG_BY_MODULE.get(sq.module, {})
    classed = FUNCTION_CATALOG_BY_MODULE_CLASSED.get(sq.module, {})
    functions: list[dict] = []
    for rank, n in enumerate(nodes):
        fn_name = n.node.metadata.get("function_name")
        if not fn_name:
            continue
        # Class-aware first: a bare-name lookup returns whichever variant of a reused
        # name (__init__, get_config, links, ports, _frame_size, drawFattree) happened
        # to load last, which can be the wrong class's entry entirely.
        node_class = n.node.metadata.get("class_name")
        entry = classed.get((node_class, fn_name)) or catalog.get(fn_name)
        if entry:
            fn = _catalog_entry_to_function(entry, sq.module)
        else:
            fn = {
                "name": fn_name,
                "signature": n.node.metadata.get("signature"),
                "class_name": n.node.metadata.get("class_name"),
                "docstring": None,
                "nl_summary": "",
                "description": "",
                "parameters": [],
                "postconditions": [],
                "modules": sq.module,
                "module": sq.module,
                "code": fetch_code_from_source(fn_name, sq.module),
                "routed_module": sq.module,
            }
        fn["match_rank"] = "BEST_MATCH" if rank == 0 else "ALTERNATE"
        fn["reranker_score"] = round(n.score or 0.0, 3)
        functions.append(fn)

    # ── Catalog validation ────────────────────────────────────────────────────
    # Was: lookup_validation.validate_function_list(). That rejected every candidate
    # with a non-null class_name (104/108 functions here), starving the coverage gate
    # and forcing generation. Now uses the in-process catalog check, which can only
    # discard a candidate that has neither a catalog entry nor resolvable source.
    valid_functions, invalid_functions = _validate_against_catalog(
        functions=functions,
        routed_module=sq.module,
    )
    if invalid_functions:
        print(
            f"[Catalog validation] Rejected {len(invalid_functions)} function(s) "
            f"for module '{sq.module}':"
        )
        for fn in invalid_functions:
            validation = fn.get("catalog_validation", {})
            print(
                f"  - {fn.get('name')} | class={fn.get('class_name')} | "
                f"reason={validation.get('reason')}"
            )

    # ── FAIL-SAFE ────────────────────────────────────────────────────────────
    # Validation is a sanity check on retrieval, not a second retrieval stage. If
    # it rejects EVERY candidate while retrieval clearly succeeded, that is a bug
    # in the validator (or a catalog/label mismatch), not evidence that the
    # codebase lacks the function. Swallowing it here silently starves
    # assess_function_coverage() — `not real` short-circuits to NO_MATCH /
    # GENERATE_STANDALONE without ever calling the LLM — so the pipeline
    # generates from scratch while the correct function sits in the catalog.
    # Fall back to the unvalidated candidates and make the anomaly loud instead.
    if functions and not valid_functions:
        print(
            f"[Catalog validation] *** ALL {len(functions)} candidate(s) rejected for "
            f"module '{sq.module}' — treating this as a VALIDATOR fault, not a miss. "
            f"Falling back to unvalidated candidates: "
            f"{[f.get('name') for f in functions]}"
        )
        valid_functions = functions

    # ── Debug: exactly what the coverage gate is about to see ────────────────
    # The single most useful line in the log. If the gold function is listed here
    # and the gate still says NO_MATCH, the problem is the assessor prompt. If it
    # is NOT listed, the problem is retrieval, routing, or validation — upstream.
    print(f"[candidates] module={sq.module} handing {len(valid_functions)} to the gate:")
    for rank, fn in enumerate(valid_functions):
        code = fn.get("code") or ""
        print(
            f"    {rank}. {fn.get('name')} | class={fn.get('class_name')} "
            f"| score={fn.get('reranker_score')} | code={len(code)} chars "
            f"| docstring={'yes' if fn.get('docstring') else 'NO'}"
        )
    if functions and not valid_functions:
        print("    (none — the gate will short-circuit to NO_MATCH/GENERATE_STANDALONE)")

    return valid_functions, top_score, second_score


# ── Stage 2b: cross-module second chance ─────────────────────────────────────
# _lookup_one() can only ever return functions from sq.module, so a classifier
# mis-route is fatal on the DO / EXPLAIN_CODE paths: the right function is
# unreachable and the pipeline generates a re-implementation of code it already
# has. EXPLAIN_CONCEPT never had this problem because _collect_doc_context()
# always searched the whole codebase. This gives the other two paths the same
# second chance, BEFORE any generation prompt runs.
#
# Set False to restore the previous routed-module-only behaviour exactly.
CROSS_MODULE_FALLBACK = True

# ── Unified whole-codebase retrieval (replaces routed-module + retry) ────────
# When True, a sub-query retrieves over ALL modules on the FIRST pass instead of
# only sq.module, so the coverage assessor sees the whole candidate pool up front.
#
# Why this removes an LLM call: the old shape was
#     _lookup_one(sq.module) -> assess_function_coverage()      [LLM call 1]
#       -> if < EXACT: _lookup_cross_module() -> assess_function_coverage()  [LLM call 2]
# The retry existed only because the first retrieval could not see outside the
# routed module. Retrieving across all modules up front makes the second
# assessment redundant, so _resolve_with_fallback() is skipped entirely and every
# sub-query costs exactly ONE coverage call regardless of how it was routed.
#
# The classifier's module choice is still used — for the answer's MODULE label and
# for the generation path's per-module retrievers — it just no longer constrains
# which functions the assessor is allowed to consider.
#
# Set False to restore the routed-module + cross-module-retry behaviour exactly.
UNIFIED_RETRIEVAL = True

# Which verdict beats which, for deciding whether the retry improved on the
# routed-module result. Only a STRICT improvement is adopted.
_MATCH_STRENGTH: Dict[MatchType, int] = {
    MatchType.NO_MATCH: 0,
    MatchType.PARTIAL_MATCH: 1,
    MatchType.EXACT_MATCH: 2,
}


def _lookup_cross_module(
    sq: SubQuery, *, unified: bool = False, algo: bool = False
) -> tuple[list[dict], float, float]:
    """Candidate selection over the WHOLE codebase (NO LLM call).

    `unified=True` means this is the PRIMARY retrieval for the sub-query (the
    UNIFIED_RETRIEVAL path), not a retry after a routed-module miss. The retrieval
    itself is identical; only the diagnostic labelling changes, so log lines and the
    per-candidate `found_via` tag do not claim a fallback happened when it did not.

    Same return contract as _lookup_one(), but retrieves with combined_retriever /
    combined_reranker — the cross-module pair already used by _collect_doc_context()
    — so a sub-query routed to the wrong module can still reach the right function.

    Two deliberate differences from _lookup_one():
      * the raw sq.text drives BOTH retrieval and reranking. augment_query() would
        anchor to sq.module, which is precisely the assumption being retried.
      * each candidate is hydrated and validated against ITS OWN module, read from
        node metadata, not sq.module — otherwise a hit in another module resolves
        to no catalog entry and looks like an unknown symbol.
    """
    # `algo=True` swaps in the retriever whose pool includes ALGO_MODULE_LABELS. The
    # reranker is unchanged and stateless, so only the candidate pool differs.
    _tag = ("algo-unified" if unified else "algo-cross-module") if algo else (
        "unified" if unified else "cross-module")
    retriever = algo_combined_retriever if algo else combined_retriever
    nodes = retriever.retrieve(sq.text)
    nodes = combined_reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(sq.text)
    )
    if not nodes:
        return [], 0.0, 0.0

    top_score = nodes[0].score or 0.0
    second_score = (nodes[1].score or 0.0) if len(nodes) > 1 else 0.0

    functions: list[dict] = []
    for rank, n in enumerate(nodes):
        fn_name = n.node.metadata.get("function_name")
        if not fn_name:
            continue
        node_module = n.node.metadata.get("module") or sq.module
        node_class = n.node.metadata.get("class_name")
        entry = (
            FUNCTION_CATALOG_BY_MODULE_CLASSED.get(node_module, {}).get((node_class, fn_name))
            or FUNCTION_CATALOG_BY_MODULE.get(node_module, {}).get(fn_name)
        )
        if entry:
            fn = _catalog_entry_to_function(entry, node_module)
        else:
            fn = {
                "name": fn_name,
                "signature": n.node.metadata.get("signature"),
                "class_name": node_class,
                "docstring": None,
                "nl_summary": "",
                "description": "",
                "parameters": [],
                "postconditions": [],
                "modules": node_module,
                "module": node_module,
                "code": fetch_code_from_source(fn_name, node_module),
                "routed_module": node_module,
            }
        fn["match_rank"] = "BEST_MATCH" if rank == 0 else "ALTERNATE"
        fn["reranker_score"] = round(n.score or 0.0, 3)
        fn["found_via"] = "unified_all_module" if unified else "cross_module_fallback"
        functions.append(fn)

    # Validate per source module, so routed_module is always the module the
    # candidate actually came from (validating a usecase_generator hit against the
    # evaluator catalog would reject it for the wrong reason).
    #
    # Group on "routed_module", NOT "module": _catalog_entry_to_function() copies the
    # catalog's own "module" field, which is a source FILENAME ("usecase_generator.py"),
    # while FUNCTION_CATALOG_BY_MODULE is keyed by module LABEL ("usecase_generator").
    # Grouping on the filename silently looks up an empty catalog, so every candidate
    # would fall through to the weaker "resolved from source" branch.
    rank_of = {id(f): i for i, f in enumerate(functions)}
    valid_functions: list[dict] = []
    for mod in sorted({f.get("routed_module") for f in functions}):
        group = [f for f in functions if f.get("routed_module") == mod]
        ok, bad = _validate_against_catalog(functions=group, routed_module=mod)
        valid_functions.extend(ok)
        for fn in bad:
            print(f"[{_tag}] rejected {fn.get('name')} ({mod}): "
                  f"{fn.get('catalog_validation', {}).get('reason')}")
    # Grouping scrambled reranker order; restore it.
    valid_functions.sort(key=lambda f: rank_of.get(id(f), 0))

    # Same fail-safe as _lookup_one(): a validator that rejects everything is a bug,
    # not evidence that the codebase lacks the function.
    if functions and not valid_functions:
        print(f"[{_tag}] *** ALL {len(functions)} candidate(s) rejected — "
              f"treating as a VALIDATOR fault, not a miss. Falling back to unvalidated.")
        valid_functions = list(functions)

    print(f"[{_tag}] {len(valid_functions)} candidate(s) across "
          f"{sorted({f.get('routed_module') for f in valid_functions})}:")
    for rank, fn in enumerate(valid_functions[:10]):
        print(f"    {rank}. {fn.get('name')} | module={fn.get('routed_module')} "
              f"| class={fn.get('class_name')} | score={fn.get('reranker_score')}")

    return valid_functions, top_score, second_score


def _fn_key(fn: dict) -> tuple:
    return (fn.get("module"), fn.get("class_name"), fn.get("name"))


def _resolve_with_fallback(
    *,
    query: str,
    sq: SubQuery,
    functions: list[dict],
    coverage: CoverageAssessment,
) -> tuple[list[dict], CoverageAssessment]:
    """Give a non-EXACT routed-module result a second chance across all modules.

    Runs BEFORE any generation prompt. Returns the (functions, coverage) the rest
    of the pipeline should use — the retry's result only when it is a STRICTLY
    better verdict, otherwise the original untouched. If the retry also fails to
    find real coverage, the verdict stays NO_MATCH / PARTIAL_MATCH and Pass 3
    proceeds to GENERATE_STANDALONE / GENERATE_WITH_HELPERS exactly as before.
    """
    if not CROSS_MODULE_FALLBACK:
        return functions, coverage
    # The parameters module is a docs index with its own LLM-free pseudo-result;
    # it has no function catalog to search cross-module.
    if sq.module == "parameters":
        return functions, coverage
    if coverage.match_type == MatchType.EXACT_MATCH:
        return functions, coverage

    seen = {_fn_key(f) for f in functions}
    x_functions, x_top, x_second = _lookup_cross_module(sq)
    fresh = [f for f in x_functions if _fn_key(f) not in seen]
    if not fresh:
        # Routing was almost certainly right — the whole-codebase search surfaced
        # nothing the assessor has not already judged. Skip the second LLM call.
        print(f"[cross-module] nothing new beyond the {len(seen)} candidate(s) already "
              f"assessed — keeping the routed-module verdict "
              f"({coverage.match_type.value}).")
        return functions, coverage

    # routed_module is the module LABEL; "module" is a source filename (see the note
    # in _lookup_cross_module).
    x_modules = sorted({f.get("routed_module") for f in x_functions})
    print(f"[cross-module] '{sq.module}' gave {coverage.match_type.value}; re-assessing "
          f"with {len(fresh)} additional candidate(s) from {x_modules}.")
    x_coverage = assess_function_coverage(
        query=query,
        sq=sq,
        functions=x_functions,
        top_score=x_top,
        second_score=x_second,
        module_label=f"{', '.join(x_modules)} (whole-codebase retry)",
    )

    old_rank = _MATCH_STRENGTH.get(coverage.match_type, 0)
    new_rank = _MATCH_STRENGTH.get(x_coverage.match_type, 0)
    if new_rank <= old_rank:
        print(f"[cross-module] discarded: {x_coverage.match_type.value} does not beat "
              f"{coverage.match_type.value} — keeping the routed-module result.")
        return functions, coverage

    x_coverage.reason = (
        f"[cross-module retry] {x_coverage.reason} "
        f"(routed module '{sq.module}' gave {coverage.match_type.value}; "
        f"re-retrieved across {', '.join(x_modules)})"
    )
    print(f"[cross-module] ADOPTED: {coverage.match_type.value} -> "
          f"{x_coverage.match_type.value}")
    return x_functions, x_coverage


# Structured "helpers dictionary" for a retrieved helper function, used to list
# available helpers for the LLM in _build_generation_context()'s request block (which both
# the specification and the implementation calls are given). Deliberately includes
# method_type alongside class_name/signature/code: instance_method, classmethod,
# staticmethod, property, and bare function each have a different real Python calling
# convention, and GENERATOR_PROMPT is instructed to call helpers by this field.
def _render_helper_for_generation(fn: dict) -> dict:
    # Name AND description. The catalog documents all 268 parameters across the three
    # modules, averaging 85 characters, and this function used to throw every one of
    # them away — leaving a bare name list that says a parameter is called `k` and
    # nothing about what `k` holds. A real run then passed `len(old)` there, a
    # network-scale port count, where the KB's own entry reads "the stream's total
    # potential (edge count) used as the baseline/maximum penalty". The fact that would
    # have prevented it was already written; it just never reached a prompt.
    params = fn.get("parameters", [])
    param_names = []

    if isinstance(params, list):
        for param in params:
            if not isinstance(param, dict) or "parameter_name" not in param:
                continue
            name = str(param["parameter_name"])
            description = str(param.get("description") or "").strip()
            param_names.append(f"{name}: {description}" if description else name)

    # The docstring FIRST, and the nl_summary only as a last resort. These two are
    # written for different jobs and are not interchangeable:
    #
    #   nl_summary is retrieval bait, and good at it — several sentences ending in
    #   "Users might ask: 'why do I get an error saying no instance is set'; ...", which
    #   is what makes the embedding match a user's phrasing. Pasted into a code-writing
    #   prompt it is 308 characters of noise per candidate, and it was what this function
    #   sent, because catalog entries carry neither "purpose" nor a non-empty
    #   "description" and the old chain fell straight through to it.
    #
    #   docstring is the one written for THIS job: 175 characters on average, saying what
    #   the member does and what it hands back. 66 of the catalog's 108 were authored
    #   specifically (the real library has no docstring at that member at all), and they
    #   carry facts nothing else in the prompt does — `initialize_problem` "Initialises
    #   TSNConstraintEvaluator after metadata has been built and selects routing/
    #   task-placement checks" is the lifecycle rule whose absence had every benchmarked
    #   run constructing a second, differently-configured evaluator by hand.
    #
    # So this is strictly better content AND ~5,400 characters smaller across a full
    # 40-candidate context. The old chain is kept below it for the one path that has no
    # docstring to offer: _lookup_cross_module's fallback branch, for a node with no
    # catalog entry, sets docstring=None and nl_summary="" (there, "code" is the only
    # real content and it is sent regardless).
    purpose = (
        fn.get("docstring")
        or fn.get("purpose")
        or fn.get("description")
        or fn.get("nl_summary")
        or ""
    )
    return {
        "name": fn.get("name"),
        "class_name": fn.get("class_name"),
        "sub_class_name": fn.get("sub_class_name"),
        "method_type": fn.get("method_type") or "unknown",
        "signature": fn.get("signature"),
        "parameters": param_names,
        "purpose": purpose,
        "source": fn.get("modules", "?"),
        "code": fn.get("code", ""),
    }


# v96: library facts that belong to ONE library member travel on that member's entry in the generation
# context, as "caveats", so the model reads them when it reads the member it is about to call. Each was
# checked against the library source. They were moved here out of the generation prompt's notes.
_SLOT_EQUALITY = ("The Slot objects involved are dataclasses whose equality includes `index`, a fresh number "
                  "from a class-wide counter given to every Slot when it is made: two Slots for the same window "
                  "made separately are never equal. Compare windows by (start_ns, end_ns, period_ns, stream_id, "
                  "port_id, queue_id), never by object (no ==, != or `in` on Slots or on values holding them), "
                  "and compare a stream's slots as a sorted collection. A Slot is unhashable: a set or a dict "
                  "key holds that tuple, never the Slot.")
_LIBRARY_CAVEATS: dict = {
    ("Schedule", "to_compact_encoding"): [
        "Raises ValueError('Schedule is empty') on a Schedule with no slots.",
        "Raises ValueError('Multiple slots found ...') when two slots share one (port, stream): the array holds one value per port and stream."],
    ("Schedule", "update_streams"): [
        "Requires every stream in `streams` to have at least two slots: a stream with no slots raises IndexError, one with exactly one raises UnboundLocalError. Usecase.addSchedule calls it, so the same holds there."],
    ("Schedule", "total_occupied_time"): ["Unusable: it calls self.compile(), which does not exist, and raises AttributeError."],
    ("Schedule", "utilization"): ["Unusable: it calls self.compile(), which does not exist, and raises AttributeError."],
}
for _member in (("Schedule", "slots_by_stream"), ("Schedule", "slots_by_port"), ("Schedule", "from_compact_encoding"),
                ("Schedule", "from_usecase"), ("Schedule", "validate_conflicts"), ("Schedule", "__init__"),
                ("Slot", "conflicts_with"), ("Slot", "all_occurrences"), ("Slot", "__init__"), ("Slot", "__lt__")):
    _LIBRARY_CAVEATS.setdefault(_member, []).append(_SLOT_EQUALITY)


def _with_caveats(functions: list[dict]) -> list[dict]:
    """The rendered helper entries with each member's caveats attached (see _LIBRARY_CAVEATS)."""
    out = []
    for fn in functions:
        entry = _render_helper_for_generation(fn)
        notes = _LIBRARY_CAVEATS.get((fn.get("class_name") or "", fn.get("name")))
        if notes:
            entry["caveats"] = list(notes)
        out.append(entry)
    return out


# v92: the generation stage is given the library code of exactly the members the blueprint's
# reuse decisions name, instead of the retrieval pool (the 40 highest-scoring candidates). In
# run 92a 4 of the 22 members the blueprint chose never reached the generation prompt, while
# most of the pool was never used. False goes back to the retrieval pool.
_GENERATION_CONTEXT_FROM_REUSE = True


def _reuse_decision_functions(blueprint: Optional[dict], candidates: list[dict]) -> list[dict]:
    """The function dicts for every member the blueprint's reuse decisions name, once each, in
    the blueprint's order: taken from the retrieved candidates when retrieval found the member,
    otherwise from the library catalogue (a member the design names but retrieval never ranked,
    such as TSNScheduling.use_instance). A name found in neither is reported and skipped."""
    by_key = {(fn.get("class_name") or "", fn.get("name")): fn for fn in candidates or []}
    out, seen = [], set()
    for decision in (blueprint or {}).get("reuse_decisions") or []:
        if not isinstance(decision, dict):
            continue
        name = str(decision.get("codebase_function") or "").strip()
        cls = str(decision.get("class_name") or "").strip()
        if "." in name:
            cls, name = cls or name.rpartition(".")[0].rpartition(".")[2], name.rpartition(".")[2]
        key = (cls, name)
        if not name or key in seen:
            continue
        seen.add(key)
        fn = by_key.get(key)
        if fn is None:
            for module, entries in FUNCTION_CATALOG_BY_MODULE_CLASSED.items():
                entry = entries.get((cls or None, name)) or entries.get((cls, name))
                if entry is not None:
                    fn = _catalog_entry_to_function(entry, module)
                    break
        if fn is None:
            print(f"[algogen] reuse decision {cls + '.' if cls else ''}{name} is in neither the "
                  f"retrieved candidates nor the library catalogue; not given to the generation stage")
            continue
        out.append(fn)
    return out


def _generation_context(blueprint: Optional[dict], candidates: list[dict]) -> str:
    """The library code the generation stage is given (see _GENERATION_CONTEXT_FROM_REUSE)."""
    chosen = (_reuse_decision_functions(blueprint, candidates)
              if _GENERATION_CONTEXT_FROM_REUSE and blueprint else candidates)
    if not chosen:
        return "No helpers found in context."
    return json.dumps(_with_caveats(chosen), indent=2)   # v96: each entry carries its library caveats


def _render_helpers_for_generation(functions: list[dict]) -> str:
    """JSON array of helper dicts (see _render_helper_for_generation), or a plain
    string when there are none — kept as a string either way since it is spliced
    straight into a text prompt, not parsed back out."""
    if not functions:
        return "No helpers found in context."
    return json.dumps([_render_helper_for_generation(fn) for fn in functions], indent=2)


# The reranker used by the generation path. reranker_algo is the dedicated one, but it is
# currently disabled to free ~2GB of VRAM (see its definition), which used to make the whole
# generation path raise RuntimeError. combined_reranker is already resident for the
# explain/grounding path, so fall back to it rather than refusing to generate; if neither is
# available, retrieval order is used as-is.
def _rerank_generation_nodes(nodes, query_text: str):
    reranker = reranker_algo if reranker_algo is not None else combined_reranker
    if reranker is None:
        return nodes
    return reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_text))


# Pipeline stage 4, shared step: build the grounding context that BOTH the spec
# extraction (4a) and the code generation (4b) calls run against. Retrieving once and
# reusing the result guarantees the implementer sees exactly the same codebase context the
# specifier did — a spec naming a helper the code generator was never shown would be
# unimplementable. Works from the sub-query alone (the original full query is intentionally
# NOT passed in). Returns (context_str, request_block).
def _build_generation_context(
    sq: SubQuery,
    helper_functions: list[dict],
    extracted_params: Optional[dict] = None,
) -> tuple[str, str]:
    augmented_text = augment_query(sq.text, sq.module)

    param_context_str = _retrieve_param_context(
        f"What parameters are relevant to: {augmented_text}"
    )
    if not param_context_str:
        param_context_str = "No relevant parameter definitions found."

    helpers_json = _render_helpers_for_generation(helper_functions)

    # Retrieve and rerank nodes directly — bypass synthesizer.
    nodes = module_gen_retrievers[sq.module].retrieve(augmented_text)
    # Raw sub-query for the reranker — see the note in _lookup_one(): the
    # "[File: ...]" prefix is shared by every node in the module and flattens
    # the cross-encoder's scores.
    nodes = _rerank_generation_nodes(nodes, sq.text)
    retrieved_context_str = "\n\n".join(n.node.get_content() for n in nodes)

    # Helpers live here (under the "Context (existing codebase helpers)" label both
    # prompts show context_str under) rather than inside request_block/query_str, which
    # both prompts frame as background/the original request — a helper's exact calling
    # convention (see method_type below) is load-bearing, not background.
    context_str = (
        f"{retrieved_context_str}\n\n"
        f"Available helper functions (each entry's \"method_type\" — instance_method, "
        f"classmethod, staticmethod, property, or function — determines how it must be "
        f"called; see the calling-convention rules in the generation instructions):\n"
        f"{helpers_json}"
    )

    request_block = (
        f"Module-specific request:\n{augmented_text}\n\n"
        f"Target module: {sq.module}\n\n"
        f"Extracted parameters:\n"
        f"{json.dumps(extracted_params or {}, indent=2)}\n\n"
        f"Relevant parameter definitions:\n{param_context_str}"
    )

    return context_str, request_block


# The keys FUNCTION_SPEC_PROMPT is contracted to return, in the order they are shown to
# the user. Used to normalize whatever the LLM emits into a stable shape so the review
# gate, the contract extractor and the code generator all see the same fields.
SPEC_KEYS = [
    "name", "type", "class", "description", "signature", "parameters",
    "preconditions", "postconditions", "callable", "dependencies", "raises",
]


# ── Spec shape enforcement: plain functions, no `self`/`cls` fiction ─────────
# FUNCTION_SPEC_PROMPT now asks for a module-level function whose state arrives as named
# parameters, but a prompt rule alone is not enforcement: measured over the recorded runs,
# specs whose signature began with `self`/`cls` passed their own tests 39-42% of the time
# against 65% for plain-parameter signatures, because the implementation and the
# (code-blind) test writer each had to reconstruct that first argument independently and
# nothing made them agree. These helpers make the shape a property of the pipeline instead
# of a request to the model: a leftover `self`/`cls` is stripped, and whatever object state
# it implied is promoted into real parameters so the function can still reach it.
def _spec_is_class(spec: dict) -> bool:
    """True when the spec describes a whole class rather than a single function.

    A class spec is legitimate (a context manager, a stateful decorator, "write a Schedule
    class") and is generated whole, so its methods keep `self` honestly: the test builds a
    real instance through the class's own __init__ rather than a stand-in.
    """
    sig = str(spec.get("signature") or "").lstrip()
    return str(spec.get("type") or "").strip().lower() == "class" or sig.startswith("class ")


def _split_signature_params(signature: str) -> tuple[str, list[str], str]:
    """Split a def line into (prefix, [param segments], suffix).

    Segments are kept verbatim so annotations and defaults survive; splitting is done at
    top-level commas only, so `Dict[str, int]` and `f(a=(1, 2))` stay in one piece.
    """
    open_idx = signature.find("(")
    close_idx = signature.rfind(")")
    if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
        return signature, [], ""
    inner = signature[open_idx + 1:close_idx]
    params, depth, current = [], 0, ""
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            params.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        params.append(current.strip())
    return signature[:open_idx + 1], [p for p in params if p], signature[close_idx:]


def _param_base_name(segment: str) -> str:
    """The bare name of one signature segment: 'chromosome: np.ndarray = None' -> 'chromosome'."""
    return segment.split("=", 1)[0].split(":", 1)[0].strip().lstrip("*")


# Attributes the spec still talks about as `self.<attr>` / `cls.<attr>`, which have to
# become parameters once the leading `self`/`cls` is gone.
_SELF_ATTR_RE = re.compile(r"\b(?:self|cls)\.([A-Za-z_]\w*)")
_MAX_PROMOTED_PARAMS = 8


def _flatten_spec_to_function(spec: dict) -> dict:
    """Rewrite a method-shaped spec into an equivalent plain-function spec.

    Strips a leading `self`/`cls` from the signature and parameter list, promotes every
    `self.<attr>` the spec's own prose mentions into an explicit parameter (so the function
    still receives the state it was written against), and rewrites those prose references
    to the bare parameter name. Class specs are returned untouched. A spec that is already
    a plain function is returned unchanged.
    """
    if _spec_is_class(spec):
        return spec

    signature = str(spec.get("signature") or "")
    prefix, segments, suffix = _split_signature_params(signature)
    if not segments or _param_base_name(segments[0]) not in {"self", "cls"}:
        return spec

    dropped, segments = segments[0], segments[1:]
    text_keys = ("description", "preconditions", "postconditions", "raises")

    # Everything the spec says the function reads off the object, in first-mention order.
    mentioned: list[str] = []
    for key in text_keys:
        val = spec.get(key)
        for chunk in (val if isinstance(val, list) else [val]):
            for attr in _SELF_ATTR_RE.findall(str(chunk or "")):
                if attr not in mentioned:
                    mentioned.append(attr)

    existing = {_param_base_name(s) for s in segments}
    promoted = [a for a in mentioned if a not in existing][:_MAX_PROMOTED_PARAMS]

    # Appending a bare name after a defaulted or star segment is a SyntaxError, so give the
    # promoted parameters a default in that case. The test writer is told to pass every
    # declared parameter either way.
    needs_default = any("=" in s or s.lstrip().startswith("*") for s in segments)
    segments += [f"{name}=None" if needs_default else name for name in promoted]

    out = dict(spec)
    out["signature"] = f"{prefix}{', '.join(segments)}{suffix}"
    out["type"] = "function"
    out["class"] = None

    params = [p for p in (spec.get("parameters") or [])
              if not (isinstance(p, dict) and str(p.get("name", "")).strip() in {"self", "cls"})]
    params += [{"name": name, "default": "None" if needs_default else None} for name in promoted]
    out["parameters"] = params

    # `self.n_ports is a 1D array` would tell the test writer to build an object that the
    # implementation never receives, so the prose has to follow the signature.
    for key in text_keys:
        val = spec.get(key)
        if isinstance(val, list):
            out[key] = [_SELF_ATTR_RE.sub(r"\1", str(v)) for v in val]
        elif isinstance(val, str):
            out[key] = _SELF_ATTR_RE.sub(r"\1", val)

    print(f"[spec] flattened method spec to a plain function: dropped '{dropped.strip()}'"
          + (f", promoted {promoted} to parameters" if promoted else ""))
    return out


def _normalize_spec(spec: dict) -> dict:
    """Coerce a raw spec object into the full SPEC_KEYS shape.

    The LLM occasionally omits an optional key or returns a scalar where a list is
    expected. Rather than failing the whole generation, fill the gaps with empty values
    of the right type so every downstream consumer (review gate, _derive_contract_from_spec,
    GENERATOR_PROMPT) can rely on the keys being present.
    """
    out: dict = {}
    list_keys = {"parameters", "preconditions", "postconditions", "raises"}

    for key in SPEC_KEYS:
        val = spec.get(key)
        if key in list_keys:
            if val is None:
                val = []
            elif not isinstance(val, list):
                val = [val]
        out[key] = val

    if not out.get("name"):
        # A nameless spec cannot produce an import hint or a test; derive one from the
        # signature if the model gave one, else fall back to a stable placeholder.
        sig_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", str(out.get("signature") or ""))
        out["name"] = sig_match.group(1) if sig_match else "generated_function"

    if not out.get("signature"):
        param_names = ", ".join(
            str(p.get("name")) for p in out["parameters"] if isinstance(p, dict) and p.get("name")
        )
        out["signature"] = f"def {out['name']}({param_names}):"

    # Enforce the plain-function shape before anything downstream reads the spec: the
    # contract, the tests and the implementation must all be derived from the same
    # signature, and this is the last point where they still share one.
    out = _flatten_spec_to_function(out)

    # A class spec keeps its own shape; everything else is a plain module-level function,
    # whether or not the model labelled it one (see _flatten_spec_to_function above).
    if _spec_is_class(out):
        out["type"] = "class"
    elif out.get("type") != "function":
        out["type"] = "function"

    if out.get("callable") is None:
        out["callable"] = True

    deps = out.get("dependencies")
    if not isinstance(deps, dict):
        deps = {}
    out["dependencies"] = {
        "internal": deps.get("internal") if isinstance(deps.get("internal"), list) else [],
        "external": deps.get("external") if isinstance(deps.get("external"), list) else [],
    }

    return out


# Pipeline stage 4a (specify): called from generate_verified_implementation() whenever
# assess_function_coverage() returned GENERATE_WITH_HELPERS or GENERATE_STANDALONE.
# Produces ONLY the specification of the function that has to be built — no code. The
# spec is an internal artefact: the contract/tests are derived from it and the
# implementation is written against it, but it is never put to the user for approval —
# the TESTS are what the user reviews. `revision` is currently UNUSED (it was the spec
# review gate's regeneration hook); kept so a revision path can be reattached cheaply.
# Returns the normalized spec dict, or None if nothing parseable came back.
def generate_function_spec(
    context_str: str,
    request_block: str,
    revision: Optional[str] = None,
) -> Optional[dict]:
    revision_block = ""
    if revision:
        revision_block = (
            "\nREVISION REQUEST from the user — apply these changes to the specification, "
            "keeping everything else that was already correct:\n"
            f"{revision}\n"
        )

    prompt = FUNCTION_SPEC_PROMPT.format(
        context_str=context_str,
        query_str=(
            f"{request_block}\n\n"
            "Specify the function that must be built to fully satisfy this request."
        ),
        revision_block=revision_block,
    )
    # max_tokens above the global default (3000): FUNCTION_SPEC_PROMPT now asks for a
    # concrete shape/dtype precondition per internal self.* dependency, and a function
    # touching many precomputed attributes (masks, cost matrices, etc.) can need more
    # room than the default budget before truncating mid-JSON -> JSONDecodeError.
    raw = Settings.llm.complete(prompt, max_tokens=llm_config.stage_max_tokens(prompt)).text.strip()

    try:
        spec = extract_json_object(raw)
    except json.JSONDecodeError as e:
        print(f"[spec] specification JSON parse failed ({e}). Retrying in plain text.")
        text_raw = _retry_as_text(prompt, e, _SPEC_TEXT_FORMAT, max_tokens=llm_config.stage_max_tokens(prompt))
        spec = _parse_spec_text(text_raw)
        if not spec.get("name"):
            print(f"[spec] plain-text retry also failed. Raw response:\n{text_raw[:2000]}")
            try:
                with open(
                    "/tmp/claude-1002/-home-admin-Danial-AI-Agent-some-files/95dc4a5e-b5dc-41f9-9500-5ae24ffabbf9/scratchpad/spec_parse_failure_dump.txt",
                    "w",
                ) as _f:
                    _f.write(f"error: {e}\nraw_len: {len(raw)}\n----- FULL RAW TEXT -----\n{raw}"
                             f"\n----- TEXT-MODE RETRY -----\n{text_raw}")
            except Exception:  # noqa: BLE001
                pass
            return None
        return _normalize_spec(spec)

    if not isinstance(spec, dict):
        print(f"[spec] specification was not a JSON object (got {type(spec).__name__}).")
        return None

    return _normalize_spec(spec)


def _spec_for_codegen(spec: dict) -> dict:
    """The approved spec as handed to GENERATOR_PROMPT — everything except "raises".

    The exception list is deliberately withheld from the implementer: the tests were
    already written from the contract (which DID see "raises"), so letting the code
    generator read it would let it shape the implementation around the error cases it
    expects to be tested on instead of around the behavior the spec describes.
    """
    return {k: v for k, v in spec.items() if k != "raises"}


def _render_spec_for_review(spec: dict) -> str:
    """Pretty-print the specification for the final answer."""
    return json.dumps(spec, indent=2)


# Pipeline stage 4b (implement): write the implementation for a spec whose tests are
# settled. Runs after the test gate, so by this point both the contract and the tests the
# code will be judged against are frozen. Returns raw Python source.
def generate_code_from_spec(
    spec: dict,
    context_str: str,
    request_block: str,
) -> str:
    prompt = GENERATOR_PROMPT.format(
        spec_json=json.dumps(_spec_for_codegen(spec), indent=2),
        context_str=context_str,
        query_str=request_block,
    )
    raw = Settings.llm.complete(prompt).text.strip()

    # The prompt asks for bare source, but models still fence it sometimes.
    code = (extract_code_block(raw) or raw).strip()

    # Syntax gate. A malformed body (a dangling `if:`/`try:` with no indented block, an
    # unterminated string) reaches pytest as a *collection* error, which fails every test
    # at once and gives the triage/repair loop a traceback about the test file rather
    # than about the defect. Catching it here buys one targeted retry that quotes the
    # actual SyntaxError, which is far more actionable than the collection failure.
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        print(f"[codegen] generated source is not valid Python ({e}); retrying once.",
              file=sys.stderr)
        retry_prompt = (
            f"{prompt}\n\n"
            f"YOUR PREVIOUS ANSWER WAS REJECTED: it was not valid Python.\n"
            f"{type(e).__name__}: {e}\n"
            f"Offending line {e.lineno}: {(e.text or '').rstrip()}\n"
            "Return the corrected, complete function. Every `if`/`for`/`while`/`try`/`def` "
            "header must be followed by a real indented block (use `pass` only if a branch "
            "genuinely does nothing), and every string/docstring must be closed."
        )
        raw = Settings.llm.complete(retry_prompt).text.strip()
        retry_code = (extract_code_block(raw) or raw).strip()
        try:
            compile(retry_code, "<generated>", "exec")
            code = retry_code
        except SyntaxError as e2:
            # Last resort before handing broken code downstream: try the one known,
            # deterministic, mechanical fix (a misplaced `from __future__ import`) rather
            # than only ever asking the LLM again. Falls through to the original
            # print-and-hand-off behavior (the normal repair loop still gets a chance at
            # it) if that doesn't happen to be what's wrong.
            fixed_retry = _fix_future_import_position(retry_code) if retry_code else ""
            if fixed_retry and _syntax_error_of(fixed_retry) is None:
                print(f"[codegen] retry still not valid Python ({e2}); repositioning a "
                      f"misplaced __future__ import fixed it.", file=sys.stderr)
                code = fixed_retry
            else:
                print(f"[codegen] retry still not valid Python ({e2}); handing it "
                      f"downstream.", file=sys.stderr)
                code = retry_code or code
    if _is_qwen_finetuned_backend():
        code = _autofix_config_ini_path(code, REPO_ROOT)
    return code


# The GENERATE_ALGORITHM counterpart to generate_code_from_spec(): identical structure
# (same one-retry syntax gate), but formats GENERATE_ALGORITHM_PROMPT instead of
# GENERATOR_PROMPT. Kept as its own function rather than a branch inside
# generate_code_from_spec so each stays single-purpose; generate_verified_implementation()
# picks between the two via its use_algorithm_prompt flag.
#
# Takes the BLUEPRINT + FROZEN INTERFACE instead of a per-function spec: the frozen
# interface already fixes every function's name/signature/docstring (what a spec used
# to provide here), so re-feeding the spec output alongside it would just be a second,
# possibly-drifted description of the same thing. The unit-test prompt is different —
# see _spec_based_tests() — because a contract's "behaviors"/"input_domains" have no
# equivalent in the blueprint, so that prompt keeps the spec-derived contract AND gains
# the blueprint/frozen interface as extra grounding.
# ---------------------------------------------------------------------------------------
# Smoke-run gate for the generated algorithm.
#
# The syntax gate only proves the file parses. Real runs showed the same prompt giving 85/89
# on the benchmark once and 6/89 the next time, and every one of the bad runs failed on the
# simplest possible inputs - a valid start that it rejected, a removed stream whose windows
# stayed behind so nothing could be added, a loop that never ended. Prompt wording did not
# make that reliable; running the code does, because the verdict is a fact about the code,
# not about whether the model followed a sentence.
#
# The probes are built ONLY from the library: a k-ary fat-tree use case from
# UsecaseGenerator.generateUC, each stream on the shortest route the generator stored in
# pathLists, start times spaced so the library evaluator accepts every stream. From that one
# valid schedule S come five generic probes (no change, remove one, add one, remove one and add
# another, one stream moving into the time another frees). The gate checks only what holds for ANY correct design - the top-level function
# returns, without raising, within a time limit - so it never needs to know the output format.
# Parameters are filled by their annotations (use case, problem, the two schedule arrays);
# when a signature cannot be filled that way, or the probes cannot be built, the gate is
# skipped rather than guessed at.
# ---------------------------------------------------------------------------------------
# Switched off 2026-10-27 at the user's request ("stop with retries and the gate"): generation
# returns its first answer, after the syntax gate only. The code is kept; set True to re-enable.
_SMOKE_GATE_ENABLED = False
_SMOKE_TIMEOUT_S = 60
_SMOKE_MAX_RETRIES = 2

_SMOKE_PROBE_CONFIG = """[topology_config]
portspeed = 1000
propdelay = 100
procdelay = 1
linkBw = 1000
queuecountperport = 8
graphconfig = {'typ': 'ft', 'k': 4}
[streams_config]
numofstreamsperinstance = 8
numplacesTalker = 1
numplacesListener = 1
numlistenersperstr = 1
routing = {'fixPath': True, 'shortestPath': True}
stride = {'active': False, 'step': 1, 'increase_factor': 1}
staggered = {'active': False, 'edgep': 1, 'podp': 1}
hotspot = {'active': False, 'increase_factor': 1}
period_range = {'start': 200, 'stop': 201}
numberframes_range = {'start': 1, 'stop': 2}
datasize_range = {'start': 125, 'stop': 126}
maxlatency_range = 0.05
expiration = {'finite_duration': False}
"""

_SMOKE_HARNESS = '''
import sys, os, random, inspect, traceback
lib_dir, impl_dir, top_name, probe, cfg_path = sys.argv[1:6]
sys.path[:0] = [impl_dir, lib_dir]
import numpy as np
from usecase_generator import UsecaseGenerator
from tsn_scheduling_problem import TSNScheduling
from evaluator import TSNConstraintEvaluator, ScheduleEncodingType
from schedule import Schedule
try:
    random.seed(0); np.random.seed(0)
    usecase = UsecaseGenerator.generateUC(cfg_path)
    builder = TSNScheduling(None)
    with builder.use_instance(usecase):
        n_ports, n_streams = builder.n_ports, builder.n_streams
        name_to_id = builder.port_name_to_id
        hops = [[name_to_id[h] for h in st["pathLists"][0] if h in name_to_id] for st in usecase.streams]
        def column(s, offset):
            col = np.zeros(n_ports, dtype=int)
            for k, port in enumerate(hops[s]):
                col[port - 1] = offset + k * 2500 + 1
            return col
        full = np.zeros((n_ports, n_streams), dtype=int)
        for s in range(n_streams):
            full[:, s] = column(s, s * 20000)
        ev = TSNConstraintEvaluator(builder, check_routing=True, check_task_placement=True,
                                    scheduling_type=ScheduleEncodingType.START_TIME)
        def valid(arr):
            present = [s for s in range(n_streams) if arr[:, s].any()]
            feasible = ev.feasible_streams(arr)
            return all(bool(feasible[s]) for s in present) and not Schedule.from_compact_encoding(
                arr, ev.windLengths, ev.stream_intervals, cycle_ns=builder.macro_cycle
            ).validate_conflicts(touching_allowed=True)
        ok = valid(full)
        # "moves into freed time": stream a is retimed onto the exact window stream b leaves
        # on a port both routes use; it is valid only once b is gone.
        moved = None
        for a in range(n_streams):
            for b in range(n_streams):
                shared = set(hops[a]) & set(hops[b])
                if a == b or not shared or moved is not None:
                    continue
                port = min(shared)
                offset = b * 20000 + (hops[b].index(port) - hops[a].index(port)) * 2500
                if offset < 0:
                    continue
                cand = full.copy(); cand[:, b] = 0; cand[:, a] = column(a, offset)
                both = cand.copy(); both[:, b] = full[:, b]
                if valid(cand) and not valid(both):
                    moved = cand
    if not ok or n_streams < 2:
        print("SKIP: could not build a valid probe schedule"); sys.exit(0)
except Exception:
    print("SKIP: probe construction failed " + traceback.format_exc()[-1500:]); sys.exit(0)

def without(arr, s):
    out = arr.copy(); out[:, s] = 0; return out
start, finish = {"no_change": (full, full),
                 "remove_one": (full, without(full, 0)),
                 "add_one": (without(full, 0), full),
                 "remove_one_add_another": (without(full, 0), without(full, 1)),
                 "move_into_freed": (full, moved)}[probe]
if finish is None:
    print("OK"); sys.exit(0)  # no pair of streams shares a port in this use case
try:
    import generated_impl as gen
    fn = getattr(gen, top_name)
except Exception:
    print("FAIL\\n" + traceback.format_exc()[-3000:]); sys.exit(0)
problem = TSNScheduling(None)
args, arrays = [], [start.copy(), finish.copy()]
for p in inspect.signature(fn).parameters.values():
    a = p.annotation
    ann = (a if isinstance(a, str) else getattr(a, "__name__", str(a))).lower()
    name = p.name.lower()
    if "usecase" in ann or name == "usecase":
        args.append(usecase)
    elif "tsnscheduling" in ann or name == "problem":
        args.append(problem)
    elif arrays and ("ndarray" in ann or "schedule" in ann or "schedule" in name):
        args.append(arrays.pop(0))
    else:
        print("SKIP: cannot fill parameter " + p.name); sys.exit(0)
if arrays:
    print("SKIP: signature takes fewer than two schedules"); sys.exit(0)
try:
    fn(*args)
    print("OK")
except Exception:
    print("FAIL\\n" + traceback.format_exc()[-3000:])
'''

_SMOKE_PROBES = (
    ("no_change", "start and finish are the same valid schedule"),
    ("remove_one", "the finish drops one stream that the start has"),
    ("add_one", "the finish adds one stream that the start lacks"),
    ("remove_one_add_another", "the finish drops one stream and adds a different one"),
    ("move_into_freed", "one stream is retimed onto the exact window another stream frees "
                        "by leaving - valid only after that stream is gone"),
)


def _smoke_gate(code: str, top_name: Optional[str]):
    """Run the generated top-level function on the library-built probes.

    Returns (passed, total, failure_text). failure_text is None when every probe passed or the
    gate could not run (no top-level name, probes not buildable, signature not fillable) -
    a skipped gate never triggers a retry. Each probe runs in its own subprocess with a time
    limit, so a crash or an endless loop in the generated code cannot affect the pipeline.
    """
    import subprocess, tempfile
    if not top_name or not re.search(r"^def " + re.escape(top_name) + r"\(", code, re.MULTILINE):
        print(f"[smoke] skipped: top-level function {top_name!r} not defined in the code",
              file=sys.stderr)
        return 0, 0, None
    work = tempfile.mkdtemp(prefix="smoke_gate_")
    with open(os.path.join(work, "generated_impl.py"), "w") as fh:
        fh.write(code)
    with open(os.path.join(work, "_smoke_harness.py"), "w") as fh:
        fh.write(_SMOKE_HARNESS)
    cfg_path = os.path.join(work, "probe.ini")
    with open(cfg_path, "w") as fh:
        fh.write(_SMOKE_PROBE_CONFIG)
    passed, failures = 0, []
    for probe, description in _SMOKE_PROBES:
        try:
            out = subprocess.run(
                [sys.executable, os.path.join(work, "_smoke_harness.py"), LIB_DIR, work,
                 top_name, probe, cfg_path],
                capture_output=True, text=True, timeout=_SMOKE_TIMEOUT_S, cwd=work,
            ).stdout.strip()
        except subprocess.TimeoutExpired:
            failures.append(f"probe '{description}': did not return within "
                            f"{_SMOKE_TIMEOUT_S} s - a loop that never ends or makes no progress.")
            continue
        if out.startswith("SKIP"):
            print(f"[smoke] skipped: {out.splitlines()[0]}", file=sys.stderr)
            return 0, 0, None
        if out.endswith("OK"):
            passed += 1
        else:
            failures.append(f"probe '{description}': it raised\n{out.removeprefix('FAIL').strip()}")
    total = len(_SMOKE_PROBES)
    print(f"[smoke] {passed}/{total} probes passed", file=sys.stderr)
    return passed, total, ("\n\n".join(failures) if failures else None)


def generate_algorithm_from_spec(
    blueprint: Optional[dict],
    frozen_interface: Optional[str],
    context_str: str,
    request_block: str,
) -> str:
    prompt = GENERATE_ALGORITHM_PROMPT.format(
        blueprint_json=json.dumps(_blueprint_for_prompt(blueprint), indent=2),
        frozen_interface=frozen_interface or "(no frozen interface available)",
        violation_row_order=_render_violation_row_order(LIB_DIR),
        context_str=context_str,
        query_str=request_block,
    )
    # max_tokens above the global default (3000): this call emits every frozen
    # function's FULL body in one file, which is strictly larger than the stub-only
    # interface freeze that already needed 4500 (build_interface_freeze) — confirmed
    # necessary after a real run truncated mid-function at the default (~12.5k chars,
    # right at the 3000-token ceiling), leaving an incomplete function body that then
    # surfaced as a spurious "undefined name" error rather than the real cause.
    raw = Settings.llm.complete(prompt, max_tokens=llm_config.stage_max_tokens(prompt)).text.strip()

    # The prompt asks for bare source, but models still fence it sometimes.
    code = (extract_code_block(raw) or raw).strip()

    # Same syntax gate as generate_code_from_spec(), same rationale: a malformed body
    # otherwise reaches pytest as a collection error instead of a targeted SyntaxError.
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        print(f"[algogen] generated source is not valid Python ({e}); retrying once.",
              file=sys.stderr)
        retry_prompt = (
            f"{prompt}\n\n"
            f"YOUR PREVIOUS ANSWER WAS REJECTED: it was not valid Python.\n"
            f"{type(e).__name__}: {e}\n"
            f"Offending line {e.lineno}: {(e.text or '').rstrip()}\n"
            "Return the corrected, complete implementation. Every `if`/`for`/`while`/"
            "`try`/`def` header must be followed by a real indented block (use `pass` "
            "only if a branch genuinely does nothing), and every string/docstring must "
            "be closed."
        )
        raw = Settings.llm.complete(retry_prompt, max_tokens=llm_config.stage_max_tokens(retry_prompt)).text.strip()
        retry_code = (extract_code_block(raw) or raw).strip()
        try:
            compile(retry_code, "<generated>", "exec")
            code = retry_code
        except SyntaxError as e2:
            # Last resort before handing broken code downstream: try the one known,
            # deterministic, mechanical fix (a misplaced `from __future__ import`) rather
            # than only ever asking the LLM again. Falls through to the original
            # print-and-hand-off behavior if that doesn't happen to be what's wrong.
            fixed_retry = _fix_future_import_position(retry_code) if retry_code else ""
            if fixed_retry and _syntax_error_of(fixed_retry) is None:
                print(f"[algogen] retry still not valid Python ({e2}); repositioning a "
                      f"misplaced __future__ import fixed it.", file=sys.stderr)
                code = fixed_retry
            else:
                print(f"[algogen] retry still not valid Python ({e2}); handing it "
                      f"downstream.", file=sys.stderr)
                code = retry_code or code
    # Smoke-run gate (see _smoke_gate): run the code; on a failure, retry with the failing
    # probe and its traceback, and keep whichever version passes the most probes, so a retry
    # that introduces a new bug can never replace a better earlier version.
    top_name = None
    if isinstance(blueprint, dict) and blueprint.get("function_inventory"):
        top_name = (blueprint["function_inventory"][0] or {}).get("name")
    passed, total, failure = _smoke_gate(code, top_name) if _SMOKE_GATE_ENABLED else (0, 0, None)
    best_code, best_passed = code, passed
    for attempt in range(1, _SMOKE_MAX_RETRIES + 1):
        if not failure:
            break
        print(f"[smoke] retry {attempt}/{_SMOKE_MAX_RETRIES}", file=sys.stderr)
        smoke_prompt = (
            f"{prompt}\n\n"
            "YOUR PREVIOUS ANSWER WAS REJECTED: it parses, but running it failed on small valid "
            "inputs built from the library (a few streams, each on its shortest route, every "
            "stream accepted by the evaluator).\n"
            f"{failure}\n\n"
            "Your previous answer, for reference:\n"
            f"```python\n{best_code}\n```\n"
            "Return the corrected, complete implementation. Find the cause of each failure "
            "and fix it where it starts, keeping every function the interface freezes."
        )
        raw = Settings.llm.complete(smoke_prompt, max_tokens=llm_config.stage_max_tokens(smoke_prompt)).text.strip()
        candidate = (extract_code_block(raw) or raw).strip()
        if _syntax_error_of(candidate) is not None:
            print("[smoke] retry is not valid Python; keeping the best version so far",
                  file=sys.stderr)
            continue
        passed, total, failure = _smoke_gate(candidate, top_name)
        if passed > best_passed:
            best_code, best_passed = candidate, passed
    code = best_code
    if _is_qwen_finetuned_backend():
        code = _autofix_config_ini_path(code, REPO_ROOT)
    return code


def _render_generated_code(generated_code: str) -> str:
    """
    Render the generator's output as clean, readable Python.

    generate_verified_implementation() returns a JSON array of {…spec…, code} objects
    (json.dumps'd), so the raw string carries the real code inside a "code" field
    with escaped newlines. Pull each function's code out and fence it. Falls back to
    extract_code_block if the string isn't that JSON shape.
    """
    try:
        items = json.loads(generated_code)
    except (json.JSONDecodeError, TypeError):
        items = None

    if isinstance(items, list) and items and all(isinstance(x, dict) for x in items):
        blocks = []
        for fn in items:
            code = (fn.get("code") or "").strip()
            if code:
                blocks.append("```python\n" + code + "\n```")
        if blocks:
            return "\n\n".join(blocks)

    # Not the expected JSON array — surface whatever code we can, fenced once.
    return "```python\n" + extract_code_block(generated_code).strip() + "\n```"


def _derive_contract_from_spec(spec: dict, user_request: str) -> dict:
    """
    Stage 5, step 1: turn the specification into a behavioral contract
    (behaviors / input_domains / invariants) that tests can be written from.

    Runs directly on the generated spec, before any code exists, so it cannot mirror an
    implementation — the contract reflects the specified intent. Unlike the code
    generator it DOES see "raises", because the error cases are part of the behavior the
    tests must pin down. Uses SPEC_EXTRACTION_PROMPT.
    """
    prompt = SPEC_EXTRACTION_PROMPT.format(
        name=spec.get("name") or "generated_function",
        signature=spec.get("signature") or "",
        description=f"User request: {user_request}\nDeclared purpose: {spec.get('description', '')}",
        parameters=json.dumps(spec.get("parameters", []), indent=2),
        preconditions=json.dumps(spec.get("preconditions", []), indent=2),
        postconditions=json.dumps(spec.get("postconditions", []), indent=2),
        raises=json.dumps(spec.get("raises", []), indent=2),
    )
    raw = Settings.llm.complete(prompt, max_tokens=llm_config.stage_max_tokens(prompt)).text.strip()
    try:
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except json.JSONDecodeError as e:
        print(f"[contract] JSON parse failed ({e}). Retrying in plain text.")
        text_raw = _retry_as_text(prompt, e, _CONTRACT_TEXT_FORMAT)
        contract = _parse_contract_text(text_raw)
        if not contract.get("behaviors"):
            print(f"[contract] plain-text retry also failed. Raw response:\n{text_raw[:2000]}")
        return contract


# ── Deterministic pre-flight checks (no LLM) ─────────────────────────────────
# Everything below is decided by parsing the source, not by asking a model. The failures
# these catch — a name used but never imported, a call that does not match the signature it
# is supposed to exercise, source that does not parse — were previously discovered only
# when pytest ran, one 90-second subprocess later, and arrived as tracebacks about the test
# file rather than as a statement of what is wrong.
def _syntax_error_of(code: str, label: str = "<generated>"):
    """The SyntaxError raised by compiling `code`, or None if it parses."""
    try:
        compile(code, label, "exec")
    except SyntaxError as e:
        return e
    return None


class _BindingCollector(ast.NodeVisitor):
    """Collects every name a module binds (imports, defs, assignments, args, ...).

    Deliberately over-collects: `_undefined_names` reports only names that appear in NO
    binding anywhere in the file, so a name bound in one scope and used in another is not
    flagged. The goal is to catch `np.zeros(...)` with no `import numpy as np` — not to
    reimplement a scope-accurate linter and produce false alarms.
    """

    def __init__(self):
        self.bound: set[str] = set()

    def _bind(self, node):
        if isinstance(node, ast.Name):
            self.bound.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._bind(elt)
        elif isinstance(node, ast.Starred):
            self._bind(node.value)

    def visit_Import(self, node):
        for a in node.names:
            self.bound.add((a.asname or a.name).split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.bound.add(a.asname or a.name)
        self.generic_visit(node)

    def _bind_args(self, args):
        for a in (args.posonlyargs + args.args + args.kwonlyargs):
            self.bound.add(a.arg)
        for a in (args.vararg, args.kwarg):
            if a:
                self.bound.add(a.arg)

    def _visit_function(self, node):
        self.bound.add(node.name)
        self._bind_args(node.args)
        self.generic_visit(node)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Lambda(self, node):
        # Anonymous, so nothing to add for the callable itself — only its parameters.
        # Without this, `sorted(xs, key=lambda x: x[1])` reports "x" as undefined: the
        # lambda's own `x` argument is bound in nothing but this visitor, since
        # `generic_visit` walks into `node.args`/`node.body` without recording bindings.
        self._bind_args(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for t in node.targets:
            self._bind(t)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._bind(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._bind(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        self._bind(node.target)
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self._bind(node.target)
        self.generic_visit(node)

    def visit_withitem(self, node):
        if node.optional_vars is not None:
            self._bind(node.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.bound.update(node.names)
        self.generic_visit(node)

    visit_Nonlocal = visit_Global


def _undefined_names(code: str, provided: tuple = ()) -> list[str]:
    """Names the source reads without ever binding or importing them, builtins excluded.

    This is the `NameError: name 'np' is not defined` class of failure: the generator is
    told to import what it uses, nothing checked that it had, and the omission only
    surfaced once pytest imported the module.

    `provided` names are treated as already bound. LIB_IMPORT_HEADER really does import
    the library modules at the top of the file this code is written into, so flagging
    `evaluator` as undefined here would force a pointless regeneration of code that runs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []          # a syntax gate handles this case; don't double-report.

    collector = _BindingCollector()
    collector.visit(tree)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    builtins_ns = set(dir(builtins)) | {"__name__", "__file__", "__doc__"} | set(provided)
    return sorted(used - collector.bound - builtins_ns)


def _wrong_module_imports(code: str, lib_dir: str) -> list[tuple[str, str]]:
    """Names imported `from <real library module> import <name>` where <name> is not
    actually defined in that module — e.g. `from tsn_scheduling_problem import
    ScheduleEncodingType`, a real name that only lives in evaluator.py.

    `_undefined_names` cannot catch this class of mistake: the name IS bound, by the
    import statement itself, so nothing looks missing until the import actually runs and
    raises `ImportError`, which aborts collection of the whole generated module — every
    test fails at once, before any of them get a chance to run, and the failure a human
    (or an automated retry) sees is buried in a pytest collection traceback instead of
    surfacing here where it's cheap to catch and name precisely.

    Verifies against the REAL modules under `lib_dir` by actually importing them, rather
    than re-deriving each one's public surface by hand — that list drifts the moment the
    library changes, an import never does. A module this can't import, or isn't one of the
    library's own `.py` files at all (stdlib, numpy, generated_impl importing itself), is
    left alone; this only judges imports it can actually check.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []           # a syntax gate handles this case; don't double-report.

    import importlib

    bad: list[tuple[str, str]] = []
    checked_modules: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module_name = node.module.split(".")[0]
        if module_name not in checked_modules:
            if not os.path.isfile(os.path.join(lib_dir, f"{module_name}.py")):
                checked_modules[module_name] = None      # not a library module; skip it
            else:
                if lib_dir not in sys.path:
                    sys.path.insert(0, lib_dir)
                try:
                    checked_modules[module_name] = importlib.import_module(module_name)
                except Exception:
                    checked_modules[module_name] = None  # module itself is broken; a
                                                           # different problem, not this one
        mod = checked_modules[module_name]
        if mod is None:
            continue
        for alias in node.names:
            if alias.name != "*" and not hasattr(mod, alias.name):
                bad.append((alias.name, module_name))
    return bad


def _autofix_wrong_module_imports(code: str, lib_dir: str) -> str:
    """Deterministically move each name `_wrong_module_imports` flags to the real
    library module that actually defines it — the same fix an LLM retry is asked to
    make, applied directly instead of hoping the retry complies.

    This exists because the retry doesn't reliably work: across repeated runs, asking
    the model to fix `from tsn_scheduling_problem import (..., ScheduleEncodingType)`
    produced the exact same bundling again, or moved it to an equally-wrong module
    (`from schedule import Schedule, Slot, ScheduleEncodingType`), and once that one
    retry attempt fails, nothing else in the pipeline — not the interactive test-review
    loop (which only ever regenerates tests, never the frozen implementation) and not
    always the post-approval repair loop either — gets another chance to touch it. A
    name either does or doesn't exist on a given real module; that's checked by
    actually importing it, not guessed, so there's no reason this needs an LLM at all.

    Edits are applied to the exact original source line range of each offending
    `ImportFrom` node (via its `lineno`/`end_lineno`), not by re-serializing the whole
    file, so everything else — comments, formatting, unrelated code — is left
    byte-identical. Returns `code` unchanged if nothing needed fixing, or if a flagged
    name can't be found in any other library module either (left for the LLM-retry /
    undefined-name path, which already ran before this and may have addressed it).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    import importlib

    module_cache: dict[str, object] = {}

    def load_module(name: str):
        if name not in module_cache:
            if not os.path.isfile(os.path.join(lib_dir, f"{name}.py")):
                module_cache[name] = None
            else:
                if lib_dir not in sys.path:
                    sys.path.insert(0, lib_dir)
                try:
                    module_cache[name] = importlib.import_module(name)
                except Exception:
                    module_cache[name] = None
        return module_cache[name]

    lib_module_names = sorted(
        f[:-3] for f in os.listdir(lib_dir) if f.endswith(".py") and f != "__init__.py"
    )

    def alias_text(alias: ast.alias) -> str:
        return alias.name if not alias.asname else f"{alias.name} as {alias.asname}"

    edits: list[tuple[int, int, str]] = []   # (start_line0, end_line0_exclusive, text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module_name = node.module.split(".")[0]
        mod = load_module(module_name)
        if mod is None:
            continue
        keep: list[str] = []
        moved: dict[str, list[str]] = {}
        for alias in node.names:
            if alias.name == "*" or hasattr(mod, alias.name):
                keep.append(alias_text(alias))
                continue
            correct_module = next(
                (c for c in lib_module_names
                 if c != module_name and (m := load_module(c)) is not None
                 and hasattr(m, alias.name)),
                None,
            )
            if correct_module:
                moved.setdefault(correct_module, []).append(alias_text(alias))
            else:
                keep.append(alias_text(alias))  # nowhere better found; leave as-is
        if not moved:
            continue
        replacement = []
        if keep:
            replacement.append(f"from {module_name} import {', '.join(keep)}")
        for correct_module, names in moved.items():
            replacement.append(f"from {correct_module} import {', '.join(names)}")
        edits.append((node.lineno - 1, node.end_lineno, "\n".join(replacement) + "\n"))

    if not edits:
        return code

    lines = code.splitlines(keepends=True)
    for start, end, replacement in sorted(edits, key=lambda e: -e[0]):
        lines[start:end] = [replacement]
    return "".join(lines)


# ── Deterministic reuse-decision resolution ──────────────────────────────────────
# The blueprint says WHICH codebase member to reuse; everything below works out HOW to
# call it, from the library's own source, so the model is never asked to recall a
# signature it merely saw in retrieved context. A recalled signature can only agree with
# the real one or silently contradict it, and the contradiction is not caught until the
# implementation crashes several stages later — confirmed on a real run, whose blueprint
# confidently prescribed a three-argument stream-update call and a conflict check said to
# return "a matrix", neither of which the real library has.
#
# Read from the AST rather than by importing and using inspect.signature, which is NOT
# equivalent here: this library's own @require_instance decorator does not use
# functools.wraps, so inspect reports every method it touches as (self, *args, **kwargs)
# — e.g. TSNScheduling.hops_costs, whose real signature is (self, frame_size). The AST
# carries the true parameter list and the decorator names both, and cannot be fooled by a
# wrapper.

# What to call a live instance of each class in a rendered call form. These are the names
# the rest of the prompts already use for these objects, so a rendered call reads like the
# surrounding guidance rather than introducing a second vocabulary.
_REUSE_INSTANCE_HINTS = {
    "TSNScheduling": "problem",
    "TSNConstraintEvaluator": "evaluator",
    "Usecase": "usecase",
    "Schedule": "schedule",
    "Slot": "slot",
    "InstancesLoader": "loader",
}

_LIBRARY_INDEX_CACHE: dict = {}

# Library objects that already exist by the time any generated function can reach them,
# and must therefore never be constructed by hand. Naming one of these as a reuse
# decision is not wrong — it IS the thing to reuse — but the call form is an attribute
# read, not a constructor, so `_resolve_reuse_decision` answers with the expression that
# reaches the existing one.
#
# Confirmed necessary on the run after the interface-freeze fix landed: with the real
# classes finally imported, the implementation went on to build its own evaluator with
# `TSNConstraintEvaluator(problem, check_routing=True)`, outside the activation block
# (every attribute its __init__ reads is instance-gated, so that raises) and with the
# default PRIORITY encoding against a start-time chromosome (so every metric it did
# return would be read wrong). Both faults disappear if the already-built one is used.
_LIBRARY_PROVIDED_INSTANCES = {
    "TSNConstraintEvaluator": {
        "expression": "problem.evaluator",
        # What to answer instead when the configuration has already settled a mode the
        # pre-built instance does not have. Without this the resolver hands a START_TIME
        # design the PRIORITY object as its authoritative call form, contradicting the
        # configuration's own environment section in the same document.
        "mode_overrides": {
            "START_TIME": "TSNConstraintEvaluator("
                          "problem, scheduling_type=ScheduleEncodingType.START_TIME)",
        },
        # Shown instead of `note` once an override has fired, i.e. once the encoding is
        # settled and the call form above already reflects it. The WRONG/RIGHT comparison
        # is then answering a question nobody still has; these constraints are not.
        "note_decided": (
            "The encoding is already decided and the call form above reflects it, so the "
            "choice of evaluator is settled. Two constraints still bind:\n"
            "  - Construct it INSIDE `with problem.use_instance(usecase):` — every "
            "attribute its __init__ reads off the problem is instance-gated and raises "
            "outside the block.\n"
            "  - Pass check_routing and check_task_placement as the configuration's "
            "evaluator step states, alongside scheduling_type. A constructed evaluator "
            "does NOT derive them: only initialize_problem calls get_task, and only for "
            "problem.evaluator. Left out, they take the constructor's defaults, "
            "check_routing=True and check_task_placement=True.\n"
            "And one library fact to stay consistent with: "
            "Schedule.from_compact_encoding and Schedule.to_compact_encoding are "
            "START_TIME in BOTH directions (a nonzero cell is start_ns plus one). There is "
            "only one chromosome format in this library and it is start-time."
        ),
        "note": (
            "ONE ALREADY EXISTS, and which one you want depends on the encoding.\n"
            "  WRONG, for any design that reasons about start times or windows:\n"
            "      evaluator = problem.evaluator\n"
            "      # problem.evaluator is ALWAYS PRIORITY: initialize_problem builds it\n"
            "      # without passing scheduling_type. Under PRIORITY the only scheduling\n"
            "      # checks are priority conflicts and priority latency — the four\n"
            "      # START_TIME checks (window sequence, macro-cycle overlap, talker\n"
            "      # offset, start-time latency) DO NOT RUN, and a chromosome of start\n"
            "      # times is read as queue priorities, so timing violations score zero.\n"
            "  RIGHT, for a start-time design:\n"
            "      with problem.use_instance(usecase):\n"
            "          evaluator = TSNConstraintEvaluator(\n"
            "              problem, scheduling_type=ScheduleEncodingType.START_TIME)\n"
            "  RIGHT, for a queue-priority design:\n"
            "      with problem.use_instance(usecase):\n"
            "          evaluator = problem.evaluator\n"
            "KNOW THIS ABOUT THE LIBRARY, because it contradicts itself and cannot\n"
            "notice: Schedule.from_compact_encoding and Schedule.to_compact_encoding are\n"
            "START_TIME in BOTH directions — a nonzero cell is start_ns plus one. There\n"
            "is only one chromosome format here and it is start-time. So if you use\n"
            "either converter anywhere, `problem.evaluator` CANNOT read your arrays and\n"
            "you must build a START_TIME evaluator yourself. Nothing outside schedule.py\n"
            "touches a compact encoding, so the library never exercises both halves\n"
            "together and never catches the mismatch for you.\n"
            "Never construct one outside the activation block — every attribute its\n"
            "__init__ reads off the problem is instance-gated and raises there. And do "
            "not rebuild one merely to set check_routing / check_task_placement: those "
            "ARE derived from the use case's config by get_task, and problem.evaluator "
            "already has them right."
        ),
    },
}


def _decorator_name(node) -> str:
    """The bare name of a decorator node — `property`, `classmethod`, `require_instance`
    — whether it is written plain, dotted, or called with arguments."""
    while isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _signature_text(args: ast.arguments, drop_first: bool) -> str:
    """The parameter list as written in the library, optionally without its leading
    `self`/`cls` — which is bound automatically and must never appear at a call site.

    This is the REFERENCE form, annotations and defaults included. It is deliberately not
    the call form: `validate_conflicts(touching_allowed: bool=True)` documents a
    signature and is not a call anyone can write. `_call_shape` below renders that.

    Defaults stay aligned when the first parameter is dropped because Python aligns
    defaults to the END of the parameter list, and a leading self/cls never carries one.
    """
    import copy as _copy

    args = _copy.deepcopy(args)
    if drop_first:
        if args.posonlyargs:
            args.posonlyargs.pop(0)
        elif args.args:
            args.args.pop(0)
    return f"({ast.unparse(args)})"


def _call_shape(args: ast.arguments, drop_first: bool) -> dict:
    """What a CALL to this member must and may supply, by parameter name.

    Separated from `_signature_text` because the two answer different questions and
    conflating them produces a call form nobody can paste anywhere: what a caller writes
    is `validate_conflicts(touching_allowed=False)`, not the signature's own
    `touching_allowed: bool=True`. Also what makes it possible to tell a WRONG proposed
    call from a merely differently-worded one — a call passing an optional argument the
    signature text happens not to show is correct, and must not be "corrected".
    """
    positional = list(args.posonlyargs) + list(args.args)
    if drop_first and positional:
        positional.pop(0)      # self / cls, bound by Python, never written at a call site
    # Defaults align to the END of the positional list, so whatever is left after
    # removing len(defaults) entries from the tail is what a caller MUST supply.
    cut = len(positional) - len(args.defaults) if args.defaults else len(positional)
    required = [a.arg for a in positional[:cut]]
    optional = [a.arg for a in positional[cut:]]
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        (optional if default is not None else required).append(arg.arg)
    return {
        "required": required,
        "optional": optional,
        "varargs": args.vararg is not None,
        "kwargs": args.kwarg is not None,
    }


def _proposed_call_is_consistent(proposed: str, kind: str, shape: dict) -> bool:
    """Whether a call form the model wrote could actually work against the real member.

    Used to decide what counts as a CORRECTION worth reporting. A plain string comparison
    against the rendered call form would flag almost everything, including calls that are
    perfectly fine — passing an optional argument, or naming a variable differently — and
    a correction count that fires on correct answers is worse than none at all, since the
    whole reason to record corrections is to measure how often a model actually gets this
    wrong.
    """
    try:
        expression = ast.parse(proposed.strip(), mode="eval").body
    except SyntaxError:
        return False

    if kind == "provided_instance":
        # Reading the existing object is always right. Building one is right ONLY when
        # it is being built differently on purpose — for the evaluator that means
        # passing scheduling_type, since the stored instance is hardwired to PRIORITY
        # and nothing else about it can be improved by hand. A bare rebuild is a
        # redundant copy of what the library already made.
        if not isinstance(expression, ast.Call):
            return True
        return any(kw.arg == "scheduling_type" for kw in expression.keywords)
    if kind == "property":
        # A property is an attribute. Calling it invokes whatever it returned.
        return not isinstance(expression, ast.Call)
    if not isinstance(expression, ast.Call):
        return False

    positional = sum(1 for a in expression.args if not isinstance(a, ast.Starred))
    if any(isinstance(a, ast.Starred) for a in expression.args) or \
            any(k.arg is None for k in expression.keywords):
        return True        # dynamic unpacking; length is not knowable statically
    named = {k.arg for k in expression.keywords}
    if not shape["kwargs"] and not named <= set(shape["required"] + shape["optional"]):
        return False
    supplied = positional + len(named)
    if supplied < len(shape["required"]):
        return False
    if not shape["varargs"] and not shape["kwargs"]:
        if supplied > len(shape["required"]) + len(shape["optional"]):
            return False
    return True


def _library_member_index(lib_dir: str) -> dict:
    """Every module-level function and every class (nested ones included) in the library,
    with each method's decorator names and real parameter list.

    Built once per directory and cached: the blueprint's reuse decisions are resolved in
    one pass, and re-parsing six library files per decision would be pure waste.
    """
    if lib_dir in _LIBRARY_INDEX_CACHE:
        return _LIBRARY_INDEX_CACHE[lib_dir]

    def class_entry(node: ast.ClassDef, qualname: str) -> dict:
        methods: dict = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # First definition wins, so `@property def x` is what a later
                # `@x.setter def x` does not overwrite.
                methods.setdefault(item.name, item)
        return {"node": node, "qualname": qualname, "methods": methods}

    index: dict = {}
    for filename in sorted(os.listdir(lib_dir)):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        module_name = filename[:-3]
        try:
            with open(os.path.join(lib_dir, filename), encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue     # an unreadable library file is a different problem, not this one
        module: dict = {"functions": {}, "classes": {}}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module["functions"][node.name] = node
            elif isinstance(node, ast.ClassDef):
                module["classes"][node.name] = class_entry(node, node.name)
                # Nested classes are reachable only through their owner
                # (TSNConstraintEvaluator.UnionFind, Schedule.TransmissionWindow), and
                # the call form has to say so — hence the dotted qualname.
                for item in node.body:
                    if isinstance(item, ast.ClassDef):
                        module["classes"][item.name] = class_entry(
                            item, f"{node.name}.{item.name}"
                        )
        index[module_name] = module

    _LIBRARY_INDEX_CACHE[lib_dir] = index
    return index


def _resolve_reuse_decision(name: str, class_name: str, lib_dir: str,
                            encoding: Optional[str] = None,
                            opens_block: Optional[bool] = None) -> dict:
    """Work out how one named library member is actually called.

    Returns a dict with "ok": True plus the resolved kind, signature, import line and
    call form — or "ok": False and a "problem" saying why the member could not be found,
    which is itself the check that a reuse decision names something real. A blueprint
    prescribing a function the library does not have is a hallucination that used to
    survive all the way to a runtime crash; here it costs nothing to catch.

    `opens_block` says whether the function that will make this call is the one that owns
    the activation block. It only matters for an instance-gated member: True (or None,
    meaning "no call graph to decide from") puts the `with` block in the call form, False
    returns a "precondition" instead and leaves the call form bare. See the requires_
    instance branch below for why the unconditional block was wrong.
    """
    index = _library_member_index(lib_dir)
    name = (name or "").strip()
    class_name = (class_name or "").strip()
    if not name:
        return {"ok": False, "problem": "no member name given"}

    # Be forgiving about the one shape the prompt asks NOT to be used: a model that
    # writes "TSNScheduling.hops_costs" as a single string has still identified the
    # member unambiguously, and rejecting that would throw away a correct answer.
    if "." in name:
        head, _, tail = name.rpartition(".")
        class_name = class_name or head.rpartition(".")[2]
        name = tail

    def find_class(cls_name: str) -> Optional[tuple[str, dict]]:
        for module_name, module in index.items():
            if cls_name in module["classes"]:
                return module_name, module["classes"][cls_name]
        return None

    def render(module_name: str, kind: str, args: Optional[ast.arguments],
               drop_first: bool, *, owner: str = "",
               requires_instance: bool = False) -> dict:
        shape = (_call_shape(args, drop_first) if args is not None
                 else {"required": [], "optional": [], "varargs": False, "kwargs": False})
        # A property has no parameter list at all, which is not the same as an empty one:
        # writing it as "()" is exactly the misreading that produces `problem.n_ports()`.
        signature = ("" if kind == "property"
                     else _signature_text(args, drop_first) if args is not None else "()")
        # What a caller WRITES: the required parameters by name, nothing else. The full
        # signature travels separately, as reference rather than as something to copy.
        call_args = f"({', '.join(shape['required'])})"
        owner_head = owner.partition(".")[0]
        reached_via = ""

        if kind == "function":
            call_form = f"{name}{call_args}"
            import_line = f"from {module_name} import {name}"
        elif kind == "constructor":
            call_form = f"{owner}{call_args}"
            import_line = f"from {module_name} import {owner_head}"
        elif kind in ("classmethod", "staticmethod"):
            call_form = f"{owner}.{name}{call_args}"
            import_line = f"from {module_name} import {owner_head}"
        else:
            instance = _REUSE_INSTANCE_HINTS.get(
                owner.rpartition(".")[2], owner.rpartition(".")[2].lower() or "instance"
            )
            call_form = (f"{instance}.{name}" if kind == "property"
                         else f"{instance}.{name}{call_args}")
            # An instance member is reached through an object this design already holds,
            # so there is no import to perform — naming one would invite constructing a
            # second, unrelated instance just to reach the method.
            import_line = ""
            reached_via = (f"a {owner} object, passed in as a parameter of your "
                           f"function. Do NOT construct one here — the caller "
                           f"supplies it. (class defined in {module_name}.py)")

        resolved = {
            "ok": True, "name": name, "module": module_name, "class_name": owner,
            "kind": kind, "signature": signature, "import": import_line,
            "reached_via": reached_via, "call_form": call_form,
            "requires_instance": requires_instance, "shape": shape,
        }
        if requires_instance:
            # These members raise ValueError("No current instance is set!") outside the
            # activation block, so the block is part of how they are reached — but WHICH
            # function writes it is not a property of the member. The blueprint prompt's
            # lifecycle rule says exactly one function opens a block, the top-level
            # orchestrator, and that helpers called inside it read gated members directly
            # (nested blocks are unsafe: leaving the inner one sets the use case back to
            # None and the outer block's remaining work raises).
            #
            # This used to prepend the block unconditionally, which handed the
            # implementation stage a worked example of a HELPER opening its own block —
            # contradicting, from inside the same run, the rule the design stage was given.
            # Confirmed on a real run: TSNScheduling.macro_cycle, used by
            # `validate_schedule`, resolved to a `with` block although validate_schedule is
            # not the orchestrator.
            #
            # opens_block carries the caller's answer: True for the one function that owns
            # the block, False for a helper running inside it, None when there is no call
            # graph to decide from (the configuration stage, whose environment step DOES
            # open it) — and None keeps the original behaviour.
            instance = _REUSE_INSTANCE_HINTS.get(owner.rpartition(".")[2], "problem")
            if opens_block is False:
                resolved["precondition"] = (
                    f"called with an active use case: the orchestrator has already "
                    f"entered `with {instance}.use_instance(usecase):` around this call. "
                    f"Read the member directly and do NOT open a block of your own — a "
                    f"nested block clears the use case when it exits and the "
                    f"orchestrator's remaining work then raises ValueError."
                )
            else:
                resolved["call_form"] = (
                    f"with {instance}.use_instance(usecase):\n    {resolved['call_form']}"
                )
        return resolved

    def from_method(module_name: str, cls: dict, method: ast.FunctionDef) -> dict:
        decorators = {_decorator_name(d) for d in method.decorator_list}
        requires_instance = "require_instance" in decorators
        if "classmethod" in decorators:
            return render(module_name, "classmethod", method.args, True,
                          owner=cls["qualname"])
        if "staticmethod" in decorators:
            return render(module_name, "staticmethod", method.args, False,
                          owner=cls["qualname"])
        if "property" in decorators:
            return render(module_name, "property", None, False, owner=cls["qualname"],
                          requires_instance=requires_instance)
        return render(module_name, "instance_method", method.args, True,
                      owner=cls["qualname"], requires_instance=requires_instance)

    def constructor(module_name: str, cls: dict) -> dict:
        # The one choke point for "something wants to build one of these", so the
        # already-built check belongs here rather than at each of the two call sites.
        provided = _LIBRARY_PROVIDED_INSTANCES.get(cls["qualname"])
        if provided:
            override = (provided.get("mode_overrides") or {}).get(
                (encoding or "").strip().upper()
            )
            init = cls["methods"].get("__init__")
            return {
                "ok": True, "name": cls["qualname"], "module": module_name,
                "class_name": cls["qualname"],
                "kind": "constructor" if override else "provided_instance",
                # When an override fires the answer IS a construction, so show the real
                # __init__: it is the line that makes the override self-explanatory,
                # because it shows scheduling_type exists and defaults to PRIORITY.
                "signature": (_signature_text(init.args, True)
                              if override and init is not None else ""),
                "import": "",
                # Once the encoding is decided the WRONG/RIGHT comparison is moot — the
                # choice has been made for this design — but the standing constraints
                # still bind, so the trimmed note keeps those and drops the fifteen
                # lines of alternatives.
                "reached_via": (provided.get("note_decided") or provided["note"])
                               if override else provided["note"],
                "call_form": override or provided["expression"],
                "requires_instance": True,
                "shape": {"required": [], "optional": [], "varargs": False,
                          "kwargs": False},
            }
        init = cls["methods"].get("__init__")
        return render(module_name, "constructor", init.args if init else None, True,
                      owner=cls["qualname"])

    if class_name:
        found = find_class(class_name)
        if found is None:
            return {"ok": False,
                    "problem": f"no class named '{class_name}' exists in the library"}
        module_name, cls = found
        if name in (class_name, "__init__"):
            return constructor(module_name, cls)
        method = cls["methods"].get(name)
        if method is not None:
            return from_method(module_name, cls, method)
        return {"ok": False,
                "problem": f"'{class_name}' has no member named '{name}'"}

    # No owning class given: a module-level function, a class used as a constructor, or a
    # method the model forgot to attribute — resolve the first two outright, and infer the
    # third only when exactly one class could have meant it.
    for module_name, module in index.items():
        if name in module["functions"]:
            return render(module_name, "function",
                          module["functions"][name].args, False)
    for module_name, module in index.items():
        if name in module["classes"]:
            return constructor(module_name, module["classes"][name])

    owners = [(module_name, cls)
              for module_name, module in index.items()
              for cls in module["classes"].values() if name in cls["methods"]]
    if len(owners) == 1:
        module_name, cls = owners[0]
        return from_method(module_name, cls, cls["methods"][name])
    if len(owners) > 1:
        return {"ok": False,
                "problem": f"'{name}' is a member of {len(owners)} different classes "
                           f"({', '.join(sorted(c['qualname'] for _, c in owners))}) — "
                           f"the reuse decision did not say which"}
    return {"ok": False, "problem": f"nothing named '{name}' exists in the library"}


def _resource_use_violations(blueprint: dict) -> list[str]:
    """Is every requirements-stage resource accounted for in the blueprint's resource_use?

    Each ledger resource must appear exactly once: either "used_as" names elements that
    exist in this blueprint (a data structure, a function, a constant, or an enum value
    written Type.VALUE), or it names none and "reason" says why. Reuse itself is the
    design's choice; leaving a name unaccounted for is what is reported.
    """
    findings: list[str] = []
    names = [str(r["name"]) for r in _REQUIREMENT_RESOURCES]
    if not names:
        return findings
    entries = [e for e in (blueprint.get("resource_use") or []) if isinstance(e, dict)]
    structures = {str(s.get("name")) for s in (blueprint.get("data_structures") or [])
                  if isinstance(s, dict)}
    structure_types = {str(s.get("name")): s.get("type") for s in (blueprint.get("data_structures") or [])
                       if isinstance(s, dict)}
    corresponds = {str(s.get("corresponds_to")).strip() for s in (blueprint.get("data_structures") or [])
                   if isinstance(s, dict) and s.get("corresponds_to")}
    functions = {str(f.get("name")) for f in (blueprint.get("function_inventory") or [])
                 if isinstance(f, dict)}
    constants = {str(c.get("name")): c for c in (blueprint.get("constants") or [])
                 if isinstance(c, dict)}
    seen: dict = {}
    for entry in entries:
        resource = str(entry.get("resource") or "")
        seen[resource] = seen.get(resource, 0) + 1
        if resource not in names:
            findings.append(f"resource_use names '{resource}', which is not a ledger resource")
            continue
        used = [str(u) for u in (entry.get("used_as") or []) if str(u).strip()]
        if not used and not str(entry.get("reason") or "").strip():
            findings.append(f"resource {resource} is used by nothing and gives no reason")
        for element in used:
            if element in structures or element in functions or element in constants:
                continue
            owner, _, value = element.partition(".")
            # Type.VALUE of an enum named Type, or of one named Type.field.
            enum_values = [v for name, c in constants.items()
                           if name == owner or name.startswith(owner + ".")
                           for v in (c.get("values") or [])]
            if value and value in enum_values:
                continue
            # The library value a declared structure stands for, written as its own
            # corresponds_to ("Usecase.streams[i]" for a stream structure): that structure.
            if element in corresponds:
                continue
            # Structure.field: a field of a declared structure, named in its type.
            if value and owner in structures and re.search(
                    rf"\b{re.escape(value)}\b", str(structure_types.get(owner) or "")):
                continue
            findings.append(f"resource {resource} is used_as '{element}', which is not a "
                            f"structure, a field of one, a function, a constant or an "
                            f"enum value of this blueprint")
    for name in names:
        if seen.get(name, 0) == 0:
            findings.append(f"resource {name} is missing from resource_use")
        elif seen[name] > 1:
            findings.append(f"resource {name} appears {seen[name]} times in resource_use")
    return findings


def _carry_evaluator_call(blueprint: dict, configuration: Optional[dict]) -> list[str]:
    """The top-level function builds the evaluator "exactly as the configuration's
    environment steps say", and the implementation stage builds what that function's
    purpose says. Models paraphrase it ("construct the START_TIME evaluator"), losing the
    problem argument and the flags, so the configuration's own call is appended to the
    purpose of the function that opens the activation block when it is not already there."""
    calls = [str(e.get("obtained_by") or "") for e in (configuration or {}).get("environment") or []
             if isinstance(e, dict) and e.get("type") == "TSNConstraintEvaluator"]
    call = next((c for c in calls if "TSNConstraintEvaluator(" in c), None)
    if not call:
        return []
    exact = call[call.index("TSNConstraintEvaluator("):]
    exact = exact[:exact.index(")") + 1] if ")" in exact else exact
    fns = [f for f in blueprint.get("function_inventory") or [] if isinstance(f, dict)]
    called = {str(c) for f in fns for c in (f.get("calls") or [])}
    tops = [f for f in fns if str(f.get("name")) not in called] or fns[:1]
    notes = []
    # The activation block the same function opens, as the configuration's InstanceContext
    # step writes it (Luna lb8 copied the use case step's text into that place instead).
    acts = [str(e.get("obtained_by") or "") for e in (configuration or {}).get("environment") or []
            if isinstance(e, dict) and e.get("type") == "InstanceContext"]
    act = next((m.group(0) for a in acts for m in [re.search(r"with\s+\w+\.use_instance\(\w+\):", a)] if m), None)
    for fn in tops[:1]:
        purpose = str(fn.get("purpose") or "")
        if act and act.replace(" ", "") not in purpose.replace(" ", ""):
            purpose = f"Opens `{act}` once around all its work. " + purpose
            notes.append(f"`{fn.get('name')}`: added the configuration's activation `{act}`")
        if exact.replace(" ", "") not in purpose.replace(" ", ""):
            purpose = purpose.rstrip(". ") + f". Builds the evaluator as the configuration states: {exact}."
            notes.append(f"`{fn.get('name')}`: appended the configuration's evaluator call {exact}")
        fn["purpose"] = purpose
    return notes


def _enums_from_literal_types(blueprint: dict) -> list[str]:
    """The blueprint's enum constants, read off its own Literal[...] types, in place.

    The prompt no longer asks for a "constants" section (2026-10-26): its enum entries only
    repeated the Literal types the structures already carry, and its formulas restated the
    purposes and could contradict them (Luna lb7: penalty "1 + the total windows" in the
    purpose, "3 * interrupted windows" in the constant). Later checks still read enum values
    from "constants" - a structure typed Literal[...] and a record field `f: Literal[...]`
    become entries named Structure and Structure.f. Existing entries are kept."""
    constants = blueprint.setdefault("constants", [])
    if not isinstance(constants, list):
        return []
    have = {str(c.get("name")) for c in constants if isinstance(c, dict)}
    added: list[str] = []
    def values(body: str) -> list[str]:
        return re.findall(r"""['"]([^'"]+)['"]""", body)
    for st in blueprint.get("data_structures") or []:
        if not isinstance(st, dict):
            continue
        name, typ = str(st.get("name") or ""), str(st.get("type") or "")
        whole = re.fullmatch(r"\s*Literal\[(.*)\]\s*", typ, re.S)
        found = [(name, values(whole.group(1)))] if whole else [
            (f"{name}.{field}", values(body))
            for field, body in re.findall(r"(\w+)\s*:\s*Literal\[([^\]]*)\]", typ)]
        for key, vals in found:
            if vals and key not in have:
                constants.append({"name": key, "kind": "enum", "values": vals})
                have.add(key)
                added.append(key)
    return added


def _form_ids_as_types(blueprint: dict) -> list[str]:
    """A PART 2 form id written where a Python type belongs - a structure's "type", or an
    annotation in a signature ("StartSchedule: compiled_schedule") - is that form's own type,
    in place. The id is the configuration's name for the form, not something the frozen
    interface can import or annotate with: given it, Luna's interfaces (2026-10-26) dropped
    the input structures and annotated with an undefined name. "corresponds_to" keeps the id.
    Only forms whose codes all name one type are rewritten (np.ndarray, Schedule)."""
    forms = {}
    for rep in _LIBRARY_FIXED_REPRESENTATIONS:
        codes = rep.get("codes") or ()
        if codes and len({c.rsplit(".", 1)[-1] for c in codes}) == 1:
            forms[rep["id"]] = codes[-1]
    notes: list[str] = []
    def sub(text: str) -> str:
        for form, code in forms.items():
            text = re.sub(rf"\b{re.escape(form)}\b(?!\s*\[:)", code, text)
        return text
    for st in blueprint.get("data_structures") or []:
        if isinstance(st, dict) and isinstance(st.get("type"), str):
            new = sub(st["type"])
            if new != st["type"]:
                notes.append(f"structure {st.get('name')}: type '{st['type'][:60]}' -> '{new[:60]}'")
                st["type"] = new
    for fn in blueprint.get("function_inventory") or []:
        if isinstance(fn, dict) and isinstance(fn.get("signature"), str):
            new = sub(fn["signature"])
            if new != fn["signature"]:
                notes.append(f"function {fn.get('name')}: form id in its signature written as its type")
                fn["signature"] = new
    return notes


def _derive_structure_flow(blueprint: dict) -> None:
    """Fill each structure's "written_by" / "read_by" from the functions' own "writes" /
    "reads", in place. The prompt asks for direction once, on the functions; asking for
    it on the structures too only gave the model a second copy to contradict (45
    disagreements in one Luna blueprint). Later stages still read both fields."""
    structures = [s for s in blueprint.get("data_structures") or [] if isinstance(s, dict)]
    names = {str(s.get("name")) for s in structures}
    flow = {n: {"written_by": [], "read_by": []} for n in names}
    # "Structure.field" in reads/writes, where the field is named in that structure's type,
    # is the structure itself: the flow is per structure, so the field part is dropped.
    types = {str(s.get("name")): str(s.get("type") or "") for s in structures}
    for f in blueprint.get("function_inventory") or []:
        if not isinstance(f, dict):
            continue
        for key in ("reads", "writes"):
            if not isinstance(f.get(key), list):
                continue
            out = []
            for x in f[key]:
                owner, _, field = str(x).partition(".")
                if field and owner in names and re.search(rf"\b{re.escape(field)}\b", types[owner]):
                    x = owner
                if x not in out:
                    out.append(x)
            f[key] = out
    # A structure named inside another's type (an ordering's entry, a plan's step) is
    # built and read with it: whoever writes or reads the container does the same to it.
    contains = {n: {m for m in names if m != n and re.search(rf"\b{re.escape(m)}\b", types[n])}
                for n in names}
    for _ in range(len(names)):
        grown = False
        for n in names:
            more = set().union(*(contains[m] for m in contains[n])) - contains[n] - {n} if contains[n] else set()
            if more:
                contains[n] |= more
                grown = True
        if not grown:
            break
    for f in blueprint.get("function_inventory") or []:
        if not isinstance(f, dict):
            continue
        for key in ("reads", "writes"):
            if isinstance(f.get(key), list):
                for x in list(f[key]):
                    for m in sorted(contains.get(str(x), ())):
                        if m not in f[key]:
                            f[key].append(m)
    # A caller receives what the functions it calls return: their "writes" are its "reads".
    # Models keep leaving these out (a check's result "read by nobody" although its caller
    # uses it), so the edge is added here rather than asked for once more.
    fns = {str(f.get("name")): f for f in blueprint.get("function_inventory") or [] if isinstance(f, dict)}
    for fn in fns.values():
        reads = [str(r) for r in (fn.get("reads") or [])]
        for callee in fn.get("calls") or []:
            for target in (fns.get(str(callee)) or {}).get("writes") or []:
                if str(target) in names and str(target) not in reads:
                    reads.append(str(target))
        fn["reads"] = reads
    for fn in blueprint.get("function_inventory") or []:
        if not isinstance(fn, dict):
            continue
        for key, field in (("writes", "written_by"), ("reads", "read_by")):
            for target in fn.get(key) or []:
                if str(target) in flow and fn.get("name") not in flow[str(target)][field]:
                    flow[str(target)][field].append(fn.get("name"))
    for st in structures:
        st.update(flow[str(st.get("name"))])


def _structure_flow_violations(blueprint: dict,
                               configuration: Optional[dict] = None) -> list[str]:
    """Which structures nothing produces, which nothing consumes, and where the design
    contradicts itself about what flows where.

    Direction is declared TWICE, deliberately: each structure names its "written_by" and
    "read_by", and each function names its own "reads" and "writes". Those are the same
    edges seen from opposite ends, so the duplication is only worth its cost if the two
    are compared — otherwise it is simply two records free to disagree. They are
    compared here, and a disagreement is reported as a finding in its own right: a
    design that lists a function as a structure's writer while that function does not
    claim to write it has not settled who produces the value, and that is exactly the
    confusion that produces a value constructed inline in three different places.

    WHY DIRECTION IS NEEDED AT ALL, from the one run this was derived against by hand:
    every "calls" list in that design was CORRECT, and four structures were still
    mis-wired, because "calls" records who invokes whom and never what flows between
    them.

      nothing reads it   — produced and dropped. MigrationStep was returned by two
                           functions and named in no parameter list anywhere, so the
                           intermediate configurations the query actually asks for were
                           built and consumed by nothing, while validate_plan's purpose
                           line claimed it "validates every intermediate state".
      nothing writes it  — appears from nowhere. ScheduleState was read by three
                           functions and returned by none, and in the generated code it
                           ended up constructed inline in three separate places because
                           the design never said who owned making one.

    Both at once is why this does not reuse v9's "a structure with no used_by is not
    part of this design — delete it". FRERState was touched in neither direction yet was
    not dead: it carried the FRER phase, the heart of that query, and survived as
    untracked locals inside one function. "Delete it" would have been the wrong fix.
    Unowned and unused look identical without direction and need opposite repairs.

    The emptiness tests take the UNION of the two views, so a structure reached by
    either one is not reported as unreached — an edge declared on one side only is a
    disagreement, which is already reported, and not an absence.

    THE ONE EXEMPTION, checked rather than assumed. A structure the environment hands in
    is legitimately produced by no function in the inventory — Usecase, TSNScheduling
    and TSNConstraintEvaluator all are. That holds exactly when "corresponds_to" names a
    library type AND the configuration obtained that type, so both are tested and the
    configuration's environment steps are read for the second. A structure of the
    design's own invention (corresponds_to null) is never exempt.

    Returns one string per finding. Silent when neither side declares anything, which is
    what the plain-text fallback produces: reporting every structure there would be
    reporting the parser, not the design.
    """
    structures = blueprint.get("data_structures")
    inventory = blueprint.get("function_inventory")
    if not isinstance(structures, list) or not isinstance(inventory, list):
        return []

    declared = [s for s in structures if isinstance(s, dict) and s.get("name")]
    functions = [f for f in inventory if isinstance(f, dict) and f.get("name")]
    if not declared or not functions:
        return []

    structure_side = any(("written_by" in s) or ("read_by" in s) for s in declared)
    function_side = any(("reads" in f) or ("writes" in f) for f in functions)
    if not structure_side and not function_side:
        return []            # fallback-parsed: the fields were never available

    names = {str(s["name"]) for s in declared}
    fn_names = {str(f["name"]) for f in functions}
    findings: list[str] = []

    def listed(entry: dict, key: str) -> list:
        return [str(x) for x in (entry.get(key) or [])]

    # The function side, inverted to per-structure.
    from_functions: dict = {"writes": {n: set() for n in names},
                            "reads": {n: set() for n in names}}
    # An object the configuration's environment supplies (the evaluator, the problem) is
    # read by name of its library class without being a structure of this design.
    environment_types = {str(step.get("type")).strip()
                         for step in ((configuration or {}).get("environment") or [])
                         if isinstance(step, dict) and step.get("type")}
    for fn in functions:
        fn_name = str(fn["name"])
        for key in ("reads", "writes"):
            for target in listed(fn, key):
                if target in names:
                    from_functions[key][target].add(fn_name)
                elif key == "reads" and target in environment_types:
                    continue
                else:
                    findings.append(
                        f"{fn_name} {key} '{target}', which is not a declared data "
                        f"structure")

    # The structure side, as declared.
    from_structures: dict = {"writes": {}, "reads": {}}
    for structure in declared:
        name = str(structure["name"])
        for key, field in (("writes", "written_by"), ("reads", "read_by")):
            claimed = set()
            for fn_name in listed(structure, field):
                if fn_name in fn_names:
                    claimed.add(fn_name)
                else:
                    findings.append(
                        f"{name}.{field} names '{fn_name}', which is not a function in "
                        f"the inventory")
            from_structures[key][name] = claimed

    # The two views compared, which is the point of declaring both.
    if structure_side and function_side:
        for key, field, verb in (("writes", "written_by", "writes"),
                                 ("reads", "read_by", "reads")):
            for name in sorted(names):
                only_structure = from_structures[key][name] - from_functions[key][name]
                only_function = from_functions[key][name] - from_structures[key][name]
                for fn_name in sorted(only_structure):
                    findings.append(
                        f"{name}.{field} lists {fn_name}, but {fn_name} does not list "
                        f"{name} under \"{verb}\" — the design disagrees with itself "
                        f"about what flows where")
                for fn_name in sorted(only_function):
                    findings.append(
                        f"{fn_name} {verb} {name}, but {name}.{field} does not list "
                        f"{fn_name} — the design disagrees with itself about what flows "
                        f"where")

    # Types the configuration actually obtained, by the type it recorded per step.
    from_environment = set()
    for step in ((configuration or {}).get("environment") or []):
        if isinstance(step, dict) and step.get("type"):
            from_environment.add(str(step["type"]).strip())

    # The value the top-level function RETURNS is read by its caller, and so is every
    # structure reachable through that value's own "type" (a plan's steps, its order).
    # The entry point is the function no other inventory function calls; its return
    # annotation is the text after "->" in its signature. Exempting these from "read by
    # nobody" is what the blueprint prompt states; a structure reached from nowhere is
    # still reported.
    called = {str(c) for fn in functions for c in (fn.get("calls") or [])}
    returned: set = set()
    for fn in functions:
        if str(fn.get("name")) in called:
            continue
        annotation = str(fn.get("signature") or "").rpartition("->")[2]
        returned |= {n for n in names if re.search(rf"\b{re.escape(n)}\b", annotation)}
    declared_types = {str(st["name"]): str(st.get("type") or "") for st in declared}
    frontier = list(returned)
    while frontier:
        inner = declared_types.get(frontier.pop(), "")
        for n in names:
            if n not in returned and re.search(rf"\b{re.escape(n)}\b", inner):
                returned.add(n)
                frontier.append(n)

    for structure in declared:
        name = str(structure["name"])
        corresponds = structure.get("corresponds_to")
        # A library class may be written module-qualified (tsn_scheduling_problem.TSNScheduling)
        # and is still the type the environment step obtained; the index into the handed-in
        # use case's streams, in the wording section 1 prescribes, comes from that use case.
        text = str(corresponds or "").strip()
        leaf = text.rsplit(".", 1)[-1] if re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)+", text) else text
        supplied = bool(corresponds) and (
            text in from_environment or leaf in from_environment
            or (re.search(r"index i into Usecase\.streams", text) is not None
                and "Usecase" in from_environment)
            or (text.startswith("Usecase.") and "Usecase" in from_environment)
            # One stream's part of a form an environment step obtains (candidate_array[:, i],
            # Schedule.slots_by_stream(i)) is read off that input by whoever reads it.
            or any(text == str(rep.get("stream_part")) and rep["id"] in from_environment
                   for rep in _LIBRARY_FIXED_REPRESENTATIONS if rep.get("stream_part")))
        writers = from_structures["writes"].get(name, set()) | from_functions["writes"][name]
        readers = from_structures["reads"].get(name, set()) | from_functions["reads"][name]

        if not writers and not supplied:
            why = (f"it corresponds to {corresponds}, but no environment step obtains "
                   f"that type" if corresponds
                   else "it is new to this design, so some function must produce it")
            findings.append(f"{name} is written by nobody and read by "
                            f"{', '.join(sorted(readers)) or 'nobody'} — {why}")
        if not readers and name not in returned:
            findings.append(f"{name} is read by nobody, though "
                            f"{', '.join(sorted(writers)) or 'nobody'} produces it — "
                            f"either a function that consumes it is missing, or it is "
                            f"not part of this design")
    return findings


def _blueprint_block_owner(blueprint: dict) -> Optional[str]:
    """Which inventory function owns the activation block, read off the design's own call
    graph: the entry point is the function nothing else in the inventory calls.

    The blueprint prompt's lifecycle rule puts the block in exactly one place — the
    top-level orchestration function — so the call graph the design itself declares is
    the authority on which function that is, and no extra field has to be asked for.

    Returns None when the answer is not knowable, which is a real and expected case: the
    configuration stage resolves reuse decisions too and has no function inventory at
    all, and a design declaring several uncalled functions has not settled on one
    orchestrator. Callers must treat None as "do not decide", not as "no block".
    """
    inventory = blueprint.get("function_inventory")
    if not isinstance(inventory, list) or not inventory:
        return None

    names = [str(fn.get("name")) for fn in inventory
             if isinstance(fn, dict) and fn.get("name")]
    called: set = set()
    for fn in inventory:
        if isinstance(fn, dict):
            called.update(str(c) for c in (fn.get("calls") or []))

    roots = [name for name in names if name not in called]
    return roots[0] if len(roots) == 1 else None


def _resolve_blueprint_reuse_decisions(blueprint: dict, lib_dir: str,
                                       encoding: Optional[str] = None) -> tuple[list, list]:
    """Fill every reuse decision's call form in from the real library, in place.

    The model's own wording is preserved as "call_form_as_proposed" whenever it differed
    from the truth, and the corrections are returned rather than silently applied: this
    pipeline exists to compare models across runs, and "how often did it get the call form
    wrong" is exactly the signal that a silent rewrite would destroy.

    Returns (corrections, unresolved) as printable strings.
    """
    corrections: list[str] = []
    unresolved: list[str] = []
    block_owner = _blueprint_block_owner(blueprint)

    for entry in blueprint.get("reuse_decisions", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("codebase_function") or "")
        class_name = str(entry.get("class_name") or "")
        proposed = str(entry.get("call_form") or "").strip()

        # None when there is no single entry point to compare against, which leaves the
        # call form as it was rather than guessing that a helper runs inside a block.
        opens_block = (None if block_owner is None
                       else str(entry.get("used_by") or "") == block_owner)
        resolved = _resolve_reuse_decision(name, class_name, lib_dir, encoding,
                                           opens_block=opens_block)
        if not resolved.get("ok"):
            entry["call_form"] = ""
            entry["unresolved"] = resolved["problem"]
            unresolved.append(
                f"'{class_name + '.' if class_name else ''}{name}' "
                f"(used by {entry.get('used_by', '?')}): {resolved['problem']}"
            )
            continue

        entry["call_form"] = resolved["call_form"]
        entry["signature"] = resolved["signature"]
        entry["kind"] = resolved["kind"]
        entry["module"] = resolved["module"]
        entry["class_name"] = resolved["class_name"]
        entry["requires_instance"] = resolved["requires_instance"]
        # Present only for a gated member used by a helper: the block is the
        # orchestrator's, so what this function needs is the precondition, not the block.
        if resolved.get("precondition"):
            entry["precondition"] = resolved["precondition"]
        else:
            entry.pop("precondition", None)
        if resolved["import"]:
            entry["import"] = resolved["import"]
        else:
            entry.pop("import", None)
            entry["reached_via"] = resolved["reached_via"]
        entry.pop("unresolved", None)
        # Only a call that could not actually work counts as a correction. A model that
        # passed an optional argument, or named its variables differently, wrote a
        # perfectly good call and must not be recorded as having got this wrong.
        if proposed and not _proposed_call_is_consistent(
            proposed, resolved["kind"], resolved["shape"]
        ):
            # Underscore-prefixed because this is reporting-only and must NOT travel into
            # a prompt: an entry carrying both the verified call form and the unworkable
            # one it replaced hands the implementation stage two candidates and an even
            # chance of copying the wrong one. _blueprint_for_prompt strips it.
            entry["_call_form_as_proposed"] = proposed
            corrections.append(
                f"{resolved['name']}: proposed `{proposed}` builds a second one by hand "
                f"-> `{resolved['call_form']}`"
                if resolved["kind"] == "provided_instance" else
                f"{resolved['name']}: proposed `{proposed}` cannot work against the real "
                f"{resolved['kind']} `{resolved['name']}{resolved['signature']}` -> "
                f"`{resolved['call_form']}`"
            )

    return corrections, unresolved


def _blueprint_for_prompt(blueprint: Optional[dict]) -> dict:
    """The blueprint as later stages must see it: resolved, and without the bookkeeping
    this pipeline keeps for its own reporting.

    `build_algorithm_blueprint` preserves the reuse decisions exactly as the model wrote
    them, so both versions can be shown side by side in the rendered answer. That record
    must never reach a prompt: handing the implementation stage a library-verified call
    form AND the unverified one it replaced gives it two sources of truth and an even
    chance of copying the wrong one — which is the whole failure this resolution exists
    to remove. Every key starting with "_" is reporting-only and is dropped here, at the
    top level AND inside each section's entries: the record of a replaced call form lives
    on the entry it belongs to, so stripping only the outer keys would leave the stale
    value sitting right beside the verified one. Confirmed by test, not assumed.
    """
    if not blueprint:
        return {}

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not k.startswith("_")}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(blueprint)


def _library_class_attributes(class_name: str, lib_dir: str) -> Optional[set]:
    """Every attribute name a real library class actually exposes, or None if the class
    cannot be resolved.

    Two sources, because neither alone is complete. `dir()` on the imported class gives
    the methods, the properties and everything inherited (schedule.Schedule subclasses
    list, so append/extend/index only show up this way). It does NOT give attributes that
    only exist once __init__ has run — `usecase`, `evaluator`, `config_matrix`,
    `instances` — so those are read off the AST as `self.X = ...` assignments anywhere in
    the class body.
    """
    import importlib

    index = _library_member_index(lib_dir)
    owner = next((m for m, mod in index.items() if class_name in mod["classes"]), None)
    if owner is None:
        return None

    attributes: set = set()
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    try:
        module = importlib.import_module(owner)
        obj = getattr(module, class_name, None)
        if isinstance(obj, type):
            attributes |= set(dir(obj))
    except Exception:
        pass                    # fall back to the AST alone; better than refusing to check

    node = index[owner]["classes"][class_name]["node"]
    attributes |= set(index[owner]["classes"][class_name]["methods"])
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                and sub.value.id == "self" and isinstance(sub.ctx, ast.Store):
            attributes.add(sub.attr)
    return attributes or None


def _unknown_kb_attributes(code: str, lib_dir: str) -> list[str]:
    """Attributes read off a parameter that its own annotation says is a library class,
    where the real class has no such attribute.

    The gap every other check here leaves open. `_undefined_names` sees the parameter
    bound by the signature. `_wrong_module_imports` sees a correct import.
    `_bad_kb_call_arity` only judges direct calls of imported names, and this is an
    attribute read on a local. So `problem.instance` — on a TSNScheduling whose __init__
    sets usecase/instances/config_file/config_matrix/uc_gen/instances_loader/
    ContextHandler and no `instance` at all — passed every gate and reached the top of
    the generated file's main path, where it raises AttributeError before a single test
    can run. Confirmed on a real run.

    Only READS are judged. Assigning a new attribute to an object is legal Python and
    routinely deliberate, so `ast.Store` contexts are skipped; so is any parameter whose
    annotation does not resolve to a real library class.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    def annotation_name(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.strip()
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    surfaces: dict = {}      # class name -> attribute set (or None when unresolvable)
    problems: list[str] = []
    seen: set = set()

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        typed = {}
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            cls = annotation_name(arg.annotation) if arg.annotation else ""
            if cls:
                typed[arg.arg] = cls
        if not typed:
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id not in typed:
                continue
            cls = typed[node.value.id]
            if cls not in surfaces:
                surfaces[cls] = _library_class_attributes(cls, lib_dir)
            surface = surfaces[cls]
            if surface is None or node.attr in surface:
                continue
            key = (cls, node.attr)
            if key in seen:
                continue
            seen.add(key)
            problems.append(
                f"line {node.lineno}: `{node.value.id}.{node.attr}` — parameter "
                f"'{node.value.id}' is annotated {cls}, and the real {cls} has no "
                f"attribute '{node.attr}'"
            )
    return problems


def _declared_call_contracts(source: str) -> dict:
    """Each function's declared "Calls:" list, read off the three-line docstring the
    interface freeze is required to emit. Returns {function_name: {callee, ...}}."""
    contracts: dict = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return contracts
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        match = re.search(r"Calls:\s*(.+)", ast.get_docstring(node) or "")
        if not match:
            continue
        claimed = {c.strip() for c in match.group(1).split(",")}
        contracts[node.name] = {c for c in claimed if c and c.lower() != "none"}
    return contracts


def _unsatisfied_call_contracts(impl_code: str, frozen_interface: str = "") -> list[str]:
    """Functions the frozen interface says call something, whose implementation never
    reaches it — directly or through any other generated function.

    The "Calls:" line is not documentation. It is the blueprint's record of HOW a
    requirement gets satisfied, and the frozen interface restates it as fixed, so an
    implementation that drops one is usually dropping the behaviour with it. Confirmed on
    a real run: `execute_stream_migration` declared `Calls: evaluate_schedule` and
    `Satisfies: R4, R7, R8, R9, R10, R17`, and called nothing — which is precisely why
    the generated FRER sequence removed the old path without ever checking the new one
    was safe, the query's own step 4.

    Reachability, not direct calls: the blueprint prompt REQUIRES a top-level
    orchestrator to list every function whose output it depends on, including ones it
    reaches through a helper, so direct-call matching would flag every orchestrator ever
    written. A claimed callee reached transitively is the contract being honoured.

    The contract is read from the frozen interface when one is available, since that is
    the artifact declared fixed; the implementation's own copy of the docstring is only a
    fallback, and a repair is free to rewrite it.
    """
    contracts = _declared_call_contracts(frozen_interface) or \
        _declared_call_contracts(impl_code)
    if not contracts:
        return []

    try:
        tree = ast.parse(impl_code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    defined = {n.name: n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    direct: dict = {}
    for name, node in defined.items():
        direct[name] = {
            call.func.id for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id in defined and call.func.id != name
        }

    def reachable(start: str) -> set:
        seen: set = set()
        stack = list(direct.get(start, ()))
        while stack:
            nxt = stack.pop()
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.extend(direct.get(nxt, ()))
        return seen

    problems: list[str] = []
    for name, claimed in sorted(contracts.items()):
        if name not in defined or not claimed:
            continue
        missing = sorted(c for c in claimed if c in defined and c not in reachable(name))
        if missing:
            problems.append(
                f"`{name}` is declared as calling {sorted(claimed)} but never reaches "
                f"{missing} — not directly and not through any other function it calls"
            )
    return problems


def _annotation_text(node) -> str:
    """An annotation as written, normalised enough to compare two spellings of the same
    type. `List[int]` and `list[int]` compare equal; whitespace never matters."""
    if node is None:
        return ""
    try:
        text = ast.unparse(node)
    except Exception:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value            # a string annotation, e.g. "MigrationPlan"
    return text.replace(" ", "").replace("typing.", "").lower().strip("'\"")


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are NOT inside brackets.

    A naive `.split(",")` tears `tuple[MigrationStep, ...]` into two entries and reports
    a correct Flow line as contradicting its own signature — confirmed as a false
    positive on three functions of a real run, every one of which was written correctly.
    """
    parts, depth, current = [], 0, ""
    for character in text:
        if character in "[({":
            depth += 1
        elif character in "])}":
            depth -= 1
        if character == "," and depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
            continue
        current += character
    if current.strip():
        parts.append(current.strip())
    return parts


def _flow_contract_violations(impl_code: str, frozen_interface: str = "") -> list[str]:
    """Functions whose declared "Flow:" line disagrees with their own signature.

    The flow line is deliberately redundant with the signature — that is what makes it
    worth having. A signature is written while thinking about the call; a flow line is
    written while thinking about what the function does. Writing both and comparing them
    turns a private confusion about a function's role into a visible contradiction, the
    same way the "Calls:" line turns a dropped call into one. It does NOT replace a type
    checker: it can only compare a function's two accounts of itself, never check that
    the value a caller actually passes is the right one.
    """
    contracts: dict = {}
    for source in (frozen_interface, impl_code):
        if not source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            match = re.search(r"Flow:\s*(.+)", ast.get_docstring(node) or "")
            if match and node.name not in contracts:
                contracts[node.name] = match.group(1).strip()
        if contracts:
            break                    # the frozen interface wins when it declares any

    if not contracts:
        return []
    try:
        tree = ast.parse(impl_code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        declared = contracts.get(node.name)
        if not declared:
            continue
        if "->" not in declared:
            problems.append(f"`{node.name}` has a Flow line with no `->`: {declared!r}")
            continue
        lhs, _, rhs = declared.partition("->")
        claimed_in = [t.replace(" ", "").replace("typing.", "").lower()
                      for t in _split_top_level(lhs)]
        claimed_out = rhs.strip().replace(" ", "").replace("typing.", "").lower()

        args = node.args
        real_in = [_annotation_text(a.annotation)
                   for a in list(args.posonlyargs) + list(args.args)]
        real_out = _annotation_text(node.returns)

        if any(not t for t in real_in) or not real_out:
            continue             # unannotated signature; nothing to compare against
        if claimed_in != real_in:
            problems.append(
                f"`{node.name}` declares Flow inputs {claimed_in} but its signature "
                f"takes {real_in}"
            )
        if claimed_out != real_out:
            problems.append(
                f"`{node.name}` declares Flow output '{claimed_out}' but its signature "
                f"returns '{real_out}'"
            )
    return problems


def _bad_internal_call_arity(code: str) -> list[str]:
    """Calls between the file's OWN generated functions whose argument count cannot match
    the callee's parameter list.

    `_bad_kb_call_arity` deliberately judges only names imported from a library module,
    so a generated function calling another generated function — which is most of the
    composition in an algorithm file — is checked by nothing at all until pytest runs and
    raises TypeError. Needs no library, no annotations and no declarations: both the
    signature and the call site are in the same file.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    defined = {n.name: n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    problems: list[str] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        target = defined.get(call.func.id)
        if target is None:
            continue
        if any(isinstance(a, ast.Starred) for a in call.args) or \
                any(k.arg is None for k in call.keywords):
            continue         # dynamic unpacking; length is not knowable statically
        shape = _call_shape(target.args, drop_first=False)
        named = {k.arg for k in call.keywords}
        supplied = len(call.args) + len(named)
        unknown = named - set(shape["required"] + shape["optional"])
        too_few = supplied < len(shape["required"])
        too_many = (not shape["varargs"] and not shape["kwargs"]
                    and supplied > len(shape["required"]) + len(shape["optional"]))
        if unknown and not shape["kwargs"]:
            problems.append(
                f"line {call.lineno}: `{call.func.id}(...)` passes unknown keyword "
                f"argument(s) {sorted(unknown)}; it takes "
                f"{shape['required'] + shape['optional']}"
            )
        elif too_few or too_many:
            problems.append(
                f"line {call.lineno}: `{call.func.id}(...)` is called with {supplied} "
                f"argument(s), but it is defined as taking "
                f"{len(shape['required'])} required"
                + (f" and {len(shape['optional'])} optional" if shape["optional"] else "")
                + f" ({', '.join(shape['required'] + shape['optional']) or 'none'})"
            )
    return problems


def _kb_names_in_scope(tree: ast.AST, lib_dir: str) -> dict:
    """Every local name in a module that refers to a real library class, however it got
    there: `from schedule import Schedule` binds one, and `def f(s: Schedule)` binds
    another of a different kind. Returns {local_name: (class_name, "class"|"instance")}.

    Shared by the two checks below so they agree on what counts as "a library object",
    instead of each re-deriving it and drifting apart.
    """
    import importlib

    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    library_modules = {f[:-3] for f in os.listdir(lib_dir)
                       if f.endswith(".py") and f != "__init__.py"}

    classes: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module.split(".")[0] not in library_modules:
            continue
        try:
            module = importlib.import_module(node.module.split(".")[0])
        except Exception:
            continue
        for alias in node.names:
            obj = getattr(module, alias.name, None)
            if isinstance(obj, type):
                classes[alias.asname or alias.name] = (alias.name, "class")
    return classes


def _resolve_library_class(name: str, lib_dir: str):
    """The real class object behind a library class name, or None."""
    import importlib

    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    index = _library_member_index(lib_dir)
    owner = next((m for m, mod in index.items() if name in mod["classes"]), None)
    if owner is None:
        return None
    try:
        return getattr(importlib.import_module(owner), name, None)
    except Exception:
        return None


def _suggest_parameter(unknown: str, candidates: list) -> Optional[str]:
    """The real parameter an unrecognised keyword was probably meant to be.

    Exists because a check that only says a keyword is WRONG gets answered by deleting
    it. Confirmed on a real run: told that `macro_cycle=` is not a parameter of
    from_compact_encoding, the repair removed the argument entirely rather than moving
    the value to `cycle_ns`, so every schedule it built afterwards had no cycle at all.
    Naming the likely destination turns "this is wrong" into "put it here".

    Scored on shared underscore-separated tokens first, since that is what actually
    distinguishes macro_cycle/cycle_ns from the five parameters it is not, and on
    sequence similarity only as a tiebreak.
    """
    import difflib

    if not candidates:
        return None
    wanted = {t for t in unknown.lower().split("_") if t}

    def score(name: str) -> tuple:
        tokens = {t for t in name.lower().split("_") if t}
        return (len(wanted & tokens),
                difflib.SequenceMatcher(None, unknown.lower(), name.lower()).ratio())

    best = max(candidates, key=score)
    shared, ratio = score(best)
    return best if shared or ratio >= 0.6 else None


def _bad_kb_member_call(code: str, lib_dir: str) -> list[str]:
    """Any call to a library CLASS MEMBER that cannot bind to its real signature —
    wrong argument count, or a keyword the member does not accept.

    General over the call FORM, which is the axis `_bad_kb_call_arity` is blind to. That
    one resolves only calls whose target is a bare `ast.Name`, so it inspects a direct
    `Schedule(...)` and nothing else; `Schedule.from_compact_encoding(...)` and
    `evaluator.evaluate(...)` are `ast.Attribute` targets and were checked by nothing at
    all. On a real run both library calls in the file were in attribute form, so the
    existing gate examined zero of them and a fatal `macro_cycle=` keyword — the real
    parameter is `cycle_ns` — reached execution.

    Two resolvable forms, both handled here and neither handled elsewhere:
      ClassName.member(...)   -> the member as reached through the class. A classmethod
                                 or staticmethod signature already excludes cls/self; a
                                 plain method reached this way is genuinely being passed
                                 its own self, so its signature is used unchanged.
      parameter.member(...)   -> where the parameter is annotated with a library class.
                                 Here self is bound implicitly, so it is dropped first.

    Verified with `inspect.signature(...).bind(...)` against the real object, never by
    re-deriving arity by hand. Dynamic `*args`/`**kwargs` unpacking is skipped: its
    length is not knowable statically, and guessing would be a fabrication.
    """
    import inspect

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    class_names = _kb_names_in_scope(tree, lib_dir)
    if not class_names:
        return []

    def annotation_class(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.strip()
        return ""

    problems: list[str] = []
    seen: set = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        instances = {}
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            cls = annotation_class(arg.annotation) if arg.annotation else ""
            if cls in class_names:
                instances[arg.arg] = class_names[cls][0]

        for call in ast.walk(fn):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            base = call.func.value
            if not isinstance(base, ast.Name):
                continue
            member = call.func.attr
            if base.id in class_names:
                owner, drop_self = class_names[base.id][0], False
            elif base.id in instances:
                owner, drop_self = instances[base.id], True
            else:
                continue
            cls = _resolve_library_class(owner, lib_dir)
            target = getattr(cls, member, None) if cls is not None else None
            if target is None or not callable(target):
                continue     # nonexistent members are the attribute check's business
            if any(isinstance(a, ast.Starred) for a in call.args) or \
                    any(k.arg is None for k in call.keywords):
                continue
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            parameters = list(signature.parameters.values())
            if drop_self and parameters:
                signature = signature.replace(parameters=parameters[1:])
            try:
                signature.bind(*[None] * len(call.args),
                               **{k.arg: None for k in call.keywords})
            except TypeError as exc:
                # Keyed by LINE as well as by member: the same wrong call repeated at
                # two sites is two sites to fix, and a whole-file repair that corrects
                # only the reported one leaves the other live. Confirmed on a real run
                # where an identical bad keyword appeared at lines 77 and 85.
                key = (call.lineno, owner, member)
                if key in seen:
                    continue
                seen.add(key)
                hint = ""
                unknown = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
                if unknown:
                    supplied = {k.arg for k in call.keywords}
                    candidates = [p for p in signature.parameters
                                  if p not in supplied and p not in ("self", "cls")]
                    target = _suggest_parameter(unknown.group(1), candidates)
                    if target:
                        hint = (f" — the real signature calls that parameter "
                                f"`{target}`; move the value there. Deleting the "
                                f"argument is NOT a fix: it was being passed for a "
                                f"reason, and dropping it silently changes behaviour.")
                problems.append(
                    f"line {call.lineno}: `{base.id}.{member}(...)` is called with "
                    f"{len(call.args)} positional and {len(call.keywords)} keyword "
                    f"argument(s), but the real `{owner}.{member}` signature is "
                    f"`{member}{signature}` — {exc}{hint}"
                )
    return problems


def _blueprint_constants(blueprint: Optional[dict]) -> tuple[dict, dict]:
    """The blueprint's CONSTANTS section, split into ({name: {values}}, {name: formula}).

    Tolerant about shape on purpose: a model may write `values` as a list of strings or
    a comma-separated string, and may put the formula under `definition` or `formula`.
    Rejecting a usable answer over its container would throw away the one record that
    stops the implementation and the tests from guessing different literals.
    """
    enums: dict = {}
    formulas: dict = {}
    for entry in (blueprint or {}).get("constants") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        kind = str(entry.get("kind") or "").strip().lower()
        raw = entry.get("values")
        if kind == "enum" or raw is not None:
            if isinstance(raw, str):
                values = {v.strip() for v in raw.split(",") if v.strip()}
            elif isinstance(raw, (list, tuple, set)):
                values = {str(v).strip() for v in raw if str(v).strip()}
            else:
                values = set()
            if values:
                enums[name] = values
                continue
        definition = entry.get("definition") or entry.get("formula")
        if kind == "formula" or definition:
            formulas[name] = str(definition or "").strip()
    return enums, formulas


def _undeclared_constant_literals(code: str, blueprint: Optional[dict]) -> list[str]:
    """String literals compared against, or assigned to, a field the blueprint declared
    as an enum — but which are not among that enum's declared values.

    The defect this catches is not a library mistake. It is the design disagreeing with
    its own implementation: the blueprint names a strategy "FRER" and the code writes
    "frer", or invents "SKIP" alongside, and because the tests are generated from the
    same blueprint separately, every test touching that value fails while each side looks
    individually reasonable.

    Judged only where the comparison is unambiguous — `x.<field> == "literal"`,
    `<field>="literal"`, `x["<field>"] == "literal"` — with the field name matching a
    declared enum. A bare string compared against an unrelated local is not evidence of
    anything and is skipped rather than guessed at.
    """
    enums, _ = _blueprint_constants(blueprint)
    if not enums:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    # Values are matched case-insensitively ONLY to report the casing mismatch itself,
    # which is the most common form of this bug and the most confusing to debug.
    lowered = {name: {v.lower(): v for v in values} for name, values in enums.items()}

    def field_of(node) -> str:
        if isinstance(node, ast.Attribute):
            return node.attr
        if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            return node.slice.value
        if isinstance(node, ast.Name):
            return node.id
        return ""

    problems: list[str] = []
    seen: set = set()

    def report(name: str, literal: str, lineno: int) -> None:
        if literal in enums[name] or (name, literal, lineno) in seen:
            return
        seen.add((name, literal, lineno))
        canonical = lowered[name].get(literal.lower())
        if canonical:
            problems.append(
                f"line {lineno}: `{name}` uses \"{literal}\" but the design declares it "
                f"as \"{canonical}\" — same value, different casing, so any comparison "
                f"against the declared spelling is always False. Use \"{canonical}\""
            )
        else:
            problems.append(
                f"line {lineno}: `{name}` is given the value \"{literal}\", which the "
                f"design does not declare. Its declared values are "
                f"{sorted(enums[name])} — use one of those, or the tests generated from "
                f"the same design will never produce this value"
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(
                node.ops[0], (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
            name = field_of(node.left)
            other = node.comparators[0]
            if name in enums and isinstance(other, ast.Constant) and isinstance(
                    other.value, str):
                report(name, other.value, node.lineno)
            name = field_of(other)
            if name in enums and isinstance(node.left, ast.Constant) and isinstance(
                    node.left.value, str):
                report(name, node.left.value, node.lineno)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (keyword.arg in enums and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)):
                    report(keyword.arg, keyword.value.value, node.lineno)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                name = field_of(target)
                if name in enums:
                    report(name, node.value.value, node.lineno)
    return problems


def _uncomputed_constant_formulas(code: str, blueprint: Optional[dict]) -> list[str]:
    """A field the blueprint declared as a formula that the implementation never computes.

    Reports the two ways a declared formula gets lost: the field is assigned a bare
    constant (0, 0.0, None) so every value is identical and any ordering built on it is
    arbitrary, or it is left behind a TODO/NotImplementedError. A design that wrote down
    "impact = hops + shared ports" and then ships `impact = 0.0` has silently dropped the
    requirement that ordering follow impact, and nothing else in this pipeline notices.
    """
    _, formulas = _blueprint_constants(blueprint)
    if not formulas:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    assigned: dict = {}
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            # `impact = 0.0` followed by `impact += hop_count(s)` is the normal way to
            # build an accumulated score, and reading only plain assignments reported
            # every such function as leaving the value uncomputed. An augmented
            # assignment counts as computing it unless it too adds a bare constant.
            targets = [node.target]
        else:
            continue
        for target in targets:
            name = (target.attr if isinstance(target, ast.Attribute) else
                    target.id if isinstance(target, ast.Name) else "")
            if name not in formulas:
                continue
            value = node.value
            trivial = (isinstance(value, ast.Constant)
                       and (value.value is None
                            or (isinstance(value.value, (int, float))
                                and not isinstance(value.value, bool))))
            # Any non-trivial assignment anywhere counts as computed: a field may well
            # be initialised to 0 and then accumulated into.
            assigned[name] = assigned.get(name, False) or not trivial

    # Only the "assigned, and never anything but a constant" case is reported. The
    # ABSENCE of an assignment is deliberately not: a formula is just as correctly
    # delivered by `def compute_impact(...): return hops + shared`, which never binds the
    # name at all, and by a comprehension, a setattr, or a differently-named local. This
    # check is gated, so a shape that merely looks unfamiliar must not block it — what is
    # reported is the one pattern that cannot be anything else.
    problems: list[str] = []
    for name, formula in formulas.items():
        if assigned.get(name) is False:
            problems.append(
                f"`{name}` is only ever assigned a constant, but the design defines it as "
                f"`{formula}`. Compute it — a field that is the same for every input "
                f"makes any ordering or comparison built on it arbitrary"
            )
    return problems


def _lifecycle_performed_members(lib_dir: str) -> dict:
    """Which library members the ACTIVATION LIFECYCLE already performs for you, derived
    from the library's own context manager rather than from a list.

    `InstanceContext.__enter__` calls `self.parent.build_metadata()`, guarded by a
    `_metadata_built_for` set keyed on the use case's id. So `with
    problem.use_instance(usecase):` builds the metadata exactly once, and an explicit
    `problem.build_metadata()` alongside it is not merely redundant — it rebuilds
    unconditionally, bypassing the cache the library maintains. A real run named
    `build_metadata` as a reuse decision and in the same breath said that entering the
    context runs it, which cannot both be acted on.

    Derived in two steps, so nothing here is specific to build_metadata:
      1. Find each class with both `__enter__` and `__init__`, note which self-attribute
         holds the object it manages, and collect the methods `__enter__` calls on it.
      2. Find the method that HANDS you that manager with `self` as the managed object.
         `TSNScheduling.use_instance` returns `self.ContextHandler(self, instance)`, so
         the manager arrives through an attribute whose class comes from `__init__`'s own
         parameter default — both that indirect form and a direct `InstanceContext(self,
         ...)` are resolved.

    Returns {(owning_class, member): "how entering already performs it"}, keyed the same
    way as _instance_gated_members: a member name alone would be ambiguous across classes.
    A manager added to the library tomorrow is covered without editing this file.
    """
    index = _library_member_index(lib_dir)

    def self_attr_assignments(init) -> dict:
        """self.<attr> = <param>  ->  {attr: param}"""
        found: dict = {}
        for node in ast.walk(init):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == "self"
                    and isinstance(node.value, ast.Name)):
                found[node.targets[0].attr] = node.value.id
        return found

    # Step 1 — context managers, and what entering one does to the object it manages.
    managers: dict = {}
    for module in index.values():
        for cls in module["classes"].values():
            enter = cls["methods"].get("__enter__")
            init = cls["methods"].get("__init__")
            if enter is None or init is None:
                continue
            holds = self_attr_assignments(init)
            performed: dict = {}
            for node in ast.walk(enter):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id == "self"
                        and node.func.value.attr in holds):
                    performed.setdefault(node.func.value.attr, set()).add(node.func.attr)
            if performed:
                managers[cls["qualname"].rsplit(".", 1)[-1]] = {
                    "qualname": cls["qualname"],
                    "params": [a.arg for a in init.args.args],
                    "holds": holds,
                    "performed": performed,
                }
    if not managers:
        return {}

    # Step 2 — the entry point: a method returning one of those managers over `self`.
    lifecycle: dict = {}
    for module in index.values():
        for cls in module["classes"].values():
            init = cls["methods"].get("__init__")
            attr_class: dict = {}
            if init is not None and init.args.defaults:
                names = [a.arg for a in init.args.args][-len(init.args.defaults):]
                defaults = dict(zip(names, init.args.defaults))
                for attr, param in self_attr_assignments(init).items():
                    default = defaults.get(param)
                    if isinstance(default, ast.Name):
                        attr_class[attr] = default.id

            for member, node in cls["methods"].items():
                for call in ast.walk(node):
                    if not isinstance(call, ast.Call):
                        continue
                    func = call.func
                    if isinstance(func, ast.Name):
                        target = func.id
                    elif (isinstance(func, ast.Attribute)
                          and isinstance(func.value, ast.Name)
                          and func.value.id == "self"):
                        target = attr_class.get(func.attr, "")
                    else:
                        continue
                    manager = managers.get(target)
                    if manager is None:
                        continue
                    for attr, methods in manager["performed"].items():
                        param = manager["holds"].get(attr)
                        if param not in manager["params"]:
                            continue
                        # __init__'s params include self at 0; the call's do not.
                        position = manager["params"].index(param) - 1
                        if not (0 <= position < len(call.args)):
                            continue
                        argument = call.args[position]
                        if not (isinstance(argument, ast.Name)
                                and argument.id == "self"):
                            continue
                        for performed in methods:
                            lifecycle[(cls["qualname"], performed)] = (
                                f"entering `with <{cls['qualname'].rsplit('.', 1)[-1]}>"
                                f".{member}(...):` already calls {performed}() "
                                f"({manager['qualname']}.__enter__ does it)"
                            )
    return lifecycle


def _lifecycle_duplicated_calls(code: str, lib_dir: str) -> list[str]:
    """Explicit calls to a member the activation lifecycle already performs.

    Only reported when the file DOES enter an activation block — a design that never
    activates anything has a different and larger problem, and `_activation_context_
    violations` is the check that names it. Reporting both would make the smaller one
    look like the cause.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    lifecycle = _lifecycle_performed_members(lib_dir)
    if not lifecycle:
        return []
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "use_instance" for node in ast.walk(tree)):
        return []

    class_names = _kb_names_in_scope(tree, lib_dir)
    if not class_names:
        return []

    def annotation_class(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.strip()
        return ""

    problems: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        instances = {}
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            cls = annotation_class(arg.annotation) if arg.annotation else ""
            if cls in class_names:
                instances[arg.arg] = class_names[cls][0]

        for call in ast.walk(fn):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            base = call.func.value
            if not isinstance(base, ast.Name):
                continue
            owner = instances.get(base.id) or (
                class_names[base.id][0] if base.id in class_names else ""
            )
            note = lifecycle.get((owner, call.func.attr))
            if note:
                problems.append(
                    f"line {call.lineno}: `{base.id}.{call.func.attr}()` is called "
                    f"explicitly, but {note}. Delete this call — it is the one case where "
                    f"removing a statement IS the fix, because the library performs it "
                    f"for you and calling it again repeats work the library deliberately "
                    f"caches"
                )
    return problems


def _instance_gated_members(lib_dir: str) -> dict:
    """Which library members cannot be touched without an active use case, derived from
    the library's own `@require_instance` decorator rather than from a list.

    Two tiers, both computed:
      direct   — the member itself carries @require_instance.
      indirect — a constructor whose __init__ reads one of those gated names off a
                 parameter, so building the object requires an active use case even
                 though nothing about the constructor says so. TSNConstraintEvaluator is
                 the real case: its __init__ reads n_nodes, n_ports, n_streams, ports,
                 streams, nodes, port_id_to_node and hops_costs off the problem it is
                 handed, every one of them gated.

    Keyed by (owning_class, member), never by member alone. That distinction is not
    pedantry: `ports` and `streams` exist on BOTH Usecase and TSNScheduling, gated on one
    and plain properties on the other, and a name-only rule flags every correct
    `usecase.ports` in the codebase. A constructor entry uses (class, "") as its key.

    Deriving it means a decorator added to the library tomorrow is enforced without
    anyone editing this file.
    """
    index = _library_member_index(lib_dir)
    gated: dict = {}
    gated_names: set = set()
    for module in index.values():
        for cls in module["classes"].values():
            for member, node in cls["methods"].items():
                decorators = {_decorator_name(d) for d in node.decorator_list}
                if "require_instance" in decorators:
                    gated[(cls["qualname"], member)] = (
                        f"{cls['qualname']}.{member} is @require_instance"
                    )
                    gated_names.add(member)
    if not gated:
        return {}

    for module in index.values():
        for cls in module["classes"].values():
            init = cls["methods"].get("__init__")
            if init is None:
                continue
            reads = {n.attr for n in ast.walk(init)
                     if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
                     and isinstance(n.value, ast.Name) and n.value.id != "self"}
            touched = sorted(reads & gated_names)
            if touched:
                gated[(cls["qualname"], "")] = (
                    f"constructing {cls['qualname']} reads {touched} off its argument, "
                    f"and those are @require_instance"
                )
    return gated


def _activation_context_violations(code: str, lib_dir: str) -> list[str]:
    """Uses of instance-gated library members that are not inside a
    `with <problem>.use_instance(...):` block.

    This is the one check here that looks for something REQUIRED TO BE PRESENT rather
    than something wrong that is present. Every other check in this file reports a
    defect it can point at; none of them can report an omission, so a generated file
    that simply never activates a use case passed all of them in silence while being
    unable to execute a single line that touches the problem.

    What counts as gated is computed by `_instance_gated_members` from the library's own
    decorators, so this is not a rule about n_ports or about the evaluator — it is a rule
    about whatever the library currently gates.

    Containment is computed through the CALL GRAPH, not lexically. A purely lexical rule
    was written first and immediately flagged the design the blueprint's own lifecycle
    rule mandates — one block opened by the orchestrator, helpers reading gated members
    directly with no block of their own. Every such helper would have been reported.
    So a function is treated as protected when it has call sites and every one of them
    sits inside a block or inside another protected function; a gated read is then fine
    if it is lexically inside a block OR inside a protected function. An entry point
    with no call sites in the file is never protected — it must open its own block.

    An access is judged only when its OWNING CLASS is known — the base is a parameter
    annotated with a library class, or the class itself. An access on an unannotated
    local is skipped rather than guessed at: `ports` on a Usecase is a plain property
    and `ports` on a TSNScheduling is gated, so a name-only rule would report every
    correct use of the former.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    gated = _instance_gated_members(lib_dir)
    if not gated:
        return []
    class_names = _kb_names_in_scope(tree, lib_dir)

    # Each block records WHICH object it activates, not merely that it exists. Proximity
    # is not protection: a real repair satisfied the earlier, proximity-only version of
    # this check by writing
    #     with TSNScheduling(usecase).use_instance(usecase):
    #         evaluator = TSNConstraintEvaluator(TSNScheduling(usecase), ...)
    # which constructs the problem TWICE — activating a throwaway and handing the
    # evaluator a fresh, unactivated one. Every gated use was lexically inside a block
    # and every one still raised. So a use counts as protected only when the object it
    # is reached through is the object the enclosing block actually activated.
    # A design may route activation through a helper of its own rather than writing the
    # library call inline — `def use_instance(problem, usecase): return
    # problem.use_instance(usecase)`, then `with use_instance(problem, usecase):`. That
    # is correct code, and a rule that only recognises the literal `.use_instance(`
    # form reports it as never activating anything. Confirmed against a real run: the
    # identical function was clean written directly and a false positive through a
    # wrapper. So local wrappers are resolved first — one level, which is all this
    # pattern needs — recording WHICH parameter carries the problem through.
    wrappers: dict = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = [a.arg for a in list(fn.args.posonlyargs) + list(fn.args.args)]
        for node in ast.walk(fn):
            if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
                continue
            inner = node.value
            if not (isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "use_instance"
                    and isinstance(inner.func.value, ast.Name)):
                continue
            if inner.func.value.id in positional:
                wrappers[fn.name] = positional.index(inner.func.value.id)

    protected: list[tuple] = []          # (start, end, {names this block activates})
    temporaries: list[int] = []          # blocks activating something nothing can name
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            receiver = None
            if isinstance(call.func, ast.Attribute) and call.func.attr == "use_instance":
                receiver = call.func.value
            elif isinstance(call.func, ast.Name) and call.func.id in wrappers:
                index = wrappers[call.func.id]
                if index < len(call.args):
                    receiver = call.args[index]
            if receiver is None:
                continue
            names: set = set()
            if isinstance(receiver, ast.Name):
                names.add(receiver.id)
            if isinstance(item.optional_vars, ast.Name):
                names.add(item.optional_vars.id)   # `as p:` binds the problem too
            if not names:
                temporaries.append(node.lineno)
            protected.append((node.lineno, node.end_lineno, names))

    def inside_any_block(lineno: int) -> bool:
        """Used only for call-graph protection: was this CALL SITE reached from inside
        some activation context? Which object is irrelevant at that granularity."""
        return any(start <= lineno <= end for start, end, _ in protected)

    def activated_here(lineno: int, name: Optional[str]) -> bool:
        """Is `name` the object an enclosing block activated?"""
        if name is None:
            return False
        return any(start <= lineno <= end and name in names
                   for start, end, names in protected)

    # --- which functions are only ever entered from inside an activation block ------
    functions = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def enclosing(lineno: int):
        best = None
        for name, node in functions.items():
            if node.lineno <= lineno <= (node.end_lineno or node.lineno):
                if best is None or node.lineno > functions[best].lineno:
                    best = name
        return best

    call_sites: dict = {name: [] for name in functions}
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                and call.func.id in call_sites:
            call_sites[call.func.id].append(call.lineno)

    # Optimistic fixed point: assume protected, then withdraw it from any function with
    # a call site that is neither inside a block nor inside a still-protected function.
    # A function nothing in this file calls is an entry point and is never protected.
    safe = {name for name, sites in call_sites.items() if sites}
    changed = True
    while changed:
        changed = False
        for name in sorted(safe):
            for lineno in call_sites[name]:
                if inside_any_block(lineno):
                    continue
                caller = enclosing(lineno)
                if caller is not None and caller in safe:
                    continue
                safe.discard(name)
                changed = True
                break

    def annotation_class(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.strip()
        return ""

    problems: list[str] = []
    seen: set = set()

    def report(key, lineno, shown, through: Optional[str]):
        """`through` is the NAME the gated object is reached by — the receiver of an
        attribute access, or the problem argument of a gated constructor. None means the
        object is an unnamed temporary, which can never be the activated one."""
        if key in seen:
            return
        if activated_here(lineno, through):
            return
        owner_fn = enclosing(lineno)
        if owner_fn is not None and owner_fn in safe:
            return                   # reached only from inside a block; see the docstring
        seen.add(key)
        if through is None:
            detail = (
                "the object it is used on is built inline, so it is not the object any "
                "block activated — and nothing else can reference it either"
            )
        elif inside_any_block(lineno):
            detail = (
                f"`{through}` is not the object the enclosing block activated; that "
                f"block activates something else, so `{through}` still has no use case"
            )
        else:
            detail = "no enclosing `with ....use_instance(...):` block activates it"
        problems.append(
            f"line {lineno}: `{shown}` needs an active use case — {gated[key]} — but "
            f"{detail}, so this raises ValueError(\"No current instance is set!\")"
        )

    # A block that activates an unnamed temporary protects nothing at all.
    for lineno in temporaries:
        problems.append(
            f"line {lineno}: this `use_instance` block activates an object built inline, "
            f"so the activated object is discarded at the end of the expression and "
            f"nothing inside the block can be using it. Bind the problem to a name once "
            f"and activate that name."
        )

    # Constructors that need an active use case, reached through the class name. The
    # object that must be active is the FIRST argument (the problem), not the class.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in class_names:
            key = (class_names[node.func.id][0], "")
            if key in gated:
                first = node.args[0] if node.args else None
                through = first.id if isinstance(first, ast.Name) else None
                report(key, node.lineno, node.func.id, through)

    # Gated members reached through a parameter whose annotation names their class.
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        typed = {}
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            cls = annotation_class(arg.annotation) if arg.annotation else ""
            if cls in class_names:
                typed[arg.arg] = class_names[cls][0]
        if not typed:
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id not in typed:
                continue
            key = (typed[node.value.id], node.attr)
            if key in gated:
                report(key, node.lineno, f"{node.value.id}.{node.attr}",
                       node.value.id)
    return problems


KB_STUB_DIR = os.path.join(REPO_ROOT, "main_files", "kb_stubs")

# mypy codes worth repairing. Everything else it emits here is strictness noise —
# "Need type annotation for x", "Function is missing a return type annotation" — which
# says nothing about whether the algorithm works and would drown the real findings.
_TYPE_ERROR_CODES = {
    "arg-type", "call-arg", "attr-defined", "return-value", "assignment",
    "index", "operator", "union-attr", "call-overload", "valid-type", "name-defined",
}


def _type_check_violations(code: str) -> list[str]:
    """Argument-type errors, found by running a real type checker over the generated
    file against the knowledge base plus the stubs in main_files/kb_stubs/.

    The one class of defect nothing else here can reach. Every other check in this file
    asks whether a name exists, whether an argument COUNT binds, or whether a pattern is
    present — all of which `TSNScheduling(usecase)` satisfies perfectly while being
    fatal, because the real parameter is a config-file PATH and passing a use case
    leaves the problem with no active instance. Arity cannot see it; only types can.

    The stubs exist because the library annotates almost nothing (1 of 108 signatures
    has a return type), so without them every library value is `Any` and nothing is
    comparable. They are deliberately PARTIAL: each class carries `__getattr__ -> Any`,
    so an unlisted member is permitted rather than reported missing. Nonexistent
    attributes are `_unknown_kb_attributes`' job; these stubs state only what a checker
    cannot infer from untyped source.

    Degrades to silence when mypy is not installed rather than failing the run — this is
    an optional dependency, and a pipeline that cannot run without it would be worse
    than one that checks less.
    """
    import subprocess
    import tempfile

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.
    del tree

    with tempfile.TemporaryDirectory() as workdir:
        target = os.path.join(workdir, "generated_impl.py")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(code)
        env = dict(os.environ)
        env["MYPYPATH"] = os.pathsep.join([KB_STUB_DIR, LIB_DIR])
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "mypy", "--no-error-summary",
                 "--follow-imports=silent", "--ignore-missing-imports",
                 "--no-color-output", target],
                capture_output=True, text=True, timeout=300, env=env,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if "No module named mypy" in (completed.stderr or ""):
            return []

    problems: list[str] = []
    for line in (completed.stdout or "").splitlines():
        if ": error:" not in line:
            continue
        code_match = re.search(r"\[([a-z-]+)\]\s*$", line)
        if code_match and code_match.group(1) not in _TYPE_ERROR_CODES:
            continue
        _, _, rest = line.partition(".py:")
        lineno, _, message = rest.partition(": error:")
        problems.append(f"line {lineno.split(':')[0]}:{message.strip()}")
    return problems


_ERASED_ANNOTATIONS = {"object", "any", "typing.any"}


def _erased_library_annotations(code: str, lib_dir: str) -> list[str]:
    """Parameters annotated `object` or `Any` that are demonstrably holding a library
    object, judged by what the body actually does with them.

    The evasion that motivated this, found in a real run: every evaluator parameter in
    the file was declared `evaluator: object` while the body called compute_metrics,
    feasible_streams and evaluate on it. Nothing was wrong with those calls — but the
    annotation is the only thing that says which class a value belongs to, so
    `_unknown_kb_attributes` and `_bad_kb_member_call` both skipped the parameter
    entirely. Verified directly: an identical bogus call is caught when the parameter is
    annotated with the real class and invisible when it is annotated `object`. A check
    the generated file can switch off by choosing a weaker annotation is not a check, so
    this one reports the erasure itself.

    A parameter is attributed to a class only when the members read off it are all real
    members of that class AND at least one of them is not something a builtin container
    would have — otherwise `results: object` with a single `.append` would be reported
    as a Schedule, which subclasses list.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    index = _library_member_index(lib_dir)
    surfaces: dict = {}
    for module in index.values():
        for name in module["classes"]:
            members = _library_class_attributes(name, lib_dir)
            if members:
                surfaces[name] = members
    if not surfaces:
        return []
    common = set(dir(list)) | set(dir(dict)) | set(dir(str)) | set(dir(object))

    problems: list[str] = []
    seen: set = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        erased = set()
        for arg in list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs):
            if arg.annotation is None:
                continue
            try:
                text = ast.unparse(arg.annotation).replace(" ", "").lower()
            except Exception:
                continue
            if text in _ERASED_ANNOTATIONS:
                erased.add(arg.arg)
        if not erased:
            continue
        used: dict = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in erased:
                used.setdefault(node.value.id, set()).add(node.attr)
        for param, members in used.items():
            distinctive = members - common
            if not distinctive:
                continue
            matches = [cls for cls, surface in surfaces.items()
                       if members <= surface and distinctive & surface]
            if len(matches) != 1 or (fn.name, param) in seen:
                continue
            seen.add((fn.name, param))
            problems.append(
                f"line {fn.lineno}: `{fn.name}` declares `{param}` as a bare "
                f"object/Any, but calls {sorted(distinctive)[:4]} on it — that is a "
                f"{matches[0]}. Annotate it `{param}: {matches[0]}`: with the type "
                f"erased, no check can verify a single call made through it"
            )
    return problems


_NUMPY_ARRAY_BUILDERS = {
    "array", "asarray", "asanyarray", "zeros_like", "ones_like", "empty_like",
    "ascontiguousarray", "atleast_2d", "asfarray",
}


def _library_value_used_as_array(code: str, lib_dir: str) -> list[str]:
    """Parameters annotated with a library class that the body treats as a numpy array,
    where the real class is not one.

    The failure this exists for is subtler than a wrong name and harder to see than a
    bare `object`. A real run annotated its parameters `start_schedule: Schedule` — the
    genuine library class, exactly as the interface-freeze rule now demands — and then
    required `numpy.asarray(value).ndim == 2`. A real `schedule.Schedule` is a list of
    Slot objects: `numpy.asarray` gives a 1-D object array, the shape test fails, and
    the function raises for the very type its own signature declares. The annotation
    went from `object` (unverifiable) to a concrete class that is WRONG, which no other
    check here can see: there is no attribute access to resolve, and handing a list to
    numpy is perfectly legal Python.

    Array treatment propagates through the file's own helpers to a fixed point, because
    that is how it was actually written — the annotated parameter went to
    `_is_schedule_array`, which called `_array`, which called `numpy.asarray`. Stopping
    at one level would have seen nothing.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    class_names = _kb_names_in_scope(tree, lib_dir)
    if not class_names:
        return []

    functions = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    positions = {name: [a.arg for a in list(fn.args.posonlyargs) + list(fn.args.args)]
                 for name, fn in functions.items()}

    def array_ops(fn, names: set) -> bool:
        """Is any name in `names` subscripted multi-dimensionally, or handed to numpy?"""
        for node in ast.walk(fn):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                    and node.value.id in names and isinstance(node.slice, ast.Tuple):
                return True
            if isinstance(node, ast.Call):
                builder = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else "")
                if builder in _NUMPY_ARRAY_BUILDERS and any(
                        isinstance(a, ast.Name) and a.id in names for a in node.args):
                    return True
        return False

    # Fixed point: which (function, parameter index) pairs end up treated as arrays.
    arrayish: set = set()
    changed = True
    while changed:
        changed = False
        for name, fn in functions.items():
            for index, param in enumerate(positions[name]):
                if (name, index) in arrayish:
                    continue
                if array_ops(fn, {param}):
                    arrayish.add((name, index))
                    changed = True
                    continue
                for node in ast.walk(fn):
                    if not (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id in functions):
                        continue
                    for position, arg in enumerate(node.args):
                        if isinstance(arg, ast.Name) and arg.id == param \
                                and (node.func.id, position) in arrayish:
                            arrayish.add((name, index))
                            changed = True
                            break
                    if (name, index) in arrayish:
                        break

    import numpy as _np

    def is_array_like(class_name: str) -> bool:
        cls = _resolve_library_class(class_name, lib_dir)
        if cls is None:
            return True                      # unknown: do not accuse
        return issubclass(cls, _np.ndarray) or hasattr(cls, "__array__")

    problems: list[str] = []
    seen: set = set()
    for name, fn in functions.items():
        for index, param in enumerate(positions[name]):
            if (name, index) not in arrayish:
                continue
            annotation = None
            all_args = list(fn.args.posonlyargs) + list(fn.args.args)
            if index < len(all_args):
                annotation = all_args[index].annotation
            if annotation is None:
                continue
            try:
                text = ast.unparse(annotation).strip("'\"")
            except Exception:
                continue
            if text not in class_names or is_array_like(class_names[text][0]):
                continue
            key = (name, param)
            if key in seen:
                continue
            seen.add(key)
            problems.append(
                f"line {fn.lineno}: `{name}` annotates `{param}` as "
                f"{class_names[text][0]} but treats it as a numpy array (indexed "
                f"[:, i] or passed to numpy). A real {class_names[text][0]} is not an "
                f"array — passing one in makes this function fail on the exact type it "
                f"declares. Annotate what the code actually accepts, or operate on the "
                f"real {class_names[text][0]} API"
            )
    return problems


# Names too common to match a library convention by name alone — a generated
# `results.count(x)` must never be judged against a library `count`.
_AMBIGUOUS_CALL_NAMES = {
    "find", "count", "index", "append", "extend", "get", "copy", "keys", "items",
    "values", "nodes", "edges", "update", "add", "remove", "insert", "pop", "sort",
    "join", "split", "format", "read", "write", "close", "next", "len", "range",
}
_AGGREGATE_FUNCTIONS = {"len", "sum", "max", "min", "count", "size"}


def _enclosing_function(tree: ast.AST, node: ast.AST):
    """The innermost function containing a node, or None at module level."""
    best = None
    for candidate in ast.walk(tree):
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if candidate.lineno <= node.lineno <= (candidate.end_lineno or candidate.lineno):
            if best is None or candidate.lineno > best.lineno:
                best = candidate
    return best


def _resolve_provenance(expr, fn, depth: int = 0):
    """Trace a bare name back to what it was assigned from, up to two hops.

    Necessary, not decorative: 83 of the 108 arguments at this library's own internal
    call sites are bare names, so comparing expressions as written would compare almost
    nothing. `group_penalty(talker_group, k)` only becomes informative once `k` is
    resolved to `stream_potential[s_idx]`.
    """
    if depth > 2 or fn is None or not isinstance(expr, ast.Name):
        return expr
    assigned = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and node.lineno < expr.lineno and any(
                isinstance(t, ast.Name) and t.id == expr.id for t in node.targets):
            assigned = node.value
        elif isinstance(node, (ast.For, ast.AsyncFor)) and \
                isinstance(node.target, ast.Name) and node.target.id == expr.id:
            return ("loop-variable", node.iter)
    return _resolve_provenance(assigned, fn, depth + 1) if assigned is not None else expr


def _expression_shape(expr) -> str:
    """A coarse class for what KIND of value an expression produces.

    Coarse on purpose. The question is never "is this the identical expression" — it is
    "is this the same kind of thing the library itself puts here": a per-element value
    pulled out of a collection, a whole-collection size, an attribute of the object, a
    constant. That distinction is what separates `stream_potential[s_idx]` from
    `len(old)` while leaving genuinely interchangeable spellings alone.
    """
    if isinstance(expr, tuple) and expr and expr[0] == "loop-variable":
        return "per-element value (loop variable)"
    if isinstance(expr, ast.Subscript):
        return "per-element value (indexed out of a collection)"
    if isinstance(expr, ast.Attribute):
        return "attribute of an object"
    if isinstance(expr, ast.Call):
        func = expr.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in _AGGREGATE_FUNCTIONS:
            return "whole-collection size/aggregate"
        return f"result of {name}()"
    if isinstance(expr, ast.Constant):
        return "constant"
    if isinstance(expr, (ast.BinOp, ast.UnaryOp)):
        return "arithmetic expression"
    if isinstance(expr, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
        return "literal collection"
    if isinstance(expr, ast.Name):
        return ""            # unresolved; carries no evidence either way
    return ""


_LIBRARY_CONVENTION_CACHE: dict = {}


def _library_call_conventions(lib_dir: str) -> dict:
    """What the library itself passes to each of its own functions, per argument
    position — the one source of truth about a parameter's intended KIND of value that
    no signature, annotation or docstring carries.

    A parameter earns a convention only when every internal call site agrees and there
    are at least two of them. Across this library that yields 17 conventions spanning 8
    functions, so the mechanism is not built around any single case: pathConvenient,
    periodConvenient, set_link_data, hop_cost_eg2eg, streamGenerator, all_occurrences
    and configure_stream_ports all contribute alongside group_penalty.

    Returns {(function_name, arg_index): (shape, example_source, n_sites)}.
    """
    if lib_dir in _LIBRARY_CONVENTION_CACHE:
        return _LIBRARY_CONVENTION_CACHE[lib_dir]

    observations: dict = {}
    for filename in sorted(os.listdir(lib_dir)):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        try:
            with open(os.path.join(lib_dir, filename), encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            name = (call.func.attr if isinstance(call.func, ast.Attribute)
                    else call.func.id if isinstance(call.func, ast.Name) else None)
            if name not in defined or name in _AMBIGUOUS_CALL_NAMES:
                continue
            enclosing = _enclosing_function(tree, call)
            for index, argument in enumerate(call.args):
                if isinstance(argument, ast.Starred):
                    continue
                resolved = _resolve_provenance(argument, enclosing)
                shape = _expression_shape(resolved)
                if not shape:
                    continue
                # Show what the argument RESOLVES to, not how it was spelled. The
                # library writes `group_penalty(talker_group, k)`; quoting `k` back at
                # a reader says nothing, while quoting `stream_potential[s_idx]` says
                # exactly what the parameter is meant to hold. A message that names
                # only the wrong thing gets answered by deleting it.
                if isinstance(resolved, tuple):
                    example = f"{ast.unparse(argument)} (each item of "\
                              f"{ast.unparse(resolved[1])})"
                else:
                    example = ast.unparse(resolved)
                    if example != ast.unparse(argument):
                        example = f"{ast.unparse(argument)} = {example}"
                observations.setdefault((name, index), []).append((shape, example))

    conventions = {}
    for key, seen in observations.items():
        shapes = {shape for shape, _ in seen}
        if len(shapes) == 1 and len(seen) >= 2:
            conventions[key] = (seen[0][0], seen[0][1], len(seen))
    _LIBRARY_CONVENTION_CACHE[lib_dir] = conventions
    return conventions


def _unconventional_library_arguments(code: str, lib_dir: str) -> list[str]:
    """Arguments to a library function whose KIND differs from what the library itself
    always passes in that position.

    The failure that motivated it binds perfectly and reads plausibly:
    `group_penalty(group, len(old))` passes a port count where every internal call
    passes `stream_potential[s_idx]`, a per-stream hop potential. The formula is
    `k - n_nodes + (n_edges - n_nodes) / 2`, so k is path-scale and a network-scale
    count silently distorts every penalty. Arity binds, types match, no attribute is
    missing — nothing else here can see it.

    Heuristic by construction, so it reports rather than gates: a convention is strong
    evidence about intent, not proof of error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    conventions = _library_call_conventions(lib_dir)
    if not conventions:
        return []

    problems: list[str] = []
    seen: set = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        name = (call.func.attr if isinstance(call.func, ast.Attribute)
                else call.func.id if isinstance(call.func, ast.Name) else None)
        if name is None or name in _AMBIGUOUS_CALL_NAMES:
            continue
        enclosing = _enclosing_function(tree, call)
        for index, argument in enumerate(call.args):
            if isinstance(argument, ast.Starred):
                continue
            convention = conventions.get((name, index))
            if convention is None:
                continue
            expected, example, sites = convention
            actual = _expression_shape(_resolve_provenance(argument, enclosing))
            if not actual or actual == expected:
                continue
            key = (call.lineno, name, index)
            if key in seen:
                continue
            seen.add(key)
            problems.append(
                f"line {call.lineno}: `{name}(...)` argument {index + 1} is "
                f"{actual} (`{ast.unparse(argument)}`), but every one of the "
                f"{sites} places the library calls `{name}` itself passes "
                f"{expected} there (e.g. `{example}`). Check what that parameter is "
                f"meant to hold before assuming these are interchangeable"
            )
    return problems


# ── A recorded KB fact: the library contradicts itself about chromosome encoding ──
#
# schedule.Schedule's two compact-encoding converters are START_TIME in both
# directions — from_compact_encoding reads a nonzero cell as `start_ns = value - 1`,
# to_compact_encoding writes `start_ns + 1`. Meanwhile
# TSNScheduling.initialize_problem builds `self.evaluator` with no scheduling_type,
# so `problem.evaluator` is always PRIORITY. There is exactly ONE chromosome format
# in this library and it is start-time; the evaluator it hands you cannot read it.
#
# The library never catches this because nothing outside schedule.py touches a
# compact encoding at all — the two halves are never exercised together, so the
# contradiction has no internal witness. It only surfaces in a consumer, and the
# migration task forces a consumer to use both. Four separate model runs paired them
# and every one was individually correct at each call site.
#
# Recorded as a fact rather than derived, because it IS a fact about this library's
# design and not a pattern in code. `_verify_start_time_converters` re-checks it
# against the source on every run so a library change makes the fact loud instead of
# silently stale.
_START_TIME_CONVERTERS = ("from_compact_encoding", "to_compact_encoding")


# The representations the library has already fixed, with the conventions and the limits
# that are true of them regardless of what any one query asks for. Supplied to the
# configuration stage as fact instead of being requested as prose: a free-text
# "canonical representation and its exact shape" field produced "rows indexed by 1-based
# port identity" on a real run, which is off by one — the array row is 0-based and the
# port id is row+1 (the library writes `mask[np.array(ports) - 1, s_idx]` and reads back
# `[p + 1 for p in np.nonzero(...)]`). A fact the model does not author is a fact it
# cannot misstate.
#
# Deliberately NOT recorded here: which representation a given design should pick. That
# is the design decision, it depends on the query, and it stays with the model. Each
# entry states what the thing is and what it cannot do; drawing the consequence for a
# particular algorithm is the model's job.
#
# LIMIT ON REUSING THIS CONSTANT. It renders ONE wording, shared by every prompt that
# interpolates it. That is the point — it is why the facts cannot drift apart — but it
# also means a prompt needing its OWN phrasing of one of these facts cannot be served
# from here. The moment a change is prompt-specific (one stage must emphasise the
# row/port mapping differently, or needs the fact folded into a sentence about something
# else), do NOT fork this constant or add a per-prompt variant to it: write that wording
# directly into the prompt that needs it, and leave this constant as the shared baseline.
# Two prompts pulling slightly different renderings out of one constant is harder to read
# and harder to fingerprint than two prompts that each plainly say their own thing.
#
# This is why the blueprint prompt's own passages on the chromosome shape (its module
# item 3) and on 1-based port ids (its module item 5) were NOT replaced by this catalogue
# when it was introduced. Checked line by line: the FACTS agree, but the wording differs
# and each passage carries material this constant does not have — what a non-zero cell
# means under each encoding, feasible_streams' return, to_compact_encoding's raise on an
# EMPTY Schedule, update_streams' two-slot precondition, and the dead
# total_occupied_time / utilization members. Substituting the catalogue there would have
# deleted all of that, so those passages stay as prompt-specific text.
_LIBRARY_FIXED_REPRESENTATIONS = (
    {
        "id": "candidate_array",
        # The code a requirement may name this form by instead of its id.
        "codes": ("numpy.ndarray", "np.ndarray"),
        # One stream's part of a schedule in this form: its route in THIS schedule
        # (which ports it uses) together with its start times there.
        "stream_part": "candidate_array[:, i]",
        "stream_part_what": "column i: stream i's route in this schedule, with its start times",
        "what": "a candidate schedule: a 2-D numpy int array of shape "
                "(n_ports, n_streams). Cell 0 means this port is not used by this "
                "stream. It is not a dict of paths.",
        "conventions": (
            "Rows and columns are 0-BASED array positions, and neither equals the id it "
            "refers to. The port id of row r is r+1 (the library writes "
            "`talker_mask[np.array(t_ports) - 1, s_idx] = 1` and reads back "
            "`[p + 1 for p in np.nonzero(chromosome[:, s_idx])[0]]`). Column c is the "
            "c-th entry of `usecase.streams` in enumeration order, not a stream id "
            "field.",
        ),
        "limits": (
            "It carries START TIMES ONLY. Window duration and period are NOT in it: "
            "`Schedule.from_compact_encoding` takes `windLengths` and "
            "`stream_intervals` as separate arguments, and the durations live on the "
            "evaluator (`windLengths[p_idx, s_idx]`). An array alone does not determine "
            "a schedule.",
            "One stream can hold AT MOST ONE slot per port. "
            "`Schedule.to_compact_encoding` RAISES ValueError('Multiple slots found for "
            "port ..., stream ...') rather than truncating. Two paths for one stream "
            "share at least their talker and listener ports, so both fit in one column "
            "only where they use the same start time on every shared port; where times "
            "differ, one is lost. Even when they fit, the column is a set of active "
            "ports, not two routes, and the evaluator judges it as one path "
            "(flow_conservation, looping).",
            "Split into views, it cannot judge a whole configuration's windows. A "
            "configuration with a stream on two routes needs at least two single-route "
            "views, and the evaluator compares only the windows inside the one array it "
            "is handed, so an overlap on a port between windows in DIFFERENT views - one "
            "stream's two routes, or its second route and another stream's route - is "
            "seen by no view. Only a compiled_schedule holding every window of the "
            "configuration, checked with `Schedule.validate_conflicts`, sees it.",
            "It cannot mark a stream as ABSENT from a configuration. A stream that is not "
            "there (not yet added, or already removed) is an all-zero column, and the "
            "evaluator reads that column as a stream that exists but has no route: "
            "`_per_stream_host_role_violations` scores `abs(talker_count - 1)` and "
            "`abs(listener_count - 1)` per column, so an empty column violates "
            "talker_not_fixed and listener_not_fixed, and `feasible_streams` returns "
            "False for it. The array itself cannot say which streams are meant to be "
            "present.",
        ),
        "witness": ("to_compact_encoding", "from_compact_encoding",
                    "_per_stream_host_role_violations",
                    "_per_stream_flow_conservation_violations", "feasible_streams"),
    },
    {
        "id": "compiled_schedule",
        "codes": ("schedule.Schedule", "Schedule"),
        "stream_part": "Schedule.slots_by_stream(i)",
        "stream_part_what": "stream i's slots: its route in this schedule, with its windows",
        "what": "a compiled schedule: a `schedule.Schedule`, which SUBCLASSES list and "
                "holds `schedule.Slot` objects directly. A Slot is one gate window — "
                "start_ns, end_ns, period_ns, stream_id, port_id, queue_id — which is "
                "what a switch is actually configured with.",
        "conventions": (
            "`Slot.port_id` is a 1-BASED port id, the same id the array's row+1 gives. "
            "`Slot.stream_id` is the array's column index. `Schedule.__init__` takes "
            "`cycle_ns` first and `basetime`, `capacity`, `iterable` after it, so an "
            "empty-looking `Schedule()` is a TypeError.",
            "`Schedule.from_compact_encoding` and `Schedule.to_compact_encoding` convert "
            "between this and the candidate array, and BOTH directions are START_TIME: a "
            "non-zero array cell is `start_ns + 1`. There is no PRIORITY converter.",
        ),
        "limits": (
            "`Schedule.update_graph` only APPENDS windows to a port's existing "
            "`port_data['schedule']` list; it never replaces them. Writing several "
            "schedules into the same use case accumulates their windows.",
            "The evaluator cannot judge it. `per_stream_all_violations`, `feasible_streams` "
            "and `evaluate` take a candidate array (they start from `chromosome.tobytes()` "
            "and numpy operations), so a configuration held only in this form must be "
            "converted with `to_compact_encoding` first, which refuses two slots for one "
            "(port, stream). Its own `validate_conflicts` compares windows per port and "
            "nothing else: not routes, talker/listener roles or latency.",
            "It cannot tell a stream's two routes apart. A Schedule is one flat list of "
            "Slots, and a Slot (start_ns, end_ns, period_ns, stream_id, port_id, queue_id, "
            "index) has no field saying which route it belongs to; two routes of one stream "
            "share at least their talker and listener ports, so the routes cannot be "
            "recovered from it either. A design that judges each route on its own needs "
            "the routes held as separate entries.",
        ),
        "witness": ("Schedule", "Slot", "per_stream_all_violations", "validate_conflicts"),
    },
    {
        "id": "problem_and_usecase",
        "codes": ("usecase_generator.Usecase", "tsn_scheduling_problem.TSNScheduling"),
        "what": "a use case is a `usecase_generator.Usecase`; the problem state around "
                "it is a `tsn_scheduling_problem.TSNScheduling`, which is a VIEW over an "
                "activated Usecase and holds no use case of its own.",
        "conventions": (
            "Both expose members named `ports` and `streams`, keyed DIFFERENTLY: on "
            "Usecase, `ports` is keyed by port NAME and a stream's `pathLists` hold port "
            "names; on TSNScheduling, `ports` is keyed by port ID and `pathLists` hold "
            "port ids. Any mention of 'ports' or 'a stream path' must say which of the "
            "two it means, or the functions built from it cannot be composed.",
            "Port id 0 is reserved and maps to the empty string.",
        ),
        "limits": (),
        "witness": ("Usecase", "TSNScheduling"),
    },
)


def _violation_row_order(lib_dir: str) -> dict:
    """The ORDER of the rows in what `TSNConstraintEvaluator.evaluate` hands back, per
    encoding, derived from the evaluator's own `__init__` and check functions.

    Why this has to be derived rather than written down. `evaluate` returns
    `per_stream_all_violations(chromosome).sum(axis=1)`, and that array is
    `np.vstack([check(chromosome) for check in self.active_checks])`. So the row order is
    the order checks were appended in `__init__`, and then each check's OWN row order —
    because a single check contributes several rows: `_per_stream_host_role_violations`
    returns a 5-row vstack. The result is 11 rows under PRIORITY and 14 under START_TIME,
    which is where those two numbers come from; they are NOT the number of checks (5 and
    7), and the prompt that said "one entry per active check" was describing them wrongly.

    And it cannot be read off the object at run time either: `self.violations_keys` is
    `relevant_weight_keys`, a SET, so its iteration order has no relationship to the rows.
    There is no other mapping in the library. Without this, "which constraint did this
    stream violate" has no answer and any design branching on violation type is guessing
    an index.

    Row labels come from the library's own trailing comments where it has them
    (`np.abs(talker_count - 1),      # talker_not_fixed`) and from the returned expression
    otherwise. Returns {"PRIORITY": [...], "START_TIME": [...]}, empty if the evaluator
    cannot be read the way this expects.
    """
    index = _library_member_index(lib_dir)
    evaluator = None
    source_lines: list[str] = []
    for module_name, module in index.items():
        cls = module["classes"].get("TSNConstraintEvaluator")
        if cls is not None:
            evaluator = cls
            path = os.path.join(lib_dir, f"{module_name}.py")
            try:
                with open(path, encoding="utf-8") as handle:
                    source_lines = handle.read().splitlines()
            except OSError:
                return {}
    if evaluator is None or not source_lines:
        return {}

    init = evaluator["methods"].get("__init__")
    if init is None:
        return {}

    def appended_checks(body) -> list[str]:
        """active_checks.append(self.X) / .extend([self.X, self.Y]) in source order."""
        found: list[str] = []
        for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if not (isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "active_checks"):
                continue
            if node.func.attr == "append" and node.args:
                target = node.args[0]
                if isinstance(target, ast.Attribute):
                    found.append(target.attr)
            elif node.func.attr == "extend" and node.args:
                arg = node.args[0]
                if isinstance(arg, (ast.List, ast.Tuple)):
                    found.extend(e.attr for e in arg.elts
                                 if isinstance(e, ast.Attribute))
        return found

    def rows_of(method_name: str) -> list[str]:
        """One label per row this check function contributes."""
        node = evaluator["methods"].get(method_name)
        if node is None:
            return [f"{method_name}[?]"]
        inner = {id(x) for d in ast.walk(node)
                 if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and d is not node
                 for x in ast.walk(d)}
        returns = [r for r in ast.walk(node)
                   if isinstance(r, ast.Return) and r.value is not None
                   and id(r) not in inner]
        # The widest return wins. A check with an early guard return of a different
        # shape is a library inconsistency, reported by the verifier below, not a
        # reason to understate the row order the normal path produces.
        best: list[str] = []
        for ret in returns:
            value = ret.value
            elements = None
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "vstack" and value.args
                    and isinstance(value.args[0], (ast.List, ast.Tuple))):
                elements = value.args[0].elts
            labels = []
            rows = elements or [value]
            for position, element in enumerate(rows):
                line = source_lines[element.lineno - 1] if element.lineno <= len(
                    source_lines) else ""
                comment = line.split("#", 1)[1].strip() if "#" in line else ""
                # Three sources, best first: the library's own trailing comment; the
                # returned variable's name, which is meaningful where the comment is
                # missing (`np.vstack((backwarding, flow_conservation, looping))`); and
                # the method name, for a row returned as an opaque expression like
                # `v[np.newaxis, :]`.
                expression = ast.unparse(element)
                named = re.fullmatch(r"[a-z_][a-z0-9_]*", expression) is not None
                fallback = (method_name if len(rows) == 1
                            else f"{method_name}[{position}]")
                labels.append(comment or (expression if named else fallback))
            if len(labels) > len(best):
                best = labels
        return best or [method_name]

    def branch_test(node: ast.If) -> str:
        text = ast.unparse(node.test)
        if "check_task_placement" in text:
            return "task_placement"
        if "check_routing" in text:
            return "routing"
        if "START_TIME" in text:
            return "START_TIME"
        if "PRIORITY" in text:
            return "PRIORITY"
        return ""

    # The two flag-guarded groups default to True, so they are always present; the
    # encoding branch is the one that differs. Order follows the source, which is the
    # order the rows come out in.
    groups: list[tuple[str, list[str]]] = []
    for node in ast.walk(init):
        if isinstance(node, ast.If):
            kind = branch_test(node)
            if kind:
                groups.append((kind, appended_checks(node.body)))

    order: dict = {}
    for encoding in ("PRIORITY", "START_TIME"):
        labels: list[str] = []
        for kind, checks in groups:
            if kind in ("task_placement", "routing") or kind == encoding:
                for check in checks:
                    labels.extend(rows_of(check))
        if labels:
            order[encoding] = labels
    return order


def _verify_violation_row_order(lib_dir: str) -> list[str]:
    """Cross-check the derived row order against the library's own `n_constr`.

    `__init__` sets `n_constr = len(relevant_weight_keys)`, built from the SAME branches
    that append the checks — so the derived row count and that key count must agree, and
    a mismatch means the derivation has gone stale. Also reports a check function whose
    returns differ in row count, because that makes the row order conditional on the
    topology rather than fixed.
    """
    index = _library_member_index(lib_dir)
    evaluator = None
    for module in index.values():
        if "TSNConstraintEvaluator" in module["classes"]:
            evaluator = module["classes"]["TSNConstraintEvaluator"]
    if evaluator is None:
        return ["recorded KB fact is stale: TSNConstraintEvaluator is gone"]
    init = evaluator["methods"].get("__init__")
    if init is None:
        return ["recorded KB fact is stale: TSNConstraintEvaluator has no __init__"]

    notes: list[str] = []
    derived = _violation_row_order(lib_dir)
    if not derived:
        return ["recorded KB fact is stale: the evaluator's violation row order can no "
                "longer be derived from its __init__"]

    # Weight keys per branch, the library's own n_constr inputs.
    keys: dict = {}
    for node in ast.walk(init):
        if not isinstance(node, ast.If):
            continue
        text = ast.unparse(node.test)
        kind = ("task_placement" if "check_task_placement" in text else
                "routing" if "check_routing" in text else
                "START_TIME" if "START_TIME" in text else
                "PRIORITY" if "PRIORITY" in text else "")
        if not kind:
            continue
        for call in ast.walk(ast.Module(body=list(node.body), type_ignores=[])):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "update" and call.args
                    and isinstance(call.args[0], (ast.List, ast.Tuple))):
                keys.setdefault(kind, set()).update(
                    e.value for e in call.args[0].elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )

    for encoding, labels in derived.items():
        expected = set()
        for kind in ("task_placement", "routing", encoding):
            expected |= keys.get(kind, set())
        if expected and len(expected) != len(labels):
            notes.append(
                f"recorded KB fact is stale: {len(labels)} violation rows derived for "
                f"{encoding} but the evaluator's own n_constr would be {len(expected)}"
            )

    for name, node in evaluator["methods"].items():
        if not name.startswith("_per_stream_"):
            continue
        inner = {id(x) for d in ast.walk(node)
                 if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and d is not node
                 for x in ast.walk(d)}
        counts = set()
        for ret in ast.walk(node):
            if not (isinstance(ret, ast.Return) and ret.value is not None
                    and id(ret) not in inner):
                continue
            value = ret.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "vstack" and value.args
                    and isinstance(value.args[0], (ast.List, ast.Tuple))):
                counts.add(len(value.args[0].elts))
            else:
                counts.add(1)
        if len(counts) > 1:
            notes.append(
                f"library inconsistency: {name} returns {sorted(counts)} rows on "
                f"different paths, so the violation row order is not fixed — a design "
                f"indexing that array by position can be off on some topologies"
            )
    return notes


def _render_violation_row_order(lib_dir: str) -> str:
    """The derived row order as prompt text, one block per encoding."""
    derived = _violation_row_order(lib_dir)
    if not derived:
        return ("   (the row order could not be derived from the library source; do not "
                "index this array by position)")
    lines: list[str] = []
    for encoding in ("PRIORITY", "START_TIME"):
        labels = derived.get(encoding)
        if not labels:
            continue
        lines.append(f"   under {encoding} it has {len(labels)} rows, in this order:")
        for position, label in enumerate(labels):
            lines.append(f"       [{position:>2}] {label}")
    return "\n".join(lines)


# Every configuration option the library actually has, with the functions each one
# switches on or off. Each description was written by READING THE BODY of the function
# named - not its docstring: five of these nine checks have none, and the other four
# state only the chromosome convention.
#
# This exists to BOUND the configuration stage. A real run produced an environment
# section that obtained schedules, converted representations, and listed evaluator
# methods as "environment" reuse decisions - all of it algorithm work. The remedy is a
# closed list that says which options are decisions. Exactly one is.
#
# Braces in this data are NOT doubled: the rendered text is substituted into the prompt
# AFTER .format() runs, so it is never reprocessed as a template.
_CONFIG_OPTION_MENU = (
    ("evaluator.py", (
        ("scheduling_type = ScheduleEncodingType.PRIORITY | ScheduleEncodingType.START_TIME",
         "YOURS TO DECIDE. The library never derives it, and its default is PRIORITY. "
         "START_TIME means a non-zero schedule cell is a transmission start time PLUS "
         "ONE - the stream begins transmitting on that port at that moment. PRIORITY "
         "means a non-zero cell is a queue priority - the stream sits in that queue on "
         "that port. The evaluator reads the cells one way or the other regardless, so "
         "a design that does not decide still gets an answer: under the wrong one every "
         "timing violation silently scores zero. If the requirements speak of schedules, "
         "windows, offsets, latency or transmission times, the answer is START_TIME.",
         (("PRIORITY -> _per_stream_priority_conflicts",
           "per stream, counts ports where another active stream holds the same cell "
           "value on that port - two streams in one queue."),
          ("PRIORITY -> _per_stream_latency_violations_priority",
           "per port, sorts active streams by cell value, accumulates all_hops_costs in "
           "that order as queueing delay, and flags a stream whose total exceeds "
           "stream_maxLatencies."),
          ("START_TIME -> _per_stream_window_sequence_violations",
           "two rows. Causality: a downstream port starting before upstream_start + "
           "hop_cost. FIFO: two streams whose arrival order at a port differs from "
           "their departure order."),
          ("START_TIME -> _per_stream_macrocycle_overlap_violations",
           "periodic collision for every stream pair sharing a port: (phase1-phase2) "
           "mod gcd(i1,i2) < window1+window2, where phase = start % interval."),
          ("START_TIME -> _per_stream_talker_offset_violations",
           "flags a stream whose talker-port start time is >= its own period."),
          ("START_TIME -> _per_stream_latency_violations_start_time",
           "end-to-end span (max start - min start) plus the largest window on any "
           "active port, flagged against stream_maxLatencies."))),
        ("check_routing = True | False",
         "NOT YOURS - derived, EXCEPT when the evaluator is constructed (step 2). "
         "initialize_problem computes not (fixPath and one-placement) from the "
         "activated use case's config, for problem.evaluator only. A constructed "
         "evaluator must be passed it: the same derivation via problem.get_task(), or "
         "True when a requirement makes the algorithm change or build routes.",
         (("_per_stream_flow_conservation_violations",
           "three rows: a port active as both egress and ingress at one node; "
           "|in-degree - out-degree| summed over switches; in-degree above 1 at a "
           "switch."),
          ("_per_stream_flow_continuity_violations",
           "union-finds the stream's active ports into groups, locates the talker's "
           "group and the listener-adjacent group, and charges group_penalty(group, "
           "stream_potential[s]) for each when the two are not the same group."))),
        ("check_task_placement = True | False",
         "NOT YOURS - derived the same way, from not one-placement, with the same "
         "exception: pass it to a constructed evaluator, derived, or True when the "
         "algorithm writes columns itself.",
         (("_per_stream_host_role_violations",
           "five rows: active talker ports != 1; active listener-adjacent ports != 1; a "
           "talker port that is also downstream of a listener; strays on each side via "
           "stray_talker_mask / stray_listener_mask."),)),
        ("weights = None | {penalty-name: number}",
         "LEAVE IT None unless a requirement names a trade-off. It merges over 15 "
         "defaults and is then filtered to only the keys the three options above left "
         "active.",
         (("TSNConstraintEvaluator.__init__",
           "sets self.weights, self.n_constr and self.violations_keys from the "
           "surviving keys; it changes no check's logic, only its cost."),)),
    )),
    ("tsn_scheduling_problem.py", (
        ("problem_definition = the config.ini path",
         "REUSE THE EXISTING ROOT config.ini BY PATH. It is the only argument that "
         "selects a scenario, and configparser.read() no-ops silently on a path that "
         "does not exist.",
         (("TSNScheduling.__init__",
           "stores it as self.config_file; reads nothing from it yet."),
          ("TSNScheduling.generate_scheduling_usecase / instantiate",
           "hands the path to UsecaseGenerator.generateUC, which builds ONE "
           "randomly-generated Usecase from it. Calling it twice gives two different "
           "scenarios."),
          ("TSNScheduling.get_task",
           "the only config reader in this file. Reads "
           "streams_config.routing['fixPath'], numplacesTalker and numplacesListener, "
           "and returns the two booleans that decide check_routing and "
           "check_task_placement."))),
    )),
    ("usecase_generator.py", (
        ("every [topology_config] and [streams_config] key",
         "NOT A DECISION HERE. These are read only while a Usecase is being generated, "
         "which happens once from the config path above. Do not restate them as "
         "environment steps.",
         (("Usecase.__init__",
           "builds self.config, a RawConfigParser, and reads the file. Every config "
           "value in the library is reached through this one object."),
          ("UsecaseGenerator.streamGenerator / generateStream",
           "reads numofstreamsperinstance, period_range, numberframes_range, "
           "datasize_range, maxlatency_range, expiration, routing to build the stream "
           "set."),
          ("UsecaseGenerator.talkerEmbedding / listenerEmbedding",
           "reads numplacesTalker, numplacesListener, numlistenersperstr, stride, "
           "staggered, hotspot to place talkers and listeners."),
          ("UsecaseGenerator.makeTopology / add_link",
           "reads graphconfig, portspeed, propdelay, procdelay, linkBw, "
           "queuecountperport to build the graph and the per-link data the evaluator's "
           "window lengths come from."))),
    )),
)


def _verify_config_option_menu(lib_dir: str) -> list[str]:
    """Confirm every function the configuration menu names still exists in the library.

    Same contract as _verify_fixed_representations: the menu is asserted as fact, so
    something has to notice when the library moves under it. Matched on the last dotted
    segment, because the menu writes some names as Class.method for readability.
    """
    index = _library_member_index(lib_dir)
    present: set = set()
    for module in index.values():
        present.update(module["functions"])
        for cls in module["classes"].values():
            present.add(cls["qualname"].rsplit(".", 1)[-1])
            present.update(cls["methods"])

    notes: list[str] = []
    for _filename, options in _CONFIG_OPTION_MENU:
        for option, _status, acts_upon in options:
            for label, _what in acts_upon:
                for piece in label.split("->")[-1].split("/"):
                    name = piece.strip().rsplit(".", 1)[-1]
                    if name and name not in present:
                        notes.append(
                            f"recorded KB fact is stale: the configuration menu names "
                            f"'{name}' under '{option.split('=')[0].strip()}', which "
                            f"the library no longer has"
                        )
    return notes


def _render_config_option_menu() -> str:
    """The menu as prompt text: file name, then the option, then what acts upon it."""
    lines: list[str] = []
    for filename, options in _CONFIG_OPTION_MENU:
        lines.append(f"   {filename}")
        for option, status, acts_upon in options:
            lines.append(f"       {option}")
            for wrapped in textwrap.wrap(f"[{status}]", 84):
                lines.append(f"           {wrapped}")
            for label, what in acts_upon:
                lines.append(f"           {label}")
                for wrapped in textwrap.wrap(what, 80):
                    lines.append(f"               {wrapped}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _verify_fixed_representations(lib_dir: str) -> list[str]:
    """Confirm each recorded representation still has its named witnesses in the library.

    The same contract as `_verify_start_time_converters`: the catalogue above is asserted
    as fact, so something has to notice when the library moves out from under it. Checks
    existence of the classes and members each entry is built on, not their full
    behaviour — a renamed or deleted witness is the drift that actually happens.
    """
    index = _library_member_index(lib_dir)
    present: set = set()
    for module in index.values():
        present.update(module["functions"])
        for cls in module["classes"].values():
            present.add(cls["qualname"].rsplit(".", 1)[-1])
            present.update(cls["methods"])

    notes: list[str] = []
    for entry in _LIBRARY_FIXED_REPRESENTATIONS:
        missing = [w for w in entry["witness"] if w not in present]
        if missing:
            notes.append(
                f"recorded KB fact is stale: representation '{entry['id']}' is "
                f"described in terms of {missing}, which the library no longer has"
            )
    return notes


def _render_fixed_representations() -> str:
    """The catalogue above as prompt text. One renderer, so the configuration prompt and
    anything else that needs these facts cannot drift apart the way four hand-copied
    blocks of KB documentation already did in this file."""
    lines: list[str] = []
    for entry in _LIBRARY_FIXED_REPRESENTATIONS:
        lines.append(f'   - "{entry["id"]}" — {entry["what"]}')
        if entry.get("codes"):
            lines.append(f'       WRITTEN AS CODE: {" or ".join(entry["codes"])}')
        if entry.get("stream_part"):
            lines.append(f'       ONE STREAM\'S PART: {entry["stream_part"]} — '
                         f'{entry["stream_part_what"]}')
        for convention in entry["conventions"]:
            lines.append(f"       HOW IT IS INDEXED: {convention}")
        for limit in entry["limits"]:
            lines.append(f"       WHAT IT CANNOT DO: {limit}")
    return "\n".join(lines)


# The two things the library keeps as plain dicts rather than classes, and therefore the
# two "types" that no class catalogue can show. Each is found by an ANCHOR KEY — a key
# that only that record has — so the key sets below are read out of the source instead of
# being written down here and left to rot.
_LIBRARY_RECORD_KINDS = (
    {
        # Named by the code that reaches the dict, not by an invented label: a label
        # ("stream record") hid that two objects of different shape answer to it.
        "name": "Usecase.streams[i]",
        "reached_as": ("the stored stream dict; i is its position in the list, which is "
                       "also candidate_array column i and Slot.stream_id. Its 'pathLists' "
                       "holds paths of egress port NAMES (as written by shortestPath / "
                       "randomPath, each ends in the listener NODE name; as written by "
                       "Schedule.update_streams, one path per listener, starting at the "
                       "talker port). TSNScheduling.streams[i] is a "
                       "fresh COPY on every read, whose 'pathLists' holds 1-based port "
                       "IDS with the last element of each path dropped"),
        "anchor": "maxLatency",
        "bases": ("stream", "streams", "_streams"),
    },
    {
        # The port family is TWO dicts, and the source touches both under the same
        # variable names, so the anchor/bases scan returns their union. The catalogue
        # splits it (see _usecase_ports_view_keys): the keys Usecase.ports builds belong
        # to that view; the rest live only on the stored graph dict.
        "name": "Usecase.ports[name]",
        "reached_as": ("a NEW dict built on every read of the property, keyed by port "
                       "NAME (node + '.' + local port name), from the graph's edges and "
                       "stored port dicts - anything written to it is lost. "
                       "TSNScheduling.ports[pid] holds the same fields keyed by 1-based "
                       "port id"),
        "anchor": "queueCount",
        "bases": ("port", "ports", "port_data", "portData", "pdata"),
        "stored_as": "usecase.graph.nodes[node]['ports'][pname]",
        "stored_reached_as": ("the per-port dict stored on the topology graph, keyed by "
                              "the node's local port name; Schedule.update_graph writes "
                              "the schedule keys into it and Usecase.removeSchedule "
                              "deletes them"),
    },
)


def _usecase_ports_view_keys(lib_dir: str) -> tuple:
    """(keys the Usecase.ports property builds, stored keys it copies in), from source.

    The property builds each entry from an edge dict literal plus an .update(...) call,
    and reads the copied fields off the stored port dict it iterates as
    `for pname, pdata in data['ports'].items()`. Returns (set(), set()) when the
    property is not shaped that way any more, and _verify_library_resources says so.
    """
    path = os.path.join(lib_dir, "usecase_generator.py")
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError):
        return set(), set()
    prop = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Usecase":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "ports":
                    prop = item
    if prop is None:
        return set(), set()
    view: set = set()
    stored_vars: set = set()
    for node in ast.walk(prop):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Subscript) for t in node.targets)):
            view |= {k.value for k in node.value.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"):
            view |= {kw.arg for kw in node.keywords if kw.arg}
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                and len(node.target.elts) == 2 and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "items"
                and isinstance(node.iter.func.value, ast.Subscript)
                and isinstance(node.iter.func.value.slice, ast.Constant)
                and node.iter.func.value.slice.value == "ports"
                and isinstance(node.target.elts[1], ast.Name)):
            stored_vars.add(node.target.elts[1].id)
    copied = {n.slice.value for n in ast.walk(prop)
              if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
              and n.value.id in stored_vars and isinstance(n.slice, ast.Constant)
              and isinstance(n.slice.value, str)}
    return view, copied


def _library_record_keys(lib_dir: str) -> dict:
    """The key set of each dict-record type in _LIBRARY_RECORD_KINDS, read from source.

    Two sources are unioned, because neither alone is complete:

      * the dict LITERALS that create the record — authoritative for the base schema,
        found by the kind's anchor key (`{...'maxLatency'...}` can only be a stream);
      * every string-literal SUBSCRIPT the library takes on a variable named after the
        record (`stream['pathLists']`), which catches the keys attached after creation —
        and several are: usecase_generator sets `pathLists`, `offset`, `accLatency` and
        `lifespanCycles` on a stream well after the literal that made it.

    So the result is "the keys this codebase actually touches", which is the honest
    claim, and not "the schema", which nothing in the library declares anywhere.
    """
    found: dict = {kind["name"]: set() for kind in _LIBRARY_RECORD_KINDS}
    by_anchor = {kind["anchor"]: kind["name"] for kind in _LIBRARY_RECORD_KINDS}
    by_base = {base: kind["name"]
               for kind in _LIBRARY_RECORD_KINDS for base in kind["bases"]}

    for filename in sorted(os.listdir(lib_dir)):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        try:
            with open(os.path.join(lib_dir, filename), encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and node.keys and all(
                    isinstance(k, ast.Constant) and isinstance(k.value, str)
                    for k in node.keys):
                keys = [k.value for k in node.keys]
                for anchor, name in by_anchor.items():
                    if anchor in keys:
                        found[name].update(keys)
            elif (isinstance(node, ast.Subscript)
                  and isinstance(node.slice, ast.Constant)
                  and isinstance(node.slice.value, str)):
                base = ast.unparse(node.value).split(".")[-1].split("[")[0]
                if base in by_base:
                    found[by_base[base]].add(node.slice.value)
    return found


def _library_resource_catalog(lib_dir: str) -> dict:
    """Every resource of the knowledge base a requirement's noun could BE, from source.

    Three groups, because a query noun can land on three different kinds of thing and
    only one of them is a class:

      forms    — what a schedule IS. Taken from _LIBRARY_FIXED_REPRESENTATIONS, which
                 _verify_fixed_representations already checks against the library, so the
                 requirements stage and the configuration stage cannot disagree about
                 what the available forms even are.
      classes  — every class the library defines, with its base and its real __init__
                 parameters. unittest.TestCase subclasses are dropped: they are the test
                 harness, not resources an algorithm is built out of, and that is
                 readable off the base list rather than being a hand-kept exclusion.
      records  — the dict-record types, from _library_record_keys above.

    Nothing here is editorial. A name in this catalogue exists in the library source, and
    _verify_library_resources re-checks that on every run.
    """
    index = _library_member_index(lib_dir)

    forms = []
    for entry in _LIBRARY_FIXED_REPRESENTATIONS:
        # First sentence only: the full conventions and limits belong to the
        # configuration prompt, which is the stage that chooses between these.
        what = entry["what"].split(". ")[0].rstrip(".")
        forms.append({"id": entry["id"], "what": what, "codes": entry.get("codes", ()),
                      "stream_part": entry.get("stream_part", ""),
                      "stream_part_what": entry.get("stream_part_what", "")})

    classes = []
    for module_name, module in sorted(index.items()):
        for class_name, cls in sorted(module["classes"].items()):
            bases = [ast.unparse(b) for b in cls["node"].bases]
            if any("TestCase" in b for b in bases):
                continue
            init = cls["methods"].get("__init__")
            if init is not None:
                args, shape = [a.arg for a in init.args.args[1:]], "init"
            else:
                # A dataclass has no __init__ in the source; its parameters ARE its
                # annotated class-level fields, and showing "__init__()" for Slot would
                # say it takes nothing when it takes six.
                args = [item.target.id for item in cls["node"].body
                        if isinstance(item, ast.AnnAssign)
                        and isinstance(item.target, ast.Name)]
                shape = "fields"
                if not args and any("Enum" in b for b in bases):
                    # An Enum's members are plain assignments, and they are the whole
                    # content of the type: ScheduleEncodingType is START_TIME or
                    # PRIORITY, and that is exactly what a requirement would bind to.
                    args = [t.id for item in cls["node"].body
                            if isinstance(item, ast.Assign)
                            for t in item.targets if isinstance(t, ast.Name)]
                    shape = "members"
            classes.append({
                "name": cls["qualname"],
                "module": module_name,
                "bases": bases,
                "init_args": args,
                "shape": shape,
            })

    records = []
    found_keys = _library_record_keys(lib_dir)
    for kind in _LIBRARY_RECORD_KINDS:
        keys = set(found_keys.get(kind["name"], ()))
        stored: set = set()
        if kind.get("stored_as"):
            view, copied = _usecase_ports_view_keys(lib_dir)
            if view:
                stored = (keys - view) | copied
                keys = view
        if keys:
            records.append({"name": kind["name"],
                            "reached_as": kind["reached_as"],
                            "keys": sorted(keys)})
        if stored:
            records.append({"name": kind["stored_as"],
                            "reached_as": kind["stored_reached_as"],
                            "keys": sorted(stored)})

    return {"forms": forms, "classes": classes, "records": records}


def _render_library_resources(lib_dir: str) -> str:
    """The catalogue above as prompt text for stage 0b."""
    catalog = _library_resource_catalog(lib_dir)
    lines: list[str] = []

    lines.append("  SCHEDULE FORMS — what a schedule IS in this codebase. These are the")
    lines.append("  only forms there are, so every schedule in the query is one of them and")
    lines.append("  none of them is null. WHICH one a design uses is settled by the")
    lines.append("  configuration stage after this one: name all the forms that would be")
    lines.append("  correct and let that stage choose between them.")
    lines.append("     IN THIS CODEBASE A SCHEDULE IS THE FORWARDING/SCHEDULING")
    lines.append("     CONFIGURATION. It is not a thing that stands for one: the windows a")
    lines.append("     schedule holds ARE what a port is configured with. So a start, a")
    lines.append("     finish, a target and an intermediate configuration are all schedules,")
    lines.append("     all one of the forms below, and there is no separate configuration")
    lines.append("     object anywhere in the library for one of them to point at.")
    # problem_and_usecase sits in _LIBRARY_FIXED_REPRESENTATIONS for the configuration
    # stage, but it is not a schedule: listed here it read as a third schedule form, and
    # models bound schedules to it.
    schedules = [f for f in catalog["forms"] if f["id"] != "problem_and_usecase"]
    others = [f for f in catalog["forms"] if f["id"] == "problem_and_usecase"]
    for form in schedules:
        lines.append(f'    {form["id"]:20} {form["what"]}')
        if form["codes"]:
            lines.append(f'    {"":20} written as code: {" or ".join(form["codes"])}')
        if form["stream_part"]:
            lines.append(f'    {"":20} one stream\'s part of it: {form["stream_part"]} '
                         f'({form["stream_part_what"]})')

    for form in others:
        lines.append("")
        lines.append("  THE USE CASE AND ITS PROBLEM — not a schedule: the network and streams a")
        lines.append("  schedule is defined against.")
        lines.append(f'    {form["what"]}')

    lines.append("")
    lines.append("  CLASSES, under the file that defines them")
    current = None
    for cls in catalog["classes"]:
        if cls["module"] != current:
            current = cls["module"]
            lines.append(f"    {current}.py")
        base = f"({', '.join(cls['bases'])})" if cls["bases"] else "(no base)"
        label = {"init": "__init__", "fields": "fields",
                 "members": "members"}[cls["shape"]]
        args = f"{label}({', '.join(cls['init_args'])})"
        lines.append(f"        {cls['name']:24} {base:28} {args}")

    if catalog["records"]:
        lines.append("")
        lines.append("  DICTS — the library keeps streams and ports as plain dicts, so they")
        lines.append("  have no class name: each is named below by the code that reaches it.")
        lines.append("  Keys below are the ones its own source touches, and a single key is")
        lines.append("  itself bindable as that code plus the key. Usecase.streams[i]['pathLists']")
        lines.append("  is the routing of the schedule CURRENTLY installed in the use case —")
        lines.append("  Schedule.update_streams rewrites it from whatever schedule is written in —")
        lines.append("  so a route that differs between two schedules is read off EACH schedule,")
        lines.append("  as that schedule's part for the stream (listed with the forms above),")
        lines.append("  never off this key.")
        for record in catalog["records"]:
            lines.append(f'    {record["name"]} — reached as {record["reached_as"]}')
            lines.append(f'        keys: {", ".join(record["keys"])}')

    return "\n".join(lines)


def _verify_library_resources(lib_dir: str) -> list[str]:
    """Confirm the catalogue still describes the library, rather than a past version.

    Cheap, and worth it: this catalogue is handed to the model as the closed set of
    things a requirement's noun may bind to, so a name in it that no longer exists would
    invite a binding nothing downstream could resolve.
    """
    notes: list[str] = []
    catalog = _library_resource_catalog(lib_dir)
    index = _library_member_index(lib_dir)

    known = {name for module in index.values() for name in module["classes"]}
    for cls in catalog["classes"]:
        leaf = cls["name"].split(".")[-1]
        if leaf not in known:
            notes.append(f"resource catalogue names class '{cls['name']}', which the "
                         f"library no longer defines")
    if not catalog["forms"]:
        notes.append("resource catalogue has no schedule forms; "
                     "_LIBRARY_FIXED_REPRESENTATIONS is empty")
    for record in catalog["records"]:
        if len(record["keys"]) < 2:
            notes.append(f"resource catalogue found only {len(record['keys'])} key(s) "
                         f"for the {record['name']}; its anchor key may have been renamed")
    if not _usecase_ports_view_keys(lib_dir)[0]:
        notes.append("recorded KB fact is stale: Usecase.ports is no longer built the way "
                     "_usecase_ports_view_keys reads it, so the catalogue cannot split the "
                     "port keys between Usecase.ports[name] and the stored graph dict")
    return notes


# What the library already DOES, as triples over the same twenty predicates the
# requirements prompt uses. Read off the source of all 130 functions in
# original_tsn_lib_files/; a member is listed only when it already does something this
# kind of query asks for, so plotting, JSON import/export, topology generation and the
# unittest helpers are absent by that test rather than by oversight.
#
# NODE NAMES ARE THE CATALOGUE'S NAMES, deliberately: "candidate_array",
# "compiled_schedule", "problem_and_usecase", "Usecase.streams[i]", "Usecase.ports[name]"
# and "usecase.graph.nodes[node]['ports'][pname]" are the same strings
# _render_library_resources() offers for corresponds_to; TSNScheduling.streams[i] and
# TSNScheduling.ports[pid] are the code of the per-read copies, named where a fact is
# about them. That is what makes
# the two halves one graph — a resource bound to candidate_array lands on the node these
# triples are about.
#
# Facts marked NOT DERIVABLE were read out of a function body and cannot be recovered
# from a signature. They are the reason this table exists, and the reason
# _verify_library_facts() re-checks every named member on every run.
_LIBRARY_FACT_TRIPLES = (
    # ---------------------------------------------------------------- activation
    ("TSNScheduling.use_instance", (
        ("hasInput", "Usecase", ""),
        ("hasOutput", "an activation context", ""),
        ("mutates", "TSNScheduling", "sets .usecase on entry and to None on exit, "
                                     "unconditionally — it is NOT reentrant, so a "
                                     "nested with-block closes the outer one"),
    )),
    ("every TSNScheduling view", (
        ("precondition", "an open activation context", "n_ports, n_streams, ports, "
         "streams, macro_cycle, port_id_to_name, port_name_to_id, node_to_port_id, "
         "hops_costs, make_stream_config, bfs_forward, bfs_backward and the rest are "
         "wrapped in @require_instance"),
        ("raises", "being read with no use case set", "no_instance_assigned() raises "
         "ValueError, it does not return a default"),
    )),
    ("TSNScheduling.build_metadata", (
        ("hasOutput", "config_matrix", ""),
        ("mutates", "Usecase.streams[i]", "attaches 'customGraph' to every stream"),
        ("precedes", "TSNScheduling.initialize_problem", "it calls it at the end, so "
         "the evaluator exists only after metadata is built"),
    )),
    ("TSNScheduling.initialize_problem", (
        ("hasOutput", "TSNConstraintEvaluator", "and it NEVER passes scheduling_type, "
         "so problem.evaluator is always PRIORITY"),
    )),
    ("TSNScheduling.generate_scheduling_usecase", (
        ("hasOutput", "Usecase", ""),
    )),

    # ------------------------------------------------- schedule <-> array conversion
    ("Schedule.from_compact_encoding", (
        ("hasInput", "candidate_array", ""),
        ("hasInput", "TSNConstraintEvaluator.windLengths", "the durations are NOT in "
         "the array; this argument supplies them"),
        ("hasInput", "TSNConstraintEvaluator.stream_intervals", "and this supplies the "
         "periods"),
        ("hasOutput", "compiled_schedule", "a nonzero cell is start_ns + 1; port_id is "
         "the row + 1; queue_id is tt_queue_id, default 0"),
    )),
    ("Schedule.to_compact_encoding", (
        ("hasInput", "compiled_schedule", ""),
        ("hasOutput", "candidate_array", "writes start_ns + 1, and skips any slot whose "
         "stream_id or port_id is None"),
        ("raises", "an empty schedule", "ValueError('Schedule is empty')"),
        ("raises", "two slots for one (port, stream)", "ValueError('Multiple slots "
         "found ...'). This is the hard limit on replicating one stream twice over the "
         "same port — it refuses rather than truncating"),
    )),
    ("Schedule.from_usecase", (
        ("hasInput", "Usecase", ""),
        ("hasInput", "TSNScheduling", "through its port_name_to_id mapping"),
        ("hasOutput", "compiled_schedule", "built from the windows already written into "
         "each usecase.graph.nodes[node]['ports'][pname], keeping the earliest slot per "
         "(port, stream, queue, phase)"),
    )),

    # ------------------------------------------- installing and removing a schedule
    ("Usecase.addSchedule", (
        ("hasInput", "compiled_schedule", ""),
        ("mutates", "Usecase", "sets .schedule, then calls update_graph and "
         "update_streams, so ONE call installs a whole configuration"),
        ("precondition", "Usecase.removeSchedule having been called first", "update_graph "
         "only ever APPENDS windows, so installing a second schedule over a first "
         "accumulates both instead of replacing"),
    )),
    ("Usecase.removeSchedule", (
        ("mutates", "Usecase", "clears .schedule and deletes 'schedule', 'basetime' "
         "and 'scheduleCycle' from every usecase.graph.nodes[node]['ports'][pname], and "
         "'offset' and 'accLatency' from every Usecase.streams[i]. It does NOT delete "
         "'pathLists'"),
    )),
    ("Schedule.update_graph", (
        ("mutates", "usecase.graph.nodes[node]['ports'][pname]", "writes 'basetime' and "
         "'scheduleCycle' and appends "
         "to 'schedule'; merges touching windows of identical state, and skips a window "
         "already present verbatim"),
    )),
    ("Schedule.update_streams", (
        ("mutates", "Usecase.streams[i]", "DERIVES 'pathLists' from the slots, plus 'offset', "
         "'accLatency', 'tlkrPssblPlcmnt' and 'lstnsrPssblPlcmnt'. This is how a path is "
         "read back out of a schedule"),
        ("raises", "a stream with one slot or none", "it reads a loop variable after the "
         "loop, so a stream with no hop beyond the talker gives NameError"),
    )),

    # ---------------------------------------------------------- reading a schedule
    ("Schedule.slots_by_stream", (
        ("hasInput", "Usecase.streams[i]", "by its index i"),
        ("hasOutput", "Slot", "every slot of one stream"),
    )),
    ("Schedule.slots_by_port", (
        ("hasOutput", "Slot", "every slot of one port"),
    )),
    ("Schedule.validate_conflicts", (
        ("hasOutput", "Slot", "the PAIRS that overlap, per port, over every "
         "occurrence in the cycle. This is the library's own answer to whether a "
         "configuration is internally valid"),
    )),
    ("Slot.conflicts_with", (
        ("hasInput", "Slot", "another slot on the same port"),
        ("hasOutput", "whether two windows overlap", "touching_allowed=True treats "
         "end == start as no conflict"),
    )),
    ("Slot.all_occurrences", (
        ("hasOutput", "every repetition of one window inside a span", ""),
    )),
    ("Slot.duration_ns", (
        ("hasOutput", "end_ns minus start_ns", ""),
    )),
    ("Slot.__lt__", (
        ("hasOutput", "whether this window ends before another starts", "this is NOT a "
         "total order: two overlapping slots are neither less nor greater, so sorting a "
         "list of Slots by it is unreliable"),
    )),

    # --------------------------------------------------------------- evaluation
    ("TSNConstraintEvaluator.__init__", (
        ("hasInput", "TSNScheduling", ""),
        ("hasInput", "ScheduleEncodingType", "PRIORITY or START_TIME; it selects which "
         "checks run and nothing else does"),
        ("precondition", "an open activation context", "every attribute it reads off "
         "the problem is instance-gated"),
    )),
    ("TSNConstraintEvaluator.evaluate", (
        ("hasInput", "candidate_array", ""),
        ("hasOutput", "violation counts per violation KIND", "already summed over "
         "streams, so per-stream attribution is NOT recoverable from it"),
        ("hasOutput", "six metrics", "congestion_score, avg_util, std_util, "
         "load_imbalance, path_len_score, avg_path_len, in that order"),
    )),
    ("TSNConstraintEvaluator.per_stream_all_violations", (
        ("hasInput", "candidate_array", ""),
        ("hasOutput", "violations per kind PER STREAM", "shape (n_constr, n_streams). "
         "This is the only way to ask which stream violated what"),
    )),
    ("TSNConstraintEvaluator.feasible_streams", (
        ("hasInput", "candidate_array", ""),
        ("hasOutput", "one boolean per stream", ""),
        ("precondition", "one egress port per stream per switch", "its routing checks "
         "are |in_degree - out_degree| and max(0, in_degree - 1) over switches, so a "
         "stream deliberately carried on TWO paths is reported infeasible. It cannot be "
         "used to approve a replicated state"),
    )),
    ("TSNConstraintEvaluator.compute_metrics", (
        ("hasOutput", "six metrics", ""),
        ("raises", "nothing, but its first three values are constants", "it divides by "
         "self.macro_cycle, which is getattr(usecase, 'macro_cycle', 0) and Usecase has "
         "NO macro_cycle — so the divisor is eps and congestion_score, avg_util and "
         "std_util are all 1.0 for every non-empty schedule. Only load_imbalance, "
         "path_len_score and avg_path_len carry information"),
    )),
    ("TSNConstraintEvaluator.macro_cycle", (
        ("hasOutput", "zero, always", "it is set as getattr(problem.usecase, "
         "'macro_cycle', 0) and Usecase has no macro_cycle, so the attribute EXISTS and "
         "holds 0. A getattr(evaluator, 'macro_cycle', <fallback>) therefore NEVER "
         "reaches its fallback — the name resolves, to the useless value. USE "
         "TSNScheduling.macro_cycle instead: that one computes the real LCM, and it is "
         "the only source of it in the library"),
    )),
    ("TSNConstraintEvaluator.group_connected_ports", (
        ("hasInput", "candidate_array"
         , ""),
        ("hasOutput", "the connected groups of one stream's active ports", "1-based "
         "port ids, grouped by the real adjacency"),
    )),
    ("TSNConstraintEvaluator.windLengths", (
        ("hasOutput", "the window duration of every (port, stream)", "shape "
         "(n_ports, n_streams); the only public source of durations"),
    )),
    ("TSNConstraintEvaluator.stream_intervals", (
        ("hasOutput", "the period of every stream", ""),
    )),
    ("TSNConstraintEvaluator.all_hops_costs", (
        ("hasOutput", "the forwarding cost of every (port, stream)", "what a start time "
         "must advance by between consecutive hops"),
    )),
    ("TSNConstraintEvaluator.next_hop_list", (
        ("hasOutput", "the egress ports reachable from each egress port", ""),
    )),

    # -------------------------------------------------------------------- paths
    ("UsecaseGenerator.shortestPath", (
        ("hasInput", "Usecase.streams[i]", ""),
        ("hasInput", "Usecase", ""),
        ("mutates", "Usecase.streams[i]", "sets 'pathLists' to shortest paths as egress port "
         "NAMES, and also rewrites 'interval' and 'latest-transmit-offset'"),
    )),
    ("UsecaseGenerator.randomPath", (
        ("mutates", "Usecase.streams[i]", "as shortestPath, but picks randomly and also "
         "rewrites 'lstnsrPssblPlcmnt'"),
    )),
    ("UsecaseGenerator.findAllPaths", (
        ("hasOutput", "every simple path between each talker and listener", "as "
         "generators keyed by the pair — the candidate set a path change chooses from"),
    )),
    ("UsecaseGenerator.pathConvenient", (
        ("hasInput", "a path as egress port names", ""),
        ("hasOutput", "whether the path's minimum latency fits maxLatency", "the "
         "library's own test of whether a candidate path is usable at all"),
        ("raises", "nothing, but it UNDERSTATES latency", "a bare except skips any hop "
         "whose edge or procDelay lookup fails, so a path can pass on partial data"),
    )),
    ("UsecaseGenerator.periodConvenient", (
        ("hasOutput", "whether one frame fits inside the cycle", ""),
        ("raises", "a path of fewer than two hops", "it indexes pathList[i+1] first"),
    )),
    ("TSNScheduling.hops_costs", (
        ("hasInput", "a frame size in bits", ""),
        ("hasOutput", "the forwarding cost of every port", "transmission + propagation "
         "+ processing"),
    )),
    ("TSNScheduling.hop_cost_eg2eg", (
        ("hasOutput", "the cost from one egress port to the next", "takes port NAMES, "
         "not ids"),
    )),
    ("TSNScheduling.bfs_forward", (
        ("hasOutput", "the minimum delay to reach each egress port from the talker", ""),
    )),
    ("TSNScheduling.bfs_backward", (
        ("hasOutput", "the delay still available at each egress port", "counted down "
         "from the stream's maxLatency, so a negative value means the deadline is "
         "already missed"),
    )),
    ("Usecase.window_length", (
        ("hasOutput", "the transmission duration of one stream on one port", ""),
    )),
    ("TSNScheduling.macro_cycle", (
        ("hasOutput", "the LCM of every stream interval", "the span a schedule must "
         "repeat over, and the ONLY place the real value is computed. Reach for it "
         "here, not off the evaluator: the evaluator has an attribute of the same name "
         "that is always 0, so a fallback written against it silently keeps the zero"),
    )),
    ("TSNScheduling.streams", (
        ("hasOutput", "TSNScheduling.streams[i]", "a fresh copy of Usecase.streams[i] with "
         "port IDS: 'pathLists' become port ids and "
         "the LAST hop of each path is dropped; on a Usecase the same key holds port "
         "NAMES and keeps every hop"),
    )),
    ("TSNScheduling.ports", (
        ("hasOutput", "TSNScheduling.ports[pid]", "keyed by 1-based port id; built from "
         "Usecase.ports[name], which is keyed by port NAME"),
    )),

    # ------------------------------------------- members that exist but cannot be used
    ("Schedule.total_occupied_time", (
        ("raises", "being read at all", "AttributeError: it calls self.compile(), and "
         "Schedule has no compile method"),
    )),
    ("Schedule.utilization", (
        ("raises", "being read at all", "it calls total_occupied_time as if it were a "
         "method, and that property is itself broken"),
    )),
    ("TSNScheduling.next_prev_hops", (
        ("raises", "being read at all", "declared @property yet takes an argument, and "
         "its body uses a local before assigning it. Nothing can call it, despite the "
         "name being exactly what a path neighbour lookup would want"),
    )),
    ("Usecase.add_link", (
        ("raises", "being called", "NameError on undefined names in its body"),
    )),
)

def _render_library_facts() -> str:
    """The fact table above as prompt text, grouped by subject so a subject is named once."""
    lines: list[str] = []
    for subject, triples in _LIBRARY_FACT_TRIPLES:
        lines.append(f"    {subject}")
        for predicate, target, note in triples:
            lines.append(f"        {predicate:14} {target}")
            if not note:
                continue
            # The note sits under its own triple, bracketed, wrapped to the prompt's
            # width and indented past the predicate column so the triples stay scannable.
            wrapped = textwrap.wrap(note, width=64)
            for position, chunk in enumerate(wrapped):
                opener = "(" if position == 0 else " "
                closer = ")" if position == len(wrapped) - 1 else ""
                lines.append(f"                      {opener}{chunk}{closer}")
    return "\n".join(lines)


def _verify_library_facts(lib_dir: str) -> list[str]:
    """Confirm every member these recorded facts name still exists in the library.

    The facts were read out of function bodies and cannot be re-derived, so this is the
    only thing standing between them and silent rot — the same reason
    _verify_fixed_representations, _verify_violation_row_order, _verify_config_option_menu
    and _verify_start_time_converters exist. It checks EXISTENCE, which is cheap and
    catches the failure that actually happens: a member renamed or removed while the
    table still describes it, so the prompt would be handing the model a fact about
    something that is no longer there.

    Subjects that are deliberately prose rather than a member ("every TSNScheduling
    view") are skipped, since there is nothing to look up.
    """
    index = _library_member_index(lib_dir)
    members: set = set()
    classes: set = set()
    for module in index.values():
        members.update(module["functions"])
        for class_name, cls in module["classes"].items():
            classes.add(class_name)
            members.update(cls["methods"])

    notes: list[str] = []
    for subject, _ in _LIBRARY_FACT_TRIPLES:
        if " " in subject:
            continue                      # a prose subject, nothing to resolve
        owner, _, member = subject.rpartition(".")
        if owner and owner not in classes:
            notes.append(f"recorded library fact names '{subject}', but the library has "
                         f"no class '{owner}' any more")
            continue
        # An attribute set in __init__ is not in the method index, so a miss there is
        # only reportable when the name is not a known attribute either.
        if member not in members and not _library_sets_attribute(lib_dir, owner, member):
            notes.append(f"recorded library fact names '{subject}', which the library "
                         f"no longer defines as a method or an attribute")
    return notes


def _library_sets_attribute(lib_dir: str, class_name: str, attribute: str) -> bool:
    """Whether a library class assigns `self.<attribute>` anywhere in its body.

    Needed because several facts are about precomputed ARRAYS rather than methods —
    TSNConstraintEvaluator.windLengths and .stream_intervals are set in __init__ and
    would otherwise read as missing.
    """
    index = _library_member_index(lib_dir)
    for module in index.values():
        cls = module["classes"].get(class_name)
        if not cls:
            continue
        for node in ast.walk(cls["node"]):
            if (isinstance(node, ast.Attribute) and node.attr == attribute
                    and isinstance(node.ctx, ast.Store)):
                return True
    return False


def _verify_start_time_converters(lib_dir: str) -> list[str]:
    """Confirm the recorded KB fact still holds: each named converter exists and still
    shifts by one against `start_ns`. Returns notes about anything that has drifted."""
    index = _library_member_index(lib_dir)
    notes: list[str] = []
    for member in _START_TIME_CONVERTERS:
        node = None
        for module in index.values():
            for cls in module["classes"].values():
                if member in cls["methods"]:
                    node = cls["methods"][member]
        if node is None:
            notes.append(f"recorded KB fact is stale: no library member named "
                         f"'{member}' exists any more")
            continue
        source = ast.unparse(node)
        if "start_ns" not in source:
            notes.append(f"recorded KB fact is stale: '{member}' no longer mentions "
                         f"start_ns, so it may no longer imply START_TIME")
    return notes


def _encoding_pairing_violations(code: str, lib_dir: str) -> list[str]:
    """Code that converts chromosomes with the library's START_TIME converters while
    scoring them with a PRIORITY evaluator.

    This replaces an earlier check that asked whether the design had RECORDED an
    encoding decision. That was the wrong question. The defect is not an unmade
    decision — it is a mismatch between two things the file uses, and it is a mismatch
    the library itself sets up by shipping a PRIORITY default for the only chromosome
    format it can produce. So this checks the PAIRING.

    Obtaining the pre-built evaluator is detected in all three forms a real run has
    used: `problem.evaluator`, `getattr(problem, "evaluator", ...)` — the form that
    defeated the previous check — and an explicit construction passing PRIORITY.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    members = _library_enum_members(lib_dir)
    priority_names = {name for name, enum in members.items() if name == "PRIORITY"}
    starttime_names = {name for name, enum in members.items() if name == "START_TIME"}

    def enum_member(node) -> Optional[str]:
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        return None

    converters: list[tuple] = []
    priority_sites: list[tuple] = []
    explicit_start_time = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = (node.func.attr if isinstance(node.func, ast.Attribute)
                      else node.func.id if isinstance(node.func, ast.Name) else None)
            if called in _START_TIME_CONVERTERS:
                converters.append((node.lineno, called))
            if called == "getattr" and len(node.args) >= 2 \
                    and isinstance(node.args[1], ast.Constant) \
                    and node.args[1].value == "evaluator":
                priority_sites.append((node.lineno, 'getattr(..., "evaluator")'))
            for keyword in node.keywords:
                if keyword.arg != "scheduling_type":
                    continue
                member = enum_member(keyword.value)
                if member in starttime_names:
                    explicit_start_time = True
                elif member in priority_names:
                    priority_sites.append(
                        (node.lineno, f"an evaluator built with {member}")
                    )
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) \
                and node.attr == "evaluator":
            priority_sites.append((node.lineno, f"{ast.unparse(node)}"))

    if not converters or not priority_sites or explicit_start_time:
        return []

    problems = _verify_start_time_converters(lib_dir)
    converter_line, converter_name = converters[0]
    site_line, site_text = priority_sites[0]
    problems.append(
        f"line {site_line}: `{site_text}` is a PRIORITY evaluator — "
        f"initialize_problem builds it without scheduling_type — but line "
        f"{converter_line} uses `{converter_name}`, which is START_TIME in both "
        f"directions (a nonzero cell is start_ns plus one). The two cannot be used "
        f"together: under PRIORITY the four timing checks never run and start times "
        f"are scored as queue priorities, so every timing violation reads zero. Build "
        f"the evaluator explicitly with scheduling_type=ScheduleEncodingType.START_TIME "
        f"inside the activation block, or stop using the compact-encoding converters"
    )
    return problems


def _library_enum_members(lib_dir: str) -> dict:
    """Every member of every enum the library defines: {member_name: EnumClass}.

    Enum members are how this library expresses a MODE — a choice that changes which
    code runs, as opposed to a value that gets computed with. Collected from the library
    so a new enum added tomorrow is covered without editing anything here.
    """
    index = _library_member_index(lib_dir)
    members: dict = {}
    for module in index.values():
        for name, cls in module["classes"].items():
            if "Enum" not in {_decorator_name(b) for b in cls["node"].bases}:
                continue
            for node in cls["node"].body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            members[target.id] = name
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    members[node.target.id] = name
    return members


def _defaulted_mode_parameters(code: str, lib_dir: str) -> list[str]:
    """Parameters that select a library mode and carry a DEFAULT.

    A default is the mechanism by which a decision goes unmade. Confirmed on a real run
    whose orchestrator declared `encoding: ScheduleEncoding = "PRIORITY"` and branched
    correctly on it — building a START_TIME evaluator in one branch and reading the
    pre-built PRIORITY one in the other. The branch was right and unreachable: nothing
    overrode the default, so every call took the PRIORITY path, the same wrong
    constraint family as the three runs before it. The model built the right door and
    left it shut, and the check of the day was satisfied because a decision WAS recorded
    somewhere in the file — just never taken. (That check has since been replaced by
    `_encoding_pairing_violations`, which asks whether the encoding MATCHES rather than
    whether a decision was written down.)

    Matched on the DEFAULT VALUE, not the annotation, because the annotation is the part
    a design is free to rename: `scheduling_type: ScheduleEncodingType =
    ScheduleEncodingType.PRIORITY` and `encoding: ScheduleEncoding = "PRIORITY"` are the
    same defect wearing different clothes, and both resolve to a library enum member.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    members = _library_enum_members(lib_dir)
    if not members:
        return []

    def default_member(node) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value if node.value in members else None
        if isinstance(node, ast.Attribute):
            return node.attr if node.attr in members else None
        if isinstance(node, ast.Name):
            return node.id if node.id in members else None
        return None

    problems: list[str] = []
    seen: set = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        positional = list(args.posonlyargs) + list(args.args)
        pairs = list(zip(positional[len(positional) - len(args.defaults):],
                         args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)
                  if d is not None]
        for arg, default in pairs:
            member = default_member(default)
            if member is None or (fn.name, arg.arg) in seen:
                continue
            seen.add((fn.name, arg.arg))
            problems.append(
                f"line {fn.lineno}: `{fn.name}` gives `{arg.arg}` the default "
                f"`{ast.unparse(default)}` — that is a member of the library enum "
                f"{members[member]}, so this parameter selects a MODE. A mode must not "
                f"have a default: the caller that omits it is not choosing, and every "
                f"branch guarded by the other value becomes unreachable. Make it a "
                f"required parameter so the choice is made where it is known"
            )
    return problems


def _library_modules_used(code: str, lib_dir: str) -> list[str]:
    """Which real library modules this file actually pulls something from.

    Counts both ways a generated file has really been seen to reach the library: the
    static form (`import evaluator`, `from schedule import Slot`) and the dynamic one
    (`importlib.import_module("evaluator")`, which one real run used exclusively — a
    static-only scan would have called that run "uses nothing" and cried wolf).

    Exists because every other check in this file asks whether the library calls a
    generated file makes are CORRECT — the names resolve, the module is the right one,
    the argument count binds — and not one of them asks whether it makes any. All three
    stay silent on a file that reaches for the library nowhere at all, which is the
    single worst outcome for a pathway whose whole premise is reuse, and was confirmed
    real: one benchmarked run never called the evaluator anywhere and no gate objected.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    library = {
        f[:-3] for f in os.listdir(lib_dir)
        if f.endswith(".py") and f != "__init__.py"
    }
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            used.update(a.name.split(".")[0] for a in node.names
                        if a.name.split(".")[0] in library)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in library:
                used.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None)
            if name == "import_module" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value in library:
                    used.add(first.value)
    return sorted(used)


def _kb_class_shadows(code: str, lib_dir: str) -> list[tuple[ast.stmt, str, str]]:
    """Top-level statements that bind a REAL library class's name to something local.

    Two forms, both seen in real runs against the same query:
        @dataclass
        class Schedule: ...          # a stand-in with invented fields
        TSNConstraintEvaluator = Any # the type erased to a placeholder outright

    Either one SHADOWS a correct `from schedule import Schedule` sitting above it, so for
    the rest of the file the real class is unreachable and every one of its methods
    raises AttributeError — while no other check here says a word. `_undefined_names`
    sees the name bound (by this very statement). `_wrong_module_imports` sees no bad
    import. `_bad_kb_call_arity` deliberately only judges names imported straight from a
    library module, so a file built entirely on same-named local stand-ins binds nothing
    it is willing to look at and reports zero problems. That combination is exactly how a
    generated "algorithm" ends up never touching the knowledge base while passing every
    gate in the pipeline.

    Only TOP-LEVEL statements count: a module-level binding is what shadows a
    module-level import, and replacing something nested inside a function with a
    module-level import would break its indentation.

    Which module owns a class is resolved by importing the real modules under `lib_dir`,
    the same way `_wrong_module_imports` does, rather than hard-coding a list that drifts
    the moment the library changes. The module that DEFINES the class wins over one that
    merely re-exports it — confirmed necessary: `tsn_scheduling_problem` imports `Usecase`
    from `usecase_generator`, so an attribute check alone credits the wrong module and
    emits an import that works today only by accident.

    Returns (node, name, owning_module) so callers can both report and rewrite without
    walking the tree twice.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    bindings: list[tuple[ast.stmt, str]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bindings.append((node, node.name))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            bindings.append((node, node.targets[0].id))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            bindings.append((node, node.target.id))
    if not bindings:
        return []

    import importlib

    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    wanted = {name for _, name in bindings}
    definers: dict[str, str] = {}     # name -> module that actually defines it
    reexporters: dict[str, str] = {}  # name -> module that merely exposes it
    for filename in sorted(os.listdir(lib_dir)):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        module_name = filename[:-3]
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue         # module itself is broken; a different problem, not this one
        for name in wanted:
            obj = getattr(module, name, None)
            if not isinstance(obj, type):
                continue
            if getattr(obj, "__module__", None) == module_name:
                definers.setdefault(name, module_name)
            else:
                reexporters.setdefault(name, module_name)

    found = []
    for node, name in bindings:
        owner = definers.get(name) or reexporters.get(name)
        if owner:
            found.append((node, name, owner))
    return found


def _redefined_kb_classes(code: str, lib_dir: str) -> list[tuple[str, str]]:
    """The (name, owning_module) pairs behind `_kb_class_shadows`, for reporting."""
    return [(name, module) for _, name, module in _kb_class_shadows(code, lib_dir)]


def _autofix_redefined_kb_classes(code: str, lib_dir: str) -> str:
    """Replace each stand-in `_kb_class_shadows` finds with an import of the real class.

    Same reasoning as `_autofix_wrong_module_imports`: which module defines a given class
    is a fact, checked by importing it, so there is nothing here an LLM needs to decide
    and no reason to spend a retry hoping it complies. The stand-in's own body is dropped
    wholesale — that is the point, since its fields are a guess at a real class whose
    actual fields are one import away.

    The statement (decorators included) is replaced IN PLACE by the import, rather than
    moving the import to the top of the file, so the substitution stays visible to the
    later stages that read this file as text — they see precisely which type stopped
    being a local invention — and every other line keeps its exact original bytes.
    """
    shadows = _kb_class_shadows(code, lib_dir)
    if not shadows:
        return code

    edits: list[tuple[int, int, str]] = []   # (start_line0, end_line0_exclusive, text)
    for node, name, module in shadows:
        # Decorators sit ABOVE the `class` line and belong to the same statement, so the
        # replaced range has to start at the first of them, not at node.lineno.
        decorators = getattr(node, "decorator_list", [])
        start = min([d.lineno for d in decorators] + [node.lineno]) - 1
        edits.append((
            start,
            node.end_lineno,
            f"from {module} import {name}  # real library class — do not redefine\n",
        ))

    lines = code.splitlines(keepends=True)
    for start, end, replacement in sorted(edits, key=lambda e: -e[0]):
        lines[start:end] = [replacement]
    return "".join(lines)


def _bad_kb_call_arity(code: str, lib_dir: str) -> list[str]:
    """Calls to a real, correctly-imported library class/function whose argument count
    cannot possibly match its actual signature — e.g. `TSNScheduling(a, b, c, ...)` with
    10 positional arguments when the real `TSNScheduling.__init__` takes one required
    argument plus `**kwargs`, or the inverse, a bare `TSNScheduling()` with none at all.

    This is a different bug from `_wrong_module_imports`: the import itself is correct
    (the name really does live in that module), so nothing about the import looks wrong.
    The mistake only shows up once the call actually runs, as a `TypeError` — usually a
    confusing one, since a construction call with 11 arguments raising "takes from 2 to
    3 positional arguments but 11 were given" reads like a deeper bug than a stale call
    site left over from an old, differently-shaped stand-in. One real run had 78 of 199
    failing tests trace back to exactly this, every one of them AFTER its import had
    already been corrected — the import fix and the call-site fix were verified as two
    separate passes instead of together, and only the first one actually landed.

    Verified with `inspect.signature(...).bind(...)` against the REAL object (imported
    from `lib_dir` the same way `_wrong_module_imports` does), rather than re-deriving
    each class's arity by hand — that drifts the moment the library changes, a bind()
    call never does. Calls using `*args`/`**kwargs` unpacking are skipped: a dynamic
    unpack's length can't be verified statically, and flagging it would be a guess, not
    a fact. A real signature that itself accepts `**kwargs` (like TSNScheduling's) will
    silently absorb any keyword name at all, including a misspelled one — that is
    correct, unfixable-by-this-check behavior inherent to `**kwargs`, not a gap in this
    function; catching a plausible-looking but wrong keyword name needs semantic
    understanding of what that parameter does, not just whether the call binds.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case; don't double-report.

    import importlib
    import inspect

    imported: dict = {}      # local_name -> (module_name, real_name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module_name = node.module.split(".")[0]
        if not os.path.isfile(os.path.join(lib_dir, f"{module_name}.py")):
            continue         # not a library module; _wrong_module_imports's business, not this
        for alias in node.names:
            if alias.name != "*":
                imported[alias.asname or alias.name] = (module_name, alias.name)

    if not imported:
        return []

    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    module_cache: dict = {}
    sig_cache: dict = {}

    def real_signature(module_name: str, real_name: str):
        key = (module_name, real_name)
        if key not in sig_cache:
            if module_name not in module_cache:
                try:
                    module_cache[module_name] = importlib.import_module(module_name)
                except Exception:
                    module_cache[module_name] = None
            mod = module_cache[module_name]
            obj = getattr(mod, real_name, None) if mod is not None else None
            try:
                sig_cache[key] = inspect.signature(obj) if obj is not None else None
            except (TypeError, ValueError):
                sig_cache[key] = None
        return sig_cache[key]

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        local_name = node.func.id
        if local_name not in imported:
            continue
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue         # dynamic *args unpack; length not knowable statically
        if any(kw.arg is None for kw in node.keywords):
            continue         # dynamic **kwargs unpack; same reason
        module_name, real_name = imported[local_name]
        sig = real_signature(module_name, real_name)
        if sig is None:
            continue
        fake_args = [None] * len(node.args)
        fake_kwargs = {kw.arg: None for kw in node.keywords}
        try:
            sig.bind(*fake_args, **fake_kwargs)
        except TypeError as e:
            problems.append(
                f"line {node.lineno}: `{local_name}(...)` is called with "
                f"{len(node.args)} positional and {len(node.keywords)} keyword "
                f"argument(s), but the real `{module_name}.{real_name}` signature is "
                f"`{real_name}{sig}` — {e}"
            )
    return problems


def _bad_kb_call_arity_in_tests(tests: list[dict], lib_dir: str) -> list[str]:
    """`_bad_kb_call_arity` over one function's whole set of generated tests at once.

    Each test's `code` is a self-contained snippet that may repeat its own imports, so
    checking them individually would miss nothing — but concatenating first and parsing
    once matches how these tests are actually concatenated for execution later, and
    keeps this a single call per function rather than one per test.
    """
    combined = "\n\n".join(t.get("code", "") for t in tests)
    return _bad_kb_call_arity(combined, lib_dir)


def _frozen_function_return_and_raise_facts(frozen_interface: str, fn_name: str) -> dict:
    """Statically extract two observable facts about one function's body in the frozen
    interface: the literal shape its return value is built from (a list/generator of
    dict literals vs of tuple literals), and which exception types it actually raises.

    Ground truth for _tests_contradict_frozen_interface() below — a spec/contract call
    has no obligation to have read this function's actual body correctly.
    """
    facts = {"return_shape": None, "raised_exceptions": set()}
    try:
        tree = ast.parse(frozen_interface)
    except SyntaxError:
        return facts

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            target = node
            break
    if target is None:
        return facts

    for node in ast.walk(target):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc_node = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc_node, ast.Name):
                facts["raised_exceptions"].add(exc_node.id)
            elif isinstance(exc_node, ast.Attribute):
                facts["raised_exceptions"].add(exc_node.attr)

    def _element_shape(elt) -> Optional[str]:
        # Covers both the literal syntax ({...} / (...)) and the equivalent
        # constructor-call form (dict(...) / tuple(...)) — confirmed necessary on a
        # real run: the actual frozen body built each entry with
        # `dict(type=label, key=key, ...)`, a Call node, not a Dict literal node.
        if isinstance(elt, ast.Dict):
            return "dict"
        if isinstance(elt, ast.Tuple):
            return "tuple"
        if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name):
            if elt.func.id == "dict":
                return "dict"
            if elt.func.id == "tuple":
                return "tuple"
        return None

    def _appended_elements(accumulator_name: str):
        # `return changes` after a loop full of `changes.append(dict(...))` is at
        # least as common a way to build a list as a literal or comprehension at the
        # return site itself — confirmed on the same real run. Every `.append(...)`
        # call anywhere in the function body targeting the same accumulator name is a
        # candidate; the first one with a recognizable shape wins.
        for n in ast.walk(target):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "append"
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == accumulator_name
                    and n.args):
                yield n.args[0]

    for node in ast.walk(target):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        shape = None
        if isinstance(value, (ast.List, ast.Set)) and value.elts:
            shape = _element_shape(value.elts[0])
        elif isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            shape = _element_shape(value.elt)
        elif isinstance(value, ast.Name):
            for elt in _appended_elements(value.id):
                shape = _element_shape(elt)
                if shape:
                    break
        if shape:
            facts["return_shape"] = shape
            break

    return facts


def _tests_contradict_frozen_interface(
    tests: list[dict], fn_name: str, frozen_interface: str
) -> list[tuple[dict, str]]:
    """Per-test, deterministic contradictions between what a generated test asserts
    about `fn_name` and what the frozen interface's own body actually does — run
    automatically before a human ever sees the approval gate, not left to manual review
    to catch by reading two separate sections of the same output.

    Confirmed causal chain on a real run this was built from: get_migration_sequence's
    approved tests expected a sorted list of 5-element (index, category, key, old, new)
    tuples with category in ADDED/DELETED/UPDATED, while the frozen function actually
    returns a list of dicts with keys type/key/stream_id/start_value/target_value and
    lowercase added/removed/modified. That mismatch is invisible to
    _wrong_module_imports/_undefined_names/_bad_kb_call_arity — every name involved is
    real and correctly imported; the contradiction is about the SHAPE of a real object,
    not an undefined or misplaced name. Left unchecked, the repair loop resolved the
    contradiction it could not satisfy (it is told to never weaken a test, and also
    that the frozen interface is fixed) by rewriting the function to match the wrong
    tests instead of the frozen design, and dropped a sibling helper function
    (classifying_change) as a side effect once its own rewritten logic no longer called
    it — even though classifying_change's own separate test file still imported it
    directly as its own unit.

    A second, unrelated pattern this also catches: migrate.__globals__['build_metadata']
    — a test reaching for a name via reflection that was never part of the frozen
    interface at all. Any `<name>.__globals__[<literal>]` subscript whose literal is not
    a real top-level name in the frozen interface is exactly as contradictory as a bad
    return-shape claim, and just as invisible to the existing checks (the name IS bound,
    by the __globals__ subscript itself, so nothing looks undefined until it runs).

    Deliberately narrow and conservative — two concrete, confirmed failure shapes, not a
    general type checker — to keep false positives near zero: an unmatched fact (e.g. a
    return whose shape could not be determined) is silently not flagged, never guessed.
    """
    facts = _frozen_function_return_and_raise_facts(frozen_interface, fn_name)
    try:
        frozen_tree = ast.parse(frozen_interface)
        top_level_names = {
            n.name for n in ast.walk(frozen_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        } | {
            t.id for n in ast.walk(frozen_tree) if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)
        }
    except SyntaxError:
        top_level_names = set()

    flagged: list[tuple[dict, str]] = []
    for test in tests:
        code = test.get("code", "")
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        reasons = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "__globals__"
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                    and node.slice.value not in top_level_names):
                reasons.append(
                    f"reaches for {ast.unparse(node.value)}[{node.slice.value!r}] via "
                    f"reflection, but '{node.slice.value}' is not defined anywhere in "
                    f"the frozen interface"
                )
            if (facts["return_shape"] and isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "isinstance"
                    and len(node.args) == 2 and isinstance(node.args[1], ast.Name)
                    and node.args[1].id in ("dict", "tuple")
                    and node.args[1].id != facts["return_shape"]):
                reasons.append(
                    f"asserts isinstance(..., {node.args[1].id}), but the frozen "
                    f"{fn_name}'s own return statement builds its result from "
                    f"{facts['return_shape']} literals, not {node.args[1].id}"
                )
        if reasons:
            flagged.append((test, "; ".join(reasons)))

    return flagged


def _drop_tests_contradicting_frozen_interface(
    tests: list[dict], fn_name: str, frozen_interface: str, stage: str = "generation",
) -> list[dict]:
    """Apply _tests_contradict_frozen_interface() and fail closed: any test it flags is
    dropped — along with its own self-contained import block, since each test's `code`
    carries its own imports rather than sharing a header — rather than approved and left
    for the execution/repair loop to somehow reconcile. Prints exactly what was dropped
    and why, so a run that loses tests here is never silently smaller than it looks.
    """
    contradictions = _tests_contradict_frozen_interface(tests, fn_name, frozen_interface)
    if not contradictions:
        return tests
    dropped_ids = set()
    for test, reason in contradictions:
        dropped_ids.add(id(test))
        print(f"[tests] {fn_name}: dropping '{test.get('test_name', '?')}' at {stage} "
              f"— contradicts the frozen interface ({reason}).", file=sys.stderr)
    return [t for t in tests if id(t) not in dropped_ids]


def _top_level_def_names(code: str) -> set[str]:
    """Every function/class name bound at module level in `code`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _names_imported_from_generated_impl(test_codes: list[str]) -> set[str]:
    """Every name any approved test imports `from generated_impl import ...` — what
    the approved suite as a whole actually needs to exist in the file under test.
    """
    names = set()
    for code in test_codes:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom) and node.module
                    and node.module.split(".")[0] == GENERATED_MODULE_NAME):
                names.update(alias.name for alias in node.names if alias.name != "*")
    return names


def _vanished_frozen_names(previous: str, current: str, test_codes: list[str]) -> set[str]:
    """Top-level names `previous` defined, that the approved tests still import from
    the generated module, that `current` no longer defines at all.

    Confirmed on a real run: a repair fixing get_migration_sequence rewrote its logic
    inline and stopped calling classifying_change, then simply left
    classifying_change's own definition out of the file it returned — even though
    classifying_change's own separate test file still imported it directly. No
    existing check catches this: the resulting file is valid Python and every import
    it contains points at a real module, so _wrong_module_imports and
    _undefined_names both see nothing wrong with it. The only way to know is
    literally diffing what got defined before against what got defined now, against
    what the approved tests still ask for.
    """
    still_needed = (
        _names_imported_from_generated_impl(test_codes) & _top_level_def_names(previous)
    )
    return still_needed - _top_level_def_names(current)


def _restore_vanished_frozen_names(
    previous: str, current: str, vanished: set[str]
) -> Optional[str]:
    """Mechanically splice each vanished name's own top-level definition, verbatim,
    from `previous` onto the end of `current` — the same "apply the fix directly
    instead of hoping a retry complies" approach as _autofix_wrong_module_imports,
    just recovering a whole dropped definition instead of one import line. Returns
    None if `previous` cannot even be parsed to recover the source from.
    """
    try:
        tree = ast.parse(previous)
    except SyntaxError:
        return None
    pieces = [current.rstrip(), ""]
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in vanished):
            segment = ast.get_source_segment(previous, node)
            if segment is None:
                return None
            pieces.append(segment)
    return "\n\n".join(pieces) + "\n"


def _hallucinated_numpy_method_calls(code: str) -> list[str]:
    """Calls to a numpy method/attribute that does not exist on `numpy.ndarray` or the
    `numpy` module itself, where the object the call is made on is statically
    recognizable as array-producing.

    Confirmed on a real, recorded run: a finetuned-model implementation called
    `(rows < half).reshape_like(cols)` inside a function turning a chromosome into its
    compact per-stream/per-port representation — `reshape_like` does not exist anywhere
    in real numpy (checked directly: `hasattr(np.ndarray, 'reshape_like')` and
    `hasattr(np, 'reshape_like')` are both False). Every one of that function's 15 tests
    failed on the resulting AttributeError, none for any other reason.

    Unlike `_bad_kb_call_arity` (checked against a real, importable object from this
    project's own library) there is no single real object to import and introspect here
    — `numpy` itself is real and always present, so the check is directly against
    `numpy.ndarray`/`numpy` via `hasattr`, the same "trust the real thing, don't
    re-derive its surface by hand" principle applied to a third-party library instead
    of this project's own KB.

    "Array-producing" is deliberately conservative to avoid false positives on
    non-array objects this has no static evidence about: a comparison (`x < y`), a
    binary op (`x + y`), or a call whose own callee is already a real numpy/ndarray
    name (`np.array(...)`, `matrix.nonzero()`, an unqualified real numpy function).
    A method called on a bare `Name` or on a literal dict/list — anything with no
    such evidence — is left alone; guessing it is a numpy array with nothing to back
    that guess up would risk flagging ordinary, unrelated method calls.

    Detector only, by design — see this function's own discussion with the user
    about why a mechanical auto-fix is not offered: unlike a KB call-arity mismatch
    or a config path, there is no single deterministic "correct" replacement for a
    hallucinated numpy method. Reconstructing what the real run above actually
    needed (`talker_count = matrix.sum(axis=0)`, not a `half`-based split nothing in
    its contract ever asked for) required reading all 15 of its tests — information
    this function does not have and a pre-flight repair call may not either. Wired
    into `_run_and_repair_algorithm`'s own repair loop specifically because THAT
    call site already has the real failing-test output and test code this needs,
    unlike the pre-flight checks earlier in the pipeline, which run before any test
    has ever executed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []            # a syntax gate handles this case elsewhere.

    import numpy as np

    def _is_array_producing(expr) -> bool:
        if isinstance(expr, (ast.Compare, ast.BinOp)):
            return True
        if isinstance(expr, ast.Call):
            f = expr.func
            if isinstance(f, ast.Attribute) and hasattr(np.ndarray, f.attr):
                return True                              # obj.some_real_method(...)
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == "np" and hasattr(np, f.attr)):
                return True                              # np.some_real_function(...)
            if isinstance(f, ast.Name) and hasattr(np, f.id):
                return True                              # some_real_function(...)
        return False

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method_name = node.func.attr
        if hasattr(np.ndarray, method_name) or hasattr(np, method_name):
            continue
        if not _is_array_producing(node.func.value):
            continue
        problems.append(
            f"line {node.lineno}: `.{method_name}(...)` is called on what looks like a "
            f"numpy array, but neither numpy.ndarray nor the numpy module itself has an "
            f"attribute named `{method_name}` in the real, installed numpy — this call "
            f"raises AttributeError."
        )
    return problems


def _is_qwen_finetuned_backend() -> bool:
    """True only when the active model is the TSN-finetuned Qwen checkpoint
    (serve_finetuned.sh's --alias qwen3.5-27b-tsn-finetuned, selected via
    TSN_LLM_MODEL) — never the base local Qwen (serve.sh's plain "qwen3.5-27b"),
    the hosted "luna" model, Claude, or the Grace Hopper Llama backend.

    Guards `_autofix_config_ini_path` below: that fix is only needed for a bug this
    one specific finetune's own training data causes (see its docstring), and has
    no reason to run — or risk of ever mattering — against any other backend.
    """
    return "finetuned" in llm_config.LLM_MODEL.lower()


def _autofix_config_ini_path(code: str, repo_root: str) -> str:
    """Replace every bare string-literal path ending in `.ini`, passed directly as a
    function-call argument, with an absolute path to this project's own real
    `config.ini` — unless it already IS that exact real path.

    The finetuned Qwen model's own training data references ~54 fixture filenames
    (fixtures/ft4_small.ini, fixtures/sw_mid_10000m_7s.ini, ...) that do not exist
    anywhere in this project (confirmed in finetune_qwen3.5_27b/dataset/train_1.jsonl).
    It reliably reproduces one of these instead of the real config.ini at this
    project's root, in generated tests and occasionally the implementation itself.
    configparser.read() silently no-ops on a missing path rather than raising, so
    this doesn't fail cleanly: it surfaces deep inside usecase_generator.py as a
    confusing NoSectionError/NoOptionError, sometimes at MODULE level — aborting
    collection of the whole test file, not just the tests that actually touch it.
    Confirmed NOT to respond reliably to prompt-level reinforcement or to natural-
    language review-round feedback repeated across multiple runs (see
    feedback_always_flag_fixture_path) — this replaces the LLM compliance this bug
    class needs with a mechanical, deterministic rewrite instead.

    Also catches the "critical correction" variant of the same bug: a bare relative
    `"config.ini"` (the right filename, wrong path) is just as broken as a wrong
    filename, since tests execute in an isolated temp directory whose CWD has no
    config.ini of its own — the check below is against the real ABSOLUTE path, so a
    bare relative reference to the correctly-spelled filename is still rewritten.

    Scoped deliberately narrow to avoid false positives: only a bare `ast.Constant`
    string ending in `.ini`, appearing directly as a positional or keyword argument
    of a `Call` node, is touched. A string used to build a `tmp_path`-derived path
    (`tmp_path / "config.ini"`) is the right operand of a `/` BinOp, not a bare Call
    argument, so the correct, already-established tmp_path/write_text pattern is left
    alone even when it happens to use a different filename than "config.ini" — and a
    later `TSNScheduling(str(p))` call passes a `Call` node (`str(p)`), not a bare
    string Constant, so that is left alone too. A `.ini` mention inside a docstring,
    comment, or unrelated literal is never a Call argument at all, so it is never a
    candidate here either.

    Known, accepted limitation: does not check whether the call sits inside a
    `pytest.raises(...)` block deliberately testing missing-file/invalid-path
    handling — replacing the literal there would remove the very condition such a
    test means to exercise. Not fixed: every real occurrence of this bug class
    observed so far has been an unconditional, unintentional hallucination of a
    "normal" config path, never a deliberate missing-file test, so tracing enclosing
    pytest.raises context is not worth the added complexity unless a real instance
    of that combination is ever actually observed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code           # a syntax gate handles this case elsewhere.

    real_abs = os.path.join(repo_root, "config.ini")
    real_norm = os.path.normpath(real_abs)
    replacement_text = repr(real_abs)

    lines = code.splitlines(keepends=True)
    spans: list[tuple[int, int, int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            if not arg.value.lower().endswith(".ini"):
                continue
            if os.path.normpath(arg.value) == real_norm:
                continue      # already the real path; nothing to fix.
            if arg.lineno != arg.end_lineno:
                continue      # a string Constant is always single-line; defensive only.
            spans.append((arg.lineno, arg.col_offset, arg.end_lineno,
                          arg.end_col_offset, replacement_text))

    if not spans:
        return code

    # Applied back-to-front so an earlier span's offset is never shifted by a later
    # edit on the same or an earlier line.
    spans.sort(key=lambda s: (s[0], s[1]), reverse=True)
    for lineno, col, end_lineno, end_col, new_text in spans:
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:col] + new_text + line[end_col:]

    new_code = "".join(lines)
    if _syntax_error_of(new_code) is not None:
        return code           # the rewrite somehow broke parsing; never ship it.
    return new_code




# A block ends at "### END TEST", at the start of the NEXT "### TEST", or at end of
# output — whichever comes first. Requiring the explicit terminator cost a whole query
# once: a completion emitted eight well-formed blocks back-to-back and not one closing
# marker, so this regex matched nothing at all — 0 of 8, not "some survived" — and with
# tests == [] both `if interactive and tests:` guards fell through, skipping the review
# gate AND execution while the answer still claimed tests were included.
#
# The terminator is a lookahead, so it is never consumed and the next block still starts
# at its own "### TEST". Doing it here rather than as a separate "retry leniently if zero
# matched" pass matters: a run where only SOME blocks lose their terminator still matches
# a non-zero number of blocks, so a zero-triggered fallback would never fire — while the
# strict pattern silently glues the unterminated block onto the next one's header and
# corrupts both. Measured on that shape: strict matched 2 blocks of which 1 parsed; this
# matches 3, all 3 parse.
_TEST_BLOCK_RE = re.compile(
    r"###\s*TEST\s*\n"
    r"BEHAVIOR_ID:\s*(?P<behavior_id>\S+)\s*\n"
    r"TEST_NAME:\s*(?P<test_name>\S+)\s*\n"
    r"CODE:\s*\n"
    r"(?P<code>.*?)"
    r"(?=\n###\s*END\s*TEST|\n###\s*TEST\s*\n|\Z)",
    re.DOTALL,
)


def _trim_to_parsable(code: str, max_dropped: int = 12) -> Optional[str]:
    """Drop trailing lines until `code` parses, or give up.

    The final block has no terminator to stop at, so anything the model appends after its
    last test ("Hope this helps!", a stray markdown fence, a summary line) lands inside
    that block's code and makes it unparseable — losing a test that was itself perfectly
    good. Only trailing lines are removed, never anything above them, and the result is
    kept only if it still contains a test function.
    """
    lines = code.rstrip().split("\n")
    for dropped in range(1, min(max_dropped, len(lines)) + 1):
        candidate = "\n".join(lines[:-dropped]).rstrip()
        if not candidate or "def test" not in candidate:
            return None
        if _syntax_error_of(candidate, "<test>") is None:
            return candidate
    return None


def _fix_future_import_position(code: str) -> str:
    """Move a `from __future__ import ...` line to the very top of the file (after only
    a leading module docstring, if any) if it is not already there.

    Python requires `__future__` imports to be the first statement(s) in a module —
    anywhere else is a hard SyntaxError, not a style nitpick. Seen for real: a repair
    round otherwise correctly fixed a failing test, but emitted `from __future__ import
    annotations` mid-file (after some dataclasses), which took collection of the whole
    generated module down — a good repair discarded over one misplaced line that has
    nothing to do with what was actually being repaired.

    Returns `code` unchanged if there is no `__future__` import, it is already
    correctly placed, or moving it does not actually produce parseable code (better to
    leave the original for the ordinary syntax gate to catch than to guess wrong).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    future_idx = [
        i for i, n in enumerate(tree.body)
        if isinstance(n, ast.ImportFrom) and n.module == "__future__"
    ]
    if not future_idx:
        return code

    def _is_docstring(node) -> bool:
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str))

    leading_ok = 1 if (tree.body and _is_docstring(tree.body[0])) else 0
    future_node_ids = {id(tree.body[i]) for i in future_idx}
    while leading_ok < len(tree.body) and id(tree.body[leading_ok]) in future_node_ids:
        leading_ok += 1
    if all(i < leading_ok for i in future_idx):
        return code   # every __future__ import is already a leading statement

    lines = code.splitlines(keepends=True)
    future_lines = []
    for i in sorted(future_idx, reverse=True):
        node = tree.body[i]
        future_lines.insert(0, "".join(lines[node.lineno - 1:node.end_lineno]))
        del lines[node.lineno - 1:node.end_lineno]

    insert_at = 0
    try:
        head = ast.parse("".join(lines))
        if head.body and _is_docstring(head.body[0]):
            insert_at = head.body[0].end_lineno
    except SyntaxError:
        pass

    fixed = "".join(lines[:insert_at]) + "".join(future_lines) + "".join(lines[insert_at:])
    return fixed if _syntax_error_of(fixed) is None else code


def _wrap_bare_test_code(code: str, test_name: str) -> str:
    """If `code`'s top level has bare executable statements outside any function — the
    "migrate" pattern seen on repeated finetuned-model runs: real setup and asserts
    written as a plain script (`problem = TSNScheduling(...)`, `with problem.use_instance(...):
    ...`) with no enclosing `def test_...():` at all — wrap them into one.

    This matters because every test's `code` is concatenated into ONE file
    (`_execute_spec_tests` / `_render_tests_for_revision`'s counterpart at execution
    time). A bare statement at module level runs the moment the file is imported, before
    pytest can even discover a single test function; if it raises, collection of the
    WHOLE FILE aborts — every other test in it, however unrelated, is reported as an
    error without ever running. Measured on a real run: one bare block's
    `configparser.NoSectionError` from a bad fixture path took all 15 of a completely
    unrelated, self-contained function's tests down with it, tests that share nothing
    with the failing block but the file they happened to be concatenated into. Isolated
    inside its own function, the identical failure is one ordinary test failure instead.

    Leaves already-correct code untouched — a single top-level `def test_...():` (with or
    without leading imports) is left exactly as it was.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    has_test_def = False
    bare_spans: list[tuple[int, int]] = []   # (start_line0, end_line0_exclusive)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                has_test_def = True
            continue
        bare_spans.append((node.lineno - 1, node.end_lineno))

    if has_test_def or not bare_spans:
        return code   # already a real test function, or nothing executable to wrap

    lines = code.splitlines(keepends=True)
    start = min(s for s, _ in bare_spans)
    end = max(e for _, e in bare_spans)
    # Some lines in [start, end) may be imports interleaved between bare statements
    # (rare, but seen) — keep the whole span together rather than cherry-picking, so
    # relative order and any shared local state is preserved exactly as written.
    body = "".join(("    " + ln if ln.strip() else ln) for ln in lines[start:end])
    wrapped = f"def {test_name}():\n{body}"
    return "".join(lines[:start]) + wrapped + "".join(lines[end:])


def _render_tests_for_revision(tests: list[dict]) -> str:
    """The current tests, written back out in the exact "### TEST" block format the model
    is asked to produce. Re-using the output format costs nothing and doubles as a worked
    example of the shape the revision must come back in."""
    return "\n\n".join(
        f"### TEST\n"
        f"BEHAVIOR_ID: {t.get('behavior_id', '?')}\n"
        f"TEST_NAME: {t.get('test_name', '?')}\n"
        f"CODE:\n{t.get('code', '')}\n"
        f"### END TEST"
        for t in tests
    )


def _spec_based_tests(contract: dict, import_hint: str, revision: str = None,
                      previous_tests: list[dict] = None, query_str: str = "",
                      blueprint: Optional[dict] = None,
                      frozen_interface: Optional[str] = None) -> list[dict]:
    """Write pytest tests from the contract ONLY (code-blind). Uses SPEC_BASED_TEST_PROMPT.

    On a revision (`revision` = the user's review comment), `previous_tests` carries the
    set the user was actually looking at, and both go into the prompt alongside the
    unchanged contract. Without them the revision call saw only the contract and the
    comment, so feedback naming a specific test ("B3 assumes the list is sorted") reached
    a model that had never read B3 — it could not act on the objection, and every test the
    user was happy with was resampled from scratch rather than carried over.

    The prompt asks for plain delimited "### TEST" / "### END TEST" blocks rather than
    JSON: each test's code is a full multi-line Python snippet (quotes, backslashes,
    docstrings, real newlines), and asking the LLM to correctly JSON-escape that as a
    string value was the single largest source of malformed-output parse failures this
    pipeline saw (Unterminated string / Expecting ',' delimiter / Expecting value errors
    from json.loads on this exact call). Plain text with unambiguous delimiters sidesteps
    the whole class of escaping bugs — nothing here needs escaping to begin with.
    """
    # The contract is what it always was — the revision material rides in its own slot
    # so the model never has to guess which part of this is the spec and which is feedback.
    contract_text = json.dumps(contract, indent=2)
    revision_block = ""
    if revision:
        sections = []
        if previous_tests:
            sections.append(
                "You already wrote the tests below from this same contract, and the user\n"
                "has now reviewed them. Rewrite the COMPLETE set: reproduce every test the\n"
                "request does not object to EXACTLY as it stands - same BEHAVIOR_ID, same\n"
                "TEST_NAME, same body - and change only what the request actually asks for.\n\n"
                "Tests you wrote previously:\n\n"
                + _render_tests_for_revision(previous_tests)
            )
        sections.append(
            "What to change (apply while still asserting ONLY contract-supported "
            "behavior; do not weaken valid assertions):\n" + revision
        )
        revision_block = "\nREVISION REQUEST\n\n" + "\n\n".join(sections) + "\n"
    function_name = import_hint.rsplit(" import ", 1)[-1].strip()
    signature = contract.get("signature") or f"{function_name}(...)  # signature unknown"
    prompt = SPEC_BASED_TEST_PROMPT.format(
        import_hint=import_hint,
        function_name=function_name,
        signature=signature,
        contract_json=contract_text,
        revision_block=revision_block,
        query_str=query_str or "not available",
        blueprint_json=json.dumps(_blueprint_for_prompt(blueprint), indent=2) if blueprint else "(no blueprint available)",
        frozen_interface=frozen_interface or "(no frozen interface available)",
        violation_row_order=_render_violation_row_order(LIB_DIR),
    )
    # "Exactly three tests per behavior" means a contract with many behaviors needs
    # proportionally more room: a 9-behavior contract (seen on the algorithm pathway,
    # where one function's contract can carry far more behaviors than the normal
    # pathway's typical 3-5) needs 27+ short pytest blocks, which overran the flat
    # 3000-token default mid-block — confirmed by a real run where the LAST test was
    # cut off with no "### END TEST" marker. Scaled per behavior instead of raised
    # flat so the common small-contract case keeps the same budget as before.
    num_behaviors = len(contract.get("behaviors", [])) or 1
    max_tokens = max(3000, num_behaviors * 900)
    raw = Settings.llm.complete(prompt, max_tokens=max_tokens).text.strip()
    tests = []
    matched = 0
    for m in _TEST_BLOCK_RE.finditer(raw):
        matched += 1
        code = m.group("code").strip("\n")
        # Syntax gate, matching the one generate_code_from_spec() applies to the
        # implementation. All test blocks are concatenated into a single file, so one
        # unparseable block is not one failed test — it is a collection error that takes
        # every other test in the file down with it.
        err = _syntax_error_of(code, "<test>")
        if err is not None:
            # Before dropping it, try trimming trailing junk: a block closed at end of
            # output swallows whatever the model wrote after its last test.
            salvaged = _trim_to_parsable(code)
            if salvaged is not None:
                print(f"[tests] '{m.group('test_name').strip()}': trimmed trailing "
                      f"non-code after the last test.", file=sys.stderr)
                code, err = salvaged, None
        if err is not None:
            print(f"[tests] dropping '{m.group('test_name').strip()}': does not parse "
                  f"({type(err).__name__}: {err.msg}, line {err.lineno}).", file=sys.stderr)
            continue
        test_name = m.group("test_name").strip()
        wrapped = _wrap_bare_test_code(code, test_name)
        if wrapped != code:
            print(f"[tests] '{test_name}' was bare top-level code, not a "
                  f"`def {test_name}():` — wrapped it in one.", file=sys.stderr)
            code = wrapped
        tests.append({
            "behavior_id": m.group("behavior_id").strip(),
            "test_name": test_name,
            "code": code,
        })

    # Loud, because the silent version of this cost a whole query: terminators missing
    # used to mean zero tests, no review gate, no execution, and an answer that still
    # said tests were included.
    closers = len(re.findall(r"###\s*END\s*TEST", raw))
    if matched and closers < matched:
        print(f"[tests] {matched - closers} of {matched} block(s) had no '### END TEST' "
              f"marker; closed them at the next block / end of output.", file=sys.stderr)
    if not matched:
        print(f"[tests] NO test blocks parsed out of {len(raw)} chars of output — the "
              f"response did not use the '### TEST' block format at all.", file=sys.stderr)

    # Wired here once rather than at every one of this function's call sites (initial
    # generation and every revision round, across both pathways): every one of them
    # returns through this exact point, so applying the fix here covers all of them.
    #
    # _autofix_wrong_module_imports was already applied to the IMPLEMENTATION on both
    # pathways' pre-flight checks, but never to tests — confirmed missing on a real
    # run: a test importing `ScheduleEncodingType` from `tsn_scheduling_problem`
    # (real name, wrong module — it only lives in evaluator) survived an explicit,
    # named review-round request to fix that exact line, aborting collection of the
    # whole concatenated file. Applied here unconditionally (every backend, not just
    # the finetuned model — a wrong-module import is not specific to it) for the
    # same reason the config-path fix above is applied here rather than scattered
    # across call sites.
    for t in tests:
        t["code"] = _autofix_wrong_module_imports(t["code"], LIB_DIR)
    if _is_qwen_finetuned_backend():
        for t in tests:
            t["code"] = _autofix_config_ini_path(t["code"], REPO_ROOT)

    return tests


# The module name the generated implementation is written under inside the sandbox.
# Deliberately NOT the target module's own name: writing it as `evaluator.py` /
# `usecase_generator.py` put a one-function file on sys.path AHEAD of the real library
# module of the same name, so every `from evaluator import ScheduleEncodingType` (or
# TSNConstraintEvaluator, ...) inside the generated code or its tests raised
# ImportError — the real symbol was shadowed and unreachable.
GENERATED_MODULE_NAME = "generated_impl"

# Imports written at the top of BOTH files created in the sandbox, so the generated code
# and its tests can use these names without having to get their own import lines right.
#
# The list is not guesswork: it is what the recorded runs in C_answers*.txt actually
# import, tallied across 9 files, kept where the count justified the line. The aliases
# matter as much as the modules — `np` appears 3669 times and `nx` 759, so `import numpy`
# alone would bind a name almost nothing references.
#
# Whole modules wherever possible, and that is deliberate: `from evaluator import Foo`
# where Foo does not exist fails while the file is still loading, which is a collection
# error that takes every test in the file down with it — measured, 0 of 14 tests ran.
# Binding the module instead defers a wrong name to the moment a test touches it, so it
# costs one failing test rather than the whole batch. The few `from X import Y` lines
# below are the exceptions the data forced: SimpleNamespace is referenced bare 2108
# times, MagicMock 252, Path 90 — a module binding would not help any of them.
#
# Cost is ~400ms cold, and most of it is already paid: usecase_generator and
# tsn_scheduling_problem themselves import numpy, networkx and matplotlib.pyplot. Every
# line here was verified to import cleanly in this environment (matplotlib resolves to
# the headless "agg" backend). That verification is the requirement for adding a line:
# an import that fails here fails EVERY run, including queries that never needed it.
SANDBOX_IMPORT_LINES = (
    # -- the real project library (tsn_scheduling_problem needs the two compatibility
    #    modules in LIB_DIR to satisfy its own lines 10-11) --
    "import evaluator",
    "import usecase_generator",
    "import tsn_scheduling_problem",
    # -- third-party, under the aliases the recorded tests actually reference --
    "import pytest",                                    # 4198 hits
    "import numpy as np",                               # 3669
    "import networkx as nx",                            #  759
    "import matplotlib",                                #  312
    "import matplotlib.pyplot as plt",                  #   94
    "from matplotlib.patches import Rectangle",         #   28
    "from scipy.sparse import csr_matrix",              #   23
    # -- stdlib, grouped to keep the injected block short --
    "import os, sys, json, math, ast, copy, time, random, inspect, io, socket",
    "import hashlib, logging, tempfile, configparser, itertools, functools",
    "import numbers, builtins, contextlib, warnings",
    "from logging.handlers import RotatingFileHandler", #   10
    "from types import SimpleNamespace",                # 2108
    "from unittest.mock import MagicMock, patch",       #  252 / 60
    "from pathlib import Path",                         #   90
    "from configparser import RawConfigParser, ConfigParser",   # 91 / 16
    "from collections import OrderedDict, deque",       #   29 / 22
    "from collections.abc import Mapping, Sequence",    #   26
    "from copy import deepcopy",                        #   18
)

LIB_IMPORT_HEADER = (
    "# --- injected by _execute_spec_tests: library + common imports ---\n"
    + "".join(line + "\n" for line in SANDBOX_IMPORT_LINES)
    + "# --- end injected header ---\n\n"
)

# Every name the header binds, derived from the header itself with the same collector
# _undefined_names() uses. Deriving it means the two can never drift: add a line above
# and the pre-flight stops flagging whatever that line provides, with nothing else to
# update. Without this, `np.zeros(...)` in generated code that relies on the header
# would be reported as an undefined name and trigger a pointless regeneration.
def _header_bound_names() -> frozenset:
    collector = _BindingCollector()
    collector.visit(ast.parse(LIB_IMPORT_HEADER))
    return frozenset(collector.bound)


LIB_PROVIDED_NAMES = _header_bound_names()


def _import_hint_for(spec: dict, module: str) -> str:
    """The import line the tests use to reach the function under test.

    _execute_spec_tests() writes the implementation to GENERATED_MODULE_NAME.py as a
    module-level definition — a plain def, or the whole class when the spec's "type" is
    "class" — so `from generated_impl import <name>` resolves it either way, and because
    that name does not collide with any real library module, the real modules stay
    importable alongside it.
    """
    return f"from {GENERATED_MODULE_NAME} import {spec.get('name')}"


def _execute_spec_tests(impl_code: str, test_codes: list[str], module: str = "") -> dict:  # noqa: ARG001
    """
    Stage 5, step 3 ("run the code it gave and check whether it ran successfully"):
    write the generated function + its spec tests to a temp dir and run pytest in a
    subprocess. Returns a dict with success / returncode / stdout / stderr / workdir.

    Two things here are load-bearing and were previously wrong:

    1. The implementation is written as GENERATED_MODULE_NAME.py, NOT as the target
       module's own filename. Writing it as `evaluator.py` placed a one-function file
       earlier on sys.path than the real library module, so the real classes and enums
       (ScheduleEncodingType, TSNConstraintEvaluator, ...) became unreachable from both
       the generated code and its tests.

    2. LIB_DIR — where the real library modules actually live — is put on PYTHONPATH.
       The previous code put only os.getcwd() there and claimed that made "real modules
       resolve"; it did not, because the sources are under original_tsn_lib_files/, not
       the repo root, so every import of a real project symbol failed.

    Note that not every real module is importable in isolation: tsn_scheduling_problem
    imports two helper modules that are not themselves the real dependency.
    usecase_generator_json now resolves via a compatibility shim
    (original_tsn_lib_files/usecase_generator_json.py) that re-exports the real
    UsecaseGenerator/Usecase classes from usecase_generator.py unchanged, so that import
    carries real behavior, not a stub. instancesLoader_json resolves too, but only to a
    placeholder InstancesLoader that constructs and does nothing else — fine in practice
    since TSNScheduling.__init__ never calls anything on the instance afterwards, but it
    fails loudly by design if anything ever does. evaluator and usecase_generator import
    cleanly on their own. Nothing here is stubbed in to paper over a genuine gap — the
    one real placeholder says so in its own docstring and fails loudly if actually used.
    """
    import tempfile, subprocess, os as _os, sys as _sys

    tmpdir = tempfile.mkdtemp(prefix="tsn_gen_")
    with open(_os.path.join(tmpdir, f"{GENERATED_MODULE_NAME}.py"), "w") as f:
        f.write(LIB_IMPORT_HEADER + (impl_code or "").rstrip() + "\n")
    test_path = _os.path.join(tmpdir, "test_generated.py")
    joined_tests = LIB_IMPORT_HEADER + "\n\n".join(test_codes).rstrip() + "\n"
    # Individually-parsing blocks can still collide once concatenated (a stray dedent, a
    # block that opened a bracket it never closed). Report it here rather than letting
    # pytest present it as a collection error for the whole file.
    joined_err = _syntax_error_of(joined_tests, "test_generated.py")
    if joined_err is not None:
        print(f"[verify] concatenated test file does not parse "
              f"({type(joined_err).__name__}: {joined_err.msg}, line {joined_err.lineno}); "
              f"pytest will report a collection error.", file=_sys.stderr)
    with open(test_path, "w") as f:
        f.write(joined_tests)

    env = dict(_os.environ)
    env["PYTHONPATH"] = _os.pathsep.join(
        [LIB_DIR, _os.getcwd()] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    try:
        proc = subprocess.run(
            [_sys.executable, "-m", "pytest", "-q", "--no-header", test_path],
            cwd=tmpdir, capture_output=True, text=True, timeout=90, env=env,
        )
        return {"success": proc.returncode == 0, "returncode": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr, "workdir": tmpdir}
    except subprocess.TimeoutExpired:
        return {"success": False, "returncode": -1, "stdout": "",
                "stderr": "pytest timed out after 90s.", "workdir": tmpdir}
    except FileNotFoundError:
        return {"success": False, "returncode": -1, "stdout": "",
                "stderr": "pytest not found — `pip install pytest`.", "workdir": tmpdir}


def _triage_failure(contract: dict, test_code: str, impl_code: str, failure_output: str) -> dict:
    """Stage 5, step 4: explain WHY the implementation fails its approved contract tests.

    No longer decides "impl vs test" — the tests are user-approved and frozen, so the only
    outcomes are IMPL_INCORRECT (repair the code) or FLAKY_OR_ENV (nothing to repair).
    Uses TRIAGE_PROMPT.
    """
    prompt = TRIAGE_PROMPT.format(
        contract_json=json.dumps(contract, indent=2),
        test_code=test_code,
        impl_code=impl_code,
        # The complete run, untruncated. It used to be cut to the first 4000 chars, and
        # because pytest prints its detail first and its tally LAST, that cut reliably
        # discarded the "N failed, M passed" line — the one fact triage most needs.
        failure_output=failure_output,
    )
    raw = Settings.llm.complete(prompt).text.strip()
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


def _repair_impl(
    impl_code: str,
    spec: dict,
    contract: dict,
    failure_output: str,
    query: str,
    guidance: str = "",
) -> str:
    """Stage 5, step 5: regenerate the IMPLEMENTATION ONLY, given the approved
    specification + contract + the failure. Returns raw code.

    Like the initial generation, the repair is shown the spec WITHOUT "raises"
    (_spec_for_codegen) — the error behavior it must satisfy is already stated in the
    contract the tests came from. `guidance` carries the triage verdict's
    required_impl_change so the retry targets the actual defect.
    """
    guidance_block = f"Triage — what must change in the implementation:\n{guidance}\n\n" if guidance else ""
    prompt = (
        "You are fixing a Python implementation that failed contract-derived tests.\n"
        "The specification, the contract and the tests were all approved by the user and\n"
        "are FIXED — only the implementation may change. Never weaken or work around a test.\n"
        "Return ONLY the corrected function code (a single module-level def ...), no fences,\n"
        "no prose.\n\n"
        f"Original user request:\n{query}\n\n"
        f"Approved specification:\n{json.dumps(_spec_for_codegen(spec), indent=2)}\n\n"
        f"Behavioral contract:\n{json.dumps(contract, indent=2)}\n\n"
        f"{guidance_block}"
        f"Current implementation:\n{impl_code}\n\n"
        f"Test failure output:\n{failure_output}\n\n"
        "Corrected implementation:"
    )
    fixed = Settings.llm.complete(prompt).text.strip()
    repaired = extract_code_block(fixed) or fixed

    # Same syntax gate the first generation gets. Without it a repair can hand unparseable
    # source straight back to pytest, which reports a collection error for the whole file —
    # so the next triage round reasons about a traceback for the test module instead of the
    # defect, and the round is wasted.
    err = _syntax_error_of(repaired)
    if err is None:
        return repaired

    print(f"[repair] repaired source is not valid Python ({err}); retrying once.", file=sys.stderr)
    retry = Settings.llm.complete(
        f"{prompt}\n\n"
        f"YOUR PREVIOUS ANSWER WAS REJECTED: it was not valid Python.\n"
        f"{type(err).__name__}: {err}\n"
        f"Offending line {err.lineno}: {(err.text or '').rstrip()}\n"
        "Return the corrected, complete function. Every `if`/`for`/`while`/`try`/`def` header "
        "must be followed by a real indented block, and every string/docstring must be closed."
    ).text.strip()
    retry_code = extract_code_block(retry) or retry
    if _syntax_error_of(retry_code) is None:
        return retry_code

    # Two unparseable attempts: keep the implementation we already had. It is the last
    # version known to parse, and replacing it with source that cannot even be imported
    # would turn a failing test into a collection error and lose the remaining rounds.
    print("[repair] retry still not valid Python; keeping the previous implementation.",
          file=sys.stderr)
    return impl_code


def _approval_gate(note: str, contract: dict, import_hint: str, tests: list[dict]) -> list[dict]:
    """
    Stage 5, step 2 ("unit tests are modified by the user and approved by the user"):
    show the note + proposed spec-based tests, then loop on input(). 'y'/Enter approves;
    anything else is treated as revision suggestions and the tests are regenerated from
    the contract, the tests currently on screen, and those suggestions together — so a
    comment aimed at one test ("B3 is wrong because...") reaches a model that can see B3.
    Note the whole set is still rewritten each round: this makes the untouched tests very
    likely to come back verbatim, but it does not pin them. Prompts go to stderr so
    redirecting stdout to a file still works.
    """
    while True:
        print("\n" + note, file=sys.stderr)
        print("\n----- PROPOSED SPEC-BASED UNIT TESTS (review before running) -----", file=sys.stderr)
        for t in tests:
            print(f"\n# {t.get('behavior_id', '?')}: {t.get('test_name', '')}", file=sys.stderr)
            print(t.get("code", ""), file=sys.stderr)
        resp = input("\nApprove these tests? 'y' to run them, or describe changes to remake: ").strip()
        if resp.lower() in {"y", "yes", ""}:
            return tests
        print("[tests] regenerating tests from the contract with your suggestions...", file=sys.stderr)
        tests = _spec_based_tests(contract, import_hint, revision=resp, previous_tests=tests)


def _approval_gate_algorithm(
    note: str,
    entries: list[dict],
    blueprint: dict,
    frozen_interface: str,
) -> list[dict]:
    """
    Combined counterpart to _approval_gate() for the GENERATE_ALGORITHM pathway: shows
    every frozen-interface function's proposed tests TOGETHER in one review and applies
    a single approve/revise cycle across the whole set, instead of build_algorithm_tests
    gating each function one at a time as its tests came off the model. A comment about
    how two functions compose — "compute_migration_plan's B2 assumes X, but
    determine_stream_order's B1 returns Y" — is not answerable when only one function's
    tests are ever on screen; this is what lets a reviewer make that comment at all.

    On revision, EVERY function's tests are regenerated from ITS OWN contract with the
    SAME comment applied to each. The comment is shared context across the whole batch,
    not routed to a single function — there is no reliable way to tell from free text
    alone which function(s) it is about, and a comment about composition is often about
    more than one. blueprint/frozen_interface are threaded through the regeneration call
    so a revision round keeps the cross-function grounding the first generation had;
    plain _approval_gate()'s regeneration call does not carry these, so this is also the
    fix for that context silently dropping out of the plain path's own revision rounds.

    Entries with no tests (spec or contract derivation failed earlier) are shown for
    visibility but excluded from the count in the prompt; each is passed through
    unchanged on a revision round rather than sent to _spec_based_tests, since there is
    no contract to regenerate from.
    """
    while True:
        print("\n" + note, file=sys.stderr)
        testable = [e for e in entries if e.get("tests")]
        for e in entries:
            print(f"\n{'=' * 70}\n{e['name']}"
                  f"{'' if e.get('tests') else '  (no tests were produced)'}"
                  f"\n{'=' * 70}", file=sys.stderr)
            if not e.get("tests"):
                continue
            print("\n----- PROPOSED SPEC-BASED UNIT TESTS (review before running) -----",
                  file=sys.stderr)
            for t in e["tests"]:
                print(f"\n# {t.get('behavior_id', '?')}: {t.get('test_name', '')}",
                      file=sys.stderr)
                print(t.get("code", ""), file=sys.stderr)

        total = sum(len(e["tests"]) for e in testable)
        resp = input(
            f"\nApprove ALL {total} test(s) across {len(testable)} function(s)? "
            f"'y' to run them, or describe changes to remake: "
        ).strip()
        if resp.lower() in {"y", "yes", ""}:
            return entries

        print("[tests] regenerating tests for every function from its own contract "
              "with your suggestions...", file=sys.stderr)
        new_entries = []
        for e in entries:
            if not e.get("contract"):
                # No contract to regenerate from at all (spec/contract derivation
                # failed upstream, before this loop ever ran) - nothing to send to
                # _spec_based_tests. This must check `contract`, not `tests`: an
                # empty test list is exactly as falsy as a genuinely-missing
                # contract, so checking `tests` here means a function whose contract
                # is fine but whose most recent attempt happened to return zero tests
                # (e.g. a confused refusal on an earlier revision round) gets frozen
                # at zero tests and skipped on every subsequent round forever, with
                # no way for it to ever regenerate again - confirmed on a real run,
                # where a function stayed stuck at 0 tests through a further revision
                # round specifically asking it to restore them, because this branch
                # never even sent that round's request to it.
                new_entries.append(e)
                continue
            new_tests = _spec_based_tests(
                e["contract"], e["import_hint"], revision=resp,
                previous_tests=e.get("tests") or [],
                query_str=e.get("query_str", ""), blueprint=blueprint,
                frozen_interface=frozen_interface,
            )
            new_entries.append({**e, "tests": new_tests})
        entries = new_entries


def _run_and_repair(impl_code, spec, contract, tests, module, query, max_rounds=2):
    """
    Stage 5, steps 3-5 combined: execute the APPROVED tests against the implementation.
    On failure, triage the round and — if the verdict is repairable — regenerate the
    IMPLEMENTATION and re-execute, up to max_rounds.

    The tests are never touched. They were derived from a specification the user approved
    and were then themselves approved by the user, so a failure can only mean the code is
    wrong; rewriting the tests here would let the pipeline pass by moving the goalposts.
    (The old "fix the test instead" branch was removed for exactly that reason.)

    Returns (execution_result, impl_code).
    """
    test_codes = [t.get("code", "") for t in tests if t.get("code")]

    def _accept_repair(repaired: str, previous: str) -> str:
        """See the identical helper in _run_and_repair_algorithm — same gap, same fix:
        never hand back repaired code that cannot even be imported without at least
        trying the one known, deterministic, mechanical fix first."""
        fixed = _fix_future_import_position(repaired)
        if _syntax_error_of(fixed) is None:
            return fixed
        print("[codegen] repair produced invalid Python; keeping the previous "
              "implementation for this round.", file=sys.stderr)
        return previous

    result = None
    for rnd in range(max_rounds + 1):
        result = _execute_spec_tests(impl_code, test_codes, module)
        if result["success"]:
            print(f"[exec] round {rnd}: all tests passed.", file=sys.stderr)
            return result, impl_code
        print(f"[exec] round {rnd}: tests failed (rc={result['returncode']}).", file=sys.stderr)
        if rnd == max_rounds:
            print("[exec] max repair rounds reached; returning last result.", file=sys.stderr)
            return result, impl_code

        # stdout carries everything pytest reports — measured, stderr comes back empty
        # even when the code under test writes to it, because pytest captures and folds
        # it into its own report. stderr is kept in the join only because _execute_spec_tests
        # writes its OWN two failures there ("pytest timed out after 90s.", "pytest not
        # found"), which appear nowhere else.
        failure_output = (result["stdout"] + "\n" + result["stderr"]).strip()
        # Sent whole, on purpose. The prompt budget is ~13k tokens after the reply
        # reservation and a bad run measured ~24k chars (~6k tokens), so it fits — but the
        # server REJECTS an oversized prompt rather than trimming it, which would cost the
        # whole round. This line is the breadcrumb for that case; it does not truncate.
        if len(failure_output) > 40000:
            print(f"[exec] failure output is {len(failure_output):,} chars — unusually "
                  f"large; if the repair call fails, this is why.", file=sys.stderr)
        try:
            triage = _triage_failure(contract, "\n\n".join(test_codes), impl_code, failure_output)
        except Exception as e:  # noqa: BLE001
            print(f"[triage] parse failed ({e}); repairing the implementation anyway.",
                  file=sys.stderr)
            impl_code = _accept_repair(
                _repair_impl(impl_code, spec, contract, failure_output, query),
                impl_code,
            )
            continue

        target = triage.get("suggested_fix_target")
        print(f"[triage] verdict={triage.get('verdict')} -> fix {target}", file=sys.stderr)
        if target == "none":
            # Environmental only (no test runner, missing package, harness timeout) —
            # regenerating the implementation cannot fix it.
            print(f"[triage] {triage.get('verdict')} — outside the implementation; stopping.",
                  file=sys.stderr)
            return result, impl_code

        impl_code = _accept_repair(
            _repair_impl(
                impl_code, spec, contract, failure_output, query,
                guidance=triage.get("required_impl_change") or triage.get("explanation", ""),
            ),
            impl_code,
        )
    return result, impl_code


# ── Stages 4-5 orchestrator: specify -> test -> approve -> implement ──────────
# The single entry point the generation branch of answer_query_deterministic() calls.
# Order matters and is deliberate: the specification is settled first, the tests are
# written from it (and approved) BEFORE any code exists, and only then is the
# implementation generated — so the code is always written against a target the user has
# already signed off on, and can never define its own success criteria. The spec itself
# passes straight through to test generation; the TESTS are the only artefact gated on
# the user, since they are what the implementation is ultimately judged against.
#
#   4a  generate_function_spec        -> the JSON specification (no code, no review)
#   5.1 _derive_contract_from_spec    -> behavioral contract from the spec
#   5.2 _spec_based_tests             -> pytest tests from the contract
#   5.2'_approval_gate                -> user reviews / amends the tests          [interactive]
#   4b  generate_code_from_spec       -> the implementation, from the spec minus "raises"
#   5.3 _run_and_repair               -> execute; repair the CODE ONLY on failure [interactive]
#
# Batch (interactive=False) runs the same sequence with the test gate skipped and no
# execution, so redirected/non-tty runs still produce spec + tests + code.
# Returns (generated_code_json, approved_test_codes, execution_result, spec).
def generate_verified_implementation(
    sq: SubQuery,
    helper_functions: list[dict],
    extracted_params: Optional[dict],
    note: str,
    interactive: bool,
    use_algorithm_prompt: bool = False,
    blueprint: Optional[dict] = None,
    frozen_interface: Optional[str] = None,
) -> tuple[str, list[str], Optional[dict], Optional[dict]]:
    user_request = sq.text

    # "parameters" is a docs-only index with no generation engine — there is no function
    # to specify, so the legacy behavior (surface the retrieved parameter context) stands.
    if sq.module == "parameters":
        return _retrieve_param_context(augment_query(sq.text, sq.module)), [], None, None
    if sq.module not in gen_engines:
        return f"NO_GENERATION_ENGINE_AVAILABLE for module: {sq.module}", [], None, None

    context_str, request_block = _build_generation_context(
        sq, helper_functions, extracted_params
    )

    # ── 4a: specify ───────────────────────────────────────────────────────────
    spec = generate_function_spec(context_str, request_block)
    if spec is None:
        print("[spec] no usable specification produced; skipping generation.", file=sys.stderr)
        return "", [], None, None

    print(f"[spec] generated: "
          f"{spec.get('name')} — {len(spec.get('postconditions', []))} postcondition(s), "
          f"{len(spec.get('raises', []))} declared exception(s).")

    # ── 5.1 / 5.2: contract + tests straight from the spec, before any code ───
    import_hint = _import_hint_for(spec, sq.module)
    contract, tests = None, []
    # TEMPORARILY COMMENTED OUT
    # try:
    #     contract = _derive_contract_from_spec(spec, user_request)
    #     # Carried through (not re-derived by SPEC_EXTRACTION_PROMPT, whose own output
    #     # schema has no signature field) so _spec_based_tests -- and any later revision
    #     # retry, which reuses this same contract object -- always sees the real def line
    #     # instead of having to guess the parameter order or whether self/cls comes first.
    #     contract["signature"] = spec.get("signature")
    #     tests = _spec_based_tests(
    #         contract, import_hint, query_str=request_block,
    #         blueprint=blueprint, frozen_interface=frozen_interface,
    #     )
    #     print(f"[tests] spec-based: {len(tests)} test(s) from "
    #           f"{len(contract.get('behaviors', []))} behavior(s).")
    #
    #     if not tests:
    #         print("[tests] no usable tests were produced.", file=sys.stderr)
    # except Exception as e:  # noqa: BLE001
    #     print(f"[tests] spec-based generation failed: {e}")
    #
    # if interactive and tests:
    #     tests = _approval_gate(note, contract, import_hint, tests)

    # ── 4b: implement the spec ───────────────────────────────────────────────
    # use_algorithm_prompt swaps GENERATOR_PROMPT (one function) for
    # GENERATE_ALGORITHM_PROMPT (an orchestration layer over reused/new capabilities) —
    # see Intent.GENERATE_ALGORITHM's branch in answer_query_deterministic(). Everything
    # else in this function (spec, contract, tests, approval gate, execute/repair) is
    # identical for both prompts.
    active_prompt = GENERATE_ALGORITHM_PROMPT if use_algorithm_prompt else GENERATOR_PROMPT
    if use_algorithm_prompt:
        impl_code = generate_algorithm_from_spec(
            blueprint, frozen_interface, context_str, request_block
        )
    else:
        impl_code = generate_code_from_spec(spec, context_str, request_block)

    # Pre-flight (no LLM): `np.zeros(...)` with no `import numpy as np`, a bare
    # ScheduleEncodingType, a `reduce(...)` never imported from functools. GENERATOR_PROMPT/
    # GENERATE_ALGORITHM_PROMPT both ask for the imports; this checks that they are
    # actually there, since the symptom otherwise is a NameError raised inside the
    # generated module at test time, which triage then has to diagnose from a traceback.
    missing = _undefined_names(impl_code, provided=LIB_PROVIDED_NAMES)
    # A second, distinct failure `_undefined_names` cannot see: a name imported `from`
    # the wrong real module (e.g. `from tsn_scheduling_problem import
    # ScheduleEncodingType`, a real name that only lives in evaluator.py). The name IS
    # bound by the import statement, so nothing looks missing until that import itself
    # raises ImportError and aborts collection of the whole generated module.
    wrong_module = _wrong_module_imports(impl_code, LIB_DIR)
    # A third, distinct failure neither check above can see: a name imported from the
    # RIGHT module but called with the wrong number of arguments for its real signature.
    # See _bad_kb_call_arity's docstring — one real run had 78 of 199 failing tests trace
    # back to exactly this pattern, every one of them AFTER its import was already fixed.
    bad_arity = _bad_kb_call_arity(impl_code, LIB_DIR)
    if missing or wrong_module or bad_arity:
        problems = []
        if missing:
            problems.append(f"used these names without defining or importing them: "
                             f"{missing}")
        if wrong_module:
            problems.append(
                "imported these names from the wrong module (the name is real, but not "
                "part of that module's public interface): " +
                ", ".join(f"'{name}' from '{mod}'" for name, mod in wrong_module)
            )
        if bad_arity:
            problems.append(
                "called a real, correctly-imported library class/function with the "
                "wrong number of arguments for its actual signature: " + "; ".join(bad_arity)
            )
        print(f"[codegen] implementation {'; and '.join(problems)}; retrying once.",
              file=sys.stderr)
        # GENERATOR_PROMPT and GENERATE_ALGORITHM_PROMPT no longer share a placeholder
        # set (the algorithm prompt takes blueprint_json/frozen_interface instead of
        # spec_json), so the retry has to format whichever one is active with its own
        # inputs rather than one shared call.
        if use_algorithm_prompt:
            retry_format_kwargs = dict(
                blueprint_json=json.dumps(_blueprint_for_prompt(blueprint), indent=2),
                frozen_interface=frozen_interface or "(no frozen interface available)",
                context_str=context_str,
                query_str=request_block,
            )
        else:
            retry_format_kwargs = dict(
                spec_json=json.dumps(_spec_for_codegen(spec), indent=2),
                context_str=context_str,
                query_str=request_block,
            )
        arity_retry_note = (
            "\nAlso fix any call listed above as having the wrong argument count/shape "
            "so it matches the real signature shown for it — check the library source "
            "under original_tsn_lib_files/ for that class/function's actual __init__ or "
            "definition rather than guessing what it takes."
            if bad_arity else ""
        )
        retry = Settings.llm.complete(
            active_prompt.format(**retry_format_kwargs)
            + f"\n\nYOUR PREVIOUS ANSWER WAS REJECTED: it {'; and '.join(problems)}.\n"
              "Return the corrected, complete function with every name it uses either "
              "imported at the top of the file (`import numpy as np`, "
              "`from evaluator import ScheduleEncodingType`, `from functools import reduce`, "
              "...) or defined in its body, each from the real module that actually "
              "defines it — check the library source under original_tsn_lib_files/ rather "
              "than guessing which module a real name lives in. Import only what you "
              "actually use." + arity_retry_note
        ).text.strip()
        retry_code = (extract_code_block(retry) or retry).strip()
        retry_ok = (_syntax_error_of(retry_code) is None
                    and not _undefined_names(retry_code, provided=LIB_PROVIDED_NAMES))
        # As on the algorithm-pathway side: the retry above doesn't reliably fix a
        # wrong-module import — it has reproduced the identical mistake, or moved it to a
        # different equally-wrong module — and that class of problem doesn't need an LLM
        # at all, since the correct module is just whichever real library file actually
        # defines the name. Fix it directly rather than hoping a second LLM call complies.
        candidate = retry_code if retry_ok else impl_code
        candidate = _autofix_wrong_module_imports(candidate, LIB_DIR)
        if (_syntax_error_of(candidate) is None
                and not _undefined_names(candidate, provided=LIB_PROVIDED_NAMES)
                and not _wrong_module_imports(candidate, LIB_DIR)
                and not _bad_kb_call_arity(candidate, LIB_DIR)):
            impl_code = candidate
        else:
            print("[codegen] retry did not resolve the import/arity problem(s); keeping "
                  "the original implementation.", file=sys.stderr)

    # The function dict handed downstream is the spec plus the code written for
    # it. Nothing here is re-read from the implementation: the spec is what the tests were
    # derived from, so it — not the generator's own account of itself — describes the result.
    # (Note that a same-named existing function is deliberately NOT substituted for the
    # generated body: this path only runs when coverage found the codebase lacking.)
    fn = {**spec, "module": sq.module, "code": impl_code}

    # ── 5.3: execute the approved tests; repair the CODE ONLY ────────────────
    execution_result = None
    if interactive and tests:
        execution_result, impl_code = _run_and_repair(
            impl_code, spec, contract, tests, sq.module, user_request
        )
        fn["code"] = impl_code   # reflect any repair back into the displayed code
    elif interactive:
        print("[verify] no spec-based tests were produced; skipping execution.", file=sys.stderr)

    test_codes = [t.get("code", "") for t in tests if t.get("code")]
    return json.dumps([fn], indent=2), test_codes, execution_result, spec


# Renders one sub-query's full result section (module/coverage banner, extracted
# parameters, generated code, unit tests, execution outcome) as plain text for the
# final answer. Used only on the generation path (needs_generation branch) of
# answer_query_deterministic() — the EXPLAIN_CONCEPT/EXPLAIN_CODE paths build their own
# sections via _explain_concept_section()/_explain_code_section() instead.
def format_subquery_answer(
    sq: SubQuery,
    extracted_params: dict,
    param_explanations: dict,
    functions: list[dict],
    coverage: CoverageAssessment,
    generated_code: Optional[str],
    execution_result: Optional[dict],
    unit_tests: list[str] = None,
    resolution = None,
    spec: Optional[dict] = None,
) -> str:
    lines = []

    lines.append(f"MODULE: {sq.module}")
    lines.append(f"SUB-QUERY: {sq.text}")
    lines.append(f"MATCH TYPE: {coverage.match_type.value}")
    lines.append(f"COVERAGE REASON: {coverage.reason}")
    lines.append(
        "MISSING REQUIREMENTS: "
        + json.dumps(coverage.missing_requirements, indent=2)
    )
    lines.append(f"RECOMMENDED ACTION: {coverage.recommended_action.value}")

    lines.append("\nEXTRACTED PARAMETERS:")
    if extracted_params:
        for name, value in extracted_params.items():
            explanation = param_explanations.get(name, {})
            lines.append(f"- {name} = {value}")

            if explanation:
                lines.append(f"  Meaning: {explanation.get('meaning', '')}")
                lines.append(f"  Used for: {explanation.get('used_for', '')}")
    else:
        lines.append("none detected")

    if resolution and (resolution.get("answer") or resolution.get("call_site")):
        if resolution.get("answer"):
            lines.append("\nANSWER:")
            lines.append(resolution["answer"])
        if resolution.get("call_site"):
            lines.append("\nUSAGE:")
            lines.append("```python")
            lines.append(resolution["call_site"])
            lines.append("```")
        if resolution.get("assumptions"):
            lines.append("\nASSUMPTIONS:")
            for a in resolution["assumptions"]:
                lines.append(f"- {a}")

    # UNUSED: would dump the retrieved helper functions' full metadata as raw JSON here,
    # with a <CODE:name> placeholder for inject_code() to splice real source into later.
    # Superseded by _explain_code_section(), which renders retrieved/helper functions as
    # a proper LLM-written explanation instead of a raw JSON metadata dump.
    # if functions:
    #         lines.append("\nFUNCTIONS FOUND:")

    #         for fn in functions:
    #             # Emit all fields except code — code is injected by Python after formatting
    #             printable = {k: v for k, v in fn.items() if k not in {
    #                 "syntax_check", "match_confidence", "reranker_score", "routed_module",
    #                 "catalog_validation", "code",  # <-- exclude code here
    #             }}
    #             lines.append(json.dumps(printable, indent=2))
    #             # Placeholder: Python will replace this with the real, untruncated code
    #             lines.append(f"    \"code\": <CODE:{fn.get('name')}>")

    # else:
    #     lines.append("\nFUNCTIONS FOUND: none")

    # The specification the code and the tests were both built from — shown so the
    # reader can check the implementation against what was actually agreed, and so
    # batch (non-interactive) runs still surface the spec that had no review gate.
    if spec:
        lines.append("\nFUNCTION SPECIFICATION:")
        lines.append("```json")
        lines.append(_render_spec_for_review(spec))
        lines.append("```")

    if generated_code:
        lines.append("\nGENERATED CODE:")
        lines.append(_render_generated_code(generated_code))

    if unit_tests:
        lines.append("\nUNIT TESTS:")
        for test_fn in unit_tests:
            lines.append("```python")
            lines.append(test_fn)
            lines.append("```")
        
    if execution_result:
        lines.append("\nEXECUTION:")
        lines.append(f"Success: {execution_result.get('success')}")
        lines.append(f"Return code: {execution_result.get('returncode')}")

        if execution_result.get("stdout"):
            lines.append(f"stdout:\n{execution_result['stdout']}")

        if execution_result.get("stderr"):
            lines.append(f"stderr:\n{execution_result['stderr']}")

    return "\n".join(lines)

# Used by _explain_concept_section() for EXPLAIN_CONCEPT sub-queries (conceptual TSN
# questions not tied to one specific function) — grounds the answer in retrieved
# parameter definitions + cross-module docstring/code context, never invention.
EXPLAIN_CONCEPT_PROMPT = """\
You are a TSN (Time-Sensitive Networking) concept explainer for this codebase.

You are given two grounding sources from THIS codebase — parameter definitions and
function docstrings/code.

Rules:
- Ground the answer in the provided context ONLY. Use it wherever it speaks to
  the question, and stay consistent with it.
- Never contradict the provided context. Be precise and concise.
- Show the code provided in the context so that the user can connect what you are saying with what is in the code.
- Answer THIS SUB-TASK specifically, but keep the OVERALL USER QUESTION in view so
  your explanation stays relevant to what the user ultimately wants. Do NOT answer
  the other parts of the overall question here — other modules handle those. Just
  make sure this piece visibly fits into the whole.

Parameter definitions (from this codebase):
{param_context}

Function docstrings / code context (from this codebase):
{doc_context}

Overall user question (CONTEXT ONLY — what the user ultimately wants; do not answer
the other parts of it here):
{overall_query}

This sub-task (the specific piece to address now):
{query}

Explanation:
"""

# Used by _explain_code_section() for EXPLAIN_CODE sub-queries AND for the exact-match
# case of DO sub-queries (RETURN_EXISTING_FUNCTIONS) — this is the "return the existing
# function plus an explanation of its parameters" half of the use case.
EXPLAIN_CODE_PROMPT = """
You are a TSN codebase explainer. Answer the user's question about the function(s)
below as directly and concisely as possible. Match the answer's scope to the
question — do not pad it out.
 
Scope:
- If the question asks for a SPECIFIC detail (a parameter's default value, the
  return value, whether it does X, one step of the logic), answer ONLY that, in
  1-3 sentences, quoting the exact part of the signature/code that shows it. Do
  NOT add Purpose / Inputs / Output / Logic / Notes sections the question did not
  ask for, and do NOT describe the other parameters or unrelated functions.
- Only if the question is a broad "explain / walk me through this function"
  request should you give a structured walkthrough (Purpose, Inputs, Output,
  Logic, Notes).
 
Rules:
- Ground every statement in the provided code and docstring. If the code does not
  show something, say so rather than guessing.
- Ignore any provided function that is not relevant to the question.
 
Function(s):
{functions_block}

Overall user request (CONTEXT ONLY — what the user ultimately wants; do not answer
the other parts of it here):
{overall_query}

The specific question to answer now:
{query}

Answer:
"""

def _collect_doc_context(sq: SubQuery) -> str:
    """Retrieve docstring/code grounding for a sub-query from the WHOLE codebase.

    Uses the combined cross-module retriever (not the routed module alone),
    reranks to the top 10, then follows each chunk's referenced symbols to the
    chunks that DEFINE them (dependency expansion). The reranked + expanded set
    IS the context — no k_nodes truncation. The DO / exact-match and generation
    paths are unaffected (they still use the per-module retrievers).
    """
    if sq.module == "parameters":
        return ""
    # Raw sub-query text on purpose: the combined retriever spans all modules, so
    # we do NOT anchor to a single file here (augment_query stays in use on the
    # per-module DO / generation paths).
    query = sq.text
    nodes = combined_retriever.retrieve(query)
    nodes = combined_reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(query)
    )
    nodes = _expand_with_definitions(nodes)
    return "\n\n".join(n.node.get_content() for n in nodes)


# Builds one EXPLAIN_CONCEPT result section: gathers parameter + cross-module doc
# grounding, then asks EXPLAIN_CONCEPT_PROMPT to answer the conceptual sub-query
# (the original full query is passed too, as overall_query, purely as context so
# the answer stays relevant to what the user ultimately wants — the LLM is told
# to answer only the sub-query, not the other parts of the overall question).
def _explain_concept_section(query: str, sq: SubQuery,
                             extracted_params: dict, param_explanations: dict) -> str:
    # 1) Parameter grounding (retrieval-only, no synthesis LLM call).
    param_ctx = _retrieve_param_context(
        f"Explain parameters and concepts relevant to: {augment_query(sq.text, sq.module)}"
    )
    if not param_ctx:
        param_ctx = "No directly relevant parameter definitions found."

    # 2) Docstring grounding (skip for the parameters module).
    doc_ctx = "" if sq.module == "parameters" else _collect_doc_context(sq)
    if not doc_ctx:
        doc_ctx = "No directly relevant function docstrings found."

    # 3) Synthesize (grounded first, LLM knowledge allowed and marked).
    prompt = EXPLAIN_CONCEPT_PROMPT.format(
        param_context=param_ctx[:8000],
        doc_context=doc_ctx[:8000],
        overall_query=query,
        query=sq.text,
    )
    explanation = Settings.llm.complete(prompt).text.strip()
    return "\n".join([
        f"MODULE: {sq.module}", f"SUB-QUERY: {sq.text}",
        "INTENT: EXPLAIN_CONCEPT", "", "EXPLANATION:", explanation,
    ])


# Builds one EXPLAIN_CODE result section — this is the "return the matched function(s)
# with a clear explanation" path of the use case, covering both genuine EXPLAIN_CODE
# sub-queries and DO sub-queries whose coverage assessment was an EXACT_MATCH (see the
# reroute in answer_query_deterministic()'s Pass 3). Falls back to whole-codebase
# doc-context retrieval if no function was matched by name.
def _explain_code_section(query: str, sq: SubQuery, functions: list[dict],
                          extracted_params: dict, param_explanations: dict) -> str:
    if functions:
        blocks = [
            f"Name: {fn.get('name')}\nSignature: {fn.get('signature')}\n"
            f"Module: {fn.get('modules')}\nDocstring: {fn.get('docstring')}\n"
            f"In plain words: {fn.get('nl_summary') or 'None'}\n"
            f"Code:\n{fn.get('code', '')}"
            for fn in functions
        ]
        functions_block = "\n\n---\n\n".join(blocks)
    else:
        # Nothing matched by name — fall back to retrieved docstrings so we stay grounded.
        functions_block = _collect_doc_context(sq) or ""
        if not functions_block:
            return "\n".join([
                f"MODULE: {sq.module}", f"SUB-QUERY: {sq.text}", "INTENT: EXPLAIN_CODE",
                "", "No matching function or code was found in this module to explain.",
            ])

    prompt = EXPLAIN_CODE_PROMPT.format(
        functions_block=functions_block[:40000],
        overall_query=query,
        query=sq.text,
    )
    explanation = Settings.llm.complete(prompt).text.strip()

    lines = [f"MODULE: {sq.module}", f"SUB-QUERY: {sq.text}", "INTENT: EXPLAIN_CODE", ""]
    if functions:
        lines.append("FUNCTIONS RETRIEVED AS CONTEXT: "
                     + ", ".join(fn.get("name") for fn in functions if fn.get("name")))
        lines.append("")
    lines += ["EXPLANATION/DIRECT ANSWER:", explanation]
    return "\n".join(lines)

# ── Final answer compiler (stage 6: merge all sub-query answers into one) ────
# Takes the per-sub-query RESULT sections built by answer_query_deterministic() and
# rewrites them into a single coherent answer to the user's original question,
# instead of leaving them as separate concatenated blocks.
FINAL_COMPILER_PROMPT = """\
You are the final answer compiler for a TSN (Time-Sensitive Networking) code
assistant. The user asked ONE overall question. The system split that question
into sub-queries and answered each one separately. Your job is to merge those
separate answers into a SINGLE, coherent response that directly and completely
answers the user's ORIGINAL question.

ORIGINAL USER QUERY:
{original_query}

INDIVIDUAL SUB-QUERY ANSWERS (in no particular order):
{subquery_answers}

WHAT TO DO:
- Write ONE unified answer addressed to the ORIGINAL user query above. Organize it
  logically (for example as an end-to-end workflow), not as a list of disconnected
  per-module blocks. Merge overlapping points and drop redundant repetition.
- Make explicit how the pieces fit together to satisfy the original request, and
  which part answers which aspect of it.
- Keep clear whether each part relies on an EXISTING library function or on NEWLY
  GENERATED code. Do not present generated code as if it already exists in the
  codebase, and do not present existing functions as if they were generated.

HARD RULES — do not violate:
- Do NOT change any fact. Function names, parameters, parameter values,
  match/coverage outcomes, execution results, and test outcomes must stay exactly
  as given. You may ONLY rephrase the surrounding wording so the whole thing reads
  as one answer to the original query — never alter, "correct", drop, or invent
  facts.
- Do NOT write, edit, translate, reformat, summarize, or remove any code. Every
  token of the form <GENCODE:number> or <CODE:name> is a stand-in for a code block.
  Copy these tokens through VERBATIM and unchanged, each placed where that code
  belongs in your answer. Never type actual code yourself and never invent new
  tokens.
- Do NOT add information, functions, caveats, or capabilities that are not present
  in the sub-query answers above.
- If two sub-answers conflict, state the conflict rather than silently choosing one.
- Do NOT add the FUNCTIONS RETRIEVED AS CONTEXT list to your answer

Write only the final compiled answer, starting immediately with the answer itself
(no preamble such as "Here is the compiled answer").

FINAL COMPILED ANSWER:
"""


# Any fenced ```...``` block: generated code, unit tests, usage snippets. Masked
# out before the final compiler sees the text so no verified/generated code can be
# rewritten by the LLM, then restored verbatim afterwards.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)


def _mask_code_blocks(text: str) -> tuple[str, dict]:
    """
    Replace every fenced ```...``` block with a unique <GENCODE:i> sentinel so the
    final compiler LLM cannot touch verified/generated code. Returns
    (masked_text, {sentinel: original_block}). Retrieved-function <CODE:name>
    placeholders are already sentinels and are left as-is (resolved later by
    inject_code).
    """
    blocks: dict = {}

    def _repl(m):
        key = f"<GENCODE:{len(blocks)}>"
        blocks[key] = m.group(0)
        return key

    return _FENCED_CODE_RE.sub(_repl, text), blocks


def _reinject_code_blocks(text: str, blocks: dict) -> str:
    """Inverse of _mask_code_blocks: restore the exact original fenced blocks."""
    for key, block in blocks.items():
        text = text.replace(key, block)
    return text


def compile_final_answer(
    original_query: str,
    subquery_answers: str,
) -> str:
    """
    Single final LLM pass. Takes the per-sub-query answer sections VERBATIM (with
    code already masked as <GENCODE:i> / <CODE:name> sentinels) and rewrites them
    into one coherent answer to the ORIGINAL user query. The prompt permits
    rephrasing prose to serve the original query but forbids changing any fact or
    touching code; the code sentinels are re-injected deterministically after the
    call, so no verified/generated code ever passes through the model.
    """
    prompt = FINAL_COMPILER_PROMPT.format(
        original_query=original_query,
        subquery_answers=subquery_answers,
    )
    return Settings.llm.complete(prompt).text.strip()


# Orchestrates stages 1-5 for one full query. NOTE: run_generated_code is accepted and
# threaded through from run_one()/run_interactive() but never actually read in the body
# below — whether generated code gets executed is controlled entirely by `interactive`
# (which Pass 3 forwards to generate_verified_implementation(), where it gates both review
# gates and the test run). Currently a no-op flag; kept as-is since removing it would
# change this function's public signature.
def answer_query_deterministic(
    query: str,
    run_generated_code: bool = True,
    interactive: bool = False,
) -> str:
    # Stage 0: decide, before any classification/sub-query splitting happens, whether
    # this is an algorithm-generation request. YES bypasses classify_query()/
    # CLASSIFICATION_PROMPT entirely and goes to the separate algorithm pathway
    # instead; NO falls through to the existing pipeline, unchanged.
    if is_algorithm_generation_query(query):
        return run_algorithm_generation_pathway(
            query, run_generated_code=run_generated_code, interactive=interactive
        )

    classification = get_or_classify_query(query)

    final_sections = []

    # ── Pass 1: collect raw results for every sub-query ───────────────────────
    # Mirrors the original per-sq loop but stores results instead of formatting
    # immediately, so Pass 2 can merge same-module entries before generation.
    raw_results = []
    for sq in classification.sub_queries:
        extracted_params = sq.parameters
        param_explanations = {}

        if sq.action == Intent.EXPLAIN_CONCEPT:
            functions = []
            coverage = CoverageAssessment(
                MatchType.NO_MATCH, "EXPLAIN_CONCEPT: coverage not assessed.",
                [], RecommendedAction.GENERATE_STANDALONE,
            )
        else:
            # DO and EXPLAIN_CODE run the SAME coverage assessor. "What function can I
            # use to <task>?" queries classify as EXPLAIN_CODE but are really task
            # requests; assessing coverage lets a genuine match still explain while a
            # no-match falls through to generation in Pass 3.
            # UNIFIED_RETRIEVAL: retrieve over the WHOLE codebase on the first and
            # only pass, so the assessor already sees every candidate the retry used
            # to surface. One coverage call per sub-query, no _resolve_with_fallback.
            # The "parameters" module keeps _lookup_one(): it is a docs index with an
            # LLM-free pseudo-result and no function catalog to search cross-module.
            if UNIFIED_RETRIEVAL and sq.module != "parameters":
                functions, top_score, second_score = _lookup_cross_module(sq, unified=True)
                coverage = assess_function_coverage(
                    query=query,
                    sq=sq,
                    functions=functions,
                    top_score=top_score,
                    second_score=second_score,
                    module_label="all modules (unified whole-codebase retrieval)",
                )
            else:
                functions, top_score, second_score = _lookup_one(sq)
                coverage = assess_function_coverage(
                    query=query,
                    sq=sq,
                    functions=functions,
                    top_score=top_score,
                    second_score=second_score,
                )
                # Anything short of EXACT_MATCH gets a second pass over the whole
                # codebase before Pass 3 is allowed to generate anything. A wrong
                # module from the classifier is recoverable here; after generation
                # starts it is not.
                functions, coverage = _resolve_with_fallback(
                    query=query, sq=sq, functions=functions, coverage=coverage,
                )

        raw_results.append({
            "sq": sq, "module": sq.module, "action": sq.action,
            "extracted_params": extracted_params, "param_explanations": param_explanations,
            "functions": functions, "coverage": coverage,
        })

    # ── Pass 2: merge same-module results ─────────────────────────────────────
    # If the classifier produced two sub-queries for the same module (despite the
    # prompt fix and deduplicate_sub_queries), their functions and missing_requirements
    # are consolidated here so generation is triggered only once per module.
    # The first sub-query for each module is kept as the representative (sq, params, etc.).
    # GENERATE_ALGORITHM sub-queries are never merged by (module, action) alone — see the
    # matching note in validate_classification(). Each keeps its own "text" in the key so
    # distinct functionalities that land in the same module stay separate RESULT sections.
    merged_by_module: Dict[tuple, dict] = {}
    for r in raw_results:
        disambiguator = r["sq"].text if r["action"] == Intent.GENERATE_ALGORITHM else None
        key = (r["module"], r["action"], disambiguator)
        if key not in merged_by_module:
            merged_by_module[key] = r
        else:
            existing_names = {fn.get("name") for fn in merged_by_module[key]["functions"]}
            for fn in r["functions"]:
                if fn.get("name") not in existing_names:
                    merged_by_module[key]["functions"].append(fn)
                    existing_names.add(fn.get("name"))
            existing_missing = set(merged_by_module[key]["coverage"].missing_requirements)
            for req in r["coverage"].missing_requirements:
                if req not in existing_missing:
                    merged_by_module[key]["coverage"].missing_requirements.append(req)
                    existing_missing.add(req)


    # Header lines — MODULES INVOLVED reflects post-merge unique modules
    final_sections.append(f"QUERY TYPE: {classification.query_type.value}")
    final_sections.append(
        "MODULES INVOLVED: "
        + ", ".join(sorted({m for (m, _a, _d) in merged_by_module.keys()}))
    )
    final_sections.append(f"NORMALIZED QUERY: {classification.normalized_query}")

    # The per-sub-query answer sections start after the header lines above. Record
    # the boundary so the final compiler is fed ONLY the sub-query answers, not the
    # QUERY TYPE / MODULES INVOLVED / NORMALIZED QUERY header lines.
    _header_count = len(final_sections)

    # ── Pass 3: generate and format one result per unique module ──────────────
    for i, (key, r) in enumerate(merged_by_module.items(), start=1):
        module, action, _disambiguator = key
        sq = r["sq"]
        functions = r["functions"]
        coverage = r["coverage"]
        extracted_params = r["extracted_params"]
        param_explanations = r["param_explanations"]

        if action == Intent.EXPLAIN_CONCEPT:
            section = _explain_concept_section(query, sq, extracted_params, param_explanations)
            final_sections.append(f"\n--- RESULT {i} ---\n{section}")
            continue

        if action == Intent.GENERATE_ALGORITHM:
            # Unlike DO, this always generates — synthesizing the orchestration layer is
            # the point even when every sub-part already exists in the codebase (the
            # prompt's own DECOMPOSE/GAP ANALYSIS steps decide what to reuse vs. implement
            # internally); it is never rerouted to the EXPLAIN_CODE path.
            param_explanations = explain_extracted_parameters(extracted_params)
            note = (
                "NOTE: This is an algorithm-synthesis request. Existing codebase "
                "functions were retrieved and analyzed for reuse; the algorithm below is "
                "an orchestration layer that calls them where possible and implements "
                "only genuinely missing functionality. Spec-based unit tests are "
                "included for review."
            )

            _gen_t0 = time.time()
            generated_code, generated_unit_tests, execution_result, spec = (
                generate_verified_implementation(
                    sq=sq,
                    helper_functions=functions,
                    extracted_params=extracted_params,
                    note=note,
                    interactive=interactive,
                    use_algorithm_prompt=True,
                )
            )
            _generation_time_log.append(time.time() - _gen_t0)

            section = format_subquery_answer(
                sq=sq,
                extracted_params=extracted_params,
                param_explanations=param_explanations,
                functions=functions,
                coverage=coverage,
                generated_code=generated_code,
                execution_result=execution_result,
                unit_tests=generated_unit_tests,
                resolution=None,
                spec=spec,
            )
            section = note + "\n\n" + section
            final_sections.append(f"\n--- RESULT {i} ---\n{section}")
            continue

        if action == Intent.EXPLAIN_CODE:
            # "Which function can I use to <task>?" and "what does <fn> do?" both classify
            # as EXPLAIN_CODE. Either way the user is asking to be POINTED AT a function,
            # not handed newly written code.
            #
            #   EXACT_MATCH   -> explain the matched function(s).
            #   PARTIAL_MATCH -> STILL EXPLAIN. Relevant candidates were retrieved; naming
            #       the best one and describing what it does answers the question. This
            #       codebase is object-oriented (77% of functions take `self`, 100% of
            #       evaluator), so the assessor routinely returns PARTIAL_MATCH for a
            #       perfectly correct method purely because calling it needs an instance.
            #       Emitting a standalone re-implementation there is strictly worse than
            #       pointing at the real function.
            #   NO_MATCH      -> nothing relevant retrieved; fall through and generate.
            if coverage.recommended_action != RecommendedAction.GENERATE_STANDALONE:
                if coverage.recommended_action == RecommendedAction.GENERATE_WITH_HELPERS:
                    print(
                        f"[explain] {sq.module}: coverage said PARTIAL_MATCH but the intent "
                        f"is EXPLAIN_CODE — explaining the retrieved function(s) instead of "
                        f"generating."
                    )
                section = _explain_code_section(query, sq, functions, extracted_params, param_explanations)
                final_sections.append(f"\n--- RESULT {i} ---\n{section}")
                continue
            action = Intent.DO   # genuine NO_MATCH; coverage is already GENERATE_STANDALONE

        # action == Intent.DO — the coverage assessor's verdict splits DO into two
        # mechanisms:
        #   GENERATE_WITH_HELPERS / GENERATE_STANDALONE -> generation (UNCHANGED)
        #   (former) RETURN_EXISTING_FUNCTIONS, i.e. an exact match -> now routed to
        #       the EXPLAIN_CODE prompt instead of resolve_exact_match, so EXPLAIN_CODE
        #       is the single retrieve-and-use-existing-functions pathway.
        needs_generation = coverage.recommended_action in {
            RecommendedAction.GENERATE_WITH_HELPERS,
            RecommendedAction.GENERATE_STANDALONE,
        }

        if needs_generation:
            param_explanations = explain_extracted_parameters(extracted_params)

            # Human-readable banner about why we generated (per the open-world path).
            # Computed BEFORE generation now: it heads both review gates (specification
            # first, then tests), so the user sees why anything is being built at all.
            note = (
                "NOTE: No existing function in the codebase satisfies this request; "
                "the specification below was written from the requirement, and the code "
                "is generated from it. Spec-based unit tests (written from the "
                "specification, not the code) are included for review."
                if coverage.recommended_action == RecommendedAction.GENERATE_STANDALONE else
                "NOTE: The codebase only partially covers this request; retrieved "
                "helper(s) were used as context and the missing functionality was "
                "specified and then generated from that specification. Spec-based unit "
                "tests are included for review."
            )

            # Specify -> (approve) -> tests -> (approve) -> implement -> (execute/repair).
            # Interactive runs both review gates and executes; batch skips the gates and
            # the execution but still produces spec, tests and code.
            _gen_t0 = time.time()
            generated_code, generated_unit_tests, execution_result, spec = (
                generate_verified_implementation(
                    sq=sq,
                    helper_functions=functions,
                    extracted_params=extracted_params,
                    note=note,
                    interactive=interactive,
                )
            )
            _generation_time_log.append(time.time() - _gen_t0)

            section = format_subquery_answer(
                sq=sq,
                extracted_params=extracted_params,
                param_explanations=param_explanations,
                functions=functions,
                coverage=coverage,
                generated_code=generated_code,
                execution_result=execution_result,
                unit_tests=generated_unit_tests,
                resolution=None,
                spec=spec,
            )
            section = note + "\n\n" + section

        else:
            # EXACT_MATCH (was RETURN_EXISTING_FUNCTIONS): retrieve-and-use existing
            # functions through the EXPLAIN_CODE prompt — no resolve_exact_match, no
            # generation. `functions` was already fetched by _lookup_one in Pass 1,
            # so the EXPLAIN_CODE path has the retrieved code to work from.
            section = _explain_code_section(
                query, sq, functions, extracted_params, param_explanations
            )

        final_sections.append(f"\n--- RESULT {i} ---\n{section}")

    raw_answer = "\n\n".join(final_sections)

    # Collect all retrieved functions across all modules for code injection
    all_functions = [
        fn
        for r in merged_by_module.values()
        for fn in r["functions"]
    ]

    # Fully-resolved per-sub-query results (retrieved-function <CODE:...> placeholders
    # replaced with real code). This is the detailed body kept BELOW the compiled
    # answer, exactly as the function used to return it.
    results_answer = inject_code(raw_answer, all_functions)

    # ── Final compile pass ────────────────────────────────────────────────────
    # One last LLM call that reads the per-sub-query answers verbatim and rewrites
    # them into a single coherent answer to the ORIGINAL user query. Runs for every
    # query (including simple ones). Code is masked out first so the model can only
    # rephrase prose, then re-injected so nothing verified is altered.
    subquery_sections = final_sections[_header_count:]
    if not subquery_sections:
        # Nothing to compile (no sub-query produced a section) — return detail only.
        return results_answer

    subquery_body = "\n\n".join(subquery_sections)
    masked_body, code_blocks = _mask_code_blocks(subquery_body)

    compiled = compile_final_answer(
        original_query=query,
        subquery_answers=masked_body,
    )

    # Restore code: <GENCODE:i> for masked fenced blocks, then <CODE:name> for any
    # retrieved-function placeholders the compiler carried through.
    compiled = _reinject_code_blocks(compiled, code_blocks)
    compiled = inject_code(compiled, all_functions)

    return (
        "FINAL COMPILED ANSWER\n"
        "=====================\n\n"
        f"{compiled}\n\n\n"
        "=====================\n"
        "PER-SUB-QUERY RESULTS (detail)\n"
        "=====================\n\n"
        f"{results_answer}"
    )


#  ── Run ───────────────────────────────────────────────────────────────────────
#
# All heavy objects (embed model, the four persisted indexes, every retriever,
# both rerankers, the LLM) are built ONCE above, at import time. Instead of
# answering a single hardcoded query and exiting — which threw that startup
# work away every run — we keep the process alive and loop. Startup is paid
# once; each subsequent query costs only its own work.
 
# One-time startup cost (everything from `start_time` at the top down to here).
# Printed to stderr so it stays visible even when stdout is redirected to a file.
print(f"[startup] models + indexes ready in {time.time() - start_time:.1f}s", file=sys.stderr)
 
 
def run_one(query: str, run_generated_code: bool = True, interactive: bool = False) -> str:
    """
    Answer a single query, resetting per-query state so each run is independent
    of the previous one — matching the old one-query-per-process behaviour:
      - reseed torch so the RNG starts from the same point every query
      - clear the [LLM] timing log so call numbering restarts at #1
    Set interactive=True to enable the approval gate + test execution + repair loop
    on the generation path (blocks on input(); terminal use only).
    """
    torch.manual_seed(42)
    _log.clear()
    _generation_time_log.clear()
    return answer_query_deterministic(
        query=query,
        run_generated_code=run_generated_code,
        interactive=interactive,
    )


def run_interactive(query: str) -> str:
    """
    Interactive single-query run. On the generation path this pauses to show the
    spec-based tests for approval (approve with 'y', or type suggestions to remake),
    then runs the approved tests against the generated code and repairs on failure.
    EXACT_MATCH / explain / concept queries behave exactly as run_one — no pause.
    """
    return run_one(query, run_generated_code=True, interactive=True)
 
 
def repl() -> None:
    # The prompt is written to stderr, NOT stdout. That way:
    #   python3 file1.py                 -> fully interactive (prompt + answers on screen)
    #   python3 file1.py > answers.txt   -> answers saved to file, prompt still on screen
    # (Don't add `2>&1` here, or the prompt is redirected into the file too.)
    print("Enter a query and press Enter. Type 'quit' / 'exit' (or Ctrl-D) to stop.",
          file=sys.stderr)
    while True:
        try:
            print("\nquery> ", end="", file=sys.stderr, flush=True)
            query = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", file=sys.stderr)
            break
 
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Exiting.", file=sys.stderr)
            break
 
        # Status to stderr so it stays visible even when stdout is redirected.
        preview = query if len(query) <= 70 else query[:70] + "..."
        print(f"[running] {preview}", file=sys.stderr, flush=True)
 
        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("=" * 60 + "\n")
 
        t0 = time.time()
        try:
            response = run_interactive(query)
        except Exception:  # one bad query shouldn't end the whole session
            import traceback
            print("[error] query failed:\n" + traceback.format_exc())           # full detail -> file
            sys.stdout.flush()                                                   # ensure the traceback is on disk now
            print(f"[error] query failed after {time.time() - t0:.1f}s "
                  f"(traceback in output)", file=sys.stderr, flush=True)         # short note -> screen
            continue

        print("\n--- FINAL ANSWER ---")
        print(response)
        elapsed = time.time() - t0
        print(f"\n[query time] {elapsed:.1f}s")
        print(f"TIME TAKEN FOR GENERATION: {sum(_generation_time_log):.1f}s")
        sys.stdout.flush()                                                       # ensure it's on disk now
        print(f"[done] {elapsed:.1f}s", file=sys.stderr, flush=True)            # completion -> screen


if __name__ == "__main__":
    repl()