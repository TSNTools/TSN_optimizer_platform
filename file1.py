# from llama_index.llms.groq import Groq
import llm_config # configures Settings.llm from env (local / tunnel / direct GH)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader,
    PromptTemplate, StorageContext, load_index_from_storage
)
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.evaluation import CorrectnessEvaluator
import os
import json
import torch
import ast
import warnings
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from deepmultilingualpunctuation import PunctuationModel
from lookup_validation import (
    validate_lookup_payload,
    validate_function_list,
    validate_single_function_claim,
)
from llama_index.core.schema import QueryBundle
from difflib import SequenceMatcher
from collections import defaultdict
from feedback_loop import run_feedback_loop_for_subquery
from spec_test_engine import (
    build_contract_from_requirements,
    generate_spec_based_tests,
)
from exact_match_resolver import resolve_exact_match
from utils import extract_code_block
import time
from llama_index.core import Settings
start_time = time.time()


_orig_complete = Settings.llm.complete          # the real BOUND method (reading is fine)
_log = []

def _timed(*args, **kw):
    t0 = time.time()
    r = _orig_complete(*args, **kw)              # exact normal call path
    _log.append(time.time() - t0)
    print(f"[LLM] #{len(_log)} {_log[-1]:5.1f}s  out~{len(getattr(r, 'text', ''))} chars")
    return r

object.__setattr__(Settings.llm, "complete", _timed)   # bypass pydantic's field guard

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────

# API             = "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"
PARAMETERS_FILE = "./parameters.md"


PYTHON_SOURCE_FILES: Dict[str, str] = {
    "usecase_generator": "./usecase_generator.py",
    "tsn_scheduling_problem": "./tsn_scheduling_problem.py",
    "evaluator": "./evaluator.py",
}

FUNCTION_CATALOG_FILE = "./function_catalog.json"

MODULE_FILES: Dict[str, str] = {
    "usecase_generator":      "./usecase_generator_functions.json",
    "tsn_scheduling_problem": "./tsn_scheduling_functions.json",
    "evaluator":              "./evaluator_functions.json",
}
MODULE_PERSIST: Dict[str, str] = {
    "usecase_generator":      "./storage_usecase",
    "tsn_scheduling_problem": "./storage_tsn",
    "evaluator":              "./storage_evaluator",
}

PARAMS_PERSIST = "./storage_params"

# The three source modules — also used as fallback broadcast targets.
CODE_MODULE_LABELS = list(MODULE_FILES.keys())

ALL_MODULE_LABELS = [
    "usecase_generator",
    "tsn_scheduling_problem",
    "evaluator",
    "parameters",
]


MODULE_FILE_LABELS: Dict[str, str] = {
    "usecase_generator":      "usecase_generator_functions.json",
    "tsn_scheduling_problem": "tsn_scheduling_functions.json",
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

CONFIDENCE_THRESHOLD = 0.4

# VALIDATION AND REPAIR FOR CLASSIFIER OUPUT
VALID_MODULES = set(ALL_MODULE_LABELS)

VALID_INTENTS = {"DO", "EXPLAIN_CONCEPT", "EXPLAIN_CODE"}

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

# ── Punctuation restoration ───────────────────────────────────────────────────

_punct_model = None


def get_punctuation_model() -> PunctuationModel:
    global _punct_model

    if _punct_model is None:
        _punct_model = PunctuationModel(
            model="oliverguhr/fullstop-punctuation-multilang-large"
        )

    return _punct_model


def restore_punctuation(text: str) -> str:
    model = get_punctuation_model()
    restored = model.restore_punctuation(text)
    print(f"[Punctuation] '{text}'\n           -> '{restored}'")
    return restored

# Classify query

class QueryType(str, Enum):
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Intent(str, Enum):
    DO = "DO"                            # build / generate / return / run / compute
    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"  # conceptual TSN question, not tied to one function
    EXPLAIN_CODE = "EXPLAIN_CODE"        # explain a specific existing function/method/class

@dataclass
class SubQuery:
    text: str
    module: str
    intent: str = "unknown"
    action: Intent = Intent.DO     # NEW: the DO / EXPLAIN axis


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

class MatchType(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    NO_MATCH = "NO_MATCH"


class RecommendedAction(str, Enum):
    RETURN_EXISTING_FUNCTIONS = "RETURN_EXISTING_FUNCTIONS"
    GENERATE_WITH_HELPERS = "GENERATE_WITH_HELPERS"
    GENERATE_STANDALONE = "GENERATE_STANDALONE"


@dataclass
class CoverageAssessment:
    match_type: MatchType
    reason: str
    missing_requirements: List[str]
    recommended_action: RecommendedAction

CLASSIFICATION_PROMPT = """\
You are a query planner for a TSN (Time-Sensitive Networking) coding assistant.

You have access to these knowledge modules:

1. usecase_generator
   - topology generation
   - use-case creation
   - stream generation
   - routing/path generation
   - graph construction

2. tsn_scheduling_problem
   - TSN (Time-Sensitive Networking) scheduling logic
   - Network topology management and graph processing
   - Stream path pruning and custom subgraph generation
   - Delay and latency calculations (transmission, propagation, and processing)
   - BFS-based delay budget analysis (forward and backward)
   - Use case instantiation and JSON metadata handling
   - Network visualization (Fat-tree layouts and stream paths)
   - Port and node mapping for egress/ingress tracking
   - Macro-cycle and interval synchronization management

3. evaluator
   - Constraint evaluation: Assesses violations across three primary categories: task placement, network routing, and temporal scheduling.
   - Encoding support: Supports both Priority-based and Start-Time-based schedule encoding types for validation.
   - Flow and connectivity checks: Detects loops, backwarding, and disconnected paths using flow conservation logic.
   - Temporal overlap detection: Implements periodic collision tests to ensure streams do not overlap within the macro-cycle.
   - Latency validation: Compares cumulative path delays against per-stream maximum latency requirements.
   - Performance metrics: Computes network congestion scores, load imbalance, and average path lengths to evaluate schedule quality.
   - Correction logic: Calculates "pressure" or bias values to suggest how to adjust start times or priorities to resolve conflicts.
   - Evaluation caching: Stores precomputed violation results and metrics to accelerate the optimization process.

4. parameters
   - parameter names
   - meanings of parameters

Your task:
Analyze the user query and produce a reliable execution plan.

Classification rules:
- SIMPLE means the query requires only ONE module.
- COMPLEX means the query requires TWO OR MORE modules.
- Split COMPLEX queries into the minimum number of sub-queries.
- Each (module, action) pair may appear at most once in sub_queries. If multiple
aspects of the query map to the same module AND the same action, merge them into a
single sub-query. Different actions on the same module stay as separate sub-queries.
- Each sub-query must map to exactly one module.
- Do not split just because there are multiple sentences instead split the query when user asks for
 actions from different modules.
- If the user asks what a parameter means, route to parameters.
- If the user asks to find/generate/use a topology or use case, route to usecase_generator.
- If the user asks to schedule/optimize/solve, route to tsn_scheduling_problem.
- If the user asks to evaluate/validate/measure/check results, route to evaluator.
- If the user asks about units, data types, output formats, field meanings, or properties
  of a specific module's outputs (e.g. "what units does UsecaseGenerator use for time and
  bandwidth", "what does generateStream return"), assign action=EXPLAIN_CONCEPT and route
  to the module that owns those outputs (usecase_generator, tsn_scheduling_problem, or
  evaluator). Only route to the parameters module when the user is asking about the
  meaning or name of a configuration parameter (e.g. "what does linkBw mean").

- For EACH sub-query also assign an "action", exactly one of:
    * DO              — user wants something built / generated / returned / run / computed
                        for that module (a function, code, a topology, a schedule, a result).
    * EXPLAIN_CONCEPT — a conceptual TSN question NOT tied to one specific function,
                        including questions about units, output formats, data types, or
                        properties of a module's output ("what is a fat-tree", "why does
                        TSN need gate control lists", "what units does UsecaseGenerator
                        use for time and bandwidth").
    * EXPLAIN_CODE    — explain or walk through a specific existing function / method /
                        class ("what does compute_metrics do", "explain k_ary_fat_tree").
  When torn between EXPLAIN_CODE and EXPLAIN_CONCEPT, prefer EXPLAIN_CODE.
- action is independent of module routing and of SIMPLE/COMPLEX. The SAME module may
  appear in two sub-queries when the user wants different actions on it (e.g. generate
  code in a module AND explain a function from that module).

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
      "intent": "<short intent label such as topology_generation, scheduling, evaluation, parameter_lookup>"
    }}
  ]
}}

User query:
{query}
"""

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
def normalize_query(query: str) -> str:
    """
    Normalizes a user query once.
    Currently restores punctuation. Later you can add whitespace cleanup,
    typo correction, etc.
    """
    query = " ".join(query.strip().split())

    if not query:
        return query

    return restore_punctuation(query)

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

def merge_results_by_module(results):
    merged = defaultdict(lambda: {"functions": [], "missing_requirements": []})
    for r in results:
        module = r["module"]
        merged[module]["functions"].extend(r.get("functions", []))
        merged[module]["missing_requirements"].extend(
            r.get("coverage", {}).get("missing_requirements", [])
        )
    # Deduplicate missing requirements
    for module in merged:
        merged[module]["missing_requirements"] = list(
            set(merged[module]["missing_requirements"])
        )
    return merged

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

# Module validation helpers
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
            action = "EXPLAIN_CONCEPT"
            warnings_list.append(
                "Rewrote parameters/DO -> EXPLAIN_CONCEPT (docs module has no callables)."
            )
        sub_queries.append(
            SubQuery(text=text, module=module, intent=intent, action=Intent(action))
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
    grouped: Dict[tuple, SubQuery] = {}
    for sq in sub_queries:
        key = (sq.module, sq.action)
        if key not in grouped:
            grouped[key] = sq
        else:
            prev = grouped[key]
            grouped[key] = SubQuery(
                text=f"{prev.text} {sq.text}".strip(),
                module=sq.module,
                intent=",".join(sorted({prev.intent, sq.intent})),
                action=sq.action,
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
    normalized_query = normalized_query or normalize_query(query)

    prompt = CLASSIFICATION_PROMPT.format(query=normalized_query)
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
module_retrievers: Dict[str, QueryFusionRetriever] = {}
for m in CODE_MODULE_LABELS:
    sem = VectorIndexRetriever(
        index=module_indexes[m],
        similarity_top_k=20,
        vector_store_query_mode=VectorStoreQueryMode.MMR,
        mmr_threshold=0.6,
    )
    bm25 = BM25Retriever.from_defaults(
        nodes=module_nodes[m],
        similarity_top_k=20,
    )
    module_retrievers[m] = QueryFusionRetriever(
        retrievers=[sem, bm25],
        similarity_top_k=20,
        mode="reciprocal_rerank",
        num_queries=1,
        use_async=False,
    )

# ── Per-module generation hybrid retrievers (pure similarity + BM25, no MMR) ─
module_gen_retrievers: Dict[str, QueryFusionRetriever] = {}
for m in CODE_MODULE_LABELS:
    sem = VectorIndexRetriever(
        index=module_indexes[m],
        similarity_top_k=10,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
    )
    bm25 = BM25Retriever.from_defaults(
        nodes=module_nodes[m],
        similarity_top_k=10,
    )
    module_gen_retrievers[m] = QueryFusionRetriever(
        retrievers=[sem, bm25],
        similarity_top_k=10,
        mode="reciprocal_rerank",
        num_queries=1,
        use_async=False,
    )

reranker_params = SentenceTransformerRerank(
    model="BAAI/bge-reranker-large", top_n=5
)
reranker_algo = SentenceTransformerRerank(
    model="BAAI/bge-reranker-large", top_n=8
)


# ── Prompt templates ─────────────────────────────────────────────────────────

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

LOOKUP_PROMPT = PromptTemplate("""
You are a precise function-lookup assistant for a TSN codebase.

Your ONLY job is to find and select functions matching the request in the context and return them
as a JSON array.

Output rules:
- Return ONLY a valid JSON array, NO PREAMBLE, NO MARKDOWN FENCES, NO EXTRA TEXT.

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
- If no function exists in the context, return exactly: []
                               
IMPORTANT NOTES:
- It can be the case that the user query cannot be answered using one function.
  In that case you MUST return multiple functions. 
- If some functions that partially answer the user query are found then include them,
  these functions will be "helper" functions which you can use to answer a query partially.

Context:
{context_str}

Request:
{query_str}

JSON array:
""")

COVERAGE_ASSESSMENT_PROMPT = """\
You are a strict TSN codebase coverage assessor.

Your job is to decide whether the retrieved function candidates fully satisfy the user's request.

You are given:
1. The user's original request.
2. One module-specific sub-query.
3. The module that was searched.
4. Retrieved function candidates from the codebase.

Decide one of:

EXACT_MATCH:
- Use this only if the retrieved function or combination of retrieved functions directly and completely satisfies the sub-query.
- The user could call the returned function(s) with appropriate parameters and achieve the requested functionality.
- No additional new function or glue code is needed.

PARTIAL_MATCH:
- Use this if retrieved functions are relevant helpers but do not fully satisfy the sub-query.
- Some new wrapper, orchestration, glue code, or missing logic is needed.
- The helper functions should be returned and used for generation.

NO_MATCH:
- Use this if no retrieved function is useful for the sub-query.
- Use this if candidates are unrelated, vague, or only match generic words.

Important:
- Retrieval confidence alone is not enough for EXACT_MATCH.
- A high reranker score can still be PARTIAL_MATCH.
- If functions are empty, the answer must be NO_MATCH.
- If the request asks for a pipeline involving multiple actions and the retrieved functions only cover one action, use PARTIAL_MATCH.
- Candidates include a "match_rank". A "BEST_MATCH" whose described behavior matches
  the request is normally EXACT_MATCH — EVEN IF its parameter names or calling
  convention differ from how the user phrased it. Judge by what the function DOES,
  not whether its inputs are named like the user's wording.
- Choose PARTIAL_MATCH only when genuine additional logic or orchestration is needed
  beyond calling the existing function(s). When a BEST_MATCH plausibly does what is
  asked, prefer EXACT_MATCH.

Return ONLY valid JSON, no markdown fences.

Required JSON schema:
{{
  "match_type": "EXACT_MATCH" | "PARTIAL_MATCH" | "NO_MATCH",
  "recommended_action": "RETURN_EXISTING_FUNCTIONS" | "GENERATE_WITH_HELPERS" | "GENERATE_STANDALONE"
}}

User request:
{original_query}

Sub-query:
{sub_query}

Searched module:
{module}

Retrieved function candidates:
{functions_json}

JSON output:
"""

GENERATOR_PROMPT = PromptTemplate("""
You are a TSN algorithm code generator. You MUST use helper functions from the provided context.

Rules:
- List every helper function you will use (exact name, parameters, source citation).
- Only after listing helpers, write the new function.
- The new function body may ONLY call functions that appear in the context or that you define inline.
- Add a docstring that explains inputs, outputs, and which helpers are used.
- If a required helper is missing from the context, say NOT_IN_CONTEXT: <what is missing> before generating.

Output rules:
- Return ONLY a valid JSON array, no preamble, no markdown fences, no extra text.

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
"code"         : "complete function code verbatim from context",
"unit_tests"   : ["Tests functions covering normal cases, edge cases, and errors (inputs, outputs, exceptions)."]

Context:
{context_str}

Request:
{query_str}

Generated function:
""")

UNIT_TEST_GENERATION_PROMPT = """\
You are an expert Python test engineer.

Given a user request, helper functions, and candidate code, write pytest-style unit tests.

Rules:
- Return ONLY valid JSON. No markdown fences. No preamble.
- The JSON must have a single key "unit_tests" whose value is a list of strings.
- Each string is a complete, self-contained Python test function (def test_...).
- Each test function must import what it needs, define or inline any stubs/mocks for
  external dependencies (networkx, numpy, etc.), and contain at least one assertion.
- Cover: normal cases, edge cases (empty input, zero, None), and expected exceptions.
- Use extracted parameter values as concrete test inputs wherever available.
- Do NOT rely on external files or network access.
- If the code under test depends on a class or helper, define a minimal stub inline.

Schema:
{{
  "unit_tests": [
    "def test_normal_case():\\n    # inline imports and stubs\\n    assert ...",
    "def test_edge_case_empty():\\n    assert ...",
    "def test_raises_on_invalid():\\n    import pytest\\n    with pytest.raises(ValueError):\\n        ..."
  ]
}}

User request:
{query}

Extracted parameters:
{params_json}

Helper functions available:
{helpers_json}

Code under test:
{code}

JSON output:
"""

def generate_unit_tests(
    query: str,
    code: str,
    helper_functions: list[dict],
    extracted_params: Optional[dict] = None,
) -> list[str]:
    """
    Asks the LLM to write pytest-style unit tests for the given code.
    Returns a list of test function source strings.
    """
    prompt = UNIT_TEST_GENERATION_PROMPT.format(
        query=query,
        code=extract_code_block(code),
        params_json=json.dumps(extracted_params or {}, indent=2),
        helpers_json=json.dumps(
            [{"name": fn.get("name"), "signature": fn.get("signature"),
              "code": str(fn.get("code") or "")[:2000]}
             for fn in helper_functions],
            indent=2,
        )[:8000],
    )

    raw = Settings.llm.complete(prompt).text.strip()

    try:
        parsed = extract_json_object(raw)
        tests = parsed.get("unit_tests", [])
        return [t for t in tests if isinstance(t, str)]
    except json.JSONDecodeError:
        print(f"[WARN] Unit test generation returned invalid JSON. Raw:\n{raw[:1000]}")
        return []

# ── Parameter extraction prompt ───────────────────────────────────────────────
# Used exclusively by extract_parameters(). Separate from PARAMS_PROMPT, which
# is used by params_engine to *explain* parameters. This prompt does the opposite:
# it reads a natural language query and maps any mentioned values onto the known
# parameter names from parameters.md. The result is always a JSON object.

PARAM_EXTRACTION_PROMPT = """\
You are a parameter extraction assistant for a TSN (Time-Sensitive Networking) system.

You are given:
1. A user query in natural language.
2. A list of known parameter definitions from the TSN codebase (name + description).

Your job:
- Read the query carefully.
- Identify every parameter from the known list that is explicitly or implicitly referenced.
- For each matched parameter, extract the concrete value the user specified.
- If no value is given for a matched parameter, use null.
- If the query references a concept that matches a parameter but uses different wording
  (e.g. "switches" → num_switches, "8 hosts" → num_hosts), still match it.
- Only include parameters that appear in the known list. Never invent parameter names.
- Output ONLY a valid JSON object, no preamble, no markdown fences.

Known parameters:
{param_definitions}

Query:
{query}

JSON output:
"""

PARAM_EXPLANATION_PROMPT = """\
You are a TSN parameter explanation assistant.

Explain only the parameters listed below using the provided context.

Return ONLY JSON.

Schema:
{{
  "<parameter_name>": {{
    "value": "<given value or null>",
    "meaning": "<brief explanation>",
    "used_for": "<brief usage explanation>"
  }}
}}

Context:
{context_str}

Parameters:
{params_json}
"""

SPEC_EXTRACTION_PROMPT = """\
You are a software specification analyst. Produce a behavioral contract
for the function described below. You are NOT given the implementation,
and you must NOT guess at implementation details.

Derive the contract ONLY from the description, signature, preconditions,
postconditions, and declared exceptions.

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


SPEC_BASED_TEST_PROMPT = """\
You are a test engineer writing pytest tests from a BEHAVIORAL CONTRACT.

CRITICAL RULES:
- You are given a contract, NOT an implementation.
- Write one test per behavior in the contract.
- Each test name must reference the behavior id (e.g. test_B1_...).
- Assert ONLY what the contract states. Do not invent behavior.
- Do NOT weaken assertions to make tests pass. The implementation does
  not exist yet from your perspective.
- For error_case behaviors, use pytest.raises with the declared exception.
- Include a docstring in each test quoting the behavior it verifies.
- Define minimal inline stubs for external deps (networkx, numpy) only if
  strictly required to call the function. Never stub the function under test.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
  "tests": [
    {{
      "behavior_id": "B1",
      "test_name": "test_B1_returns_sorted_paths",
      "code": "def test_B1_returns_sorted_paths():\\n    \\"\\"\\"B1: ...\\"\\"\\"\\n    ..."
    }}
  ]
}}

Function to import: {import_hint}

Behavioral contract:
{contract_json}

JSON output:
"""

TRIAGE_PROMPT = """\
You are a test-failure triage analyst.

A test derived from a behavioral contract failed against an implementation.
Decide the ROOT CAUSE. Be skeptical: the default assumption is that the
IMPLEMENTATION is wrong, because the test was written from the contract
before seeing the implementation.

Only declare TEST_INCORRECT if you can prove the test contradicts the
contract or makes an assertion the contract does not support.

Return ONLY valid JSON. No markdown fences.

Schema:
{{
  "verdict": "IMPL_INCORRECT" | "TEST_INCORRECT" | "SPEC_AMBIGUOUS" | "FLAKY_OR_ENV",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "explanation": "...",
  "evidence_from_contract": "quote the relevant behavior",
  "suggested_fix_target": "implementation" | "test" | "spec" | "none"
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
        node_postprocessors=[reranker_algo]
    )
    le.update_prompts({"response_synthesizer:text_qa_template": LOOKUP_PROMPT})
    lookup_engines[m] = le

    ge = RetrieverQueryEngine(
        retriever=module_gen_retrievers[m],
        node_postprocessors=[reranker_algo]
    )
    ge.update_prompts({"response_synthesizer:text_qa_template": GENERATOR_PROMPT})
    gen_engines[m] = ge


# ── Agent tool functions ──────────────────────────────────────────────────────

def _format_sources(source_nodes) -> str:
    return "\n".join(
        f"[Source: {n.node.metadata.get('file_name', '?')} | "
        f"Section: {n.node.metadata.get('header_path', n.node.metadata.get('section', '?'))} | "
        f"Score: {f'{n.score:.3f}' if n.score is not None else 'N/A'}]"
        for n in source_nodes
    )

def _top_score(source_nodes) -> float:
    return max((n.score or 0.0) for n in source_nodes) if source_nodes else 0.0


# Cache for parameter definitions text — built once on first extraction call.
# Avoids re-querying params_engine on every user query.
_param_definitions_cache: Dict[str, str] = {}

def _get_param_definitions() -> str:
    """
    Returns the full parameter definitions text from parameters.md.
    Fetched once via params_engine and cached in memory for the session.
    """
    if "definitions" not in _param_definitions_cache:
        response = params_engine.query(
            "List every parameter name and its description."
        )
        result = str(response).strip()
        if not result or "Empty Response" in result:
            raise RuntimeError(
                "[Startup] Failed to load parameter definitions from index. "
                "Check that parameters.md is indexed and storage_params exists."
            )
        _param_definitions_cache["definitions"] = result
        return result
    return _param_definitions_cache["definitions"]

def explain_extracted_parameters(params: dict) -> dict:
    if not params:
        return {}

    names = ", ".join(params.keys())
    response = params_engine.query(f"Explain these parameters: {names}")

    prompt = PARAM_EXPLANATION_PROMPT.format(
        context_str=str(response),
        params_json=json.dumps(params, indent=2),
    )

    raw = Settings.llm.complete(prompt).text.strip()

    try:
        parsed = extract_json_object(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {
            name: {
                "value": value,
                "meaning": "Parameter explanation could not be parsed.",
                "used_for": "",
            }
            for name, value in params.items()
        }

def explain_function_parameters(functions: list[dict]) -> tuple[dict, dict]:
    """
    Build ({param: value}, {param: {value, meaning, used_for}}) for the SIGNATURE
    parameters of the returned function(s), explained from each function's own
    docstring / signature / code. Same shape and Meaning/Used-for format as
    explain_extracted_parameters, so format_subquery_answer renders it unchanged.
    """
    func_params: dict = {}
    context_parts: list[str] = []

    for fn in functions:
        context_parts.append(
            f"Function: {fn.get('name')}\n"
            f"Signature: {fn.get('signature') or ''}\n"
            f"Description: {fn.get('description') or ''}\n"
            f"Code:\n{str(fn.get('code') or '')[:2000]}"
        )
        for p in (fn.get("parameters") or []):
            name = p.get("name")
            if not name or name == "self":          # skip the bound receiver
                continue
            default = p.get("default")
            func_params[name] = None if default in (None, "None", "null") else default

    if not func_params:
        return {}, {}

    prompt = PARAM_EXPLANATION_PROMPT.format(
        context_str="\n\n".join(context_parts),
        params_json=json.dumps(func_params, indent=2),
    )
    raw = Settings.llm.complete(prompt).text.strip()
    try:
        explanations = extract_json_object(raw)
        if not isinstance(explanations, dict):
            explanations = {}
    except json.JSONDecodeError:
        explanations = {
            name: {"value": val, "meaning": "Parameter explanation could not be parsed.", "used_for": ""}
            for name, val in func_params.items()
        }
    return func_params, explanations

# Simple dict cache for ClassifiedQuery objects — avoids repeated LLM classification.
_classification_cache: Dict[str, ClassifiedQuery] = {}
def get_or_classify_query(query: str) -> ClassifiedQuery:
    """
    Normalizes and classifies a query, reusing the cached classification
    when the same normalized query was already processed.
    """
    normalized_query = normalize_query(query)

    if normalized_query in _classification_cache:
        return _classification_cache[normalized_query]

    classification = classify_query(query, normalized_query=normalized_query)
    _classification_cache[classification.normalized_query] = classification
    return classification

def debug_classification(query: str) -> None:
    classification = get_or_classify_query(query)

    print("\n--- QUERY CLASSIFICATION DEBUG ---")
    print(f"Original:   {classification.original_query}")
    print(f"Normalized: {classification.normalized_query}")
    print(f"Type:       {classification.query_type.value}")
    print(f"Confidence: {classification.classification_confidence.value}")
    print(f"Reason:     {classification.reason}")

    print("\nSub-queries:")
    for i, sq in enumerate(classification.sub_queries, 1):
        print(f"  {i}. Module: {sq.module}")
        print(f"     Intent:  {sq.intent}")
        print(f"     Text:    {sq.text}")

    if classification.warnings:
        print("\nWarnings:")
        for warning in classification.warnings:
            print(f"  - {warning}")

    print("----------------------------------\n")

def extract_parameters(query: str) -> str:
    """
    Classifies the query (SIMPLE or COMPLEX), then for each sub-query extracts
    any parameter names and values explicitly or implicitly mentioned by the user,
    matched against the known parameters from parameters.md.

    Returns a JSON string per sub-query. For SIMPLE queries a single JSON object
    is returned. For COMPLEX queries one JSON object per sub-query is returned,
    labelled by module.

    This tool should be called FIRST — before lookup_and_understand — so that
    the extracted parameter context can sharpen the subsequent function lookup.

    Example:
        Input : "Test TSN with 8 switches and 8 hosts"
        Output: {"num_switches": 8, "num_hosts": 8}
    """
    classification = get_or_classify_query(query)

    param_defs = _get_param_definitions()

    def _extract_one(text: str, module: str) -> dict:
        augmented_text = augment_query(text, module)
        prompt = PARAM_EXTRACTION_PROMPT.format(
            param_definitions=param_defs,
            query=augmented_text,
        )
        raw = Settings.llm.complete(prompt).text.strip()

        try:
            parsed = extract_json_object(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError as e:
            print(f"[WARN] Parameter extraction parse failed ({e}). Returning empty.")
            return {}

    if classification.query_type == QueryType.SIMPLE:
        sq     = classification.sub_queries[0]
        result = _extract_one(sq.text, sq.module)
        print(f"[Params] SIMPLE [{sq.module}] → {result}")
        return json.dumps(result, indent=2)

    # COMPLEX: extract per sub-query and combine under module keys
    combined = {}
    for sq in classification.sub_queries:
        result = _extract_one(sq.text, sq.module)
        print(f"[Params] COMPLEX [{sq.module}] → {result}")
        if result:
            combined[sq.module] = result

    return json.dumps(combined, indent=2)

def extract_parameters_for_subquery(sq: SubQuery) -> dict:
    param_defs = _get_param_definitions()

    prompt = PARAM_EXTRACTION_PROMPT.format(
        param_definitions=param_defs,
        query=augment_query(sq.text, sq.module),
    )

    raw = Settings.llm.complete(prompt).text.strip()

    try:
        parsed = extract_json_object(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as e:
        print(f"[WARN] Parameter extraction parse failed ({e}). Returning empty.")
        return {}

def _parse_lookup_json(raw: str) -> list[dict]:
    """
    Parses the JSON array returned by LOOKUP_PROMPT.
    Returns an empty list on any parse failure.
    """
    try:
        raw = raw.replace("\\'", "'")
        result = extract_json_array(raw)
        return [item for item in result if isinstance(item, dict)]
    except json.JSONDecodeError as e:
        print(f"[WARN] Lookup JSON parse failed ({e}). Raw response:\n{raw[:2000]}")
        return []

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

def _validate_coverage_assessment(parsed: dict, functions: List[Dict]) -> CoverageAssessment:
    raw_match_type = parsed.get("match_type", "NO_MATCH")
    raw_action = parsed.get("recommended_action", "GENERATE_STANDALONE")

    if raw_match_type not in {
        "EXACT_MATCH",
        "PARTIAL_MATCH",
        "NO_MATCH",
    }:
        raw_match_type = "NO_MATCH"

    if raw_action not in {
        "RETURN_EXISTING_FUNCTIONS",
        "GENERATE_WITH_HELPERS",
        "GENERATE_STANDALONE",
    }:
        raw_action = "GENERATE_STANDALONE"

    reason = str(parsed.get("reason", "")).strip()
    missing_requirements = parsed.get("missing_requirements", [])

    if not isinstance(missing_requirements, list):
        missing_requirements = [str(missing_requirements)]

    missing_requirements = [str(item) for item in missing_requirements]

    # Hard safety rule:
    # If there are no functions, coverage cannot be EXACT or PARTIAL.
    if not functions:
        raw_match_type = "NO_MATCH"
        raw_action = "GENERATE_STANDALONE"
        if not reason:
            reason = "No retrieved functions were returned."

    # Hard safety rule:
    # If exact match is claimed, there must be at least one function.
    if raw_match_type == "EXACT_MATCH" and not functions:
        raw_match_type = "NO_MATCH"
        raw_action = "GENERATE_STANDALONE"
        reason = "Exact match was claimed, but no functions were returned."

    # Keep match_type and recommended_action consistent.
    if raw_match_type == "EXACT_MATCH":
        raw_action = "RETURN_EXISTING_FUNCTIONS"
    elif raw_match_type == "PARTIAL_MATCH":
        raw_action = "GENERATE_WITH_HELPERS"
    elif raw_match_type == "NO_MATCH":
        raw_action = "GENERATE_STANDALONE"

    if not reason:
        reason = f"Coverage assessed as {raw_match_type}."

    return CoverageAssessment(
        match_type=MatchType(raw_match_type),
        reason=reason,
        missing_requirements=missing_requirements,
        recommended_action=RecommendedAction(raw_action),
    )

def assess_function_coverage(
    original_query: str,
    sq: SubQuery,
    functions: List[Dict],
    reranker_score: float,
) -> CoverageAssessment:
    """
    Decides whether retrieved functions fully satisfy the sub-query,
    partially satisfy it as helpers, or do not match at all.
    """
    confidence = (
        "HIGH_CONFIDENCE"
        if reranker_score >= CONFIDENCE_THRESHOLD
        else "LOW_CONFIDENCE"
    )

    real_callables = [f for f in functions if f.get("code")]
    if sq.module == "parameters" or not real_callables:
        return CoverageAssessment(
            match_type=MatchType.NO_MATCH,
            reason="No callable with source code was retrieved; "
                   "routing to explanation/generation instead of exact match.",
            missing_requirements=["No existing callable function candidate was found."],
            recommended_action=RecommendedAction.GENERATE_STANDALONE,
        )

    # Deterministic no-match rule.
    if not functions:
        return CoverageAssessment(
            match_type=MatchType.NO_MATCH,
            reason="No functions were returned by lookup.",
            missing_requirements=["No existing function candidate was found."],
            recommended_action=RecommendedAction.GENERATE_STANDALONE,
        )

    RETURN_EXISTING_THRESHOLD = 0.5   # calibrate from logged scores; >= CONFIDENCE_THRESHOLD

    best = next((f for f in functions if f.get("match_rank") == "BEST_MATCH"), None)
    if best is not None and reranker_score >= RETURN_EXISTING_THRESHOLD:
        return CoverageAssessment(
            match_type=MatchType.EXACT_MATCH,
            reason=f"BEST_MATCH '{best.get('name')}' retrieved with high confidence; returning existing function.",
            missing_requirements=[],
            recommended_action=RecommendedAction.RETURN_EXISTING_FUNCTIONS,
        )

    # Deterministic low-confidence downgrade:
    # Low-confidence retrieval should not be considered exact.
    if confidence == "LOW_CONFIDENCE":
        return CoverageAssessment(
            match_type=MatchType.PARTIAL_MATCH,
            reason=(
                "Functions were retrieved, but reranker confidence is low. "
                "Treating them as possible helpers rather than exact matches."
            ),
            missing_requirements=[
                "Need generated code or further verification because retrieval confidence is low."
            ],
            recommended_action=RecommendedAction.GENERATE_WITH_HELPERS,
        )

    functions_for_prompt = json.dumps(
        [
            {
                "name": fn.get("name"),
                "purpose": fn.get("purpose"),
                "signature": fn.get("signature"),
                "parameters": fn.get("parameters"),
                "preconditions": fn.get("preconditions"),
                "postconditions": fn.get("postconditions"),
                "modules": fn.get("modules"),
                "match_rank": fn.get("match_rank"),
                "code": str(fn.get("code") or "")[:3000],
            }
            for fn in functions
        ],
        indent=2,
    )

    prompt = COVERAGE_ASSESSMENT_PROMPT.format(
        original_query=original_query,
        sub_query=sq.text,
        module=sq.module,
        functions_json=functions_for_prompt,
    )

    raw = Settings.llm.complete(prompt).text.strip()

    try:
        parsed = extract_json_object(raw)
        return _validate_coverage_assessment(parsed, functions)
    except json.JSONDecodeError as e:
        print(f"[WARN] Coverage assessment parse failed ({e}). Using fallback.")

        return CoverageAssessment(
            match_type=MatchType.PARTIAL_MATCH,
            reason=(
                "Coverage assessment failed to parse, but functions were retrieved. "
                "Treating retrieved functions as helpers."
            ),
            missing_requirements=[
                "Unable to verify that retrieved functions fully satisfy the request."
            ],
            recommended_action=RecommendedAction.GENERATE_WITH_HELPERS,
        )

def coverage_to_dict(coverage: CoverageAssessment) -> Dict:
    return {
        "match_type": coverage.match_type.value,
        "reason": coverage.reason,
        "missing_requirements": coverage.missing_requirements,
        "recommended_action": coverage.recommended_action.value,
    }

def lookup_result_needs_generation(lookup_data: Dict) -> bool:
    """
    Returns True if any part of lookup result requires generation.
    """
    if lookup_data.get("query_type") == "SIMPLE":
        coverage = lookup_data.get("coverage", {})
        return coverage.get("recommended_action") in {
            "GENERATE_WITH_HELPERS",
            "GENERATE_STANDALONE",
        }

    for sq in lookup_data.get("sub_queries", []):
        coverage = sq.get("coverage", {})
        if coverage.get("recommended_action") in {
            "GENERATE_WITH_HELPERS",
            "GENERATE_STANDALONE",
        }:
            return True

    return False

def _lookup_one(sq: SubQuery, original_query: str = "") -> tuple[list[dict], float]:
    """
    Queries the correct per-module engine, parses the JSON array returned by
    LOOKUP_PROMPT, validates all returned functions against the hard-coded
    catalog, and returns only valid functions.
    """
    augmented_text = augment_query(sq.text, sq.module)

    if sq.module == "parameters":
        response = params_engine.query(augmented_text)
        score = _top_score(response.source_nodes)
        return [
            {
                "name": "name of the parameter",
                "class_name": None,
                "purpose": str(response),
                "library calls used": [],
                "preconditions": [],
                "postconditions": [],
                "signature": None,
                "docstring": None,
                "parameters": [],
                "modules": "parameters.md",
                "code": None,
                "match_rank": "BEST_MATCH",
                "match_confidence": (
                    "HIGH_CONFIDENCE"
                    if score >= CONFIDENCE_THRESHOLD
                    else "LOW_CONFIDENCE"
                ),
                "reranker_score": round(score, 3),
                "routed_module": "parameters",
                "catalog_validation": {
                    "valid": True,
                    "reason": "Parameter pseudo-result bypasses function catalog validation.",
                    "matched_entry": None,
                    "alternatives": [],
                },
            }
        ], score

    if sq.module not in lookup_engines:
        print(f"[WARN] No lookup engine for module: {sq.module}")
        return [], 0.0

    # Retrieve
    nodes = module_retrievers[sq.module].retrieve(augmented_text)

    # Rerank
    nodes = reranker_algo.postprocess_nodes(
        nodes, query_bundle=QueryBundle(augmented_text)
    )

    score = max((n.score or 0.0) for n in nodes) if nodes else 0.0
    confidence = (
        "HIGH_CONFIDENCE"
        if score >= CONFIDENCE_THRESHOLD
        else "LOW_CONFIDENCE"
    )

    # Build context string and call LLM directly — no synthesizer
    context_str = "\n\n".join(n.node.get_content() for n in nodes)
    
    query_str = (
        f"Original user query (full context):\n{original_query}\n\n"
        f"Module-specific request:\n{augmented_text}"
    ) if original_query else augmented_text
    
    prompt = LOOKUP_PROMPT.format(
        context_str=context_str,
        query_str=query_str,
    )
    raw = Settings.llm.complete(prompt).text.strip()

    # print(f"\n[DEBUG] Raw LLM output from LOOKUP_PROMPT:\n{raw[:2000]}")

    functions = _parse_lookup_json(raw)  # now raw is actual JSON, not prose

    # Fetch real code from source files — replaces the stripped "code_ref" field
    for fn in functions:
        fn_name = fn.get("name") or fn.get("code_ref", "")
        fn["code"] = fetch_code_from_source(fn_name, sq.module)

    for fn in functions:
        fn["match_confidence"] = confidence
        fn["reranker_score"] = round(score, 3)
        fn["routed_module"] = sq.module

    valid_functions, invalid_functions = validate_function_list(
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

    return valid_functions, score


def lookup_and_understand(query: str) -> str:
    """
    Restores punctuation, classifies the query, routes each sub-query to its
    dedicated per-module engine, and returns a JSON string.

    For SIMPLE queries: returns a JSON object with keys "query_type", "module",
    "match_confidence", "reranker_score", and "functions" (list of function dicts).

    For COMPLEX queries: returns a JSON object with keys "query_type" and
    "sub_queries" — a list where each entry has "module", "match_confidence",
    "reranker_score", and "functions".

    Each function dict has the full schema:
      name, class_name, purpose, "library calls used", preconditions, postconditions,
      signature, docstring, parameters, modules, code, match_rank,
      match_confidence, reranker_score, routed_module.

    An empty "functions" list means NO_FUNCTION_FOUND for that sub-query.
    """
    classification = get_or_classify_query(query)

    if classification.query_type == QueryType.SIMPLE:
        sq = classification.sub_queries[0]
        print(f"[Intent] SIMPLE → module: {sq.module}")

        functions, score = _lookup_one(sq, original_query=classification.original_query)
        for fn in functions:
            print(f"[DEBUG] Function: {fn.get('name')}")
            print(f"[DEBUG] Code field:\n{fn.get('code')}")
        confidence = (
            "HIGH_CONFIDENCE"
            if score >= CONFIDENCE_THRESHOLD
            else "LOW_CONFIDENCE"
        )

        coverage = assess_function_coverage(
            original_query=classification.original_query,
            sq=sq,
            functions=functions,
            reranker_score=score,
        )

        result = {
            "query_type": classification.query_type.value,
            "classification_confidence": classification.classification_confidence.value,
            "classification_reason": classification.reason,
            "normalized_query": classification.normalized_query,
            "module": sq.module,
            "intent": sq.intent,
            "match_confidence": confidence,
            "reranker_score": round(score, 3),
            "coverage": coverage_to_dict(coverage),
            "functions": functions,
        }

        return json.dumps(result, indent=2)

    print(f"[Intent] COMPLEX → {len(classification.sub_queries)} sub-queries")
    sub_results = []
    for i, sq in enumerate(classification.sub_queries, 1):
        print(f"  Sub-query {i}: [{sq.module}] {sq.text}")

        functions, score = _lookup_one(sq, original_query=classification.original_query)
        confidence = (
            "HIGH_CONFIDENCE"
            if score >= CONFIDENCE_THRESHOLD
            else "LOW_CONFIDENCE"
        )

        coverage = assess_function_coverage(
            original_query=classification.original_query,
            sq=sq,
            functions=functions,
            reranker_score=score,
        )

        sub_results.append({
            "module": sq.module,
            "intent": sq.intent,
            "sub_query_text": sq.text,
            "match_confidence": confidence,
            "reranker_score": round(score, 3),
            "coverage": coverage_to_dict(coverage),
            "functions": functions,
        })

    result = {
        "query_type": classification.query_type.value,
        "classification_confidence": classification.classification_confidence.value,
        "classification_reason": classification.reason,
        "normalized_query": classification.normalized_query,
        "sub_queries": sub_results,
    }
    return json.dumps(result, indent=2)

def _format_helper_signature(fn: dict) -> str:
    params = fn.get("parameters", [])
    param_names = []

    if isinstance(params, list):
        for param in params:
            if isinstance(param, dict) and "name" in param:
                param_names.append(str(param["name"]))

    return (
        f"- {fn.get('name')}({', '.join(param_names)}): "
        f"{fn.get('purpose', '')} [source: {fn.get('modules', '?')}]"
    )

def _generate_for_subquery(sq: SubQuery) -> str:
    """Generates new code for a single sub-query using the module-specific gen engine."""
    if sq.module == "parameters":
        response = params_engine.query(augment_query(sq.text, sq.module))
        return (
            "[Module: parameters]\n"
            "This sub-query is about parameter explanation, not code generation.\n\n"
            f"{response}"
        )

    if sq.module not in gen_engines:
        return (
            f"[Module: {sq.module}]\n"
            "NO_GENERATION_ENGINE_AVAILABLE for this module."
        )

    augmented_text = augment_query(sq.text, sq.module)

    param_context_str = str(params_engine.query(
        f"What parameters are relevant to: {augmented_text}"
    )).strip()
    if not param_context_str or "Empty Response" in param_context_str:
        param_context_str = "No relevant parameter definitions found."

    helper_functions, _ = _lookup_one(sq)
    helper_summary = (
        "\n".join(_format_helper_signature(fn) for fn in helper_functions)
        if helper_functions
        else "No helpers found in context."
    )

    # Retrieve and rerank nodes directly — bypass synthesizer
    nodes = module_gen_retrievers[sq.module].retrieve(augmented_text)
    nodes = reranker_algo.postprocess_nodes(
        nodes, query_bundle=QueryBundle(augmented_text)
    )
    sources = _format_sources(nodes)

    context_str = "\n\n".join(n.node.get_content() for n in nodes)
    augmented_request = (
        f"{augmented_text}\n\n"
        f"Relevant parameter definitions:\n{param_context_str}\n\n"
        f"Available helper functions:\n{helper_summary}"
    )
    prompt = GENERATOR_PROMPT.format(
        context_str=context_str,
        query_str=augmented_request,
    )
    raw = Settings.llm.complete(prompt).text.strip()

    return f"[Module: {sq.module}]\n{raw}\n\n--- Context sources ---\n{sources}"

def generate_code_for_subquery(
    query: str,
    sq: SubQuery,
    helper_functions: list[dict],
    extracted_params: Optional[dict] = None,
) -> tuple[str, str]:
    if sq.module == "parameters":
        response = params_engine.query(augment_query(sq.text, sq.module))
        return str(response), ""

    if sq.module not in gen_engines:
        return f"NO_GENERATION_ENGINE_AVAILABLE for module: {sq.module}", ""

    augmented_text = augment_query(sq.text, sq.module)

    param_context_str = str(params_engine.query(
    f"What parameters are relevant to: {augmented_text}"
    )).strip()
    if not param_context_str or "Empty Response" in param_context_str:
        param_context_str = "No relevant parameter definitions found."

    helper_summary = (
        "\n".join(_format_helper_signature(fn) for fn in helper_functions)
        if helper_functions
        else "No helpers found in context."
    )

    helper_code = "\n\n".join(
        fn.get("code", "")
        for fn in helper_functions
        if fn.get("code")
    )

    # Retrieve and rerank nodes directly — bypass synthesizer
    nodes = module_gen_retrievers[sq.module].retrieve(augmented_text)
    nodes = reranker_algo.postprocess_nodes(
        nodes, query_bundle=QueryBundle(augmented_text)
    )
    context_str = "\n\n".join(n.node.get_content() for n in nodes)

    augmented_request = (
        f"Original user query:\n{query}\n\n"
        f"Module-specific request:\n{augmented_text}\n\n"
        f"Extracted parameters:\n"
        f"{json.dumps(extracted_params or {}, indent=2)}\n\n"
        f"Relevant parameter definitions:\n{param_context_str}\n\n"
        f"Available helper functions:\n{helper_summary}\n\n"
        f"Helper function code:\n{helper_code}\n\n"
        "Generate the missing code needed to fully satisfy the request."
    )

    prompt = GENERATOR_PROMPT.format(
        context_str=context_str,
        query_str=augmented_request,
    )
    raw = Settings.llm.complete(prompt).text.strip()

    # Try to parse the generated JSON and inject real helper codes
    
    generated_functions = _parse_lookup_json(raw)

    if not generated_functions:
        salvaged = extract_code_block(raw)
        if salvaged.strip():
            name_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", salvaged)
            generated_functions = [{
                "name": name_match.group(1) if name_match else None,
                "module": sq.module,
                "code": salvaged,
                "description": "Recovered from non-JSON generator output.",
            }]
        else:
            return raw, context_str  # nothing parseable at all — surface raw

    for fn in generated_functions:
        fn_name = fn.get("name") or fn.get("code_ref", "")
        fetched = fetch_code_from_source(fn_name, sq.module)
        if fetched:
            fn["code"] = fetched

    return json.dumps(generated_functions, indent=2), context_str

def generate_function(query: str) -> str:
    """
    Generates a new TSN function grounded in helper functions from the codebase.
    Call this ONLY when lookup_and_understand returned LOW_CONFIDENCE or NO_FUNCTION_FOUND.

    Restores punctuation first, then reuses the ClassifiedQuery from the cache
    (populated by lookup_and_understand) to avoid a redundant LLM classification call.
    """
    classification = get_or_classify_query(query)
    if classification.query_type == QueryType.SIMPLE:
        return _generate_for_subquery(classification.sub_queries[0])

    parts = []
    for i, sq in enumerate(classification.sub_queries, 1):
        print(f"  Generating for sub-query {i}: [{sq.module}]")
        parts.append(f"=== Part {i} [{sq.module}] ===\n{_generate_for_subquery(sq)}")
    return "\n\n".join(parts)

def format_subquery_answer(
    sq: SubQuery,
    extracted_params: dict,
    param_explanations: dict,
    functions: list[dict],
    coverage: CoverageAssessment,
    generated_code: Optional[str],
    execution_result: Optional[dict],
    unit_tests: list[str] = None,
    resolution = None
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

    if functions:
            lines.append("\nFUNCTIONS FOUND:")

            for fn in functions:
                # Emit all fields except code — code is injected by Python after formatting
                printable = {k: v for k, v in fn.items() if k not in {
                    "syntax_check", "match_confidence", "reranker_score", "routed_module",
                    "catalog_validation", "code",  # <-- exclude code here
                }}
                lines.append(json.dumps(printable, indent=2))
                # Placeholder: Python will replace this with the real, untruncated code
                lines.append(f"    \"code\": <CODE:{fn.get('name')}>")

    else:
        lines.append("\nFUNCTIONS FOUND: none")

    if generated_code:
        lines.append("\nGENERATED CODE:")
        lines.append("```python")
        lines.append(extract_code_block(generated_code))
        lines.append("```")
    
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

EXPLAIN_CONCEPT_PROMPT = """\
You are a TSN (Time-Sensitive Networking) concept explainer for this codebase.

You are given two grounding sources from THIS codebase — parameter definitions and
function docstrings/code — and you may also use your own general TSN knowledge.

Rules:
- Ground the answer in the provided context FIRST. Prefer it wherever it speaks to
  the question, and stay consistent with it.
- You MAY add your own general TSN knowledge to fill gaps or give intuition, but when
  you go beyond the provided context, signal it (e.g. "More broadly in TSN, ...").
- Never contradict the provided context. Be precise and concise.

Parameter definitions (from this codebase):
{param_context}

Function docstrings / code context (from this codebase):
{doc_context}

Question:
{query}

Explanation:
"""

EXPLAIN_CODE_PROMPT = """\
You are a TSN codebase explainer. Explain the function(s) below clearly and accurately.

For each function, cover:
- Purpose: what it does and why it exists.
- Inputs: each parameter and its role.
- Output: the return value and its structure.
- Logic: a short step-by-step walkthrough.
- Notes: side effects, preconditions, key library calls, raised exceptions.

Rules:
- Ground every statement in the provided code and docstring.
- If the code does not show something, say so rather than guessing.
- You may add brief TSN background to clarify intent, but mark it as background.

Function(s):
{functions_block}

User request:
{query}

Explanation:
"""


def _collect_doc_context(sq: SubQuery, k_nodes: int = 6) -> str:
    """Retrieve docstring/code context from the routed code module."""
    if sq.module not in module_retrievers:
        return ""
    augmented = augment_query(sq.text, sq.module)
    nodes = module_retrievers[sq.module].retrieve(augmented)
    nodes = reranker_algo.postprocess_nodes(nodes, query_bundle=QueryBundle(augmented))
    return "\n\n".join(n.node.get_content() for n in nodes[:k_nodes])


def _explain_concept_section(query: str, sq: SubQuery,
                             extracted_params: dict, param_explanations: dict) -> str:
    # 1) Parameter grounding.
    param_ctx = str(params_engine.query(
        f"Explain parameters and concepts relevant to: {augment_query(sq.text, sq.module)}"
    )).strip()
    if not param_ctx or "Empty Response" in param_ctx:
        param_ctx = "No directly relevant parameter definitions found."

    # 2) Docstring grounding (skip for the parameters module).
    doc_ctx = "" if sq.module == "parameters" else _collect_doc_context(sq)
    if not doc_ctx:
        doc_ctx = "No directly relevant function docstrings found."

    # 3) Synthesize (grounded first, LLM knowledge allowed and marked).
    prompt = EXPLAIN_CONCEPT_PROMPT.format(
        param_context=param_ctx[:8000],
        doc_context=doc_ctx[:8000],
        query=sq.text,
    )
    explanation = Settings.llm.complete(prompt).text.strip()
    return "\n".join([
        f"MODULE: {sq.module}", f"SUB-QUERY: {sq.text}",
        "INTENT: EXPLAIN_CONCEPT", "", "EXPLANATION:", explanation,
    ])


def _explain_code_section(query: str, sq: SubQuery, functions: list[dict],
                          extracted_params: dict, param_explanations: dict) -> str:
    if functions:
        blocks = [
            f"Name: {fn.get('name')}\nSignature: {fn.get('signature')}\n"
            f"Module: {fn.get('modules')}\nDocstring: {fn.get('docstring')}\n"
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

    prompt = EXPLAIN_CODE_PROMPT.format(functions_block=functions_block[:12000], query=sq.text)
    explanation = Settings.llm.complete(prompt).text.strip()

    lines = [f"MODULE: {sq.module}", f"SUB-QUERY: {sq.text}", "INTENT: EXPLAIN_CODE", ""]
    if functions:
        lines.append("FUNCTIONS EXPLAINED: "
                     + ", ".join(fn.get("name") for fn in functions if fn.get("name")))
        lines.append("")
    lines += ["EXPLANATION:", explanation]
    return "\n".join(lines)

def answer_query_deterministic(
    query: str,
    run_generated_code: bool = True,
) -> str:
    classification = get_or_classify_query(query)

    final_sections = []

    # ── Pass 1: collect raw results for every sub-query ───────────────────────
    # Mirrors the original per-sq loop but stores results instead of formatting
    # immediately, so Pass 2 can merge same-module entries before generation.
    raw_results = []
    for sq in classification.sub_queries:
        extracted_params = extract_parameters_for_subquery(sq)
        param_explanations = {}

        if sq.action == Intent.EXPLAIN_CONCEPT:
            functions, score = [], 0.0
            coverage = CoverageAssessment(
                MatchType.NO_MATCH, "EXPLAIN_CONCEPT: coverage not assessed.",
                [], RecommendedAction.GENERATE_STANDALONE,
            )
        else:
            functions, score = _lookup_one(sq, original_query=query)
            if sq.action == Intent.DO:
                coverage = assess_function_coverage(
                    original_query=query, sq=sq,
                    functions=functions, reranker_score=score,
                )
            else:  # EXPLAIN_CODE
                coverage = CoverageAssessment(
                    MatchType.NO_MATCH, "EXPLAIN_CODE: coverage not assessed.",
                    [], RecommendedAction.GENERATE_STANDALONE,
                )

        raw_results.append({
            "sq": sq, "module": sq.module, "action": sq.action,
            "extracted_params": extracted_params, "param_explanations": param_explanations,
            "functions": functions, "score": score, "coverage": coverage,
        })

    # ── Pass 2: merge same-module results ─────────────────────────────────────
    # If the classifier produced two sub-queries for the same module (despite the
    # prompt fix and deduplicate_sub_queries), their functions and missing_requirements
    # are consolidated here so generation is triggered only once per module.
    # The first sub-query for each module is kept as the representative (sq, params, etc.).
    merged_by_module: Dict[tuple, dict] = {}
    for r in raw_results:
        key = (r["module"], r["action"])
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

            # Merge missing_requirements — deduplicate across the two coverage objects
            existing_missing = set(merged_by_module[module]["coverage"].missing_requirements)
            for req in r["coverage"].missing_requirements:
                if req not in existing_missing:
                    merged_by_module[module]["coverage"].missing_requirements.append(req)
                    existing_missing.add(req)

    # Header lines — MODULES INVOLVED reflects post-merge unique modules
    final_sections.append(f"QUERY TYPE: {classification.query_type.value}")
    final_sections.append(
        "MODULES INVOLVED: "
        + ", ".join(sorted({m for (m, _a) in merged_by_module.keys()}))
    )
    final_sections.append(f"NORMALIZED QUERY: {classification.normalized_query}")


    # ── Pass 3: generate and format one result per unique module ──────────────
    for i, (key, r) in enumerate(merged_by_module.items(), start=1):
        module, action = key
        sq = r["sq"]
        functions = r["functions"]
        coverage = r["coverage"]
        extracted_params = r["extracted_params"]
        param_explanations = r["param_explanations"]

        if action == Intent.EXPLAIN_CONCEPT:
            section = _explain_concept_section(query, sq, extracted_params, param_explanations)
            final_sections.append(f"\n--- RESULT {i} ---\n{section}")
            continue

        if action == Intent.EXPLAIN_CODE:
            section = _explain_code_section(query, sq, functions, extracted_params, param_explanations)
            final_sections.append(f"\n--- RESULT {i} ---\n{section}")
            continue

        # action == Intent.DO — your existing generated_code / coverage / resolve_exact_match
        # block continues UNCHANGED from here down.
        generated_code = None
        execution_result = None
        generated_unit_tests = []
        resolution = None                      # <-- initialize for both paths

        needs_generation = coverage.recommended_action in {
            RecommendedAction.GENERATE_WITH_HELPERS,
            RecommendedAction.GENERATE_STANDALONE,
        }

        if needs_generation:
            param_explanations = explain_extracted_parameters(extracted_params)
            generated_code, retrieved_source = generate_code_for_subquery(
                query=query, sq=sq,
                helper_functions=functions, extracted_params=extracted_params,
            )
            try:
                parsed_gen = extract_json_array(generated_code)
                code_for_tests = "\n\n".join(
                    item.get("code", "") for item in parsed_gen if isinstance(item, dict)
                ) or generated_code
            except (json.JSONDecodeError, Exception):
                code_for_tests = generated_code

            # Name the intended entry point so the spec tests can call it directly.
            target_name = "solution"
            try:
                _names = [f.get("name") for f in extract_json_array(generated_code)
                          if isinstance(f, dict) and f.get("name")]
                if _names:
                    target_name = _names[0]
            except Exception:
                pass

            # Contract from the REQUIREMENTS, before judging the generated code.
            contract = build_contract_from_requirements(
                query=query,
                sub_query=sq.text,
                missing_requirements=coverage.missing_requirements,
                extracted_params=extracted_params,
                target_name=target_name,
            )
            generated_unit_tests = generate_spec_based_tests(contract)

            if run_generated_code and generated_unit_tests:
                execution_result = run_feedback_loop_for_subquery(
                    query=query,
                    generated_code=code_for_tests,
                    unit_tests=generated_unit_tests,
                    helper_functions=functions,
                    retrieved_code=retrieved_source,
                    contract=contract,          # enables triage + spec-aimed repair
                )
                # Show the code that actually ran/passed, not the pre-repair draft.
                _final = (execution_result or {}).get("final_code")
                if _final and _final.strip():
                    generated_code = _final
                repaired = (execution_result or {}).get("final_code")  # adjust to its real key
                if repaired:
                    generated_code = repaired

        else:
            # EXACT_MATCH / RETURN_EXISTING_FUNCTIONS:
            # apply the retrieved function to the user's request — no tests, no generation.
            resolution = resolve_exact_match(
                query=query,
                functions=functions,
                extracted_params=extracted_params,
            )
            # Show the RETURNED FUNCTION's own signature parameters, in the same
            # Meaning/Used-for style — not the global parameter-catalog dump.
            extracted_params, param_explanations = explain_function_parameters(functions)

        section = format_subquery_answer(
            sq=sq,
            extracted_params=extracted_params,
            param_explanations=param_explanations,
            functions=functions,
            coverage=coverage,
            generated_code=generated_code,
            execution_result=execution_result,
            unit_tests=generated_unit_tests,
            resolution=resolution,              # <-- pass it so ANSWER/USAGE render
        )

        final_sections.append(f"\n--- RESULT {i} ---\n{section}")

    raw_answer = "\n\n".join(final_sections)

    # Collect all retrieved functions across all modules for code injection
    all_functions = [
        fn
        for r in merged_by_module.values()
        for fn in r["functions"]
    ]
    return inject_code(raw_answer, all_functions)


# ── Run ───────────────────────────────────────────────────────────────────────
print("EXECUTION METHOD")
# print([m for m in dir(agent) if not m.startswith("_")])

query = (
    "What are `remove_bias`, `add_bias`, and `gene_value_bias` in `suggest_corrections`" 
    "intended for?"
)

print("\n" + "=" * 60)
print("QUERY:", query)
print("=" * 60 + "\n")

import asyncio

async def main():
    response = answer_query_deterministic(
        query=query,
        run_generated_code=True,
    )

    print("\n--- FINAL ANSWER ---")
    print(response)


if __name__ == "__main__":
    # run_classification_tests()
    asyncio.run(main())

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.4f} seconds")