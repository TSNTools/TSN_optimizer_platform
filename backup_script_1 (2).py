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
import os
import json
import torch
import ast
import warnings
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
# from deepmultilingualpunctuation import PunctuationModel  # REMOVED: punctuation restoration unused (only called at line ~611 which is commented out); frees VRAM
from lookup_validation import (
    validate_lookup_payload,
    validate_function_list,
    validate_single_function_claim,
)
from llama_index.core.schema import QueryBundle
from difflib import SequenceMatcher
from collections import defaultdict
from exact_match_resolver import resolve_exact_match
from utils import extract_code_block
import time
from llama_index.core import Settings
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

# Initialize the debug handler
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([llama_debug])

start_time = time.time()


_orig_complete = Settings.llm.complete          # the real BOUND method (reading is fine)
_log = []

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

# ── Config ───────────────────────────────────────────────────────────────────

# API             = "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"
PARAMETERS_FILE = "./parameters.md"

# Tier 2: precomputed parameter explanations (built ONCE offline by build_param_map.py).
# Shape: { "<param_name>": {"meaning": "...", "used_for": "..."} }
# Loaded at startup with zero LLM cost; used for all parameter explanations at runtime.
PARAM_EXPLANATIONS_FILE = "./parameter_explanations.json"

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


PYTHON_SOURCE_FILES: Dict[str, str] = {
    "usecase_generator": "./usecase_generator.py",
    "tsn_scheduling_problem": "./tsn_scheduling_problem.py",
    "evaluator": "./evaluator.py",
}

FUNCTION_CATALOG_FILE = "./function_catalog.json"

MODULE_FILES: Dict[str, str] = {
    "usecase_generator":      "./usecase_generator_functions.json",
    "tsn_scheduling_problem": "./tsn_scheduling_problem_functions.json",
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

# ── Deterministic retrieval-confidence gate (calibration knobs) ───────────────
# bge-reranker-large scores are NOT calibrated probabilities. The [gate] log line
# in assess_function_coverage prints top/2nd/margin per sub-query — collect ~30-50
# real queries and set T_EXACT where correct vs wrong top-1 hits separate.
# Keep T_EXACT conservatively HIGH: a mis-routed exact-match degrades safely into
# DO-gen-with-helpers, whereas a weak match promoted to DO-exact applies the wrong
# function confidently.
T_EXACT   = 0.50   # top-1 must clear this to be eligible for DO-exact.
MARGIN    = 0.00   # AND top-1 must beat top-2 by at least this (disambiguation).
T_HELPERS = CONFIDENCE_THRESHOLD   # below T_EXACT but >= this -> generate-with-helpers.


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
                if isinstance(item, dict) and item.get("name"):
                    by_name[item["name"]] = item
        except FileNotFoundError:
            print(f"[WARN] catalog not found: {path} — run build_indexes.py.")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] could not load catalog {path}: {e}")
        catalogs[module] = by_name
    return catalogs


FUNCTION_CATALOG_BY_MODULE: Dict[str, Dict[str, dict]] = _load_function_catalogs(MODULE_FILES)


def _catalog_entry_to_function(entry: dict, module: str) -> dict:
    """Build the function dict downstream consumers expect, from a catalog entry."""
    name = entry.get("name")
    code = fetch_code_from_source(name, module) or entry.get("code", "")
    return {
        "name": name,
        "class_name": entry.get("class_name"),
        "sub_class_name": entry.get("sub_class_name"),
        "signature": entry.get("signature"),
        "docstring": entry.get("docstring"),
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

VALID_INTENTS = {"DO", "EXPLAIN_CONCEPT", "EXPLAIN_CODE"}

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
    action: Intent = Intent.DO
    parameters: dict = field(default_factory=dict)


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
   - The Usecase object and its GRAPH-LEVEL views: hosts, switches, links, ports (as
     graph attributes keyed by port NAME), window_length, distance_to_hosts.
   - RAW (de)serialization of a Usecase: to_file, to_json, from_json, topology2json,
     guess_config, get_config.
   - Drawing the BASE use-case topology: drawTopology, drawFattree.
   Route here when the user wants to build, generate, randomize, sample, or serialize a
   use case / topology / streams / routes.
   NOT here: integer port-id mappings, per-stream pruned subgraphs, delay budgets,
   macro-cycle value -> tsn_scheduling_problem. Checking a candidate schedule ->
   evaluator.

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
- Each (module, action) pair may appear at most once in sub_queries. If multiple
aspects of the query map to the same module AND the same action, merge them into a
single sub-query. Different actions on the same module stay as separate sub-queries.
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
  the module that owns it with action=EXPLAIN_CODE — even if the query uses the
  words "parameter", "default", or "argument".
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
    * DO              — user wants something new built or generated or computed 
    * EXPLAIN_CONCEPT — a conceptual TSN question NOT tied to one specific function,
                        including questions about units, output formats, data types, or
                        properties of a module's output ("what is a fat-tree", "why does
                        TSN need gate control lists", "what units does UsecaseGenerator
                        use for time and bandwidth").
    * EXPLAIN_CODE    — explain or walk through a specific existing function, method AND/OR
                        class ("what does compute_metrics do", "explain k_ary_fat_tree").
  When torn between EXPLAIN_CODE and EXPLAIN_CONCEPT, prefer EXPLAIN_CODE.
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
    # model="BAAI/bge-reranker-large", top_n=5   # HEAVY (~2GB) — swapped for lighter base model
    model="BAAI/bge-reranker-base", top_n=5
)
reranker_algo = SentenceTransformerRerank(
    # model="BAAI/bge-reranker-large", top_n=8   # HEAVY (~2GB) — swapped for lighter base model
    model="BAAI/bge-reranker-base", top_n=8
)


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
        similarity_top_k=20,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
    )
    for m in CODE_MODULE_LABELS
]
_combined_bm25 = [
    BM25Retriever.from_defaults(nodes=module_nodes[m], similarity_top_k=20)
    for m in CODE_MODULE_LABELS
]
combined_retriever = QueryFusionRetriever(
    retrievers=[*_combined_semantic, *_combined_bm25],
    similarity_top_k=20,                 # fused candidates pooled across ALL modules
    mode="reciprocal_rerank",
    num_queries=1,                       # no LLM query expansion
    use_async=False,
)

# Cross-module reranker: keep the top 10 fused chunks (replaces the old k_nodes=6
# truncation — the reranked set itself is the context).
combined_reranker = SentenceTransformerRerank(
    # model="BAAI/bge-reranker-large", top_n=10   # HEAVY (~2GB) — swapped for lighter base model
    model="BAAI/bge-reranker-base", top_n=10
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

Return ONLY a valid JSON object, NO MARKDOWN, NO PREAMBLE:
{
  "functions": [ <function objects as specified above, or [] if none> ],
  "coverage": {
    "match_type": "EXACT_MATCH" | "PARTIAL_MATCH" | "NO_MATCH",
    "missing_requirements": ["<capability requested but NOT provided by the functions>"]
  }
}

Coverage rules:
- EXACT_MATCH: the returned function(s), called with suitable params, fully satisfy the request; no glue code needed; missing_requirements = [].
- PARTIAL_MATCH: returned functions are useful helpers but new wrapper/orchestration/logic is still needed; list what's missing.
- NO_MATCH: nothing useful; functions = [].
- High relevance rank alone does NOT justify EXACT_MATCH.
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
- Judge each candidate by what its code and docstring actually DO versus the request,
  not by retrieval order or how confident retrieval seemed.
- If functions are empty, the answer must be NO_MATCH.
- If the request asks for a pipeline involving multiple actions and the retrieved functions only cover one action, use PARTIAL_MATCH.
- When a candidate plausibly performs the requested task, prefer EXACT_MATCH even if its
  parameter names or calling convention differ from how the user phrased it.
- Choose PARTIAL_MATCH only when genuine additional logic or orchestration beyond calling
  the existing function(s) is required.

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
You are a TSN algorithm code generator. Your job is to produce a complete, working Python
implementation that fully satisfies the request. You ALWAYS return code — never a refusal.

How to use the context:
- The context may contain existing helper functions from this codebase. USE them when they
  fit the request, calling them by their exact names, and list them under dependencies.internal.
- If the functionality required by the request is NOT present in the context, GENERATE it from
  scratch. Write real, runnable logic — do NOT refuse and do NOT emit "NOT_IN_CONTEXT".
- When the codebase does not provide a needed capability, you MAY use the Python standard
  library and well-known third-party libraries (e.g. networkx, numpy, lxml/xml, ncclient).
  Declare any third-party libraries under dependencies.external.

Rules:
- Write a complete function: a def signature, a docstring (inputs, outputs, which helpers are
  used, and any assumptions you made when generating from scratch), and a real body.
- Prefer runnable logic over stubs. If you must assume the shape of an input the context does
  not define, state that assumption in the docstring and proceed with a sensible implementation.

Output rules:
- Return ONLY a valid JSON array, no preamble, no markdown fences, no extra text.

"name"          : "the function name (the name you gave it if newly generated, or the exact context name if reused)",
"type"          : "classmethod | staticmethod | instance_method | function | property",
"module"        : "target source filename this belongs to (e.g. \"evaluator.py\")",
"class"         : "class the function belongs to, or null for module-level functions",
"description"   : "one sentence describing what the function does",
"signature"     : "the full def line of the function (generated or reused), e.g. \"def export_to_netconf(self, schedule, out_path):\"",
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
                            "context helpers / project functions this function calls",
                            "may include calls like nx.shortest_path, np.zeros, etc."
                        ],
                        "external": [
                            "third-party libraries used that are not implemented in the codebase"
                        ]
                    },
"raises"       : ["exceptions explicitly raised or propagated by the function"],
"example_use"  : "short realistic example showing how the function is called",
"code"         : "the complete function code — reused verbatim if it came from context, otherwise the newly generated implementation",
"unit_tests"   : ["Tests covering normal cases, edge cases, and errors (inputs, outputs, exceptions)."]

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

    explanations = _explain_params_from_map(func_params)
    return func_params, explanations

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

def _parse_lookup_payload(raw: str) -> tuple[list[dict], Optional[dict]]:
    raw = raw.replace("\\'", "'")
    try:
        obj = extract_json_object(raw)
        if isinstance(obj, dict) and "functions" in obj:
            funcs = [f for f in obj.get("functions", []) if isinstance(f, dict)]
            cov = obj.get("coverage") if isinstance(obj.get("coverage"), dict) else None
            return funcs, cov
    except json.JSONDecodeError:
        pass
    return _parse_lookup_json(raw), None   # legacy bare-array fallback

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
    Behavioural evidence ONLY for the coverage LLM — name, signature, docstring, code.
    Deliberately omits match_rank and reranker_score: match_rank is a derived relevance
    label and the reranker score is saturated (~0.95+ for anything on-topic), so neither
    is trustworthy evidence. The assessor judges purely on what each function does.
    """
    rendered = []
    for fn in functions:
        rendered.append({
            "name": fn.get("name"),
            "signature": fn.get("signature"),
            "docstring": fn.get("docstring"),
            "code": (fn.get("code") or "")[:100],   # code for every candidate, not just a presumed "best" #originally was [:1500]
        })
    return json.dumps(rendered, indent=2)


def assess_function_coverage(
    *,
    query: str,
    sq: SubQuery,
    functions: List[Dict],
    top_score: float,
    second_score: float,
) -> CoverageAssessment:
    """
    Decide DO-exact vs DO-gen-with-helpers vs DO-gen-standalone.

    LLM coverage assessor. The old pure score+margin gate is unusable here: the
    bge-reranker scores saturate (~0.92-0.98) for anything on-topic, so top_score >=
    T_EXACT was always true and every DO query was (wrongly) EXACT_MATCH -> never
    generated. The LLM reads the candidates' actual code and judges whether they
    implement the request.

    DO-only: this is called solely from the DO branch of Pass 1. EXPLAIN_CODE and
    EXPLAIN_CONCEPT never reach it, so their behaviour is unchanged.
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

    # ── OLD deterministic score gate — DISABLED. Reranker scores don't discriminate
    #    (they saturate near 1.0 for anything topical), so no threshold on top_score
    #    can separate "implements the request" from "merely related". Kept for
    #    reference; T_EXACT / T_HELPERS / MARGIN are now unused by this function.
    #
    # if top_score >= T_EXACT and margin >= MARGIN:
    #     return CoverageAssessment(
    #         match_type=MatchType.EXACT_MATCH,
    #         reason=(f"Top candidate '{real[0].get('name')}' cleared T_EXACT={T_EXACT} "
    #                 f"with margin {margin:.3f} >= {MARGIN}; returning existing function."),
    #         missing_requirements=[],
    #         recommended_action=RecommendedAction.RETURN_EXISTING_FUNCTIONS,
    #     )
    # if top_score >= T_HELPERS:
    #     why = ("strong but ambiguous (margin below threshold)"
    #            if top_score >= T_EXACT else "medium confidence")
    #     return CoverageAssessment(
    #         match_type=MatchType.PARTIAL_MATCH,
    #         reason=f"Retrieved candidates are {why}; using them as helpers and generating.",
    #         missing_requirements=["Retrieval not confident enough for an exact match; "
    #                               "generate to fill gaps."],
    #         recommended_action=RecommendedAction.GENERATE_WITH_HELPERS,
    #     )
    # return CoverageAssessment(
    #     match_type=MatchType.NO_MATCH,
    #     reason="Retrieval confidence below the helper floor; generating standalone.",
    #     missing_requirements=["No sufficiently relevant existing function found."],
    #     recommended_action=RecommendedAction.GENERATE_STANDALONE,
    # )

    # ── LLM coverage assessment (evidence-only: name / signature / docstring / code) ──
    prompt = COVERAGE_ASSESSMENT_PROMPT.format(
        original_query=query,
        sub_query=sq.text,
        module=sq.module,
        functions_json=_render_candidates_for_coverage(real),
    )
    try:
        raw = Settings.llm.complete(prompt).text.strip()
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        match_type = MatchType(data["match_type"])
    except Exception as e:  # noqa: BLE001 — fail toward generation, never a false exact-match
        print(f"[gate] LLM coverage parse failed ({e}); defaulting to GENERATE_WITH_HELPERS.")
        return CoverageAssessment(
            match_type=MatchType.PARTIAL_MATCH,
            reason="LLM coverage assessment unparseable; generating with retrieved helpers.",
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
                "match_confidence": (
                    "HIGH_CONFIDENCE" if score >= CONFIDENCE_THRESHOLD else "LOW_CONFIDENCE"
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
        ], score, 0.0

    if sq.module not in module_retrievers:
        print(f"[WARN] No retriever for module: {sq.module}")
        return [], 0.0, 0.0

    nodes = module_retrievers[sq.module].retrieve(augmented_text)
    nodes = reranker_algo.postprocess_nodes(
        nodes, query_bundle=QueryBundle(augmented_text)
    )
    if not nodes:
        return [], 0.0, 0.0

    nodes = nodes[:5]
    top_score = nodes[0].score or 0.0
    second_score = (nodes[1].score or 0.0) if len(nodes) > 1 else 0.0
    confidence = (
        "HIGH_CONFIDENCE" if top_score >= CONFIDENCE_THRESHOLD else "LOW_CONFIDENCE"
    )

    catalog = FUNCTION_CATALOG_BY_MODULE.get(sq.module, {})
    functions: list[dict] = []
    for rank, n in enumerate(nodes):
        fn_name = n.node.metadata.get("function_name")
        if not fn_name:
            continue
        entry = catalog.get(fn_name)
        if entry:
            fn = _catalog_entry_to_function(entry, sq.module)
        else:
            fn = {
                "name": fn_name,
                "signature": n.node.metadata.get("signature"),
                "class_name": n.node.metadata.get("class_name"),
                "docstring": None,
                "description": "",
                "parameters": [],
                "postconditions": [],
                "modules": sq.module,
                "module": sq.module,
                "code": fetch_code_from_source(fn_name, sq.module),
                "routed_module": sq.module,
            }
        fn["match_rank"] = "BEST_MATCH" if rank == 0 else "ALTERNATE"
        fn["match_confidence"] = confidence
        fn["reranker_score"] = round(n.score or 0.0, 3)
        functions.append(fn)

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

    return valid_functions, top_score, second_score




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


def generate_code_for_subquery(
    query: str,
    sq: SubQuery,
    helper_functions: list[dict],
    extracted_params: Optional[dict] = None,
) -> tuple[str, str]:
    if sq.module == "parameters":
        return _retrieve_param_context(augment_query(sq.text, sq.module)), ""

    if sq.module not in gen_engines:
        return f"NO_GENERATION_ENGINE_AVAILABLE for module: {sq.module}", ""

    augmented_text = augment_query(sq.text, sq.module)

    param_context_str = _retrieve_param_context(
        f"What parameters are relevant to: {augmented_text}"
    )
    if not param_context_str:
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


def _render_generated_code(generated_code: str) -> str:
    """
    Render the generator's output as clean, readable Python.

    generate_code_for_subquery returns a JSON array of {name, code, ...} objects
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


def _extract_spec(fn: dict, user_request: str) -> dict:
    """
    Derive a behavioral contract from the function's DECLARED interface + the user
    request. Never receives the function body, so the contract reflects intent, not
    implementation. Uses the dormant SPEC_EXTRACTION_PROMPT.
    """
    prompt = SPEC_EXTRACTION_PROMPT.format(
        name=fn.get("name") or "generated_function",
        signature=fn.get("signature") or "",
        description=f"User request: {user_request}\nDeclared purpose: {fn.get('description', '')}",
        parameters=json.dumps(fn.get("parameters", []), indent=2),
        preconditions=json.dumps(fn.get("preconditions", []), indent=2),
        postconditions=json.dumps(fn.get("postconditions", []), indent=2),
        raises=json.dumps(fn.get("raises", []), indent=2),
    )
    raw = Settings.llm.complete(prompt).text.strip()
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


def _spec_based_tests(contract: dict, import_hint: str, revision: str = None) -> list[dict]:
    """Write pytest tests from the contract ONLY (code-blind). Uses SPEC_BASED_TEST_PROMPT.
    If `revision` is given (user suggestions or triage feedback), it is appended so the
    tests are remade while still only asserting contract-supported behavior."""
    contract_text = json.dumps(contract, indent=2)
    if revision:
        contract_text += (
            "\n\nREVISION REQUEST (apply while still asserting ONLY contract-supported "
            f"behavior; do not weaken valid assertions):\n{revision}"
        )
    prompt = SPEC_BASED_TEST_PROMPT.format(
        import_hint=import_hint,
        contract_json=contract_text,
    )
    raw = Settings.llm.complete(prompt).text.strip()
    data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    return data.get("tests", [])


def generate_spec_based_tests_for(generated_code: str, user_request: str, module: str) -> list[dict]:
    """
    Spec-first, code-blind unit test generation for a generated function.

    Flow: parse the generated function's declared interface -> derive a behavioral
    contract from that interface + the user request (NOT the code body) -> write
    pytest tests from the contract. Because the test writer never sees the
    implementation, the tests judge the code against intent, not against itself.
    Returns a list of {behavior_id, test_name, code} dicts (possibly empty).
    """
    try:
        fns = json.loads(generated_code)
    except (json.JSONDecodeError, TypeError):
        return []
    if not (isinstance(fns, list) and fns and isinstance(fns[0], dict)):
        return []
    fn = fns[0]
    try:
        contract = _extract_spec(fn, user_request)
        mod = module[:-3] if module.endswith(".py") else module
        import_hint = f"from {mod} import {fn.get('name')}"
        tests = _spec_based_tests(contract, import_hint)
        print(f"[tests] spec-based: {len(tests)} test(s) from "
              f"{len(contract.get('behaviors', []))} behavior(s).")
        return tests
    except Exception as e:  # noqa: BLE001
        print(f"[tests] spec-based generation failed: {e}")
        return []


def _execute_spec_tests(impl_code: str, test_codes: list[str], module: str) -> dict:
    """
    Write the generated function + its spec tests to a temp dir and run pytest in a
    subprocess. The original codebase dir (CWD) is put on PYTHONPATH so real modules
    and third-party libs the generated code imports still resolve. Returns a dict with
    success / returncode / stdout / stderr / workdir.
    """
    import tempfile, subprocess, os as _os, sys as _sys

    mod_name = module[:-3] if module.endswith(".py") else module
    tmpdir = tempfile.mkdtemp(prefix="tsn_gen_")
    with open(_os.path.join(tmpdir, f"{mod_name}.py"), "w") as f:
        f.write((impl_code or "").rstrip() + "\n")
    test_path = _os.path.join(tmpdir, "test_generated.py")
    with open(test_path, "w") as f:
        f.write("\n\n".join(test_codes).rstrip() + "\n")

    env = dict(_os.environ)
    env["PYTHONPATH"] = _os.getcwd() + _os.pathsep + env.get("PYTHONPATH", "")
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
    """Decide impl-wrong vs test-wrong for a failed contract test. Uses TRIAGE_PROMPT."""
    prompt = TRIAGE_PROMPT.format(
        contract_json=json.dumps(contract, indent=2),
        test_code=test_code,
        impl_code=impl_code,
        failure_output=failure_output[:4000],
    )
    raw = Settings.llm.complete(prompt).text.strip()
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


def _repair_impl(impl_code: str, contract: dict, failure_output: str, query: str) -> str:
    """Regenerate the implementation given the contract + failure. Returns raw code."""
    prompt = (
        "You are fixing a Python implementation that failed contract-derived tests.\n"
        "Return ONLY the corrected function code (a single def ...), no fences, no prose.\n\n"
        f"Original user request:\n{query}\n\n"
        f"Behavioral contract:\n{json.dumps(contract, indent=2)}\n\n"
        f"Current implementation:\n{impl_code}\n\n"
        f"Test failure output:\n{failure_output[:3000]}\n\n"
        "Corrected implementation:"
    )
    fixed = Settings.llm.complete(prompt).text.strip()
    return extract_code_block(fixed) or fixed


def _approval_gate(note: str, contract: dict, import_hint: str, tests: list[dict]) -> list[dict]:
    """
    Interactive gate: show the note + proposed spec-based tests, then loop on input().
    'y'/Enter approves; anything else is treated as revision suggestions and the tests
    are regenerated from the contract with those suggestions folded in. Prompts go to
    stderr so redirecting stdout to a file still works.
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
        tests = _spec_based_tests(contract, import_hint, revision=resp)


def _run_and_repair(impl_code, contract, tests, module, query, import_hint, max_rounds=2):
    """
    Execute the approved tests against the implementation. On failure, triage each round
    (impl-wrong vs test-wrong per TRIAGE_PROMPT), repair the guilty side, and re-execute,
    up to max_rounds. Returns (execution_result, impl_code, tests).
    """
    test_codes = [t.get("code", "") for t in tests if t.get("code")]
    result = None
    for rnd in range(max_rounds + 1):
        result = _execute_spec_tests(impl_code, test_codes, module)
        if result["success"]:
            print(f"[exec] round {rnd}: all tests passed.", file=sys.stderr)
            return result, impl_code, tests
        print(f"[exec] round {rnd}: tests failed (rc={result['returncode']}).", file=sys.stderr)
        if rnd == max_rounds:
            print("[exec] max repair rounds reached; returning last result.", file=sys.stderr)
            return result, impl_code, tests

        failure_output = (result["stdout"] + "\n" + result["stderr"]).strip()
        try:
            triage = _triage_failure(contract, "\n\n".join(test_codes), impl_code, failure_output)
        except Exception as e:  # noqa: BLE001
            print(f"[triage] parse failed ({e}); stopping repair.", file=sys.stderr)
            return result, impl_code, tests

        target = triage.get("suggested_fix_target")
        print(f"[triage] verdict={triage.get('verdict')} -> fix {target}", file=sys.stderr)
        if target == "implementation":
            impl_code = _repair_impl(impl_code, contract, failure_output, query)
        elif target == "test":
            tests = _spec_based_tests(
                contract, import_hint,
                revision=(f"A test failed. Triage: {triage.get('explanation', '')}. "
                          "Fix ONLY if the test contradicts the contract; never weaken a "
                          "valid assertion just to make it pass."),
            )
            test_codes = [t.get("code", "") for t in tests if t.get("code")]
        else:
            print(f"[triage] {triage.get('verdict')} — not auto-repairable; stopping.", file=sys.stderr)
            return result, impl_code, tests
    return result, impl_code, tests


def _interactive_verify(generated_code: str, user_request: str, module: str, note: str):
    """
    Full interactive verification for a generated function:
      derive contract -> spec-based tests -> approval gate -> execute -> triage/repair.
    Returns (generated_code, approved_test_codes, execution_result). If anything can't be
    prepared, returns the inputs unverified so the flow still completes.
    """
    try:
        fns = json.loads(generated_code)
        fn = fns[0]
    except Exception:  # noqa: BLE001
        print("[verify] generated code not parseable as JSON; skipping verification.", file=sys.stderr)
        return generated_code, [], None

    impl_code = fn.get("code", "")
    contract = _extract_spec(fn, user_request)
    mod = module[:-3] if module.endswith(".py") else module
    import_hint = f"from {mod} import {fn.get('name')}"
    tests = _spec_based_tests(contract, import_hint)

    if not tests:
        print("[verify] no spec-based tests were produced; skipping execution.", file=sys.stderr)
        return generated_code, [], None

    tests = _approval_gate(note, contract, import_hint, tests)
    result, impl_code, tests = _run_and_repair(
        impl_code, contract, tests, module, user_request, import_hint
    )

    # Reflect any repaired implementation back into the displayed code.
    fn["code"] = impl_code
    generated_code = json.dumps(fns, indent=2)
    test_codes = [t.get("code", "") for t in tests if t.get("code")]
    return generated_code, test_codes, result


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
  your explanation stays relevant to what the user ultimately wants. Do not answer
  the other parts of the overall question here — just make sure this piece fits it.

Parameter definitions (from this codebase):
{param_context}

Function docstrings / code context (from this codebase):
{doc_context}

Overall user question (for context — what the user ultimately wants):
{overall_query}

This sub-task (the specific piece to address now):
{query}

Explanation:
"""

EXPLAIN_CODE_PROMPT = """\
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
 
Overall user request (for context — what the user ultimately wants):
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

    prompt = EXPLAIN_CODE_PROMPT.format(
        functions_block=functions_block[:12000],
        overall_query=query,
        query=sq.text,
    )
    explanation = Settings.llm.complete(prompt).text.strip()

    lines = [f"MODULE: {sq.module}", f"SUB-QUERY: {sq.text}", "INTENT: EXPLAIN_CODE", ""]
    if functions:
        lines.append("FUNCTIONS RETRIEVED AS CONTEXT: "
                     + ", ".join(fn.get("name") for fn in functions if fn.get("name")))
        lines.append("")
    lines += ["EXPLANATION/ANSWER:", explanation]
    return "\n".join(lines)

def answer_query_deterministic(
    query: str,
    run_generated_code: bool = True,
    interactive: bool = False,
) -> str:
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
            functions, score = [], 0.0
            coverage = CoverageAssessment(
                MatchType.NO_MATCH, "EXPLAIN_CONCEPT: coverage not assessed.",
                [], RecommendedAction.GENERATE_STANDALONE,
            )
        else:
            functions, top_score, second_score = _lookup_one(sq)
            score = top_score
            if sq.action == Intent.DO:
                coverage = assess_function_coverage(
                    query=query,
                    sq=sq,
                    functions=functions,
                    top_score=top_score,
                    second_score=second_score,
                )

            else:  # EXPLAIN_CODE
                # Run the SAME coverage assessor as DO. "What function can I use to
                # <task>?" queries classify as EXPLAIN_CODE but are really task
                # requests; assessing coverage lets a genuine match still explain while
                # a no-match falls through to generation in Pass 3.
                coverage = assess_function_coverage(
                    query=query,
                    sq=sq,
                    functions=functions,
                    top_score=top_score,
                    second_score=second_score,
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
            # Genuine "explain this existing function" -> assessor returns EXACT_MATCH
            # -> explain exactly as before (behaviour of _explain_code_section preserved).
            # A "what function can I use to <task>?" miss -> NO_MATCH / PARTIAL_MATCH ->
            # fall through to the DO generation logic below instead of dead-ending.
            if coverage.recommended_action == RecommendedAction.RETURN_EXISTING_FUNCTIONS:
                section = _explain_code_section(query, sq, functions, extracted_params, param_explanations)
                final_sections.append(f"\n--- RESULT {i} ---\n{section}")
                continue
            action = Intent.DO   # reroute the misses; coverage is already GENERATE_*

        # action == Intent.DO — the score gate splits DO into two mechanisms:
        #   GENERATE_WITH_HELPERS / GENERATE_STANDALONE -> generation (UNCHANGED)
        #   (former) RETURN_EXISTING_FUNCTIONS, i.e. an exact match -> now routed to
        #       the EXPLAIN_CODE prompt instead of resolve_exact_match, so EXPLAIN_CODE
        #       is the single retrieve-and-use-existing-functions pathway.
        needs_generation = coverage.recommended_action in {
            RecommendedAction.GENERATE_WITH_HELPERS,
            RecommendedAction.GENERATE_STANDALONE,
        }

        if needs_generation:
            generated_code = None
            execution_result = None
            generated_unit_tests = []

            param_explanations = explain_extracted_parameters(extracted_params)
            generated_code, retrieved_source = generate_code_for_subquery(
                query=query, sq=sq,
                helper_functions=functions, extracted_params=extracted_params,
            )

            # Human-readable banner about why we generated (per the open-world path).
            note = (
                "NOTE: No existing function in the codebase satisfies this request; "
                "the code below was generated from scratch. Spec-based unit tests "
                "(written from the requirement, not the code) are included for review."
                if coverage.recommended_action == RecommendedAction.GENERATE_STANDALONE else
                "NOTE: The codebase only partially covers this request; retrieved "
                "helper(s) were used as context and the missing logic was generated. "
                "Spec-based unit tests are included for review."
            )

            if interactive:
                # Full flow: derive contract -> spec tests -> approval gate ->
                # execute -> triage/repair. May update generated_code (if repaired).
                generated_code, generated_unit_tests, execution_result = _interactive_verify(
                    generated_code, sq.text, sq.module, note
                )
            else:
                # Batch: generate spec-based tests and show them; no pause, no execution.
                spec_tests = generate_spec_based_tests_for(generated_code, sq.text, sq.module)
                generated_unit_tests = [t.get("code", "") for t in spec_tests if t.get("code")]

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
    return inject_code(raw_answer, all_functions)


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
        sys.stdout.flush()                                                       # ensure it's on disk now
        print(f"[done] {elapsed:.1f}s", file=sys.stderr, flush=True)            # completion -> screen


if __name__ == "__main__":
    repl()