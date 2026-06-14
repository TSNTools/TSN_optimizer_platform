"""
lookup_validation.py
--------------------
Deterministic anti-hallucination guardrail for the lookup step.

After the LLM proposes candidate functions for a query, it can confidently name
functions that don't exist, or place a real function in the wrong module/class.
This module checks every claimed function against a hand-maintained ground-truth
catalog of the real TSN codebase and separates the genuine matches from the
hallucinated ones before they reach the rest of the pipeline.

Ground truth ------------
HARDCODED_CATALOG maps each module ("evaluator", "tsn_scheduling_problem",
"usecase_generator") to the list of functions/methods that actually exist in it.
Each is a CatalogEntry recording module, function name, owning class (or None for
module-level functions), and kind ("method", "classmethod", "staticmethod",
"property", "function"). At import time build_catalog_indexes() turns this into
two lookup tables: one keyed by (module, function_name), one keyed by name alone
(the latter powers "exists, but elsewhere" suggestions).

Core check ----------
validate_function(name, module, class_name) -> ValidationResult, covering:
  * missing/invalid name, or unknown module
  * name not in the catalog at all
  * name exists but in a DIFFERENT module  -> invalid, with real locations
                                               returned as `alternatives`
  * name exists in the module but not under the claimed class
  * clean match
The `alternatives` field means a wrong-module/class claim returns where the
function really lives, rather than a bare rejection.


MAINTENANCE -----------
HARDCODED_CATALOG is maintained by hand, so it is only as accurate as you keep
it. If you add, rename, move, or delete a function in evaluator.py,
tsn_scheduling_problem.py, or usecase_generator.py, update the catalog too —
otherwise this validator will reject functions that genuinely exist (false
negatives) or fail to flag ones that no longer do. If the source modules change
often, consider generating the catalog by AST-walking them instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CatalogEntry:
    module: str
    function_name: str
    class_name: Optional[str]
    kind: str


@dataclass
class ValidationResult:
    valid: bool
    reason: str
    matched_entry: Optional[CatalogEntry]
    alternatives: List[Dict[str, Any]]


def _entry(
    module: str,
    function_name: str,
    class_name: Optional[str] = None,
    kind: str = "method",
) -> CatalogEntry:
    return CatalogEntry(
        module=module,
        function_name=function_name,
        class_name=class_name,
        kind=kind,
    )


HARDCODED_CATALOG: Dict[str, List[CatalogEntry]] = {
    "evaluator": [
        _entry("evaluator", "__init__", "TSNConstraintEvaluator"),
        _entry(
            "evaluator",
            "_per_stream_host_role_violations",
            "TSNConstraintEvaluator",
        ),
        _entry("evaluator", "group_connected_ports", "TSNConstraintEvaluator"),
        _entry("evaluator", "talker_listener_groups", "TSNConstraintEvaluator"),
        _entry("evaluator", "group_penalty", "TSNConstraintEvaluator"),
        _entry(
            "evaluator",
            "_per_stream_flow_continuity_violations",
            "TSNConstraintEvaluator",
        ),
        _entry(
            "evaluator",
            "_per_stream_flow_conservation_violations",
            "TSNConstraintEvaluator",
        ),
        _entry("evaluator", "_per_stream_priority_conflicts", "TSNConstraintEvaluator"),
        _entry(
            "evaluator",
            "_per_stream_latency_violations_priority",
            "TSNConstraintEvaluator",
        ),
        _entry(
            "evaluator",
            "_per_stream_latency_violations_start_time",
            "TSNConstraintEvaluator",
        ),
        _entry(
            "evaluator",
            "_per_stream_talker_offset_violations",
            "TSNConstraintEvaluator",
        ),
        _entry(
            "evaluator",
            "_per_stream_macrocycle_overlap_violations",
            "TSNConstraintEvaluator",
        ),
        _entry(
            "evaluator",
            "_per_stream_window_sequence_violations",
            "TSNConstraintEvaluator",
        ),
        _entry("evaluator", "per_stream_all_violations", "TSNConstraintEvaluator"),
        _entry("evaluator", "feasible_streams", "TSNConstraintEvaluator"),
        _entry("evaluator", "compute_metrics", "TSNConstraintEvaluator"),
        _entry("evaluator", "evaluate", "TSNConstraintEvaluator"),
        _entry("evaluator", "suggest_corrections", "TSNConstraintEvaluator"),
        _entry("evaluator", "__init__", "UnionFind"),
        _entry("evaluator", "find", "UnionFind"),
        _entry("evaluator", "union", "UnionFind"),
    ],
    "tsn_scheduling_problem": [
        _entry("tsn_scheduling_problem", "require_instance", None, "function"),
        _entry("tsn_scheduling_problem", "__init__", "InstanceContext"),
        _entry("tsn_scheduling_problem", "__enter__", "InstanceContext"),
        _entry("tsn_scheduling_problem", "__exit__", "InstanceContext"),
        _entry("tsn_scheduling_problem", "__init__", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "build_metadata", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "initialize_problem", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "use_instance", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "no_instance_assigned", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "instantiate", "TSNScheduling"),
        _entry(
            "tsn_scheduling_problem",
            "import_scheduling_usecase",
            "TSNScheduling",
        ),
        _entry(
            "tsn_scheduling_problem",
            "generate_scheduling_usecase",
            "TSNScheduling",
        ),
        _entry(
            "tsn_scheduling_problem",
            "generate_incremental_usecases",
            "TSNScheduling",
        ),
        _entry("tsn_scheduling_problem", "get_config", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "get_task", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "n_ports", "TSNScheduling", "property"),
        _entry("tsn_scheduling_problem", "n_streams", "TSNScheduling", "property"),
        _entry(
            "tsn_scheduling_problem",
            "port_id_to_name",
            "TSNScheduling",
            "property",
        ),
        _entry(
            "tsn_scheduling_problem",
            "port_name_to_id",
            "TSNScheduling",
            "property",
        ),
        _entry("tsn_scheduling_problem", "ports", "TSNScheduling", "property"),
        _entry(
            "tsn_scheduling_problem",
            "port_id_to_node",
            "TSNScheduling",
            "property",
        ),
        _entry("tsn_scheduling_problem", "nodes", "TSNScheduling", "property"),
        _entry("tsn_scheduling_problem", "n_nodes", "TSNScheduling", "property"),
        _entry(
            "tsn_scheduling_problem",
            "node_to_port_id",
            "TSNScheduling",
            "property",
        ),
        _entry(
            "tsn_scheduling_problem",
            "next_prev_hops",
            "TSNScheduling",
            "property",
        ),
        _entry("tsn_scheduling_problem", "macro_cycle", "TSNScheduling", "property"),
        _entry("tsn_scheduling_problem", "streams", "TSNScheduling", "property"),
        _entry("tsn_scheduling_problem", "links", "TSNScheduling", "property"),
        _entry("tsn_scheduling_problem", "_frame_size", "TSNScheduling", "classmethod"),
        _entry("tsn_scheduling_problem", "hops_costs", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "network_borders", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "configure_stream_ports", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "get_concurrent_ports", "TSNScheduling"),
        _entry(
            "tsn_scheduling_problem",
            "make_default_stream_config",
            "TSNScheduling",
        ),
        _entry("tsn_scheduling_problem", "make_stream_config", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "hop_cost_eg2eg", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "bfs_forward", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "bfs_backward", "TSNScheduling"),
        _entry("tsn_scheduling_problem", "drawFattree", "TSNScheduling", "staticmethod"),
        _entry(
            "tsn_scheduling_problem",
            "plot_all_stream_custom_graphs",
            "TSNScheduling",
            "staticmethod",
        ),
    ],
    "usecase_generator": [
        _entry("usecase_generator", "drawFattree", None, "function"),
        _entry("usecase_generator", "__init__", "UsecaseGenerator"),
        _entry("usecase_generator", "generateUC", "UsecaseGenerator", "classmethod"),
        _entry("usecase_generator", "ucGenerator", "UsecaseGenerator", "classmethod"),
        _entry("usecase_generator", "empty_uc", "UsecaseGenerator", "classmethod"),
        _entry(
            "usecase_generator",
            "streamGenerator",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "generateStream",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "talkerEmbedding",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "listenerEmbedding",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "findAllPaths",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "shortestPath",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry("usecase_generator", "randomPath", "UsecaseGenerator", "classmethod"),
        _entry(
            "usecase_generator",
            "pathConvenient",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "periodConvenient",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "makeTopology",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "set_link_data",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "k_ary_fat_tree",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "makeRandomGraph",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry(
            "usecase_generator",
            "changeDegree",
            "UsecaseGenerator",
            "classmethod",
        ),
        _entry("usecase_generator", "__init__", "Usecase"),
        _entry("usecase_generator", "hosts", "Usecase", "property"),
        _entry("usecase_generator", "_frame_size", "Usecase"),
        _entry("usecase_generator", "window_length", "Usecase"),
        _entry("usecase_generator", "distance_to_hosts", "Usecase", "property"),
        _entry("usecase_generator", "switches", "Usecase", "property"),
        _entry("usecase_generator", "links", "Usecase", "property"),
        _entry("usecase_generator", "ports", "Usecase", "property"),
        _entry("usecase_generator", "port_name_id_mapping", "Usecase", "property"),
        _entry("usecase_generator", "addSchedule", "Usecase"),
        _entry("usecase_generator", "removeSchedule", "Usecase"),
        _entry("usecase_generator", "addRouting", "Usecase"),
        _entry("usecase_generator", "removeRouting", "Usecase"),
        _entry("usecase_generator", "fixTalkers", "Usecase"),
        _entry("usecase_generator", "fixListeners", "Usecase"),
        _entry("usecase_generator", "remove_stream", "Usecase"),
        _entry("usecase_generator", "add_stream", "Usecase"),
        _entry("usecase_generator", "remove_link", "Usecase"),
        _entry("usecase_generator", "add_link", "Usecase"),
        _entry("usecase_generator", "to_file", "Usecase"),
        _entry("usecase_generator", "to_json", "Usecase"),
        _entry("usecase_generator", "topology2json", "Usecase"),
        _entry("usecase_generator", "from_json", "Usecase", "classmethod"),
        _entry("usecase_generator", "guess_config", "Usecase", "classmethod"),
        _entry("usecase_generator", "get_config", "Usecase"),
        _entry(
            "usecase_generator",
            "generate_fattree_regions",
            "Usecase",
            "staticmethod",
        ),
        _entry("usecase_generator", "drawTopology", "Usecase"),
    ],
}


def build_catalog_indexes(
    catalog: Dict[str, List[CatalogEntry]],
) -> Tuple[Dict[Tuple[str, str], List[CatalogEntry]], Dict[str, List[CatalogEntry]]]:
    by_module_and_name: Dict[Tuple[str, str], List[CatalogEntry]] = {}
    by_name: Dict[str, List[CatalogEntry]] = {}

    for module_name, entries in catalog.items():
        for entry in entries:
            by_module_and_name.setdefault(
                (module_name, entry.function_name), []
            ).append(entry)
            by_name.setdefault(entry.function_name, []).append(entry)

    return by_module_and_name, by_name


CATALOG_BY_MODULE_AND_NAME, CATALOG_BY_NAME = build_catalog_indexes(
    HARDCODED_CATALOG
)


def catalog_entry_to_dict(entry: CatalogEntry) -> Dict[str, Any]:
    return {
        "module": entry.module,
        "function_name": entry.function_name,
        "class_name": entry.class_name,
        "kind": entry.kind,
    }


def validate_function(
    function_name: Optional[str],
    module_name: Optional[str],
    class_name: Optional[str] = None,
) -> ValidationResult:
    if not function_name or not isinstance(function_name, str):
        return ValidationResult(
            valid=False,
            reason="Missing or invalid function name.",
            matched_entry=None,
            alternatives=[],
        )

    if not module_name or module_name not in HARDCODED_CATALOG:
        return ValidationResult(
            valid=False,
            reason=f"Missing or invalid module '{module_name}'.",
            matched_entry=None,
            alternatives=[],
        )

    candidates = CATALOG_BY_MODULE_AND_NAME.get((module_name, function_name), [])

    if not candidates:
        same_name_elsewhere = CATALOG_BY_NAME.get(function_name, [])

        if same_name_elsewhere:
            return ValidationResult(
                valid=False,
                reason=(
                    f"Function '{function_name}' exists in the catalog, but not in "
                    f"module '{module_name}'."
                ),
                matched_entry=None,
                alternatives=[
                    catalog_entry_to_dict(entry) for entry in same_name_elsewhere
                ],
            )

        return ValidationResult(
            valid=False,
            reason=f"Function '{function_name}' does not exist in the catalog.",
            matched_entry=None,
            alternatives=[],
        )

    if class_name not in (None, "", "null"):
        class_matches = [entry for entry in candidates if entry.class_name == class_name]

        if not class_matches:
            return ValidationResult(
                valid=False,
                reason=(
                    f"Function '{function_name}' exists in module '{module_name}', "
                    f"but not under class '{class_name}'."
                ),
                matched_entry=None,
                alternatives=[catalog_entry_to_dict(entry) for entry in candidates],
            )

        return ValidationResult(
            valid=True,
            reason="Function name, module, and class match the catalog.",
            matched_entry=class_matches[0],
            alternatives=[catalog_entry_to_dict(entry) for entry in class_matches],
        )

    return ValidationResult(
        valid=True,
        reason="Function name and module match the catalog.",
        matched_entry=candidates[0],
        alternatives=[catalog_entry_to_dict(entry) for entry in candidates],
    )


def annotate_function_result(
    function_obj: Dict[str, Any],
    routed_module: str,
) -> Dict[str, Any]:
    annotated = dict(function_obj)

    result = validate_function(
        function_name=annotated.get("name"),
        module_name=routed_module,
        class_name=annotated.get("class_name"),
    )

    annotated["catalog_validation"] = {
        "valid": result.valid,
        "reason": result.reason,
        "matched_entry": (
            catalog_entry_to_dict(result.matched_entry)
            if result.matched_entry is not None
            else None
        ),
        "alternatives": result.alternatives,
    }

    return annotated


def validate_function_list(
    functions: List[Dict[str, Any]],
    routed_module: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    annotated = [
        annotate_function_result(fn, routed_module=routed_module)
        for fn in functions
    ]

    valid_functions = [
        fn
        for fn in annotated
        if fn.get("catalog_validation", {}).get("valid") is True
    ]
    invalid_functions = [
        fn
        for fn in annotated
        if fn.get("catalog_validation", {}).get("valid") is not True
    ]

    return valid_functions, invalid_functions


def validate_lookup_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the full JSON-style payload returned by the lookup step.

    Supports:
    - SIMPLE results:
      {
        "query_type": "SIMPLE",
        "module": "...",
        "functions": [...]
      }

    - COMPLEX results:
      {
        "query_type": "COMPLEX",
        "sub_queries": [
          {
            "module": "...",
            "functions": [...]
          }
        ]
      }
    """
    result = dict(payload)

    if payload.get("query_type") == "SIMPLE":
        module = payload.get("module")
        functions = payload.get("functions", [])

        valid_functions, invalid_functions = validate_function_list(
            functions=functions,
            routed_module=module,
        )

        result["functions"] = valid_functions
        result["invalid_functions"] = invalid_functions
        result["all_functions_valid"] = len(invalid_functions) == 0
        return result

    sub_queries = []
    all_valid = True

    for sq in payload.get("sub_queries", []):
        sq_copy = dict(sq)
        module = sq.get("module")
        functions = sq.get("functions", [])

        valid_functions, invalid_functions = validate_function_list(
            functions=functions,
            routed_module=module,
        )

        sq_copy["functions"] = valid_functions
        sq_copy["invalid_functions"] = invalid_functions
        sq_copy["all_functions_valid"] = len(invalid_functions) == 0

        if invalid_functions:
            all_valid = False

        sub_queries.append(sq_copy)

    result["sub_queries"] = sub_queries
    result["all_functions_valid"] = all_valid
    return result


def validate_single_function_claim(
    name: str,
    module: str,
    class_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Useful when you only want to validate one claimed function.
    """
    result = validate_function(
        function_name=name,
        module_name=module,
        class_name=class_name,
    )

    return {
        "valid": result.valid,
        "reason": result.reason,
        "matched_entry": (
            catalog_entry_to_dict(result.matched_entry)
            if result.matched_entry is not None
            else None
        ),
        "alternatives": result.alternatives,
    }

