"""
build_indexes.py

Builds the vector storage for the coding modules and parameter files.
"""

import os
import ast
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Config ───────────────────────────────────────────────────────────────────

PARAMETERS_FILE = "./parameters.md"

MODULE_FILES: Dict[str, str] = {
    "usecase_generator": "./usecase_generator_functions.json",
    "tsn_scheduling_problem": "./tsn_scheduling_problem_functions.json",
    "evaluator": "./evaluator_functions.json",
}

MODULE_PERSIST: Dict[str, str] = {
    "usecase_generator": "./storage_usecase",
    "tsn_scheduling_problem": "./storage_tsn",
    "evaluator": "./storage_evaluator",
}

PARAMS_PERSIST = "./storage_params"

# ── Parameter-explanation knowledge base ─────────────────────────────────────
# parameter_explanations.json is the source of truth for parameter meanings.
# Shape: { "<param_name>": {"meaning": "...", "used_for": "..."} }
# Indexed so the program RETRIEVES these explanations instead of generating them.
PARAM_EXPLANATIONS_FILE = "./parameter_explanations.json"
PARAM_EXPL_PERSIST = "./storage_param_explanations"

# schedule.py — code grounding for how parameters are actually used.
# (Disabled for now — uncomment to re-enable schedule.py indexing.)
# SCHEDULE_FILE = "./schedule.py"
# SCHEDULE_PERSIST = "./storage_schedule"

HASH_FILE_PARAM_EXPL = os.path.join(PARAM_EXPL_PERSIST, "file_hashes.json")
# HASH_FILE_SCHEDULE = os.path.join(SCHEDULE_PERSIST, "file_hashes.json")

HASH_FILE_PARAMS = os.path.join(PARAMS_PERSIST, "file_hashes.json")
MODULE_HASH_FILES: Dict[str, str] = {
    module: os.path.join(MODULE_PERSIST[module], "file_hashes.json")
    for module in MODULE_FILES
}

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"
)


# ── Hash helpers ─────────────────────────────────────────────────────────────

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_saved_hash(hash_file: str) -> str:
    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            return json.load(f).get("hash", "")
    return ""


def save_hash(hash_file: str, file_hash: str) -> None:
    os.makedirs(os.path.dirname(hash_file), exist_ok=True)
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump({"hash": file_hash}, f, indent=2)


def load_existing_index(
    persist_dir: str,
) -> Tuple[VectorStoreIndex, List[Any]]:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    nodes = list(storage_context.docstore.docs.values())
    return index, nodes


# ── Markdown index builder ───────────────────────────────────────────────────

def build_markdown_index(
    source_file: str,
    persist_dir: str,
    hash_file: str,
    module_label: str = "parameters",
) -> Tuple[VectorStoreIndex, List[Any]]:
    current_hash = compute_file_hash(source_file)
    index_exists = os.path.exists(persist_dir) and bool(os.listdir(persist_dir))

    if index_exists and current_hash == load_saved_hash(hash_file):
        print(f"Loading existing markdown index: {persist_dir}")
        return load_existing_index(persist_dir)

    print(f"Building markdown index from: {source_file}")

    reader = SimpleDirectoryReader(input_files=[source_file])
    documents = reader.load_data()

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    file_basename = os.path.basename(source_file)

    for node in nodes:
        existing_heading = (
            node.metadata.get("header_path")
            or node.metadata.get("section")
            or node.metadata.get("heading")
        )

        if not existing_heading:
            for line in node.text.splitlines()[:6]:
                if line.strip().startswith("##"):
                    existing_heading = line.strip().lstrip("#").strip()
                    break

        node.metadata["file_name"] = file_basename
        node.metadata["module"] = module_label
        node.metadata["section"] = existing_heading or "unknown"

    print(
        f"  Nodes created: {len(nodes)} | "
        f"file: {file_basename} | module: {module_label}"
    )

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    save_hash(hash_file, current_hash)
    return index, nodes


# ── JSON function-catalog index builder ──────────────────────────────────────

def _format_parameters(parameters: Any) -> str:
    if not isinstance(parameters, list):
        return "None"

    formatted = []
    for param in parameters:
        if not isinstance(param, dict):
            continue
        name = param.get("name")
        default = param.get("default")
        formatted.append(f"{name}={default}")
    return ", ".join(formatted) if formatted else "None"


def _build_function_text(item: dict, fallback_module_name: str) -> str:
    name = item.get("name", "")
    class_name = item.get("class_name", "")
    sub_class_name = item.get("sub_class_name", "")
    signature = item.get("signature", "")
    docstring = item.get("docstring", "")
    parameters = item.get("parameters", [])
    module_name = item.get("module", fallback_module_name)
    code = item.get("code", "")

    parts = [
        f"Function name: {name or 'None'}",
        f"Class name: {class_name or 'None'}",
        f"Sub class name: {sub_class_name or 'None'}",
        f"Signature: {signature or 'None'}",
        f"Docstring: {docstring or 'None'}",
        f"Parameters: {_format_parameters(parameters)}",
        f"Module: {module_name or fallback_module_name}",
        "Code:",
        code or "",
    ]
    return "\n".join(parts)


def _build_section_name(item: dict, index: int) -> str:
    name = item.get("name", "")
    class_name = item.get("class_name", "")
    sub_class_name = item.get("sub_class_name", "")

    section = " :: ".join(
        part for part in [class_name, sub_class_name, name] if part
    )
    return section or f"function_{index}"


def build_function_json_index(
    source_file: str,
    persist_dir: str,
    hash_file: str,
    module_label: str,
) -> Tuple[VectorStoreIndex, List[TextNode]]:
    current_hash = compute_file_hash(source_file)
    index_exists = os.path.exists(persist_dir) and bool(os.listdir(persist_dir))

    if index_exists and current_hash == load_saved_hash(hash_file):
        print(f"Loading existing JSON function index: {persist_dir}")
        return load_existing_index(persist_dir)

    print(f"Building JSON function index from: {source_file}")

    with open(source_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError(
            f"Expected top-level JSON array in {source_file}, "
            f"got {type(items).__name__}"
        )

    file_basename = os.path.basename(source_file)
    nodes: List[TextNode] = []

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        text = _build_function_text(item, fallback_module_name=file_basename)
        section = _build_section_name(item, i)

        node = TextNode(
            text=text,
            metadata={
                "file_name": file_basename,
                "module": module_label,
                "source_module": item.get("module", ""),
                "function_name": item.get("name", ""),
                "class_name": item.get("class_name", ""),
                "sub_class_name": item.get("sub_class_name", ""),
                "signature": item.get("signature", ""),
                "section": section,
            },
        )
        nodes.append(node)

    print(
        f"  Nodes created: {len(nodes)} | "
        f"file: {file_basename} | module: {module_label}"
    )

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    save_hash(hash_file, current_hash)
    return index, nodes


# ── Parameter-explanations index builder ─────────────────────────────────────
# parameter_explanations.json is a dict { "<param_name>": {"meaning", "used_for"} }.
# One node per parameter, so a semantic query ("what controls the slot size?")
# retrieves the right parameter even without its exact name. The stored meaning /
# used-for text is returned AS-IS — this is the knowledge base, not a prompt.

def _build_param_explanation_text(param_name: str, info: Any) -> str:
    if not isinstance(info, dict):
        return f"Parameter: {param_name}\nMeaning: {info or 'None'}"

    meaning = info.get("meaning", "")
    used_for = info.get("used_for", "")
    parts = [
        f"Parameter: {param_name}",
        f"Meaning: {meaning or 'None'}",
        f"Used for: {used_for or 'None'}",
    ]
    # Preserve any extra keys the file may carry (units, default, range, ...).
    for key, value in info.items():
        if key in ("meaning", "used_for"):
            continue
        parts.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    return "\n".join(parts)


def build_param_explanations_index(
    source_file: str,
    persist_dir: str,
    hash_file: str,
    module_label: str = "parameters",
) -> Tuple[VectorStoreIndex, List[TextNode]]:
    current_hash = compute_file_hash(source_file)
    index_exists = os.path.exists(persist_dir) and bool(os.listdir(persist_dir))

    if index_exists and current_hash == load_saved_hash(hash_file):
        print(f"Loading existing parameter-explanations index: {persist_dir}")
        return load_existing_index(persist_dir)

    print(f"Building parameter-explanations index from: {source_file}")

    with open(source_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept the documented dict shape OR a list of {"name": ..., ...} objects.
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [
            (d.get("name", f"param_{i}"), d)
            for i, d in enumerate(data)
            if isinstance(d, dict)
        ]
    else:
        raise ValueError(
            f"Expected a JSON object or array in {source_file}, "
            f"got {type(data).__name__}"
        )

    file_basename = os.path.basename(source_file)
    nodes: List[TextNode] = []

    for param_name, info in items:
        nodes.append(
            TextNode(
                text=_build_param_explanation_text(param_name, info),
                metadata={
                    "file_name": file_basename,
                    "module": module_label,
                    "parameter_name": param_name,
                    "section": param_name,
                },
            )
        )

    print(
        f"  Nodes created: {len(nodes)} | "
        f"file: {file_basename} | module: {module_label}"
    )

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    save_hash(hash_file, current_hash)
    return index, nodes


# ── Python source index builder (AST) ────────────────────────────────────────
# Turns a raw .py file (e.g. schedule.py) into one node per function/method, with
# the same metadata shape as the JSON function catalogs (function_name, signature,
# class_name, section) so the existing retrieval path can read it identically.

def _signature_from_func(node: ast.AST) -> str:
    name = getattr(node, "name", "")
    try:
        return f"{name}({ast.unparse(node.args)})"   # Python 3.9+
    except Exception:
        return f"{name}(...)"


def _collect_py_functions(source: str) -> List[dict]:
    """Return [{name, class_name, signature, docstring, code}] for top-level
    functions and class methods (nested inner functions are skipped)."""
    tree = ast.parse(source)
    found: List[dict] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)          # descend to capture methods
            self.class_stack.pop()

        def _handle(self, node: ast.AST) -> None:
            found.append({
                "name": node.name,
                "class_name": self.class_stack[-1] if self.class_stack else "",
                "signature": _signature_from_func(node),
                "docstring": ast.get_docstring(node) or "",
                "code": ast.get_source_segment(source, node) or "",
            })
            # Do NOT generic_visit here: skip nested inner functions.

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._handle(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._handle(node)

    _Visitor().visit(tree)
    return found


def _build_py_function_text(item: dict, module_name: str) -> str:
    parts = [
        f"Function name: {item.get('name') or 'None'}",
        f"Class name: {item.get('class_name') or 'None'}",
        f"Signature: {item.get('signature') or 'None'}",
        f"Docstring: {item.get('docstring') or 'None'}",
        f"Module: {module_name}",
        "Code:",
        item.get("code", ""),
    ]
    return "\n".join(parts)


def build_python_source_index(
    source_file: str,
    persist_dir: str,
    hash_file: str,
    module_label: str,
) -> Tuple[VectorStoreIndex, List[TextNode]]:
    current_hash = compute_file_hash(source_file)
    index_exists = os.path.exists(persist_dir) and bool(os.listdir(persist_dir))

    if index_exists and current_hash == load_saved_hash(hash_file):
        print(f"Loading existing python-source index: {persist_dir}")
        return load_existing_index(persist_dir)

    print(f"Building python-source index from: {source_file}")

    with open(source_file, "r", encoding="utf-8") as f:
        source = f.read()

    file_basename = os.path.basename(source_file)
    functions = _collect_py_functions(source)
    nodes: List[TextNode] = []

    for i, item in enumerate(functions):
        section = " :: ".join(p for p in [item["class_name"], item["name"]] if p)
        nodes.append(
            TextNode(
                text=_build_py_function_text(item, module_label),
                metadata={
                    "file_name": file_basename,
                    "module": module_label,
                    "source_module": module_label,
                    "function_name": item["name"],
                    "class_name": item["class_name"],
                    "signature": item["signature"],
                    "section": section or f"function_{i}",
                },
            )
        )

    # Fallback: a script with no defs still gets indexed as one whole-file node.
    if not nodes:
        nodes.append(
            TextNode(
                text=source,
                metadata={
                    "file_name": file_basename,
                    "module": module_label,
                    "source_module": module_label,
                    "function_name": "",
                    "section": file_basename,
                },
            )
        )

    print(
        f"  Nodes created: {len(nodes)} | "
        f"file: {file_basename} | module: {module_label}"
    )

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    save_hash(hash_file, current_hash)
    return index, nodes


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Building parameters index ===")
    build_markdown_index(
        source_file=PARAMETERS_FILE,
        persist_dir=PARAMS_PERSIST,
        hash_file=HASH_FILE_PARAMS,
        module_label="parameters",
    )

    print("\n=== Building parameter-explanations knowledge base ===")
    if os.path.exists(PARAM_EXPLANATIONS_FILE):
        build_param_explanations_index(
            source_file=PARAM_EXPLANATIONS_FILE,
            persist_dir=PARAM_EXPL_PERSIST,
            hash_file=HASH_FILE_PARAM_EXPL,
            module_label="parameters",
        )
    else:
        print(f"  [skip] {PARAM_EXPLANATIONS_FILE} not found.")

    # print("\n=== Building schedule.py source index ===")
    # if os.path.exists(SCHEDULE_FILE):
    #     build_python_source_index(
    #         source_file=SCHEDULE_FILE,
    #         persist_dir=SCHEDULE_PERSIST,
    #         hash_file=HASH_FILE_SCHEDULE,
    #         module_label="schedule",
    #     )
    # else:
    #     print(f"  [skip] {SCHEDULE_FILE} not found.")

    print("\n=== Building function-catalog indexes ===")
    for module_label, source_file in MODULE_FILES.items():
        build_function_json_index(
            source_file=source_file,
            persist_dir=MODULE_PERSIST[module_label],
            hash_file=MODULE_HASH_FILES[module_label],
            module_label=module_label,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
