import os
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Building parameters index ===")
    build_markdown_index(
        source_file=PARAMETERS_FILE,
        persist_dir=PARAMS_PERSIST,
        hash_file=HASH_FILE_PARAMS,
        module_label="parameters",
    )

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