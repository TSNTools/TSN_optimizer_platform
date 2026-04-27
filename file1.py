from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader,
    PromptTemplate, StorageContext, load_index_from_storage
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.question_gen import LLMQuestionGenerator
import os, hashlib, json, torch, warnings

warnings.filterwarnings('ignore')


# ── Config ───────────────────────────────────────────────────────────────────

API             = "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"
PARAMETERS_FILE = "./parameters.md"
ALGORITHM_FILE  = "./some_functions.md"
PERSIST_DIR     = "./storage"
PARAMS_PERSIST  = "./storage_params"
ALGO_PERSIST    = "./storage_algo"


# ── LLM & Embeddings ─────────────────────────────────────────────────────────

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=API,
    temperature=0,
    # max_tokens=8192
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")


# ── Hash-based cache invalidation ────────────────────────────────────────────

HASH_FILE_PARAMS = os.path.join(PARAMS_PERSIST, "file_hashes.json")
HASH_FILE_ALGO   = os.path.join(ALGO_PERSIST,   "file_hashes.json")

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_saved_hash(hash_file: str) -> str:
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return json.load(f).get("hash", "")
    return ""

def save_hash(hash_file: str, file_hash: str):
    os.makedirs(os.path.dirname(hash_file), exist_ok=True)
    with open(hash_file, "w") as f:
        json.dump({"hash": file_hash}, f)

def build_or_load_index(source_file: str, persist_dir: str, hash_file: str):
    current_hash = compute_file_hash(source_file)
    index_exists = os.path.exists(persist_dir) and os.listdir(persist_dir)

    if index_exists and current_hash == load_saved_hash(hash_file):
        print(f"Loading index from {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)

    reason = "Building index..." if not index_exists else "File changed, rebuilding..."
    print(f"{reason} ({source_file})")

    reader = SimpleDirectoryReader(input_files=[source_file])
    data   = reader.load_data()
    parser = MarkdownNodeParser()
    nodes  = parser.get_nodes_from_documents(data)
    print(f"  Nodes created: {len(nodes)}")

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    save_hash(hash_file, current_hash)
    return index


# ── Build two separate indexes — one per file ────────────────────────────────
# Separating them means retrieval stays within the right knowledge domain.
# Parameter queries don't pull algorithm chunks and vice versa.

params_index = build_or_load_index(PARAMETERS_FILE, PARAMS_PERSIST, HASH_FILE_PARAMS)
algo_index   = build_or_load_index(ALGORITHM_FILE,  ALGO_PERSIST,   HASH_FILE_ALGO)


# ── Retriever + reranker (one per index) ─────────────────────────────────────

torch.manual_seed(42)

params_retriever = VectorIndexRetriever(index=params_index, similarity_top_k=10)
algo_retriever   = VectorIndexRetriever(index=algo_index,   similarity_top_k=40)

reranker_params = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=4
)
reranker_algo = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=10
)


# ── Prompt templates ─────────────────────────────────────────────────────────

PARAMS_PROMPT = PromptTemplate("""
You are a TSN parameter reference assistant.

Your job is to explain what the requested parameters mean, their types, valid ranges,
and how they are used in TSN optimization.

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

Your ONLY job is to find whether functions matching the request exist in the context.

Rules:
- Output the exact function name as it appears in the context. Do not paraphrase it.
- Output every matching function's parameters verbatim.
- Cite the source file and section for each function.
- If no function exists, say exactly: NO_FUNCTION_FOUND
- Do not generate new code. Do not suggest modifications. Just look up.

Context:
{context_str}

Request:
{query_str}

Answer (function name, parameters, source):
""")

GENERATOR_PROMPT = PromptTemplate("""
You are a TSN algorithm code generator. You MUST use helper functions from the provided context.

Rules:
- List every helper function you will use (exact name, parameters, source citation).
- Only after listing helpers, write the new function.
- The new function body may ONLY call functions that appear in the context or that you define inline.
- Add a docstring that explains inputs, outputs, and which helpers are used.
- If a required helper is missing from the context, say NOT_IN_CONTEXT: <what is missing> before generating.

Context:
{context_str}

Request:
{query_str}

Generated function:
""")


# ── Four query engines ───────────────────────────────────────────────────────

params_engine = RetrieverQueryEngine(
    retriever=params_retriever,
    node_postprocessors=[reranker_params]
)
params_engine.update_prompts({"response_synthesizer:text_qa_template": PARAMS_PROMPT})

lookup_engine = RetrieverQueryEngine(
    retriever=algo_retriever,
    node_postprocessors=[reranker_algo]
)
lookup_engine.update_prompts({"response_synthesizer:text_qa_template": LOOKUP_PROMPT})

generator_engine = RetrieverQueryEngine(
    retriever=algo_retriever,
    node_postprocessors=[reranker_algo]
)
generator_engine.update_prompts({"response_synthesizer:text_qa_template": GENERATOR_PROMPT})


# ── SubQuestionQueryEngine ───────────────────────────────────────────────────
# This sits between the agent and the two knowledge sources.
# Given a query it automatically decomposes it into sub-questions,
# routes each to the right engine, then synthesizes the answers into one response.
#
# Example decomposition for "compute paths using latency parameter":
#   Sub-Q 1 → params_engine:  "What is the latency parameter and its type?"
#   Sub-Q 2 → lookup_engine:  "Is there a function that computes paths using latency?"

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool(
            query_engine=params_engine,
            metadata=ToolMetadata(
                name="parameters",
                description=(
                    "Use this for any question about TSN parameter definitions, "
                    "types, valid ranges, or meaning. "
                    "Input: a question about one or more parameter names. "
                    "Output: precise parameter definitions from the parameters file."
                )
            )
        ),
        QueryEngineTool(
            query_engine=lookup_engine,
            metadata=ToolMetadata(
                name="algorithms",
                description=(
                    "Use this to find existing TSN functions or algorithms. "
                    "Input: a description of desired functionality. "
                    "Output: exact function names, parameters, and source citations, "
                    "or NO_FUNCTION_FOUND."
                )
            )
        ),
    ],
    question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),
    verbose=True,
)


# ── Agent tool functions ──────────────────────────────────────────────────────

def lookup_and_understand(query: str) -> str:
    """
    Decomposes the query into sub-questions, resolves parameter meanings
    and algorithm existence in parallel, then returns a unified answer.
    Use this FIRST for every query.
    """
    response = sub_question_engine.query(query)
    sources = "\n".join(
        f"[Source: {n.node.metadata.get('file_name', '?')} | "
        f"Section: {n.node.metadata.get('header_path', n.node.metadata.get('section', '?'))} | "
        f"Score: {f'{n.score:.3f}' if n.score is not None else 'N/A'}]"
        for n in response.source_nodes
    )
    return f"{response}\n\n--- Retrieved context ---\n{sources}"


def generate_function(query: str) -> str:
    """
    Generates a new TSN function grounded in helper functions from the codebase.
    Call this ONLY when lookup_and_understand returned NO_FUNCTION_FOUND or a PARTIAL match.
    Internally resolves parameter meanings and finds helper functions before generating.
    """
    # Resolve parameter meanings first
    param_context = params_engine.query(
        f"What parameters are relevant to: {query}"
    )
    # Find helper functions
    helper_context = lookup_engine.query(
        f"Find all helper functions relevant to: {query}"
    )
    augmented_query = (
        f"{query}\n\n"
        f"Relevant parameter definitions:\n{param_context}\n\n"
        f"Available helper functions:\n{helper_context}"
    )
    response = generator_engine.query(augmented_query)
    sources = "\n".join(
        f"[Source: {n.node.metadata.get('file_name', '?')} | "
        f"Section: {n.node.metadata.get('header_path', n.node.metadata.get('section', '?'))}| "
        f"Score: {f'{n.score:.3f}' if n.score is not None else 'N/A'}]"
        for n in response.source_nodes
    )
    return f"{response}\n\n--- Context sources ---\n{sources}"


# ── Agent tools ───────────────────────────────────────────────────────────────

lookup_tool = FunctionTool.from_defaults(
    fn=lookup_and_understand,
    name="lookup_and_understand",
    description=(
        "Use this tool FIRST for every query. "
        "Automatically resolves parameter meanings AND searches for existing functions. "
        "Input: the user's full query as-is. "
        "Output: parameter definitions + matching function names with citations, "
        "or NO_FUNCTION_FOUND."
    )
)

generate_tool = FunctionTool.from_defaults(
    fn=generate_function,
    name="generate_function",
    description=(
        "Use this tool ONLY when lookup_and_understand returned NO_FUNCTION_FOUND "
        "or a PARTIAL match. "
        "Generates a new function grounded in helper functions and parameter definitions. "
        "Input: the user's full query as-is. "
        "Output: new function code with helper citations."
    )
)


# ── ReAct agent ───────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """
You are an expert TSN (Time-Sensitive Networking) algorithm assistant.

You have two tools:
1. lookup_and_understand — always call this first.
   It resolves parameter meanings AND checks if a matching function exists.
2. generate_function — call this only if lookup_and_understand returned
   NO_FUNCTION_FOUND or a PARTIAL match.

Mandatory reasoning chain for every query:

STEP 1 — Call lookup_and_understand with the user's full query.
  If a function is found (EXACT MATCH):
    → Go to STEP 2 directly.
  If NO_FUNCTION_FOUND:
    → Call generate_function, then go to STEP 2.
  If PARTIAL MATCH:
    → Call generate_function for the missing part, then go to STEP 2.

STEP 2 — Format the final answer:
---
MATCH TYPE: EXACT | PARTIAL | GENERATED

PARAMETER CONTEXT:
  <relevant parameter definitions used>

FUNCTION(S):
  Name: <exact name as in context>
  Parameters: <verbatim from context>
  Source: <file, section>

[If GENERATED, also include:]
HELPER FUNCTIONS USED:
  - <name>(<params>) [Source: <file, section>]

CODE:
<function code>
---

Never make up function names. If a name is not in the retrieved context, say so explicitly.
"""

agent = ReActAgent.from_tools(
    tools=[lookup_tool, generate_tool],
    llm=Settings.llm,
    verbose=True,
    max_iterations=8,
    system_prompt=AGENT_SYSTEM_PROMPT,
)


# ── Run ───────────────────────────────────────────────────────────────────────

query = (
    "I need to generate a fat-tree topology with parameter k=4 and directed links." 
    "Which function provides this functionality?"
)

print("\n" + "="*60)
print("QUERY:", query)
print("="*60 + "\n")

response = agent.chat(query)
print("\n--- FINAL ANSWER ---")
print(response)

debug_response = lookup_engine.query(
    "generate a fat-tree topology with parameter k and directed links"
)
print("\n--- DEBUG: Raw retrieved chunks ---")
for i, node in enumerate(debug_response.source_nodes):
    print(f"\nChunk {i+1} | Score: {node.score:.3f}")
    print(f"Section: {node.node.metadata.get('section', '?')}")
    print(node.node.text[:300])

# ── Evaluation ────────────────────────────────────────────────────────────────

evaluator = CorrectnessEvaluator(llm=Settings.llm)

last_lookup  = lookup_and_understand(query)
context_text = last_lookup.split("--- Retrieved context ---")[-1] if "---" in last_lookup else last_lookup

correctness_result = evaluator.evaluate(
    query=query,
    response=str(response),
    contexts=[context_text]
)

print(f"\nCorrectness — Score: {correctness_result.score}")
print(f"Feedback: {correctness_result.feedback}")