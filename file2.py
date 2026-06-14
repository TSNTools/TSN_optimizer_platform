from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os
import hashlib
import json
import torch
import warnings

warnings.filterwarnings('ignore')


API = "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"
PARAMETERS_FILE = "./parameters.md"
ALGORITHM_FILE = "./some_functions.md"
PERSIST_DIR = "./storage"


# Groq LLM
Settings.llm = Groq(
    model="openai/gpt-oss-20b",
    api_key=API,
    temperature=0,
    # max_tokens=2048
)

# Storing hash of source files. This will compare the current file hashes to the saved ones.
# This way each time a modification is made to the source files then chunking and embedding
# will be done again. 

HASH_FILE = os.path.join(PERSIST_DIR, "file_hashes.json")

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def get_current_hashes() -> dict:
    return {
        PARAMETERS_FILE: compute_file_hash(PARAMETERS_FILE),
        ALGORITHM_FILE: compute_file_hash(ALGORITHM_FILE),
    }

def load_saved_hashes() -> dict:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hashes: dict):
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f)

def files_have_changed() -> bool:
    return get_current_hashes() != load_saved_hashes()

# Embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

index_exists = os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)

if index_exists and not files_have_changed():
    # Nothing changed — load from disk
    print("Loading index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    # Files changed or first run — rebuild
    reason = "Building index from scratch..." if not index_exists else "Files changed, rebuilding index..."
    print(reason)

    reader = SimpleDirectoryReader(input_files=[PARAMETERS_FILE, ALGORITHM_FILE])
    data = reader.load_data()

    # chunking
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(data)
    print(f"Total nodes created: {len(nodes)}")

    # embedding and storage
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    # Save new hashes after successful build
    save_hashes(get_current_hashes())

# Retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=20
)

torch.manual_seed(42)
# Reranking (post-processing)
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=8
)

# System Prompt
qa_prompt = PromptTemplate(
    """
    You are an assistant that helps in constructing and testing TSN optimization algorithms.

    STRICT RULES:
    - You MUST use ONLY the provided context.
    - Every function name and parameter you mention MUST exactly match the context.
    - If something is missing from the context, you MUST explicitly say so.
    - You MUST follow the step-by-step methodology below EXACTLY and in order.
    - You MUST NOT skip any step.
    - You MUST NOT merge steps.

    --------------------------------------------------
    ANSWERING METHODOLOGY (MANDATORY)
    --------------------------------------------------

    STEP 1 (INTERNAL ONLY — DO NOT OUTPUT):
    Expand the user's request into:
    - required inputs
    - expected outputs
    - constraints (latency, scheduling, etc.)
    - relevant TSN concepts

    If the request is too vague to map to a function:
    Mark it as ambiguous internally and proceed, but reflect this in STEP 2.

    --------------------------------------------------

    STEP 2: FUNCTION EXISTENCE CHECK

    Start this step with EXACTLY one of the following:

    YES — a function exists
    PARTIAL — function exists but does not fully satisfy the request
    NO — no function exists

    Then:

    - If YES:
    List ALL matching functions with:
    Function: <exact name>
    Parameters: <exact parameters>

    - If PARTIAL:
    List the closest function(s):
    Function: <exact name>
    Parameters: <exact parameters>
    Covers: <what it does>
    Missing: <what is not covered>

    - If NO:
    Explicitly state:
    "There is no function in the context that provides this functionality."

    --------------------------------------------------

    STEP 3: USE EXISTING FUNCTION (ONLY if YES or PARTIAL)

    - You MUST reuse ONLY the exact function names from STEP 2.
    - You MUST restate the function name EXACTLY before using it.
    - If you accidentally change the function name, STOP and correct yourself.

    For YES:
    Provide:
    Function: <exact name>
    Parameters: <parameters>
    Example Usage:
    <code snippet using ONLY context-defined functions>

    For PARTIAL:
    Provide:
    Function: <exact name>
    Covers: <what it handles>
    Missing: <gap>

    Then continue to STEP 4 for the missing part.

    --------------------------------------------------

    STEP 4: HANDLE MISSING FUNCTIONALITY (ONLY if PARTIAL or NO)

    If NO or PARTIAL:

    1. Identify HELPER FUNCTIONS from the context:
    For each:
    Function: <exact name>
    Parameters: <parameters>
    Purpose: <one sentence>
    Cite each using:
    [Source: <source_file>, Section: <section_name>]

    2. Then you MUST generate a new function:

    GENERATED FUNCTION:
    Name: <name>
    Parameters: <typed parameters>
    Uses: <which helper functions and why>
    Code:
    <implementation using ONLY helper functions from context>

    --------------------------------------------------

    FINAL RULE:
    - If u miss any of the above mentioned steps then your answer will be incorrect.
    - If ANY required information is missing from the context, include:

    NOT IN CONTEXT:
    <explicitly state what is missing>

    --------------------------------------------------

    Context:
    {context_str}

    Question:
    {query_str}

    Answer:
    """
)

# Response generation
response_synthesizer = get_response_synthesizer(text_qa_template=qa_prompt)
base_query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[reranker]
)

query = ("I want to distribute streams across the network to avoid link congestion while respecting "
"shortest path constraints. Determine whether such functionality exists. "
"If not, generate a function that implements this algorithm.")

response = base_query_engine.query(query)
print(response)

# # Inspect retrieved sources
# for source_node in response.source_nodes:
#     print(f"---- {source_node.node.metadata.get('source', 'Unknown')}")
#     print(f"Preview: {source_node.node.text[:100]}...")
#     print(f"Score: {source_node.score:.3f}")

# evaluation
from llama_index.core.evaluation import CorrectnessEvaluator

evaluator = CorrectnessEvaluator(llm=Settings.llm)
correctness_result = evaluator.evaluate(
    query=query,
    response=response.response,
    contexts=[node.node.text for node in response.source_nodes]
)

print(f"Correctness — Score: {correctness_result.score}")
print(f"Feedback: {correctness_result.feedback}\n")
