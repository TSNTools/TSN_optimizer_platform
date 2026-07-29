"""
llm_config.py
=============
Single source of truth for the LLM endpoint used by the TSN agent.

Every module in the agent calls `Settings.llm.complete(...)`. This module sets
that singleton once, in one place, so the endpoint — a local llama.cpp server,
an SSH-tunnelled remote, or a direct Grace Hopper IP — is configured by
environment variables instead of being pinned inside file1.py.

DEFAULT: local llama.cpp server (./serve.sh) on http://127.0.0.1:8080/v1.

USAGE
-----
At the top of your entry-point module (backup_script_1.py), simply:

    import llm_config            # configures Settings.llm on import

The other modules (feedback_loop.py, spec_test_engine.py,
unit_test_generator.py) need no change — they already use Settings.llm — but you
may add the same `import llm_config` line to each as a safety net so the LLM is
configured no matter which module is imported first.

To run the agent against the remote Grace Hopper server, just use the launcher,
which sets the env vars below for you and health-checks the endpoint first:

    ./run_gracehopper.sh

SWITCHING LOCAL <-> REMOTE
--------------------------
The only thing that differs between modes is the base URL (and model id):

    # LOCAL llama.cpp on this machine (start it with:  ./serve.sh)
    #   -> leave everything unset; the defaults below point at 127.0.0.1:8080.

    # REMOTE — Grace Hopper (GH200) vLLM server @ 10.1.65.168:8000.
    # NOTE ON THE MODEL ID: the server runs
    #   hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
    # but was launched with `--served-model-name llama-3.1-70b-awq`, so it
    # advertises (and only answers to) the short alias below. Sending the full
    # Hugging Face repo id returns a 404 "model does not exist".

    # REMOTE direct (port reachable on the LAN, no firewall):
    export TSN_LLM_BASE="http://10.1.65.168:8000/v1"
    export TSN_LLM_MODEL="llama-3.1-70b-awq"

    # REMOTE via SSH tunnel  (ssh -N -L 8000:localhost:8000 admin@10.1.65.168):
    export TSN_LLM_BASE="http://127.0.0.1:8000/v1"
    export TSN_LLM_MODEL="llama-3.1-70b-awq"

HEALTH CHECK
------------
Run this file directly to verify the endpoint before launching the agent:

    python llm_config.py
"""

from __future__ import annotations

import os

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike


# ── Configuration (env-driven, with sensible defaults) ────────────────────────

# Base URL of the OpenAI-compatible server.
#   - Local llama.cpp (./serve.sh):  http://127.0.0.1:8080/v1   (default)
#   - Remote SSH tunnel / direct:    set TSN_LLM_BASE (see docstring above)
LLM_BASE_URL: str = os.environ.get("TSN_LLM_BASE", "http://127.0.0.1:8080/v1")

# Model id the server advertises.
#   - Local llama.cpp (default): the --alias set in serve.sh -> "qwen3.5-27b".
#   - Remote Grace Hopper vLLM:  the --served-model-name -> "llama-3.1-70b-awq"
#     (set via TSN_LLM_MODEL, e.g. by ./run_gracehopper.sh). The full HF repo id
#     is NOT accepted by that server; use the short alias.
LLM_MODEL: str = os.environ.get(
    "TSN_LLM_MODEL",
    "qwen3.5-27b",
)

# vLLM ignores the key, but the OpenAI client requires a non-empty string.
LLM_API_KEY: str = os.environ.get("TSN_LLM_API_KEY", "not-needed")

# Generation cap. NOT 200 — the agent's prompts return JSON arrays / full test
# bodies and will be silently truncated (-> JSONDecodeError) if this is small.
LLM_MAX_TOKENS: int = int(os.environ.get("TSN_LLM_MAX_TOKENS", "3000"))

# Must match the server's --max-model-len. If prompt + max_tokens exceeds the
# server's real limit, vLLM returns a 400 rather than truncating.
LLM_CONTEXT_WINDOW: int = int(os.environ.get("TSN_LLM_CONTEXT_WINDOW", "16384"))

# Per-request timeout (seconds). 70B AWQ over a tunnel can be slow to first
# token on long prompts; keep this generous.
LLM_TIMEOUT: float = float(os.environ.get("TSN_LLM_TIMEOUT", "600"))

# Deterministic output for parsing-heavy prompts.
LLM_TEMPERATURE: float = float(os.environ.get("TSN_LLM_TEMPERATURE", "0"))

# Optional system prompt applied to chat-mode calls.
LLM_SYSTEM_PROMPT: str = os.environ.get(
    "TSN_LLM_SYSTEM_PROMPT",
    "You are an expert Python programmer for a TSN (Time-Sensitive Networking) codebase.",
)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_llm(**overrides) -> OpenAILike:
    """
    Construct the OpenAILike LLM from the module-level config. Any keyword
    override (e.g. build_llm(temperature=0.2)) wins over the env/default value.
    Construction does no network I/O — the first request hits the server.
    """
    kwargs = dict(
        model=LLM_MODEL,
        api_base=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        is_chat_model=True,          # routes .complete() through /v1/chat/completions
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        context_window=LLM_CONTEXT_WINDOW,
        timeout=LLM_TIMEOUT,
        system_prompt=LLM_SYSTEM_PROMPT,
    )
    kwargs.update(overrides)
    return OpenAILike(**kwargs)


_LLM: "OpenAILike | None" = None  # our own record of what we configured


def configure(force: bool = False, **overrides) -> OpenAILike:
    """
    Set Settings.llm to the configured OpenAILike instance and return it.

    Idempotent: builds the LLM once and reuses it on subsequent calls. Pass
    force=True (or any overrides) to rebuild.

    NOTE: we deliberately do NOT read Settings.llm to decide whether to
    configure. In llama_index, Settings.llm is a lazy property that, when
    unset, tries to construct a default OpenAI model and raises if there is no
    OPENAI_API_KEY. We track state with our own _LLM reference and only ever
    *assign* Settings.llm.
    """
    global _LLM
    if _LLM is None or force or overrides:
        _LLM = build_llm(**overrides)
        Settings.llm = _LLM
    return _LLM


# Configure on import so a bare `import llm_config` is enough for every module.
LLM: OpenAILike = configure()


# ── Health check ────────────────────────────────────────────────────────────

def check_connection(prompt: str = "Reply with the single word: pong",
                     max_tokens: int = 8) -> tuple[bool, str]:
    """
    Send a tiny completion to verify the endpoint is reachable and responding.
    Returns (ok, message). Use before launching the full agent to confirm the
    tunnel / direct connection is live.
    """
    try:
        text = LLM.complete(prompt, max_tokens=max_tokens).text.strip()
        return True, text or "(empty response)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def describe() -> str:
    """Human-readable summary of the active configuration (no secrets)."""
    return (
        "TSN LLM configuration\n"
        f"  base_url       : {LLM_BASE_URL}\n"
        f"  model          : {LLM_MODEL}\n"
        f"  max_tokens     : {LLM_MAX_TOKENS}\n"
        f"  context_window : {LLM_CONTEXT_WINDOW}\n"
        f"  timeout        : {LLM_TIMEOUT}s\n"
        f"  temperature    : {LLM_TEMPERATURE}\n"
    )


if __name__ == "__main__":
    print(describe())
    print("Pinging endpoint ...")
    ok, msg = check_connection()
    print(f"  {'OK  ->' if ok else 'FAIL ->'} {msg}")
    raise SystemExit(0 if ok else 1)