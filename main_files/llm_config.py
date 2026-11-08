"""
llm_config.py
=============
Single source of truth for the LLM endpoint used by the TSN agent.

Every module in the agent calls `Settings.llm.complete(...)`. This module sets
that singleton once, in one place, so the backend — a local llama.cpp server,
an SSH-tunnelled remote, a direct Grace Hopper IP, or the hosted Anthropic API
— is configured by environment variables instead of being pinned inside
file1.py.

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

    # REMOTE — Grace Hopper (GH200) vLLM server @ <grace-hopper-host>:8000.
    # NOTE ON THE MODEL ID: the server runs
    #   hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
    # but was launched with `--served-model-name llama-3.1-70b-awq`, so it
    # advertises (and only answers to) the short alias below. Sending the full
    # Hugging Face repo id returns a 404 "model does not exist".

    # REMOTE direct (port reachable on the LAN, no firewall):
    export TSN_LLM_BASE="http://<grace-hopper-host>:8000/v1"
    export TSN_LLM_MODEL="llama-3.1-70b-awq"

    # REMOTE via SSH tunnel  (ssh -N -L 8000:localhost:8000 admin@<grace-hopper-host>):
    export TSN_LLM_BASE="http://127.0.0.1:8000/v1"
    export TSN_LLM_MODEL="llama-3.1-70b-awq"

SWITCHING TO CLAUDE (hosted Anthropic API)
------------------------------------------
Set TSN_LLM_PROVIDER=anthropic. Everything else in the agent is unchanged —
the modules still call Settings.llm.complete(), they just reach a different
backend. Use the launcher, which sets the vars and health-checks first:

    ./run_claude.sh

The Claude branch deliberately does NOT reuse TSN_LLM_MAX_TOKENS or
TSN_LLM_CONTEXT_WINDOW; those are sized for the 16k-context vLLM server and
would cripple / truncate a 1M-context Claude model. See the notes on
CLAUDE_MAX_TOKENS below.

HEALTH CHECK
------------
Run this file directly to verify the endpoint before launching the agent:

    python llm_config.py
"""

from __future__ import annotations

import os

from llama_index.core import Settings
from llama_index.core.llms import LLM as _LLMBase  # aliased: `LLM` below is the instance
from llama_index.llms.openai_like import OpenAILike


# ── Configuration (env-driven, with sensible defaults) ────────────────────────

# Which backend to build. "openai_like" (default) preserves the existing
# behaviour exactly — local llama.cpp or the remote vLLM server. "anthropic"
# swaps in the hosted Claude API. Nothing else in the agent changes: every
# module still calls Settings.llm.complete().
LLM_PROVIDER: str = os.environ.get("TSN_LLM_PROVIDER", "openai_like").lower()

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

# Output cap for the algorithm pathway's design and generation stages (configuration,
# blueprint, interface, spec, implementation, repair). Their answers are long - a Luna
# blueprint is ~8k tokens, a full implementation ~10k - and the old per-stage caps
# (3500-7000) cut them off. For openai_like the pipeline clips this to what the context
# window leaves after the prompt (stage_max_tokens below), since vLLM 400s otherwise.
STAGE_MAX_TOKENS: int = int(os.environ.get("TSN_STAGE_MAX_TOKENS", "32768"))

# Must match the server's --max-model-len. If prompt + max_tokens exceeds the
# server's real limit, vLLM returns a 400 rather than truncating.
LLM_CONTEXT_WINDOW: int = int(os.environ.get("TSN_LLM_CONTEXT_WINDOW", "98304"))

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


# ── Claude / Anthropic configuration (only read when TSN_LLM_PROVIDER=anthropic)

# Model id. claude-opus-5 is the strongest for agentic work; claude-sonnet-5 is
# cheaper and claude-haiku-4-5 cheaper still — use one of those for smoke tests
# before spending Opus tokens on a full run.
CLAUDE_MODEL: str = os.environ.get("TSN_CLAUDE_MODEL", "claude-opus-5")

# NOT the same knob as LLM_MAX_TOKENS (3000, sized for the vLLM server).
# On Claude 5-family models extended thinking is on by default and max_tokens
# caps thinking + answer *together*, so a 3000 budget can be spent thinking and
# leave the JSON array / test body truncated -> JSONDecodeError, the same
# failure mode the LLM_MAX_TOKENS comment above warns about. Keep this roomy.
CLAUDE_MAX_TOKENS: int = int(os.environ.get("TSN_CLAUDE_MAX_TOKENS", "16000"))

# Left as None on purpose: the anthropic SDK falls back to the ANTHROPIC_API_KEY
# environment variable, so the key never has to be handed around in code.
CLAUDE_API_KEY: "str | None" = os.environ.get("ANTHROPIC_API_KEY") or None

# Set TSN_CLAUDE_THINKING=disabled to turn extended thinking off — a real
# cost/latency lever given the agent makes ~60 .complete() calls per run.
# Anything else leaves thinking at the model default (adaptive).
CLAUDE_THINKING: str = os.environ.get("TSN_CLAUDE_THINKING", "default").lower()


# ── OpenAI configuration (only read when TSN_LLM_PROVIDER=openai) ─────────────

# Model id. Must be one your account can actually see — run
#   python main_files/llm_config.py --list-models
# to print exactly what this key has access to. A wrong id fails with a 404.
OPENAI_MODEL: str = os.environ.get("TSN_OPENAI_MODEL", "gpt-5.6-luna")

# Left as None on purpose: the openai SDK falls back to the OPENAI_API_KEY
# environment variable. Deliberately NOT falling back to TSN_LLM_API_KEY, whose
# default is the placeholder "not-needed" — that would send junk and 401.
OPENAI_API_KEY: "str | None" = os.environ.get("OPENAI_API_KEY") or None

# Same reasoning as CLAUDE_MAX_TOKENS, not LLM_MAX_TOKENS: on the reasoning
# models this budget covers reasoning tokens *and* the visible answer (it is
# sent as max_completion_tokens), so a 3000 cap can be spent reasoning and leave
# the JSON array / test body truncated -> JSONDecodeError.
OPENAI_MAX_TOKENS: int = int(os.environ.get("TSN_OPENAI_MAX_TOKENS", "16000"))

# Cost/quality lever for the reasoning models (gpt-5 family, o-series), the
# rough analogue of TSN_CLAUDE_THINKING. Ignored by non-reasoning models.
# One of: none, minimal, low, medium, high, xhigh. Unset = provider default.
OPENAI_REASONING_EFFORT: "str | None" = (
    os.environ.get("TSN_OPENAI_REASONING_EFFORT") or None
)


# ── Builder ───────────────────────────────────────────────────────────────────

def _build_openai_like(**overrides) -> OpenAILike:
    """The original backend: any OpenAI-compatible server (llama.cpp / vLLM)."""
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


def _build_anthropic(**overrides) -> _LLMBase:
    """
    The hosted Claude API. Imported lazily so that
    `pip install llama-index-llms-anthropic` is only required by people who
    actually set TSN_LLM_PROVIDER=anthropic — the local/vLLM path keeps working
    on an environment that has never heard of Claude.
    """
    try:
        from llama_index.llms.anthropic import Anthropic
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "TSN_LLM_PROVIDER=anthropic needs the Anthropic integration:\n"
            "    pip install llama-index-llms-anthropic"
        ) from e

    kwargs = dict(
        model=CLAUDE_MODEL,
        api_key=CLAUDE_API_KEY,      # None -> SDK reads ANTHROPIC_API_KEY
        max_tokens=CLAUDE_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
        system_prompt=LLM_SYSTEM_PROMPT,
        # Claude 5-family models reject `temperature` with a 400; the
        # integration strips it for exactly those ids and honours it on the
        # older ones, so passing our 0 is safe either way.
        temperature=LLM_TEMPERATURE,
        # context_window is deliberately NOT passed: the integration derives the
        # real window from the model id (1M for the 5-family). Forcing our
        # 16384 here would throw away almost all of it.
    )
    if CLAUDE_THINKING == "disabled":
        kwargs["thinking_dict"] = {"type": "disabled"}
    kwargs.update(overrides)
    return Anthropic(**kwargs)


def _build_openai(**overrides) -> _LLMBase:
    """
    The hosted OpenAI API, via the dedicated integration rather than the generic
    OpenAILike used for llama.cpp/vLLM. That matters: the dedicated class knows
    the model catalogue, so it resolves the real context window from the model
    id and, on the reasoning models (gpt-5 family, o-series), automatically
    forces temperature to 1 and sends max_completion_tokens instead of
    max_tokens. Pointing OpenAILike at api.openai.com would work but would leave
    our temperature=0 in place and 400 on exactly those models.
    """
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "TSN_LLM_PROVIDER=openai needs the OpenAI integration:\n"
            "    pip install llama-index-llms-openai"
        ) from e

    kwargs = dict(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,      # None -> SDK reads OPENAI_API_KEY
        max_tokens=OPENAI_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
        system_prompt=LLM_SYSTEM_PROMPT,
        temperature=LLM_TEMPERATURE,  # overridden to 1.0 on reasoning models
        # context_window is deliberately NOT passed: resolved from the model id.
    )
    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT
    kwargs.update(overrides)
    return OpenAI(**kwargs)


def stage_max_tokens(prompt: str) -> int:
    """STAGE_MAX_TOKENS, clipped for a self-hosted server to the room the context window
    leaves after this prompt. The prompt is estimated at 3.5 characters a token (these
    code-heavy prompts measure ~4), so the estimate errs towards a smaller budget rather
    than a request the server rejects. The hosted APIs resolve their own, far larger
    windows and are not clipped."""
    if LLM_PROVIDER != "openai_like":
        return STAGE_MAX_TOKENS
    room = LLM_CONTEXT_WINDOW - int(len(prompt) / 3.5) - 1024
    return max(1024, min(STAGE_MAX_TOKENS, room))


def build_llm(**overrides) -> _LLMBase:
    """
    Construct the LLM selected by TSN_LLM_PROVIDER from the module-level config.
    Any keyword override (e.g. build_llm(temperature=0.2)) wins over the
    env/default value. Construction does no network I/O — the first request
    hits the server.
    """
    if LLM_PROVIDER == "anthropic":
        return _build_anthropic(**overrides)
    if LLM_PROVIDER == "openai":
        return _build_openai(**overrides)
    if LLM_PROVIDER == "openai_like":
        return _build_openai_like(**overrides)
    raise ValueError(
        f"Unknown TSN_LLM_PROVIDER={LLM_PROVIDER!r}; expected "
        "'openai_like' (default), 'anthropic', or 'openai'."
    )


_LLM: "_LLMBase | None" = None  # our own record of what we configured


def configure(force: bool = False, **overrides) -> _LLMBase:
    """
    Set Settings.llm to the configured LLM instance and return it.

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
LLM: _LLMBase = configure()


# ── Health check ────────────────────────────────────────────────────────────

def check_connection(prompt: str = "Reply with the single word: pong",
                     max_tokens: "int | None" = None) -> tuple[bool, str]:
    """
    Send a tiny completion to verify the endpoint is reachable and responding.
    Returns (ok, message). Use before launching the full agent to confirm the
    tunnel / direct connection is live.

    max_tokens defaults per provider: 8 is plenty for a raw OpenAI-compatible
    server, but on Claude (thinking) and on OpenAI's reasoning models the budget
    covers the hidden reasoning too, so an 8-token cap would be spent before any
    text and report a misleading "(empty response)" instead of a real failure.
    """
    if max_tokens is None:
        max_tokens = 2048 if LLM_PROVIDER in ("anthropic", "openai") else 8
    try:
        text = LLM.complete(prompt, max_tokens=max_tokens).text.strip()
        return True, text or "(empty response)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def describe() -> str:
    """Human-readable summary of the active configuration (no secrets)."""
    if LLM_PROVIDER == "anthropic":
        thinking = "disabled" if CLAUDE_THINKING == "disabled" else "model default"
        return (
            "TSN LLM configuration\n"
            f"  provider       : anthropic (hosted Claude API)\n"
            f"  model          : {CLAUDE_MODEL}\n"
            f"  max_tokens     : {CLAUDE_MAX_TOKENS}\n"
            f"  context_window : from model id\n"
            f"  thinking       : {thinking}\n"
            f"  timeout        : {LLM_TIMEOUT}s\n"
            f"  api key        : {'ANTHROPIC_API_KEY set' if CLAUDE_API_KEY else 'NOT SET'}\n"
        )
    if LLM_PROVIDER == "openai":
        return (
            "TSN LLM configuration\n"
            f"  provider       : openai (hosted OpenAI API)\n"
            f"  model          : {OPENAI_MODEL}\n"
            f"  max_tokens     : {OPENAI_MAX_TOKENS}\n"
            f"  context_window : from model id\n"
            f"  reasoning      : {OPENAI_REASONING_EFFORT or 'provider default'}\n"
            f"  timeout        : {LLM_TIMEOUT}s\n"
            f"  api key        : {'OPENAI_API_KEY set' if OPENAI_API_KEY else 'NOT SET'}\n"
        )
    return (
        "TSN LLM configuration\n"
        f"  provider       : openai_like\n"
        f"  base_url       : {LLM_BASE_URL}\n"
        f"  model          : {LLM_MODEL}\n"
        f"  max_tokens     : {LLM_MAX_TOKENS}\n"
        f"  context_window : {LLM_CONTEXT_WINDOW}\n"
        f"  timeout        : {LLM_TIMEOUT}s\n"
        f"  temperature    : {LLM_TEMPERATURE}\n"
    )


def list_models() -> list[str]:
    """
    Ask the provider which models this key can actually see. The single most
    useful thing to run when a health check fails with a 404 — model catalogues
    change and differ per account, so guessing an id is not worth the round trip.
    """
    if LLM_PROVIDER == "openai":
        from openai import OpenAI as _OpenAIClient
        client = _OpenAIClient(api_key=OPENAI_API_KEY)
    elif LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic as _AnthropicClient
        client = _AnthropicClient(api_key=CLAUDE_API_KEY)
    else:
        # llama.cpp and vLLM both expose the OpenAI-compatible /v1/models route.
        from openai import OpenAI as _OpenAIClient
        client = _OpenAIClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return sorted(m.id for m in client.models.list())


if __name__ == "__main__":
    import sys

    if "--list-models" in sys.argv:
        print(f"Models visible to this key (provider={LLM_PROVIDER}):")
        try:
            names = list_models()
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED -> {type(e).__name__}: {e}")
            raise SystemExit(1)

        # Having the model on your account is necessary but NOT sufficient: the
        # llama_index integration carries its own hard-coded model catalogue and
        # raises ValueError locally, before any request, for ids it doesn't know.
        # Annotate so the two failure modes are never confused again.
        known = None
        if LLM_PROVIDER == "openai":
            try:
                from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS
                known = set(ALL_AVAILABLE_MODELS)
            except Exception:  # noqa: BLE001
                known = None

        usable = []
        for name in names:
            if known is None:
                print(f"  {name}")
            elif name in known:
                print(f"  {name}")
                usable.append(name)
            else:
                print(f"  {name}   [on your account, but too new for the installed "
                      f"llama-index-llms-openai — would raise ValueError]")
        if known is not None:
            print(f"\n  -> {len(usable)} of {len(names)} are usable right now.")
            print("     To use a newer one, upgrade the integration (see run_openai.sh).")
        raise SystemExit(0)

    print(describe())
    print("Pinging endpoint ...")
    ok, msg = check_connection()
    print(f"  {'OK  ->' if ok else 'FAIL ->'} {msg}")
    raise SystemExit(0 if ok else 1)