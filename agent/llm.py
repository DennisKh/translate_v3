"""Provider-aware LLM factory for the LangGraph agent.

Public surface:
  - ``build_chat_model(cfg)`` — returns a ``ChatModelBundle`` wrapping a
    ``BaseChatModel`` configured for the requested provider.
  - ``apply_prompt_cache(system_prompt, supports_caching)`` — returns the
    correct shape for a system message: a plain string for non-Anthropic
    providers, or a list-of-content-blocks with cache_control for Anthropic.
  - ``turn_usage_from_ai_message(msg)`` — converts AIMessage.usage_metadata
    to TurnUsage, handling the nested input_token_details path for cache
    tokens and providers that emit no usage metadata at all.

Anthropic-specific perks (prompt caching, adaptive thinking) are isolated
here. Callers see a plain ``BaseChatModel`` and don't branch on provider.

Spike findings that shaped this module (see PLAN_LANGGRAPH.md §10):
- 10.1: ChatOllama ignores tool_choice — forces_tool_call=False for Ollama.
- 10.2: cache_control via content-block dict, not middleware.
- 10.3: AIMessage.tool_calls normalized across providers — no caller branching.
- 10.4: cache tokens in usage_metadata["input_token_details"] — not top-level.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from agent.cost import TurnUsage
from project.config import Config


class ExceptionKind(str, Enum):
    """Provider-agnostic classification of exceptions raised during a model call.

    The session runner in ``agent/translator.py`` dispatches on this to decide
    whether to retry (soft) with rate-limit accounting, checkpoint the session,
    fail fast, or re-raise. Keeping the mapping in ``classify_exception`` below
    means the runner never imports a provider SDK.
    """
    RATE_LIMIT = "rate_limit"     # 429 / 529 / overloaded — soft retry, streak++
    TIMEOUT = "timeout"           # per-turn wall clock or connection timeout — checkpoint
    TRANSIENT = "transient"       # connection reset, 5xx-not-overload — treat as timeout
    BAD_REQUEST = "bad_request"   # 400 / auth / user config error — re-raise, do NOT swallow
    AUTH = "auth"                 # 401 / 403 — re-raise, user must fix API key
    UNKNOWN = "unknown"           # anything else — session-level parse_failure


def classify_exception(exc: BaseException) -> ExceptionKind:
    """Map a provider SDK exception to a provider-agnostic ``ExceptionKind``.

    Optionally imports each provider's SDK module inside the function so
    absence of an SDK (e.g. no anthropic install) doesn't prevent classifying
    other providers' exceptions. This keeps the module import list free of
    provider-specific SDKs.

    Order of checks:
    1. Anthropic (most likely default provider)
    2. OpenAI (used for direct API AND for LM Studio / vLLM / OpenRouter)
    3. Fallback: match on exception class name substring for Ollama / httpx,
       which don't ship a rich exception hierarchy.
    """
    try:
        import anthropic
        if isinstance(exc, anthropic.RateLimitError):
            return ExceptionKind.RATE_LIMIT
        if isinstance(exc, anthropic.InternalServerError):
            # 500 and 529 (overloaded) both land here.
            return ExceptionKind.RATE_LIMIT
        if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
            return ExceptionKind.TIMEOUT
        if isinstance(exc, anthropic.AuthenticationError):
            return ExceptionKind.AUTH
        if isinstance(exc, anthropic.BadRequestError):
            return ExceptionKind.BAD_REQUEST
        if isinstance(exc, anthropic.APIError):
            return ExceptionKind.TRANSIENT
    except ImportError:
        pass

    try:
        import openai
        if isinstance(exc, openai.RateLimitError):
            return ExceptionKind.RATE_LIMIT
        if isinstance(exc, openai.InternalServerError):
            return ExceptionKind.RATE_LIMIT
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return ExceptionKind.TIMEOUT
        if isinstance(exc, openai.AuthenticationError):
            return ExceptionKind.AUTH
        if isinstance(exc, openai.BadRequestError):
            return ExceptionKind.BAD_REQUEST
        if isinstance(exc, openai.APIError):
            return ExceptionKind.TRANSIENT
    except ImportError:
        pass

    # Ollama / LangChain-Ollama surface errors via httpx.ReadTimeout,
    # httpx.ConnectError, and OllamaError (from langchain_ollama). Rather than
    # add hard imports for each, match on class name — httpx status is stable.
    name = type(exc).__name__
    if name in ("ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"):
        return ExceptionKind.TIMEOUT
    if name in ("ConnectError", "RemoteProtocolError", "NetworkError"):
        return ExceptionKind.TRANSIENT
    # Ollama returns 400 for context-too-large; httpx wraps as HTTPStatusError.
    if name == "HTTPStatusError":
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            return ExceptionKind.RATE_LIMIT
        if status in (500, 502, 503, 504):
            return ExceptionKind.TRANSIENT
        if status in (401, 403):
            return ExceptionKind.AUTH
        if status == 400:
            return ExceptionKind.BAD_REQUEST

    return ExceptionKind.UNKNOWN


@dataclass(frozen=True)
class ChatModelBundle:
    chat: BaseChatModel
    # True when the provider reliably enforces tool_choice="any" (Anthropic,
    # OpenAI). False for Ollama, where tool_choice is explicitly ignored in
    # ChatOllama.bind_tools(). Callers use this to decide whether the
    # no-tool-calls→END branch in route_after_agent is a primary exit
    # (Ollama) or a defensive fallback (Anthropic/OpenAI).
    forces_tool_call: bool
    # True only for Anthropic. Other providers silently drop cache_control
    # blocks in message content, making apply_prompt_cache() a no-op for them.
    supports_prompt_caching: bool
    # True only for Anthropic. thinking= parameter is a ChatAnthropic-specific
    # field that has no equivalent on OpenAI or Ollama.
    supports_adaptive_thinking: bool
    # Provider-specific API payload for "must call some tool". Injected into
    # model_kwargs by _make_agent_node (via model_copy) when forces_tool_call
    # is True. None when forces_tool_call is False (Ollama — skipped entirely).
    #
    # Anthropic API: {"type": "any"}
    # OpenAI API:    "required"   (OpenAI rejects {"type":"any"} — that form
    #                requires a "function" key; "required" is the correct string)
    # Ollama:        None
    tool_choice_any_payload: dict | str | None


def build_chat_model(cfg: Config) -> ChatModelBundle:
    """Build a provider-configured BaseChatModel from Config.

    Reads cfg.llm for provider, model, temperature, max_tokens, and base_url.
    Reads cfg.agent for thinking_budget_tokens (Anthropic only).

    Raises ValueError on an unknown provider.
    """
    provider = cfg.llm.provider

    if provider == "anthropic":
        return _build_anthropic(cfg)
    if provider == "openai":
        return _build_openai(cfg)
    if provider == "ollama":
        return _build_ollama(cfg)

    raise ValueError(
        f"Unknown LLM provider {provider!r}. "
        f"Supported providers: 'anthropic', 'openai', 'ollama'."
    )


def turn_usage_from_ai_message(msg: AIMessage) -> TurnUsage:
    """Convert AIMessage.usage_metadata to TurnUsage.

    Handles two provider shapes:
    - Anthropic: usage_metadata["input_token_details"]["cache_creation"] and
      ["cache_read"] carry prompt-cache accounting (spike 10.4).
    - OpenAI / Ollama: usage_metadata has no input_token_details; cache tokens
      default to 0.
    - No metadata at all (dry-run mocks, providers that omit it): all zeros.
    """
    meta = msg.usage_metadata
    if meta is None:
        return TurnUsage()

    details = meta.get("input_token_details") or {}
    return TurnUsage(
        input_tokens=meta.get("input_tokens") or 0,
        output_tokens=meta.get("output_tokens") or 0,
        cache_write_tokens=details.get("cache_creation") or 0,
        cache_read_tokens=details.get("cache_read") or 0,
    )


def _extract_text_from_ai_message(msg: AIMessage) -> str:
    """Extract plain text from AIMessage.content regardless of shape.

    When thinking is enabled, langchain-anthropic sets content to a list of
    dicts (thinking blocks + text blocks). Without thinking, content is a str.
    Both shapes are handled here.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    # list[str | dict] — collect text from all text-typed blocks
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def apply_prompt_cache(
    system_prompt: str,
    supports_caching: bool,
) -> str | list[dict]:
    """Return system-message content in the correct shape for the provider.

    For Anthropic (supports_caching=True): returns a list-of-content-blocks
    with cache_control so the system prompt is written to Anthropic's prompt
    cache on the first call and read from it on subsequent calls.

    For all other providers (supports_caching=False): returns the prompt
    unchanged as a string. LangChain's non-Anthropic backends don't understand
    cache_control blocks and would either error or silently ignore them.

    Usage:
        content = apply_prompt_cache(system_prompt, bundle.supports_prompt_caching)
        messages = [SystemMessage(content=content), ...]
    """
    if not supports_caching:
        return system_prompt

    return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]


def _build_anthropic(cfg: Config) -> ChatModelBundle:
    from langchain_anthropic import ChatAnthropic

    if cfg.agent.thinking_budget_tokens > 0:
        thinking: dict = {
            "type": "enabled",
            "budget_tokens": cfg.agent.thinking_budget_tokens,
        }
    else:
        thinking = {"type": "adaptive"}

    chat = ChatAnthropic(
        model=cfg.llm.model,
        temperature=None,
        max_tokens=cfg.llm.max_tokens,
        thinking=thinking,
    )
    return ChatModelBundle(
        chat=chat,
        forces_tool_call=True,
        supports_prompt_caching=True,
        supports_adaptive_thinking=True,
        tool_choice_any_payload={"type": "any"},
    )


def _build_openai(cfg: Config) -> ChatModelBundle:
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": cfg.llm.model,
        "temperature": cfg.llm.temperature,
        "max_tokens": cfg.llm.max_tokens,
    }
    if cfg.llm.base_url:
        kwargs["base_url"] = cfg.llm.base_url
    if cfg.llm.reasoning_effort:
        kwargs["reasoning_effort"] = cfg.llm.reasoning_effort

    chat = ChatOpenAI(**kwargs)
    return ChatModelBundle(
        chat=chat,
        forces_tool_call=True,
        supports_prompt_caching=False,
        supports_adaptive_thinking=False,
        tool_choice_any_payload="required",
    )


def _build_ollama(cfg: Config) -> ChatModelBundle:
    from langchain_ollama import ChatOllama

    base_url = cfg.llm.base_url or "http://localhost:11434"

    kwargs: dict = {
        "model": cfg.llm.model,
        "temperature": cfg.llm.temperature,
        "base_url": base_url,
    }
    if cfg.llm.num_ctx is not None:
        kwargs["num_ctx"] = cfg.llm.num_ctx
    if cfg.llm.num_predict is not None:
        kwargs["num_predict"] = cfg.llm.num_predict

    chat = ChatOllama(**kwargs)
    return ChatModelBundle(
        chat=chat,
        forces_tool_call=False,
        supports_prompt_caching=False,
        supports_adaptive_thinking=False,
        tool_choice_any_payload=None,
    )
