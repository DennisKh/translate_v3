"""Unit tests for agent/llm.py — LLM factory and prompt-cache helper.

All tests mock the underlying ChatModel classes; no network calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from agent.cost import TurnUsage
from agent.llm import ChatModelBundle, apply_prompt_cache, build_chat_model, turn_usage_from_ai_message
from project.config import Config, LLMConfig, AgentConfig, build_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(provider: str = "anthropic", model: str = "claude-opus-4-7",
         temperature: float = 0.0, max_tokens: int = 64_000,
         base_url: str | None = None,
         num_ctx: int | None = None,
         num_predict: int | None = None,
         reasoning_effort: str | None = None,
         cost_zero: bool = False,
         thinking_budget_tokens: int = 0) -> Config:
    """Build a minimal Config for the given provider."""
    return Config(
        llm=LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            num_ctx=num_ctx,
            num_predict=num_predict,
            reasoning_effort=reasoning_effort,
            cost_zero=cost_zero,
        ),
        agent=AgentConfig(thinking_budget_tokens=thinking_budget_tokens),
    )


# ---------------------------------------------------------------------------
# Factory — returns correct ChatModel class
# ---------------------------------------------------------------------------

class TestBuildChatModelClass:
    def test_anthropic_returns_chat_anthropic(self):
        mock_instance = MagicMock()
        # Patch at the source module because the factory does a lazy local import.
        with patch("langchain_anthropic.ChatAnthropic", return_value=mock_instance) as MockCls:
            bundle = build_chat_model(_cfg("anthropic"))

        MockCls.assert_called_once()
        assert bundle.chat is mock_instance

    def test_openai_returns_chat_openai(self):
        mock_instance = MagicMock()
        with patch("langchain_openai.ChatOpenAI", return_value=mock_instance) as MockCls:
            bundle = build_chat_model(_cfg("openai", model="gpt-4o"))

        MockCls.assert_called_once()
        assert bundle.chat is mock_instance

    def test_ollama_returns_chat_ollama(self):
        mock_instance = MagicMock()
        with patch("langchain_ollama.ChatOllama", return_value=mock_instance) as MockCls:
            bundle = build_chat_model(_cfg("ollama", model="llama3.1:8b"))

        MockCls.assert_called_once()
        assert bundle.chat is mock_instance

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider 'gemini'"):
            build_chat_model(_cfg("gemini"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Factory — capability flags
# ---------------------------------------------------------------------------

class TestChatModelBundleFlags:
    def test_anthropic_flags(self):
        with patch("langchain_anthropic.ChatAnthropic", return_value=MagicMock()):
            bundle = build_chat_model(_cfg("anthropic"))

        assert bundle.forces_tool_call is True
        assert bundle.supports_prompt_caching is True
        assert bundle.supports_adaptive_thinking is True

    def test_openai_flags(self):
        with patch("langchain_openai.ChatOpenAI", return_value=MagicMock()):
            bundle = build_chat_model(_cfg("openai", model="gpt-4o"))

        assert bundle.forces_tool_call is True
        assert bundle.supports_prompt_caching is False
        assert bundle.supports_adaptive_thinking is False

    def test_ollama_flags(self):
        with patch("langchain_ollama.ChatOllama", return_value=MagicMock()):
            bundle = build_chat_model(_cfg("ollama", model="llama3.1:8b"))

        assert bundle.forces_tool_call is False
        assert bundle.supports_prompt_caching is False
        assert bundle.supports_adaptive_thinking is False


# ---------------------------------------------------------------------------
# Factory — Anthropic thinking config
# ---------------------------------------------------------------------------

class TestAnthropicThinkingConfig:
    def test_adaptive_thinking_when_budget_is_zero(self):
        with patch("langchain_anthropic.ChatAnthropic") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("anthropic", thinking_budget_tokens=0))

        _, kwargs = MockCls.call_args
        assert kwargs["thinking"] == {"type": "adaptive"}

    def test_explicit_budget_when_thinking_budget_tokens_set(self):
        with patch("langchain_anthropic.ChatAnthropic") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("anthropic", thinking_budget_tokens=20_000))

        _, kwargs = MockCls.call_args
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 20_000}


# ---------------------------------------------------------------------------
# Factory — Anthropic model/temperature/max_tokens forwarding
# ---------------------------------------------------------------------------

class TestAnthropicKwargs:
    def test_model_and_max_tokens_forwarded(self):
        """model and max_tokens are forwarded; temperature is always None for Anthropic.

        The Anthropic API requires temperature=1 (or omitted) when thinking is
        enabled. We always pass temperature=None so the API uses its implicit
        default of 1. cfg.llm.temperature is ignored for Anthropic.
        """
        with patch("langchain_anthropic.ChatAnthropic") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg(
                "anthropic",
                model="claude-sonnet-4-6",
                temperature=0.5,  # ignored for Anthropic+thinking
                max_tokens=16_000,
            ))

        _, kwargs = MockCls.call_args
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["temperature"] is None  # always None; API defaults to 1
        assert kwargs["max_tokens"] == 16_000

    def test_no_tool_choice_in_model_kwargs(self):
        """tool_choice is NOT in model_kwargs on the base ChatAnthropic instance.

        tool_choice='any' is injected at bind_tools() time in _make_agent_node
        via model_copy(update={'model_kwargs': ...}). This keeps bundle.chat
        clean for one-shot calls (readme_translator, test_writer) that invoke
        the model without tools and would receive a 400 if tool_choice were set.
        """
        with patch("langchain_anthropic.ChatAnthropic") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("anthropic", model="claude-opus-4-7"))

        _, kwargs = MockCls.call_args
        assert "tool_choice" not in kwargs.get("model_kwargs", {})


# ---------------------------------------------------------------------------
# Factory — Ollama base_url handling
# ---------------------------------------------------------------------------

class TestOllamaBaseUrl:
    def test_default_base_url_when_not_configured(self):
        with patch("langchain_ollama.ChatOllama") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("ollama", base_url=None))

        _, kwargs = MockCls.call_args
        assert kwargs["base_url"] == "http://localhost:11434"

    def test_custom_base_url_forwarded(self):
        with patch("langchain_ollama.ChatOllama") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("ollama", base_url="http://my-server:11434"))

        _, kwargs = MockCls.call_args
        assert kwargs["base_url"] == "http://my-server:11434"


# ---------------------------------------------------------------------------
# apply_prompt_cache
# ---------------------------------------------------------------------------

class TestApplyPromptCache:
    def test_returns_string_when_caching_not_supported(self):
        result = apply_prompt_cache("Hello world", supports_caching=False)
        assert result == "Hello world"

    def test_returns_content_block_when_caching_supported(self):
        result = apply_prompt_cache("Hello world", supports_caching=True)
        assert isinstance(result, list)
        assert len(result) == 1
        block = result[0]
        assert block["type"] == "text"
        assert block["text"] == "Hello world"
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_content_block_preserves_full_prompt(self):
        long_prompt = "System: " + ("x" * 10_000)
        result = apply_prompt_cache(long_prompt, supports_caching=True)
        assert result[0]["text"] == long_prompt

    def test_no_cache_flag_returns_plain_string_type(self):
        result = apply_prompt_cache("prompt", supports_caching=False)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Config parsing — [llm] section
# ---------------------------------------------------------------------------

def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "translate.toml"
    p.write_text(content)
    return p


class TestLLMConfigParsing:
    def test_defaults_when_llm_section_absent(self, tmp_path):
        cfg, warnings = build_config(
            source_root=Path("/src"), target_root=Path("/tgt"),
        )
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-opus-4-7"
        assert cfg.llm.temperature == 0.0
        assert cfg.llm.max_tokens == 64_000
        assert cfg.llm.base_url is None
        assert warnings == []

    def test_llm_section_parsed_from_toml(self, tmp_path):
        toml = _write_toml(tmp_path, """
[llm]
provider = "ollama"
model = "llama3.1:8b"
temperature = 0.7
max_tokens = 4096
base_url = "http://remote:11434"
""")
        cfg, warnings = build_config(
            source_root=Path("/s"), target_root=Path("/t"),
            config_file=toml,
        )
        assert cfg.llm.provider == "ollama"
        assert cfg.llm.model == "llama3.1:8b"
        assert cfg.llm.temperature == 0.7
        assert cfg.llm.max_tokens == 4096
        assert cfg.llm.base_url == "http://remote:11434"
        assert warnings == []

    def test_partial_llm_section_fills_defaults(self, tmp_path):
        toml = _write_toml(tmp_path, """
[llm]
provider = "openai"
model = "gpt-4o"
""")
        cfg, _ = build_config(
            source_root=Path("/s"), target_root=Path("/t"),
            config_file=toml,
        )
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4o"
        assert cfg.llm.temperature == 0.0
        assert cfg.llm.max_tokens == 64_000
        assert cfg.llm.base_url is None

    def test_unknown_llm_key_produces_warning(self, tmp_path):
        toml = _write_toml(tmp_path, """
[llm]
provider = "anthropic"
invented_key = "oops"
""")
        _, warnings = build_config(
            source_root=Path("/s"), target_root=Path("/t"),
            config_file=toml,
        )
        assert any("invented_key" in w for w in warnings)

    def test_cli_model_flag_updates_llm_model(self, tmp_path):
        toml = _write_toml(tmp_path, """
[llm]
provider = "anthropic"
model = "claude-opus-4-7"
""")
        cfg, _ = build_config(
            source_root=Path("/s"), target_root=Path("/t"),
            config_file=toml,
            model="claude-sonnet-4-6",
        )
        assert cfg.llm.model == "claude-sonnet-4-6"

    def test_llm_section_does_not_produce_unknown_top_level_warning(self, tmp_path):
        toml = _write_toml(tmp_path, """
[llm]
provider = "anthropic"
""")
        _, warnings = build_config(
            source_root=Path("/s"), target_root=Path("/t"),
            config_file=toml,
        )
        assert not any("[llm]" in w for w in warnings)


# ---------------------------------------------------------------------------
# ChatModelBundle is frozen
# ---------------------------------------------------------------------------

def test_bundle_is_frozen():
    bundle = ChatModelBundle(
        chat=MagicMock(),
        forces_tool_call=True,
        supports_prompt_caching=True,
        supports_adaptive_thinking=True,
        tool_choice_any_payload={"type": "any"},
    )
    with pytest.raises((AttributeError, TypeError)):
        bundle.forces_tool_call = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# turn_usage_from_ai_message
# ---------------------------------------------------------------------------

class TestTurnUsageFromAiMessage:
    def test_anthropic_shape_with_cache_tokens(self):
        # Anthropic's usage_metadata nests cache tokens under input_token_details
        # (spike 10.4 finding).
        msg = AIMessage(
            content="result",
            usage_metadata={
                "input_tokens": 500,
                "output_tokens": 200,
                "total_tokens": 700,
                "input_token_details": {
                    "cache_creation": 100,
                    "cache_read": 50,
                },
                "output_token_details": {},
            },
        )
        usage = turn_usage_from_ai_message(msg)
        assert usage.input_tokens == 500
        assert usage.output_tokens == 200
        assert usage.cache_write_tokens == 100
        assert usage.cache_read_tokens == 50

    def test_openai_shape_without_cache_tokens(self):
        # OpenAI / Ollama providers emit no input_token_details.
        msg = AIMessage(
            content="result",
            usage_metadata={
                "input_tokens": 300,
                "output_tokens": 80,
                "total_tokens": 380,
            },
        )
        usage = turn_usage_from_ai_message(msg)
        assert usage.input_tokens == 300
        assert usage.output_tokens == 80
        assert usage.cache_write_tokens == 0
        assert usage.cache_read_tokens == 0

    def test_missing_metadata_returns_zeros(self):
        # Dry-run mocks or providers that omit usage_metadata entirely.
        msg = AIMessage(content="result", usage_metadata=None)
        usage = turn_usage_from_ai_message(msg)
        assert usage == TurnUsage()

    def test_partial_input_token_details_missing_cache_keys(self):
        # Defensive: input_token_details exists but has only 'audio', not cache keys.
        msg = AIMessage(
            content="result",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 40,
                "total_tokens": 140,
                "input_token_details": {"audio": 10},
            },
        )
        usage = turn_usage_from_ai_message(msg)
        assert usage.cache_write_tokens == 0
        assert usage.cache_read_tokens == 0


# ---------------------------------------------------------------------------
# apply_prompt_cache + SystemMessage construction
# ---------------------------------------------------------------------------

class TestExtractTextFromAiMessage:
    def test_string_content(self):
        from agent.llm import _extract_text_from_ai_message
        msg = AIMessage(content="just a string")
        assert _extract_text_from_ai_message(msg) == "just a string"

    def test_list_content_with_text_block(self):
        from agent.llm import _extract_text_from_ai_message
        msg = AIMessage(content=[{"type": "text", "text": "hello"}])
        assert _extract_text_from_ai_message(msg) == "hello"

    def test_list_content_thinking_and_text_blocks(self):
        """Anthropic with thinking enabled: thinking + text blocks; only text extracted."""
        from agent.llm import _extract_text_from_ai_message
        msg = AIMessage(content=[
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "text", "text": "answer"},
        ])
        assert _extract_text_from_ai_message(msg) == "answer"

    def test_list_content_only_thinking_returns_empty(self):
        """Edge case: only thinking block, no text. Callers must handle empty."""
        from agent.llm import _extract_text_from_ai_message
        msg = AIMessage(content=[{"type": "thinking", "thinking": "still thinking"}])
        assert _extract_text_from_ai_message(msg) == ""

    def test_list_content_multiple_text_blocks_concatenated(self):
        from agent.llm import _extract_text_from_ai_message
        msg = AIMessage(content=[
            {"type": "text", "text": "part 1 "},
            {"type": "text", "text": "part 2"},
        ])
        assert _extract_text_from_ai_message(msg) == "part 1 part 2"

    def test_list_content_with_string_element(self):
        """Defensive: a raw str in the list is preserved."""
        from agent.llm import _extract_text_from_ai_message
        msg = AIMessage(content=["bare string"])
        assert _extract_text_from_ai_message(msg) == "bare string"


# ---------------------------------------------------------------------------
# ChatModelBundle.tool_choice_any_payload — provider-specific API payload
# ---------------------------------------------------------------------------

class TestToolChoiceAnyPayload:
    def test_anthropic_returns_type_any_dict(self):
        """Anthropic API expects {"type": "any"} for forced tool use."""
        with patch("langchain_anthropic.ChatAnthropic", return_value=MagicMock()):
            bundle = build_chat_model(_cfg("anthropic"))
        assert bundle.tool_choice_any_payload == {"type": "any"}

    def test_openai_returns_required_string(self):
        """OpenAI API expects the string "required". {"type":"any"} causes a 400
        because OpenAI interprets dicts as the function-specific form, which
        requires a "function.name" key."""
        with patch("langchain_openai.ChatOpenAI", return_value=MagicMock()):
            bundle = build_chat_model(_cfg("openai", model="gpt-5-mini"))
        assert bundle.tool_choice_any_payload == "required"

    def test_ollama_returns_none(self):
        """Ollama ignores tool_choice entirely; payload is None (skipped in _make_agent_node)."""
        with patch("langchain_ollama.ChatOllama", return_value=MagicMock()):
            bundle = build_chat_model(_cfg("ollama", model="llama3.1:8b"))
        assert bundle.tool_choice_any_payload is None


class TestApplyPromptCacheEmptyString:
    def test_empty_string_no_caching_returns_empty_string(self):
        assert apply_prompt_cache("", supports_caching=False) == ""

    def test_empty_string_with_caching_returns_empty_text_block(self):
        result = apply_prompt_cache("", supports_caching=True)
        assert isinstance(result, list)
        assert result[0]["text"] == ""
        assert result[0]["cache_control"] == {"type": "ephemeral"}


class TestApplyPromptCacheSystemMessageIntegration:
    def test_string_content_accepted_by_system_message(self):
        # Non-Anthropic path: returns plain str; SystemMessage must accept it.
        content = apply_prompt_cache("system prompt text", supports_caching=False)
        msg = SystemMessage(content=content)
        assert msg.content == "system prompt text"

    def test_list_content_with_cache_control_accepted_by_system_message(self):
        # Anthropic path: returns list[dict] with cache_control; SystemMessage must accept it.
        content = apply_prompt_cache("system prompt text", supports_caching=True)
        msg = SystemMessage(content=content)
        assert isinstance(msg.content, list)
        assert msg.content[0]["cache_control"] == {"type": "ephemeral"}
        assert msg.content[0]["text"] == "system prompt text"


# ---------------------------------------------------------------------------
# OpenAI provider-specific kwargs (base_url for OpenAI-compat, reasoning_effort)
# ---------------------------------------------------------------------------

class TestOpenAIKwargs:
    def test_base_url_forwarded_when_set(self):
        """base_url is critical for LM Studio / vLLM / OpenRouter routing."""
        with patch("langchain_openai.ChatOpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("openai", model="qwen2.5", base_url="http://localhost:1234/v1"))

        _, kwargs = MockCls.call_args
        assert kwargs["base_url"] == "http://localhost:1234/v1"

    def test_base_url_omitted_when_unset(self):
        """No base_url in the TOML → don't pass it (SDK default = api.openai.com)."""
        with patch("langchain_openai.ChatOpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("openai", model="gpt-5-mini"))

        _, kwargs = MockCls.call_args
        assert "base_url" not in kwargs

    def test_reasoning_effort_forwarded_when_set(self):
        with patch("langchain_openai.ChatOpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("openai", model="gpt-5-mini", reasoning_effort="high"))

        _, kwargs = MockCls.call_args
        assert kwargs["reasoning_effort"] == "high"

    def test_reasoning_effort_omitted_when_unset(self):
        with patch("langchain_openai.ChatOpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("openai", model="gpt-5-mini"))

        _, kwargs = MockCls.call_args
        assert "reasoning_effort" not in kwargs


# ---------------------------------------------------------------------------
# Ollama provider-specific kwargs (num_ctx, num_predict)
# ---------------------------------------------------------------------------

class TestOllamaKwargs:
    def test_num_ctx_forwarded_when_set(self):
        """num_ctx overrides the Modelfile default — fixes 400 on big prompts."""
        with patch("langchain_ollama.ChatOllama") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("ollama", model="llama3.1:8b", num_ctx=32768))

        _, kwargs = MockCls.call_args
        assert kwargs["num_ctx"] == 32768

    def test_num_ctx_omitted_when_unset(self):
        with patch("langchain_ollama.ChatOllama") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("ollama", model="llama3.1:8b"))

        _, kwargs = MockCls.call_args
        assert "num_ctx" not in kwargs

    def test_num_predict_forwarded_when_set(self):
        with patch("langchain_ollama.ChatOllama") as MockCls:
            MockCls.return_value = MagicMock()
            build_chat_model(_cfg("ollama", model="llama3.1:8b", num_predict=4096))

        _, kwargs = MockCls.call_args
        assert kwargs["num_predict"] == 4096


# ---------------------------------------------------------------------------
# classify_exception — provider-agnostic exception dispatch
# ---------------------------------------------------------------------------

class TestClassifyException:
    """Every provider SDK exception the session runner cares about must map
    to the correct ExceptionKind. If this breaks, the outer loop misclassifies
    transient failures as parse_failure and burns the run on the first hiccup.
    """

    def test_anthropic_rate_limit(self):
        import anthropic
        from agent.llm import ExceptionKind, classify_exception
        exc = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        assert classify_exception(exc) == ExceptionKind.RATE_LIMIT

    def test_anthropic_internal_server_error_is_rate_limit(self):
        # 529 (overloaded) lives under InternalServerError in the anthropic SDK.
        import anthropic
        from agent.llm import ExceptionKind, classify_exception
        exc = anthropic.InternalServerError(
            message="overloaded",
            response=MagicMock(status_code=529),
            body=None,
        )
        assert classify_exception(exc) == ExceptionKind.RATE_LIMIT

    def test_anthropic_timeout(self):
        import anthropic
        from agent.llm import ExceptionKind, classify_exception
        exc = anthropic.APITimeoutError(request=MagicMock())
        assert classify_exception(exc) == ExceptionKind.TIMEOUT

    def test_anthropic_bad_request(self):
        import anthropic
        from agent.llm import ExceptionKind, classify_exception
        exc = anthropic.BadRequestError(
            message="bad payload",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert classify_exception(exc) == ExceptionKind.BAD_REQUEST

    def test_openai_rate_limit(self):
        # Regression guard for the OpenAI path: prior code caught only
        # anthropic.RateLimitError, so an openai.RateLimitError silently
        # became "parse_failure" — no streak accounting, no retry.
        import openai
        from agent.llm import ExceptionKind, classify_exception
        exc = openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        assert classify_exception(exc) == ExceptionKind.RATE_LIMIT

    def test_openai_timeout(self):
        import openai
        from agent.llm import ExceptionKind, classify_exception
        exc = openai.APITimeoutError(request=MagicMock())
        assert classify_exception(exc) == ExceptionKind.TIMEOUT

    def test_openai_bad_request(self):
        import openai
        from agent.llm import ExceptionKind, classify_exception
        exc = openai.BadRequestError(
            message="bad payload",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert classify_exception(exc) == ExceptionKind.BAD_REQUEST

    def test_httpx_read_timeout_is_timeout(self):
        # Ollama / other httpx-backed providers surface read timeouts this way.
        import httpx
        from agent.llm import ExceptionKind, classify_exception
        exc = httpx.ReadTimeout("read timeout")
        assert classify_exception(exc) == ExceptionKind.TIMEOUT

    def test_httpx_400_status_is_bad_request(self):
        # Ollama context-too-large returns HTTP 400 wrapped in HTTPStatusError.
        # This was the recent bug: mapped to UNKNOWN → parse_failure.
        import httpx
        from agent.llm import ExceptionKind, classify_exception
        response = MagicMock(status_code=400)
        exc = httpx.HTTPStatusError("bad request", request=MagicMock(), response=response)
        assert classify_exception(exc) == ExceptionKind.BAD_REQUEST

    def test_unknown_exception_returns_unknown(self):
        from agent.llm import ExceptionKind, classify_exception
        assert classify_exception(RuntimeError("wat")) == ExceptionKind.UNKNOWN


# ---------------------------------------------------------------------------
# LLMConfig new fields — TOML parsing
# ---------------------------------------------------------------------------

class TestLLMConfigProviderKwargs:
    def test_num_ctx_parsed(self, tmp_path):
        toml = _write_toml(tmp_path, """
[llm]
provider = "ollama"
model = "llama3.1:8b"
num_ctx = 16384
""")
        cfg, warnings = build_config(
            source_root=Path("/s"), target_root=Path("/t"), config_file=toml,
        )
        assert cfg.llm.num_ctx == 16384
        assert warnings == []

    def test_reasoning_effort_parsed(self, tmp_path):
        toml = _write_toml(tmp_path, """
[llm]
provider = "openai"
model = "gpt-5-mini"
reasoning_effort = "low"
""")
        cfg, warnings = build_config(
            source_root=Path("/s"), target_root=Path("/t"), config_file=toml,
        )
        assert cfg.llm.reasoning_effort == "low"
        assert warnings == []

    def test_cost_zero_parsed(self, tmp_path):
        toml = _write_toml(tmp_path, """
[llm]
provider = "openai"
model = "custom-local-model"
cost_zero = true
""")
        cfg, warnings = build_config(
            source_root=Path("/s"), target_root=Path("/t"), config_file=toml,
        )
        assert cfg.llm.cost_zero is True
        assert warnings == []

    def test_defaults_when_all_optional_fields_omitted(self, tmp_path):
        cfg, _ = build_config(source_root=Path("/s"), target_root=Path("/t"))
        assert cfg.llm.num_ctx is None
        assert cfg.llm.num_predict is None
        assert cfg.llm.reasoning_effort is None
        assert cfg.llm.cost_zero is False

    def test_safety_section_now_unknown_top_level(self, tmp_path):
        # SafetyConfig was deleted; users with legacy [safety] blocks should
        # see an unknown-section warning — the correct signal that the key
        # never did anything useful.
        toml = _write_toml(tmp_path, """
[safety]
task_budget_beta = false
""")
        _, warnings = build_config(
            source_root=Path("/s"), target_root=Path("/t"), config_file=toml,
        )
        assert any("[safety]" in w for w in warnings)
