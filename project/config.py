"""Config loading with CLI > TOML > defaults precedence.

Two required positional args (SOURCE_ROOT, TARGET_ROOT). Everything else
optional. Maven layout auto-discovered under SOURCE_ROOT; overrides via
config or flags.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

Severity = Literal["higher", "high", "normal", "low", "info"]
Provider = Literal["anthropic", "openai", "ollama"]


@dataclass(frozen=True)
class SourceConfig:
    language: str = "java"
    root: Path | None = None
    sources_glob: str = "src/main/java/**/*.java"
    tests_glob: str = "src/test/java/**/*.java"
    resources_glob: str = "src/main/resources/**/*"


@dataclass(frozen=True)
class TargetConfig:
    language: str = "elixir"
    root: Path | None = None
    app_name: str | None = None                # None => defer to `mix new`
    module_name: str | None = None             # None => defer to `mix new`
    mix_deps: dict[str, str] = field(default_factory=lambda: {"decimal": "~> 2.0"})
    dev_deps: dict[str, str] = field(default_factory=lambda: {"credo": "~> 1.7"})
    # Non-code files to copy verbatim from source_root → target_root at
    # scaffold time. Legal/attribution files (LICENSE, NOTICE) are essential
    # for Apache-2.0 and similar licenses. Uses shell-style globs relative
    # to source_root. Missing files are silently ignored.
    copy_files: list[str] = field(default_factory=lambda: [
        "LICENSE*", "NOTICE*", "COPYING*", "AUTHORS*", "CONTRIBUTORS*",
    ])
    # Translate README.md as a post-Phase-A step, once validation is green.
    # Uses the actual Elixir module APIs to produce valid code examples.
    translate_readme: bool = True


@dataclass(frozen=True)
class LLMConfig:
    provider: Provider = "anthropic"
    model: str = "claude-opus-4-7"
    temperature: float = 0.0
    # Maximum tokens the model may output per turn. Anthropic default is 64 000
    # (the current per-turn ceiling for Opus 4.7). OpenAI and Ollama interpret
    # None as "model default", which is fine since they don't have the same
    # thinking-token overhead concerns.
    max_tokens: int = 64_000
    # OpenAI-compat + Ollama. For Ollama defaults to http://localhost:11434.
    # For OpenAI direct, leave None (uses api.openai.com). For LM Studio /
    # vLLM / OpenRouter / any OpenAI-compatible endpoint, set the endpoint's
    # base URL (e.g. "http://localhost:1234/v1"). Ignored by Anthropic.
    base_url: str | None = None
    # Ollama only. Overrides the model's default context window. Necessary
    # for large-context Ollama runs (default is often 2048/4096 which trips
    # HTTP 400 on our system prompt). Ignored by other providers.
    num_ctx: int | None = None
    # Ollama only. Caps output tokens per turn. Ignored by other providers.
    num_predict: int | None = None
    # OpenAI only. Reasoning-mode dial for GPT-5 family: "minimal" | "low" |
    # "medium" | "high". Not the same axis as `agent.effort` (which drives
    # our prompt-side effort). Ignored by other providers.
    reasoning_effort: str | None = None
    # Skip the "unknown model — treating as $0/token" warning for custom
    # local models whose name doesn't match a known free-model prefix.
    # Set for local OpenAI-compat servers (LM Studio etc.) and custom
    # Ollama Modelfiles. Ollama provider auto-enables this behavior via
    # prefix matching, but the escape hatch is here for edge cases.
    cost_zero: bool = False


@dataclass(frozen=True)
class AgentConfig:
    model_version_policy: str = "pinned"
    # Default `medium` — translation with a detailed system prompt does NOT
    # need deep deliberation. `high` on Sonnet 4.6 caused 30-min single-turn
    # hangs on the hardest files (BigMoney), producing thinking overhead
    # without commensurate output value. Override via config for genuinely
    # hard tasks that need more reasoning.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_budget_tokens: int = 3_000_000
    max_tool_calls: int = 500
    max_wall_seconds: int = 5400
    session_reset_after_files: int = 10
    session_reset_after_tokens: int = 500_000
    # Per-turn hard wall-clock ceiling. If a single API turn exceeds this,
    # the SDK aborts the request and we treat it as a failed turn. 900s
    # (15 min) is generous even for the biggest translations at medium effort.
    max_turn_seconds: float = 900.0
    # Explicit thinking-token cap. Uses `thinking: {type: "enabled", budget}`
    # instead of adaptive when set. Prevents runaway deliberation.
    # 0 = keep adaptive (no cap); >0 = hard cap in tokens.
    thinking_budget_tokens: int = 0
    # Set False to skip the prompt in scripted runs that share a TTY.
    hitl_on_wall_cap: bool = True


@dataclass(frozen=True)
class PhasesConfig:
    translate: bool = True
    generate_tests: bool = False


@dataclass(frozen=True)
class ValidationConfig:
    run_format: bool = True
    run_credo: bool = True
    credo_block_severity: Severity = "high"
    run_tests: bool = False


@dataclass(frozen=True)
class LangfuseConfig:
    enabled: bool = False
    # 0.0–1.0 sampling rate for traces. 1.0 = capture everything.
    sample_rate: float = 1.0


@dataclass(frozen=True)
class Config:
    source: SourceConfig = field(default_factory=SourceConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    phases: PhasesConfig = field(default_factory=PhasesConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)


_KNOWN_TOP_LEVEL = {"source", "target", "agent", "llm", "phases", "validation", "langfuse"}
_KNOWN_KEYS = {
    "source": {"language", "root", "sources_glob", "tests_glob", "resources_glob"},
    "target": {"language", "root", "app_name", "module_name", "mix_deps", "dev_deps"},
    "agent": {"model_version_policy", "effort", "max_budget_tokens",
              "max_tool_calls", "max_wall_seconds",
              "session_reset_after_files", "session_reset_after_tokens",
              "max_turn_seconds", "thinking_budget_tokens",
              "hitl_on_wall_cap"},
    "llm": {"provider", "model", "temperature", "max_tokens", "base_url",
            "num_ctx", "num_predict", "reasoning_effort", "cost_zero"},
    "phases": {"translate", "generate_tests"},
    "validation": {"run_format", "run_credo", "credo_block_severity", "run_tests"},
    "langfuse": {"enabled", "sample_rate"},
}


class ConfigValidationError(ValueError):
    """Raised when the TOML has structural problems (unknown keys, bad types)."""


def _validate_raw(raw: dict) -> list[str]:
    """Return a list of validation warnings for unknown keys.

    Unknown keys become warnings, not errors — a v3 user should be able to add
    forward-compatible keys without the tool refusing to run. But they should
    know their typo silently didn't take effect.
    """
    warnings: list[str] = []
    for top_key in raw:
        if top_key not in _KNOWN_TOP_LEVEL:
            warnings.append(f"unknown top-level section [{top_key}] — ignored")
            continue
        section = raw[top_key]
        if not isinstance(section, dict):
            continue
        for k in section:
            # Nested tables like `[target.mix_deps]` show up as dict values, allow them
            if k in _KNOWN_KEYS.get(top_key, set()):
                continue
            if isinstance(section[k], dict):
                # `[target.mix_deps]` is a table; treat inner keys as free-form dep name/version
                continue
            warnings.append(f"unknown key [{top_key}].{k} — ignored")
    return warnings


def load_toml(path: Path) -> tuple[dict, list[str]]:
    """Load a TOML file and return (parsed, warnings).

    Warnings surface unknown keys so silent typos don't get past the user.
    """
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return raw, _validate_raw(raw)


def _from_toml(raw: dict) -> Config:
    """Build a Config from a parsed TOML dict, filling defaults where absent."""
    src = raw.get("source", {})
    tgt = raw.get("target", {})
    agt = raw.get("agent", {})
    llm = raw.get("llm", {})
    phs = raw.get("phases", {})
    val = raw.get("validation", {})
    lfs = raw.get("langfuse", {})

    return Config(
        source=SourceConfig(
            language=src.get("language", "java"),
            root=Path(src["root"]) if src.get("root") else None,
            sources_glob=src.get("sources_glob", "src/main/java/**/*.java"),
            tests_glob=src.get("tests_glob", "src/test/java/**/*.java"),
            resources_glob=src.get("resources_glob", "src/main/resources/**/*"),
        ),
        target=TargetConfig(
            language=tgt.get("language", "elixir"),
            root=Path(tgt["root"]) if tgt.get("root") else None,
            app_name=tgt.get("app_name"),
            module_name=tgt.get("module_name"),
            mix_deps=tgt.get("mix_deps", {"decimal": "~> 2.0"}),
            dev_deps=tgt.get("dev_deps", {"credo": "~> 1.7"}),
            copy_files=tgt.get("copy_files", [
                "LICENSE*", "NOTICE*", "COPYING*", "AUTHORS*", "CONTRIBUTORS*",
            ]),
            translate_readme=tgt.get("translate_readme", True),
        ),
        agent=AgentConfig(
            model_version_policy=agt.get("model_version_policy", "pinned"),
            effort=agt.get("effort", "medium"),
            max_budget_tokens=agt.get("max_budget_tokens", 3_000_000),
            max_tool_calls=agt.get("max_tool_calls", 500),
            max_wall_seconds=agt.get("max_wall_seconds", 5400),
            session_reset_after_files=agt.get("session_reset_after_files", 10),
            session_reset_after_tokens=agt.get("session_reset_after_tokens", 500_000),
            max_turn_seconds=agt.get("max_turn_seconds", 900.0),
            thinking_budget_tokens=agt.get("thinking_budget_tokens", 0),
        ),
        llm=LLMConfig(
            provider=llm.get("provider", "anthropic"),
            model=llm.get("model", "claude-opus-4-7"),
            temperature=llm.get("temperature", 0.0),
            max_tokens=llm.get("max_tokens", 64_000),
            base_url=llm.get("base_url"),
            num_ctx=llm.get("num_ctx"),
            num_predict=llm.get("num_predict"),
            reasoning_effort=llm.get("reasoning_effort"),
            cost_zero=llm.get("cost_zero", False),
        ),
        phases=PhasesConfig(
            translate=phs.get("translate", True),
            generate_tests=phs.get("generate_tests", False),
        ),
        validation=ValidationConfig(
            run_format=val.get("run_format", True),
            run_credo=val.get("run_credo", True),
            credo_block_severity=val.get("credo_block_severity", "high"),
            run_tests=val.get("run_tests", False),
        ),
        langfuse=LangfuseConfig(
            enabled=lfs.get("enabled", False),
            sample_rate=lfs.get("sample_rate", 1.0),
        ),
    )


def build_config(
    source_root: Path,
    target_root: Path,
    *,
    config_file: Path | None = None,
    app_name: str | None = None,
    module_name: str | None = None,
    model: str | None = None,
    max_budget_tokens: int | None = None,
    phase: Literal["A", "B", "both"] = "A",
    skip_format: bool = False,
    skip_credo: bool = False,
    strict_validation: bool = False,
) -> tuple[Config, list[str]]:
    """Compose a Config with precedence: CLI > config file > defaults.

    Positional args (source_root, target_root) are always applied last so they
    override whatever the config file says. Same for the CLI overrides.

    Returns (config, warnings). `warnings` is a list of human-readable
    strings (e.g. unknown TOML keys). Empty if all clean.
    """
    warnings: list[str] = []
    if config_file:
        raw, warnings = load_toml(config_file)
        base = _from_toml(raw)
    else:
        base = Config()

    # Positional args always win
    base = replace(base,
        source=replace(base.source, root=source_root),
        target=replace(base.target, root=target_root),
    )

    # CLI overrides
    if app_name is not None:
        base = replace(base, target=replace(base.target, app_name=app_name))
    if module_name is not None:
        base = replace(base, target=replace(base.target, module_name=module_name))
    if model is not None:
        base = replace(base, llm=replace(base.llm, model=model))
    if max_budget_tokens is not None:
        base = replace(base, agent=replace(base.agent, max_budget_tokens=max_budget_tokens))

    # Phase gating
    if phase == "A":
        base = replace(base, phases=PhasesConfig(translate=True, generate_tests=False))
    elif phase == "B":
        base = replace(base, phases=PhasesConfig(translate=False, generate_tests=True))
    elif phase == "both":
        base = replace(base, phases=PhasesConfig(translate=True, generate_tests=True))

    # Validation-gate CLI overrides
    if skip_format or skip_credo or strict_validation:
        v = base.validation
        base = replace(
            base,
            validation=replace(
                v,
                run_format=v.run_format and not skip_format,
                run_credo=v.run_credo and not skip_credo,
                # --strict-validation lowers the block threshold to "info" so any
                # Credo issue at all is treated as a failure. Otherwise leave as
                # configured (default "high").
                credo_block_severity="info" if strict_validation else v.credo_block_severity,
            ),
        )

    return base, warnings
