"""Config loading + precedence + validation warnings."""

from pathlib import Path

import pytest

from project.config import Config, build_config, load_toml


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "translate.toml"
    p.write_text(content)
    return p


def test_defaults_only(tmp_path):
    cfg, warnings = build_config(
        source_root=Path("/src"), target_root=Path("/tgt"),
    )
    assert cfg.source.root == Path("/src")
    assert cfg.target.root == Path("/tgt")
    assert cfg.source.language == "java"
    assert cfg.target.language == "elixir"
    assert cfg.llm.model == "claude-opus-4-7"
    # Default effort is `medium` — chosen after testing showed `high` on
    # Sonnet 4.6 with adaptive thinking caused 30-min single-turn hangs on
    # the hardest files without commensurate output value.
    assert cfg.agent.effort == "medium"
    assert warnings == []


def test_positional_overrides_toml(tmp_path):
    toml = _write_toml(tmp_path, """
[source]
root = "/wrong/src"

[target]
root = "/wrong/tgt"
""")
    cfg, _ = build_config(
        source_root=Path("/right/src"), target_root=Path("/right/tgt"),
        config_file=toml,
    )
    assert cfg.source.root == Path("/right/src")
    assert cfg.target.root == Path("/right/tgt")


def test_cli_overrides_toml(tmp_path):
    toml = _write_toml(tmp_path, """
[llm]
model = "claude-sonnet-4-6"

[agent]
max_budget_tokens = 500_000
""")
    cfg, _ = build_config(
        source_root=Path("/s"), target_root=Path("/t"),
        config_file=toml,
        model="claude-haiku-4-5",           # CLI wins
        max_budget_tokens=100_000,          # CLI wins
    )
    assert cfg.llm.model == "claude-haiku-4-5"
    assert cfg.agent.max_budget_tokens == 100_000


def test_unknown_key_produces_warning(tmp_path):
    toml = _write_toml(tmp_path, """
[agent]
totally_made_up_key = "hi"
""")
    _, warnings = build_config(
        source_root=Path("/s"), target_root=Path("/t"), config_file=toml,
    )
    assert any("totally_made_up_key" in w for w in warnings)


def test_unknown_top_section_produces_warning(tmp_path):
    toml = _write_toml(tmp_path, """
[wat]
foo = "bar"
""")
    _, warnings = build_config(
        source_root=Path("/s"), target_root=Path("/t"), config_file=toml,
    )
    assert any("[wat]" in w for w in warnings)


def test_nested_target_deps_supported(tmp_path):
    toml = _write_toml(tmp_path, """
[target.mix_deps]
decimal = "~> 2.0"
other_dep = "~> 1.0"

[target.dev_deps]
credo = "~> 1.7"
""")
    cfg, warnings = build_config(
        source_root=Path("/s"), target_root=Path("/t"), config_file=toml,
    )
    assert cfg.target.mix_deps == {"decimal": "~> 2.0", "other_dep": "~> 1.0"}
    assert cfg.target.dev_deps == {"credo": "~> 1.7"}
    # `[target.mix_deps]` is a dict-valued key — should NOT produce an unknown-key warning
    assert not any("mix_deps" in w for w in warnings)


def test_phase_selection():
    cfg_a, _ = build_config(Path("/s"), Path("/t"), phase="A")
    cfg_b, _ = build_config(Path("/s"), Path("/t"), phase="B")
    cfg_both, _ = build_config(Path("/s"), Path("/t"), phase="both")

    assert cfg_a.phases.translate and not cfg_a.phases.generate_tests
    assert not cfg_b.phases.translate and cfg_b.phases.generate_tests
    assert cfg_both.phases.translate and cfg_both.phases.generate_tests
