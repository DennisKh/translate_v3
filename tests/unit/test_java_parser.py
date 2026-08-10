"""language/java.py — package/dep discovery."""

from pathlib import Path

from language.java import (
    JavaClass,
    discover_classes,
    discover_deps,
    find_test_files,
    parse_package,
    strip_noise,
)


def _write(dir: Path, name: str, content: str) -> Path:
    p = dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_parse_package():
    assert parse_package("package org.joda.money;\nclass X {}") == "org.joda.money"
    assert parse_package("class X {}") == ""


def test_strip_noise_removes_comments_and_strings():
    src = '/* hi */ class X { String s = "Money"; // Money in comment\n }'
    stripped = strip_noise(src)
    assert "Money" not in stripped


def test_discover_classes_skips_module_info(tmp_path):
    _write(tmp_path, "src/main/java/org/joda/money/Money.java",
           "package org.joda.money;\nclass Money {}\n")
    _write(tmp_path, "src/main/java/module-info.java", "module m {}\n")

    classes = discover_classes(tmp_path, "src/main/java/**/*.java")
    stems = {c.class_name for c in classes}
    assert stems == {"Money"}
    m = next(c for c in classes if c.class_name == "Money")
    assert m.package == "org.joda.money"
    assert m.fqn == "org.joda.money.Money"


def test_discover_deps_word_boundaries(tmp_path):
    # Money should not match inside MoneyFormatter
    _write(tmp_path, "src/main/java/pkg/Money.java",
           "package pkg;\nclass Money { }\n")
    _write(tmp_path, "src/main/java/pkg/MoneyFormatter.java",
           "package pkg;\nclass MoneyFormatter { Money m; }\n")
    _write(tmp_path, "src/main/java/pkg/Other.java",
           "package pkg;\nclass Other { }\n")

    classes = discover_classes(tmp_path, "src/main/java/**/*.java")
    deps = discover_deps(classes)

    # Money has no deps
    assert deps["Money"] == set()
    # MoneyFormatter references Money (not itself, not Other)
    assert deps["MoneyFormatter"] == {"Money"}
    # Other has nothing
    assert deps["Other"] == set()


def test_discover_deps_ignores_comments_and_strings(tmp_path):
    _write(tmp_path, "src/main/java/pkg/Widget.java",
           "package pkg;\nclass Widget { }\n")
    _write(tmp_path, "src/main/java/pkg/Other.java",
           'package pkg;\nclass Other { String s = "Widget"; // Widget in comment\n}\n')

    classes = discover_classes(tmp_path, "src/main/java/**/*.java")
    deps = discover_deps(classes)
    assert deps["Other"] == set()  # Widget only appears in string+comment


def test_find_test_files_matches_conventions(tmp_path):
    money = JavaClass(path=tmp_path / "Money.java", package="p",
                      class_name="Money", fqn="p.Money")
    tests_root = tmp_path / "tests"
    _write(tests_root, "TestMoney.java", "class TestMoney {}")
    _write(tests_root, "TestMoney_Extra.java", "class TestMoney_Extra {}")
    _write(tests_root, "TestMoneyUtils.java", "class TestMoneyUtils {}")  # NOT for Money
    _write(tests_root, "MoneyTest.java", "class MoneyTest {}")            # SomethingTest form

    result = {p.name for p in find_test_files(money, tests_root)}
    assert "TestMoney.java" in result
    assert "TestMoney_Extra.java" in result
    assert "MoneyTest.java" in result
    assert "TestMoneyUtils.java" not in result  # word-boundary saves us
