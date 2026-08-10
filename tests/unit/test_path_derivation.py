"""main.py — _common_package_prefix + _lib_path_for.

These are the small pure functions that turn Java package/class into Elixir
module + lib path. Live in main.py so imported via runpy indirection to keep
main.py's shell out of import cycles for tests.
"""

import importlib.util
import sys
from pathlib import Path


def _load_main():
    """main.py isn't a package member; load it as a module."""
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_main", root / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_main = _load_main()


class _FakeJC:
    def __init__(self, package, class_name):
        self.package = package
        self.class_name = class_name


def test_common_prefix_all_same_package():
    classes = [_FakeJC("org.joda.money", "Money"),
               _FakeJC("org.joda.money", "BigMoney"),
               _FakeJC("org.joda.money", "CurrencyUnit")]
    assert _main._common_package_prefix(classes) == "org.joda.money"


def test_common_prefix_with_subpackages():
    classes = [_FakeJC("org.joda.money", "Money"),
               _FakeJC("org.joda.money.format", "MoneyFormatter")]
    assert _main._common_package_prefix(classes) == "org.joda.money"


def test_common_prefix_disjoint():
    classes = [_FakeJC("com.acme.foo", "Foo"),
               _FakeJC("org.bar.baz", "Baz")]
    assert _main._common_package_prefix(classes) == ""


def test_common_prefix_empty_when_no_classes():
    assert _main._common_package_prefix([]) == ""


def test_sub_package_root():
    jc = _FakeJC("org.joda.money", "Money")
    assert _main._sub_package(jc, "org.joda.money") == []


def test_sub_package_nested():
    jc = _FakeJC("org.joda.money.format", "MoneyFormatter")
    assert _main._sub_package(jc, "org.joda.money") == ["format"]


def test_lib_path_root_class():
    jc = _FakeJC("org.joda.money", "Money")
    module, lib, test = _main._lib_path_for(
        jc, "org.joda.money", "joda_money", "JodaMoney",
    )
    assert module == "JodaMoney.Money"
    assert lib == "lib/joda_money/money.ex"
    assert test == "test/joda_money/money_test.exs"


def test_lib_path_nested_class():
    jc = _FakeJC("org.joda.money.format", "MoneyFormatter")
    module, lib, test = _main._lib_path_for(
        jc, "org.joda.money", "joda_money", "JodaMoney",
    )
    assert module == "JodaMoney.Format.MoneyFormatter"
    assert lib == "lib/joda_money/format/money_formatter.ex"
    assert test == "test/joda_money/format/money_formatter_test.exs"


def test_lib_path_class_with_multiple_capitals():
    jc = _FakeJC("org.joda.money", "BigMoneyProvider")
    module, lib, _ = _main._lib_path_for(
        jc, "org.joda.money", "joda_money", "JodaMoney",
    )
    assert module == "JodaMoney.BigMoneyProvider"
    assert lib == "lib/joda_money/big_money_provider.ex"


# ---------------------------------------------------------------------------
# _resolve_tests_root — Maven convention discovery
# ---------------------------------------------------------------------------

def test_resolve_tests_root_maven_layout(tmp_path):
    from types import SimpleNamespace
    (tmp_path / "src" / "test" / "java" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "test" / "java" / "pkg" / "Foo.java").write_text("class Foo {}")

    cfg = SimpleNamespace(
        source=SimpleNamespace(root=tmp_path, tests_glob="src/test/java/**/*.java")
    )
    root = _main._resolve_tests_root(cfg)
    assert root == tmp_path / "src" / "test" / "java"


def test_resolve_tests_root_missing_dir(tmp_path):
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        source=SimpleNamespace(root=tmp_path, tests_glob="src/test/java/**/*.java")
    )
    assert _main._resolve_tests_root(cfg) is None


def test_resolve_tests_root_none_source_root():
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        source=SimpleNamespace(root=None, tests_glob="src/test/java/**/*.java")
    )
    assert _main._resolve_tests_root(cfg) is None
