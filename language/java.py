"""Java source parsing — package/class discovery, imports, test-file matching.

Pure Python regex-based. Deliberately simple. Handles what v2 handled:
- Extract package + primary class name
- Find in-project dep references (word-bounded class-name matches on stripped source)
- Locate test files by convention (`FooTest.java` or `TestFoo*.java`)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JavaClass:
    """One tracked Java source file."""
    path: Path                # absolute
    package: str              # e.g. "org.joda.money"
    class_name: str           # e.g. "BigMoney"
    fqn: str                  # e.g. "org.joda.money.BigMoney"


_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)

_STRING_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_noise(java_source: str) -> str:
    """Remove comments and string literals so identifier searches don't false-match."""
    s = _BLOCK_COMMENT.sub(" ", java_source)
    s = _LINE_COMMENT.sub(" ", s)
    s = _STRING_LITERAL.sub(" ", s)
    return s


def parse_package(java_source: str) -> str:
    """Extract the Java package declaration; empty string if none found."""
    m = _PACKAGE_RE.search(java_source)
    return m.group(1) if m else ""


def discover_classes(source_root: Path, sources_glob: str) -> list[JavaClass]:
    """Scan the source root and produce a JavaClass per .java file.

    `module-info.java` is skipped. Class name is the file stem.
    """
    result: list[JavaClass] = []
    for path in sorted(source_root.glob(sources_glob)):
        if path.name == "module-info.java":
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        pkg = parse_package(text)
        class_name = path.stem
        fqn = f"{pkg}.{class_name}" if pkg else class_name
        result.append(JavaClass(path=path.resolve(), package=pkg,
                                class_name=class_name, fqn=fqn))
    return result


def discover_deps(classes: list[JavaClass]) -> dict[str, set[str]]:
    """For each class, find which OTHER in-project classes it references.

    Word-boundary regex on stripped source, so `Money` doesn't match
    `MoneyFormatter`. Returns {class_name: {dep_class_name, ...}}.
    """
    stem_to_class = {c.class_name: c for c in classes}
    deps: dict[str, set[str]] = {}
    for c in classes:
        try:
            stripped = strip_noise(c.path.read_text())
        except OSError:
            deps[c.class_name] = set()
            continue
        my_deps: set[str] = set()
        for name in stem_to_class:
            if name == c.class_name:
                continue
            if re.search(rf"\b{re.escape(name)}\b", stripped):
                my_deps.add(name)
        deps[c.class_name] = my_deps
    return deps


def find_test_files(java_class: JavaClass, tests_root: Path) -> list[Path]:
    """Find Java test files that cover `java_class`.

    Two conventions:
      - `FooTest.java`             (SomethingTest)
      - `TestFoo.java` / `TestFoo_*.java` / `TestFoo<digits>.java`
    We match by class-stem, walking `tests_root` recursively for `.java` files.
    Word-boundary guards prevent `TestMoney` from also matching `TestMoneyUtils`.
    """
    if not tests_root.exists():
        return []
    stem = java_class.class_name
    prefix = f"Test{stem}"
    matched: list[Path] = []
    for tf in tests_root.rglob("*.java"):
        if tf.stem == f"{stem}Test":
            matched.append(tf.resolve())
        elif tf.stem.startswith(prefix):
            rest = tf.stem[len(prefix):]
            if not rest or rest[0] in ("_", *"0123456789"):
                matched.append(tf.resolve())
    return sorted(matched)
