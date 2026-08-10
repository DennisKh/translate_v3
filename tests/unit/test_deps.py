"""project/deps.py — topological level computation + SCC detection."""

from project.deps import build


def test_pure_leaves_are_level_zero():
    # A, B, C all have no deps → all at level 0
    graph = build({"A": set(), "B": set(), "C": set()})
    assert set(graph.levels[0]) == {"A", "B", "C"}
    assert graph.sccs == []


def test_chain_produces_ascending_levels():
    # A → B → C  (C is a leaf, then B, then A)
    graph = build({"A": {"B"}, "B": {"C"}, "C": set()})
    assert graph.levels[0] == ["C"]
    assert graph.levels[1] == ["B"]
    assert graph.levels[2] == ["A"]
    assert graph.level_of == {"A": 2, "B": 1, "C": 0}


def test_diamond_dependency():
    # D depends on B and C; both depend on A
    #     A
    #    / \
    #   B   C
    #    \ /
    #     D
    graph = build({
        "A": set(),
        "B": {"A"},
        "C": {"A"},
        "D": {"B", "C"},
    })
    assert graph.level_of["A"] == 0
    assert graph.level_of["B"] == 1
    assert graph.level_of["C"] == 1
    assert graph.level_of["D"] == 2


def test_cycle_members_share_a_level():
    # A <-> B (mutual reference)
    graph = build({"A": {"B"}, "B": {"A"}})
    # Both in same SCC, same level
    assert graph.level_of["A"] == graph.level_of["B"]
    assert graph.sccs == [["A", "B"]]


def test_cycle_plus_dependent():
    # A <-> B, C depends on A
    graph = build({"A": {"B"}, "B": {"A"}, "C": {"A"}})
    assert graph.level_of["A"] == graph.level_of["B"] == 0
    assert graph.level_of["C"] == 1
    assert graph.sccs == [["A", "B"]]


def test_dep_list_sorted():
    graph = build({"A": {"B", "C", "D"}, "B": set(), "C": set(), "D": set()})
    assert graph.dep_list("A") == ["B", "C", "D"]


def test_levels_within_level_are_sorted():
    graph = build({"Zoo": set(), "Apple": set(), "Mango": set()})
    assert graph.levels[0] == ["Apple", "Mango", "Zoo"]
