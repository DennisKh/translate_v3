"""Dependency graph analysis — topological levels + SCC-aware cycle handling.

Ported from v2 (translate.py) with cleaner interfaces. Java files that mutually
reference each other (cycles are common in mature Java libs) get assigned the
same level; SCCs of size > 1 are surfaced so the agent knows to expect them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepGraph:
    """Immutable dep graph over class stems (short names)."""
    deps: dict[str, set[str]]                     # stem -> set of stems it depends on
    levels: list[list[str]]                       # level_index -> sorted stems at that level
    sccs: list[list[str]]                         # cycles (size > 1), sorted
    level_of: dict[str, int]                      # stem -> its level

    def dep_list(self, stem: str) -> list[str]:
        return sorted(self.deps.get(stem, set()))


def _find_sccs(deps: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC — returns ALL SCCs including singletons."""
    index_counter = [0]
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in deps.get(v, ()):
            if w not in deps:
                continue
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w is v:
                    break
            sccs.append(scc)

    for v in deps:
        if v not in index:
            strongconnect(v)
    return sccs


def build(deps: dict[str, set[str]]) -> DepGraph:
    """Compute levels for a dep graph.

    All files in the same SCC share a level. Level of an SCC = 1 + max level
    of any other SCC it depends on. Level 0 = pure leaves (or leaf SCCs).

    Within a level, files are alphabetically sorted for determinism.
    """
    sccs = _find_sccs(deps)
    file_to_scc = {stem: i for i, scc in enumerate(sccs) for stem in scc}

    # Condense to SCC DAG
    scc_deps: dict[int, set[int]] = {i: set() for i in range(len(sccs))}
    for stem, stem_deps in deps.items():
        i = file_to_scc[stem]
        for d in stem_deps:
            if d not in file_to_scc:
                continue
            j = file_to_scc[d]
            if i != j:
                scc_deps[i].add(j)

    scc_level: dict[int, int] = {}

    def level_of_scc(i: int) -> int:
        if i in scc_level:
            return scc_level[i]
        if not scc_deps[i]:
            lvl = 0
        else:
            lvl = 1 + max(level_of_scc(j) for j in scc_deps[i])
        scc_level[i] = lvl
        return lvl

    for i in range(len(sccs)):
        level_of_scc(i)

    max_level = max(scc_level.values(), default=-1)
    grouped: list[list[str]] = [[] for _ in range(max_level + 1)]
    level_of: dict[str, int] = {}
    for i, scc in enumerate(sccs):
        lvl = scc_level[i]
        for stem in scc:
            grouped[lvl].append(stem)
            level_of[stem] = lvl
    for level in grouped:
        level.sort()

    real_sccs = sorted([sorted(s) for s in sccs if len(s) > 1])

    return DepGraph(deps=deps, levels=grouped, sccs=real_sccs, level_of=level_of)
