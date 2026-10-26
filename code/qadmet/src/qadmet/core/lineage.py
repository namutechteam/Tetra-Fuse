"""Tracks artifact dependency graph.

Each artifact's manifest has ``inputs: [{kind, name, run_id}, ...]``.
``trace_dependencies`` expands that graph via BFS and returns all ancestor artifacts.
``to_dot`` generates a graphviz dot string so humans can visualize it.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from qadmet.core.artifacts import Artifact, ArtifactStore


def trace_dependencies(
    store: ArtifactStore,
    artifact: Artifact,
    *,
    max_depth: int = 32,
) -> list[Artifact]:
    """Returns all ancestors of the artifact (including itself, in BFS order)."""
    seen: set[tuple[str, str, str]] = set()
    order: list[Artifact] = []
    queue: deque[tuple[Artifact, int]] = deque([(artifact, 0)])
    while queue:
        node, depth = queue.popleft()
        key = (node.kind, node.name, node.run_id)
        if key in seen:
            continue
        seen.add(key)
        order.append(node)
        if depth >= max_depth:
            continue
        for inp in node.manifest.get("inputs", []) or []:
            try:
                child = store.load(inp["kind"], inp["name"], inp["run_id"])
            except FileNotFoundError:
                continue
            queue.append((child, depth + 1))
    return order


def to_dot(nodes: Iterable[Artifact]) -> str:
    """Outputs a list of artifacts as a graphviz dot graph."""
    nodes = list(nodes)
    lines = ["digraph qadmet_lineage {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    seen: set[str] = set()
    for n in nodes:
        nid = f"{n.kind}__{n.name}__{n.run_id}"
        if nid in seen:
            continue
        seen.add(nid)
        label = f"{n.kind}\\n{n.name}\\n{n.run_id}"
        lines.append(f'  "{nid}" [label="{label}"];')
    for n in nodes:
        nid = f"{n.kind}__{n.name}__{n.run_id}"
        for inp in n.manifest.get("inputs", []) or []:
            iid = f"{inp['kind']}__{inp['name']}__{inp['run_id']}"
            lines.append(f'  "{iid}" -> "{nid}";')
    lines.append("}")
    return "\n".join(lines)
