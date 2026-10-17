"""Groups active MTM entries for the Consolidator, so the model never sees all memory at once.

Two MTM entries are linked when their cosine similarity reaches `threshold`, or when they share
a topic (other than "other") or an entity. Connected components become clusters; components
larger than `max_size` are split. Entries with no links are batched together so they can still
be promoted or forgotten. Each cluster also carries related LTM entries as context, so a new
fact can UPDATE consolidated knowledge.
"""

from dataclasses import dataclass, field

import numpy as np

from app.memory.embed import from_bytes
from app.memory.models import MemoryEntry


@dataclass
class Cluster:
    mtm: list[MemoryEntry]
    ltm: list[MemoryEntry] = field(default_factory=list)
    linked: bool = True  # False => a batch of unrelated singletons


def _keys(e: MemoryEntry) -> set[str]:
    topics = {f"t:{t}" for t in e.topics if t != "other"}
    entities = {f"e:{x.strip().lower()}" for x in e.entities if x.strip()}
    return topics | entities


def _matrix(entries: list[MemoryEntry]) -> np.ndarray:
    return np.stack([from_bytes(e.embedding) for e in entries])


def _components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_clusters(
    mtm: list[MemoryEntry],
    ltm: list[MemoryEntry],
    *,
    threshold: float,
    max_size: int,
    max_ltm_context: int,
) -> list[Cluster]:
    if not mtm:
        return []
    sims = _matrix(mtm) @ _matrix(mtm).T
    keys = [_keys(e) for e in mtm]
    edges = [
        (i, j)
        for i in range(len(mtm))
        for j in range(i + 1, len(mtm))
        if sims[i, j] >= threshold or keys[i] & keys[j]
    ]
    linked: list[list[MemoryEntry]] = []
    singletons: list[MemoryEntry] = []
    for comp in _components(len(mtm), edges):
        members = sorted((mtm[i] for i in comp), key=lambda e: (e.created_at, e.id))
        if len(members) > 1:
            linked.append(members)
        else:
            singletons.extend(members)

    clusters = [Cluster(mtm=c) for comp in linked for c in _chunks(comp, max_size)]
    singletons.sort(key=lambda e: (e.created_at, e.id))
    clusters += [Cluster(mtm=c, linked=False) for c in _chunks(singletons, max_size)]

    if ltm:
        ltm_mat, ltm_keys = _matrix(ltm), [_keys(e) for e in ltm]
        for c in clusters:
            score = (_matrix(c.mtm) @ ltm_mat.T).max(axis=0)
            ckeys = set().union(*(_keys(e) for e in c.mtm))
            related = [
                (float(score[j]), ltm[j]) for j in range(len(ltm))
                if score[j] >= threshold or ltm_keys[j] & ckeys
            ]
            related.sort(key=lambda x: -x[0])
            c.ltm = [e for _, e in related[:max_ltm_context]]
    return clusters
