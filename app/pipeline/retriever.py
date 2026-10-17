"""Retriever (code, no LLM): cosine top-k over the requested tiers, merged across queries."""

from dataclasses import dataclass

from app.memory.embed import Embedder
from app.memory.models import MemoryEntry, StoredTier
from app.memory.store import MemoryStore


@dataclass
class Candidate:
    entry: MemoryEntry
    score: float  # best cosine over all queries


async def retrieve(
    store: MemoryStore, embedder: Embedder, queries: list[str], tiers: list[StoredTier], k: int
) -> list[Candidate]:
    if not queries or not tiers or k <= 0:
        return []
    best: dict[str, Candidate] = {}
    for qvec in await embedder.embed(queries):
        for entry, score in store.search(qvec, tiers, k):
            if entry.id not in best or score > best[entry.id].score:
                best[entry.id] = Candidate(entry, score)
    return sorted(best.values(), key=lambda c: c.score, reverse=True)[:k]
