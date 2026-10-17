"""Applies validated ops. Nothing is physically deleted: replaced entries are marked
`superseded` (pointing to their successor) and forgotten entries are soft-deleted.

Embeddings for new texts are computed first (`prepare`, async); all writes then happen in one
synchronous transaction (`commit`), so a failure leaves memory untouched.
"""

from dataclasses import dataclass

import numpy as np

from app.dream.validator import Rules, has_evidence
from app.memory.embed import Embedder, to_bytes
from app.memory.models import MemoryEntry, Op, StoredTier
from app.memory.store import MemoryStore


@dataclass
class Prepared:
    op: Op
    targets: list[MemoryEntry]
    new_text: str | None
    vector: np.ndarray | None


def _union(lists: list[list[str]]) -> list[str]:
    return list(dict.fromkeys(x for xs in lists for x in xs))


def result_tier(op: Op, targets: list[MemoryEntry], rules: Rules, merge_promotes: bool) -> StoredTier:
    if op.op == "PROMOTE" or any(e.tier == "LTM" for e in targets):
        return "LTM"
    if merge_promotes and has_evidence(targets, rules):
        return "LTM"
    return "MTM"


async def prepare(store: MemoryStore, embedder: Embedder, ops: list[Op]) -> list[Prepared]:
    prepared = []
    for op in ops:
        targets = [store.get_entry(i) for i in op.target_ids]
        text = None
        if op.op in ("MERGE", "UPDATE"):
            text = op.new_text.strip()
        elif op.op == "PROMOTE":
            text = (op.new_text or "").strip() or targets[0].text
        prepared.append(Prepared(op, targets, text, None))
    to_embed = [p for p in prepared if p.new_text]
    if to_embed:
        for p, v in zip(to_embed, await embedder.embed([p.new_text for p in to_embed]), strict=True):
            p.vector = v
    return prepared


def commit(
    store: MemoryStore, prepared: list[Prepared], rules: Rules, merge_promotes: bool
) -> list[dict]:
    """Must run inside `store.transaction()`."""
    applied = []
    for p in prepared:
        op, targets = p.op, p.targets
        record = {
            "op": op.op,
            "target_ids": op.target_ids,
            "targets": [{"id": e.id, "tier": e.tier, "text": e.text} for e in targets],
            "reason": op.reason,
            "new_id": None, "new_text": None, "new_tier": None,
        }
        if op.op == "FORGET":
            for e in targets:
                e.status = "forgotten"
                store.update_entry(e)
        else:
            newest = max(targets, key=lambda e: e.created_at)
            kinds = {e.kind for e in targets}
            tier = result_tier(op, targets, rules, merge_promotes)
            new = MemoryEntry(
                tier=tier,
                text=p.new_text,
                kind=newest.kind if op.op == "UPDATE" or len(kinds) > 1 else kinds.pop(),
                entities=_union([e.entities for e in targets]),
                topics=_union([e.topics for e in targets]),
                source_turn_ids=_union([e.source_turn_ids for e in targets]),
                access_count=(newest.access_count if op.op == "UPDATE"
                              else sum(e.access_count for e in targets)),
                last_accessed_at=max((e.last_accessed_at for e in targets if e.last_accessed_at),
                                     default=None),
                # The fact is as old as its newest source, not as old as the Dream: otherwise a
                # consolidated entry looks newer than facts learned after it (Phase 3 bug).
                created_at=newest.created_at,
                embedding=to_bytes(p.vector),
            )
            store.add_entry(new)
            for e in targets:
                e.status, e.superseded_by = "superseded", new.id
                store.update_entry(e)
            record.update(new_id=new.id, new_text=new.text, new_tier=tier)
        applied.append(record)
    return applied
