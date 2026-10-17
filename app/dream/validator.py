"""Validates consolidation ops. Pure and deterministic: the model proposes, the code disposes.

Ops are checked in order; an entry may be touched by at most one op per Dream, so a later op
that reuses an id is rejected as a conflict.
"""

import re
import unicodedata
from dataclasses import dataclass

from app.memory.models import MemoryEntry, Op


@dataclass(frozen=True)
class Rules:
    promote_min_turns: int = 2
    promote_min_access: int = 3
    max_entry_chars: int = 300


@dataclass
class Rejection:
    op: Op
    error: str


def has_evidence(entries: list[MemoryEntry], rules: Rules) -> bool:
    """A fact is stable enough for LTM if it was stated in several turns or used often."""
    turns = {t for e in entries for t in e.source_turn_ids}
    return (len(turns) >= rules.promote_min_turns
            or sum(e.access_count for e in entries) >= rules.promote_min_access)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


# Capitalised words (not sentence-initial "The") and numbers: the details a model can invent.
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-ZÁÉÍÓÚÑ][\wáéíóúñü]+)")
_NUMBER = re.compile(r"\d+")
_STOP = {"the", "user", "user's", "users"}


def ungrounded_terms(new_text: str, targets: list[MemoryEntry]) -> list[str]:
    """Names and numbers in `new_text` that appear in none of the target entries.

    Catches prompt-example leakage (Phase 3: the 7B proposed "The user lives in Bilbao
    (previously lived in Lima)", copied from the Consolidator's own example) and invented
    details. Deliberately strict: it grounds against the targets only, not the whole cluster."""
    source = _fold(" ".join([e.text for e in targets] + [x for e in targets for x in e.entities]))
    found = _NAME.findall(new_text) + _NUMBER.findall(new_text)
    return [t for t in dict.fromkeys(found) if _fold(t) not in _STOP and _fold(t) not in source]


def _mentions(text: str, term: str) -> bool:
    text, term = _fold(text), _fold(term).strip()
    if not term or term in text:
        return True
    words = [w for w in re.findall(r"\w+", term) if len(w) >= 3]
    return bool(words) and all(w in text for w in words)


def dropped_current_values(new_text: str, targets: list[MemoryEntry]) -> list[str]:
    """For each topic, the newest target holding it carries the current value; its entities
    must survive in `new_text`. Phase 3: a location UPDATE superseded an entry that also held
    the user's current job ("data analyst") and silently dropped it."""
    by_topic: dict[str, MemoryEntry] = {}
    for e in sorted(targets, key=lambda e: e.created_at):
        for t in e.topics or ["other"]:
            by_topic[t] = e
    current = {x for e in by_topic.values() for x in e.entities}
    return sorted(x for x in current if not _mentions(new_text, x))


def dropped_names(new_text: str, targets: list[MemoryEntry]) -> list[str]:
    """Names and numbers of any target missing from a MERGE result ("keep every detail")."""
    found = [t for e in targets for t in _NAME.findall(e.text) + _NUMBER.findall(e.text)]
    return [t for t in dict.fromkeys(found)
            if _fold(t) not in _STOP and not _mentions(new_text, t)]


def check_op(op: Op, targets: list[MemoryEntry], rules: Rules) -> str | None:
    """Op-specific rules; `targets` already exist, are active and unused."""
    text = (op.new_text or "").strip()
    tiers = {e.tier for e in targets}
    if op.op in ("MERGE", "UPDATE"):
        if len(targets) < 2:
            return f"{op.op} needs at least 2 entries"
        if not text:
            return f"{op.op} needs new_text"
        if len(text) > rules.max_entry_chars:
            return "new_text too long"
        if invented := ungrounded_terms(text, targets):
            return f"new_text not grounded in its targets: {invented}"
    if op.op == "MERGE" and len(tiers) > 1:
        return "MERGE across tiers"
    if op.op == "MERGE" and (lost := dropped_names(text, targets)):
        return f"MERGE loses details: {lost}"
    if op.op == "UPDATE" and "MTM" not in tiers:
        # Rewriting consolidated knowledge needs new evidence from conversation.
        return "UPDATE needs at least one MTM entry"
    if op.op == "UPDATE" and (lost := dropped_current_values(text, targets)):
        return f"UPDATE drops current value(s): {lost}"
    if op.op == "PROMOTE":
        if len(targets) != 1:
            return "PROMOTE takes exactly 1 entry"
        if targets[0].tier != "MTM":
            return "PROMOTE of an entry that is not MTM"
        if text and len(text) > rules.max_entry_chars:
            return "new_text too long"
        if text and (invented := ungrounded_terms(text, targets)):
            return f"new_text not grounded in its targets: {invented}"
        if not has_evidence(targets, rules):
            return (f"PROMOTE without evidence (needs >= {rules.promote_min_turns} turns "
                    f"or >= {rules.promote_min_access} accesses)")
    if op.op == "FORGET":
        if "LTM" in tiers:
            return "FORGET over LTM is not allowed"
        reinforced = [e for e in targets if len(e.source_turn_ids) >= rules.promote_min_turns]
        if reinforced:
            return "FORGET of a reinforced entry (stated in several turns)"
    return None


def stale_warnings(applied_ops: list[Op], entries: dict[str, MemoryEntry],
                   cluster_ids: list[set[str]]) -> list[dict]:
    """After an UPDATE, an older active entry in the same cluster that shares a topic with it
    probably holds the old value (Phase 3: "lives in Quito" survived the UPDATE to Madrid).
    Reported, not fixed: sharing a topic does not prove a contradiction."""
    used = {i for op in applied_ops for i in op.target_ids}
    warnings = []
    for op in applied_ops:
        if op.op != "UPDATE":
            continue
        targets = [entries[i] for i in op.target_ids]
        topics = {t for e in targets for t in e.topics if t != "other"}
        newest = max(e.created_at for e in targets)
        scope = next((c for c in cluster_ids if set(op.target_ids) <= c), set())
        for i in scope - used:
            e = entries[i]
            if e.is_active and e.created_at < newest and topics & set(e.topics):
                warnings.append({"entry_id": e.id, "text": e.text,
                                 "shared_topics": sorted(topics & set(e.topics)),
                                 "update_text": op.new_text})
    return warnings


def validate_ops(
    ops: list[Op],
    entries: dict[str, MemoryEntry],
    rules: Rules,
    allowed_ids: dict[int, set[str]] | None = None,
    op_scope: list[int] | None = None,
) -> tuple[list[Op], list[Rejection]]:
    """`entries`: every entry by id (any status). `allowed_ids[scope]`: ids the model was
    shown for that cluster; `op_scope[i]` says which cluster op i came from."""
    valid: list[Op] = []
    rejected: list[Rejection] = []
    used: set[str] = set()
    for i, op in enumerate(ops):
        ids = op.target_ids
        allowed = allowed_ids[op_scope[i]] if allowed_ids is not None and op_scope else None
        error = None
        if not ids:
            error = "no targets"
        elif len(set(ids)) != len(ids):
            error = "duplicate target"
        elif missing := [x for x in ids if x not in entries or (allowed and x not in allowed)]:
            error = f"unknown id(s): {missing}"
        elif inactive := [x for x in ids if not entries[x].is_active]:
            error = f"inactive entry(ies): {inactive}"
        elif clash := [x for x in ids if x in used]:
            error = f"conflict: entry(ies) already changed in this Dream: {clash}"
        else:
            error = check_op(op, [entries[x] for x in ids], rules)
        if error:
            rejected.append(Rejection(op, error))
        else:
            valid.append(op)
            used.update(ids)
    return valid, rejected
