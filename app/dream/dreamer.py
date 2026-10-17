"""The Dream: cluster → propose (LLM) → validate (code) → apply (transaction) → diff."""

import time

from app.config import Settings
from app.dream.applier import commit, prepare
from app.dream.cluster import build_clusters
from app.dream.consolidator import propose_ops
from app.dream.validator import Rules, stale_warnings, validate_ops
from app.llm import OllamaClient
from app.memory.embed import Embedder
from app.memory.models import new_id
from app.memory.store import MemoryStore


def rules_from(settings: Settings) -> Rules:
    return Rules(promote_min_turns=settings.promote_min_turns,
                 promote_min_access=settings.promote_min_access,
                 max_entry_chars=settings.max_entry_chars)


async def run_dream(
    store: MemoryStore, embedder: Embedder, llm: OllamaClient, settings: Settings
) -> dict:
    t0 = time.perf_counter()
    rules = rules_from(settings)
    stats_before, before = store.stats(), store.snapshot()

    clusters = build_clusters(
        store.list_entries(["MTM"]), store.list_entries(["LTM"]),
        threshold=settings.cluster_threshold, max_size=settings.max_cluster_size,
        max_ltm_context=settings.max_ltm_context,
    )

    ops, scope, allowed, llm_calls, fallbacks = [], [], {}, 0, 0
    for i, cluster in enumerate(clusters):
        proposal = await propose_ops(llm, cluster, rules)
        llm_calls += len(proposal.result.attempts)
        fallbacks += int(proposal.result.fallback_used)
        allowed[i] = set(proposal.labels.values())
        ops += proposal.ops
        scope += [i] * len(proposal.ops)

    entries = {e.id: e for e in store.list_entries(include_inactive=True)}
    valid, rejected = validate_ops(ops, entries, rules, allowed, scope)
    warnings = stale_warnings(valid, entries, [{e.id for e in c.mtm + c.ltm} for c in clusters])
    prepared = await prepare(store, embedder, valid)

    dream_id = new_id()
    with store.transaction():
        applied = commit(store, prepared, rules, settings.merge_promotes)
        diff = {
            "dream_id": dream_id,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            "llm_calls": llm_calls,
            "fallbacks": fallbacks,
            "model": llm.model,
            "clusters": [
                {"mtm": [e.id for e in c.mtm], "ltm": [e.id for e in c.ltm], "linked": c.linked}
                for c in clusters
            ],
            "ops_proposed": len(ops),
            "ops_applied": applied,
            "ops_rejected": [
                {"op": r.op.op, "target_ids": r.op.target_ids,
                 "targets": [entries[x].text if x in entries else x for x in r.op.target_ids],
                 "new_text": r.op.new_text, "reason": r.op.reason, "error": r.error}
                for r in rejected
            ],
            "warnings": warnings,
            "stats_before": stats_before,
            "stats_after": store.stats(),
        }
        store.save_dream(dream_id, diff, before, store.snapshot())
    return diff
