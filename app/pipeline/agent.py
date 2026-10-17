"""Online pipeline orchestration: Controller → Retriever → Selector → Responder → Writer."""

import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm import LLMResult, OllamaClient
from app.memory.embed import Embedder, to_bytes
from app.memory.models import MemoryEntry, Turn
from app.memory.store import MemoryStore
from app.pipeline.controller import run_controller
from app.pipeline.prompts import load_prompt
from app.pipeline.responder import run_responder
from app.pipeline.retriever import retrieve
from app.pipeline.selector import Selection, filter_context, run_selector
from app.pipeline.writer import run_writer

PROMPT_NAMES = ("controller", "selector", "responder", "writer")


@dataclass
class ChatResult:
    reply: str
    trace: dict[str, Any]


def format_stm(turns: list[Turn]) -> str:
    return "\n".join(f"{t.role.capitalize()}: {t.content}" for t in turns)


def _llm_info(r: LLMResult) -> dict[str, Any]:
    return {"ok": r.ok, "attempts": r.attempts, "error": r.error}


class Agent:
    def __init__(
        self, llm: OllamaClient, embedder: Embedder, store: MemoryStore, settings: Settings,
        *, index_turns: bool = False,
    ) -> None:
        """`index_turns`: also embed each user turn, so a naive-RAG answer can search the
        raw conversation (the app's mode selector). Off in the benchmark."""
        self.index_turns = index_turns
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.settings = settings

    async def chat(self, session_id: str, message: str, *, learn: bool = True) -> ChatResult:
        """Answer, then (if `learn`) store the turn and write memory. `learn=False` is for
        benchmark questions, which must not change the memory they are measuring."""
        res = await self.answer(session_id, message)
        if learn:
            learned = await self.learn(session_id, message, res.reply,
                                       shares_new_info=res.trace["controller"]["shares_new_info"])
            merge_learn_trace(res.trace, learned)
        else:
            res.trace["writer"] = {"skipped": True, "entries": [], "reinforced": [], "dropped": 0}
            res.trace["latency_ms"]["total"] = res.trace["latency_ms"]["reply"]
        return res

    async def answer(self, session_id: str, message: str) -> ChatResult:
        """Controller → Retriever → Selector → Responder. Reads memory, never writes it."""
        s = self.settings
        t_start = time.perf_counter()
        lat: dict[str, float] = {}
        fallbacks: list[str] = []
        repairs: dict[str, list[str]] = {}
        llm_calls = 0

        def tick(stage: str, t0: float) -> None:
            lat[stage] = round((time.perf_counter() - t0) * 1000, 1)

        stm = self.store.recent_turns(session_id, s.stm_turns)

        # 1. Controller
        t0 = time.perf_counter()
        decision = await run_controller(self.llm, message, format_stm(stm))
        tick("controller", t0)
        llm_calls += len(decision.result.attempts)
        if decision.result.fallback_used:
            fallbacks.append("controller")
        if decision.repairs:
            repairs["controller"] = decision.repairs

        # 2-3. Retriever + Selector. In "controller" mode they are skipped when the Controller
        # says memory isn't needed; in "always" mode the user's message becomes the query.
        candidates, selection = [], None
        queries, tiers = decision.out.queries, decision.out.tiers
        forced = s.retrieval_mode == "always" and not decision.out.needs_memory
        if forced:
            queries, tiers = [message], ["MTM", "LTM"]
        if decision.out.needs_memory or forced:
            t0 = time.perf_counter()
            candidates = await retrieve(self.store, self.embedder, queries, tiers, s.top_k)
            tick("retriever", t0)

            t0 = time.perf_counter()
            if forced and s.selector_mode == "filter":
                # The Controller saw no need for memory: skip the Selector's LLM call and pass
                # the best-scored entries as light background.
                kept, context = filter_context([], candidates, s.selector_min_keep)
                selection = Selection(kept=kept, context=context, result=None, repairs=[])
            else:
                selection = await run_selector(
                    self.llm, message, candidates, s.selector_fallback_k,
                    mode=s.selector_mode, min_keep=s.selector_min_keep)
            tick("selector", t0)
            if selection.result is not None:
                llm_calls += len(selection.result.attempts)
                if selection.result.fallback_used:
                    fallbacks.append("selector")
            if selection.repairs:
                repairs["selector"] = selection.repairs
            # access_count counts entries that were actually *used*, not merely retrieved.
            self.store.touch([c.entry.id for c in selection.kept])

        memory_context = selection.context if selection else ""

        # 4. Responder
        t0 = time.perf_counter()
        reply = await run_responder(self.llm, memory_context, stm, message)
        tick("responder", t0)
        llm_calls += 1
        lat["reply"] = round((time.perf_counter() - t_start) * 1000, 1)

        trace = {
            "mode": "lightmem",
            "session_id": session_id,
            "model": self.llm.model,
            "controller": {**decision.out.model_dump(), **_llm_info(decision.result),
                           "forced_retrieval": forced},
            "retrieval": {
                "candidates": [
                    {"id": c.entry.id, "tier": c.entry.tier, "text": c.entry.text,
                     "score": round(c.score, 3)}
                    for c in candidates
                ],
            },
            "selector": None if selection is None else {
                "skipped": selection.result is None,
                "kept_ids": [c.entry.id for c in selection.kept],
                "context": selection.context,
                **(_llm_info(selection.result) if selection.result else {}),
            },
            "stm_turns": len(stm),
            "fallbacks": fallbacks,
            "repairs": repairs,
            "llm_calls": llm_calls,
            "latency_ms": lat,
            "prompt_versions": {n: load_prompt(n).version for n in PROMPT_NAMES},
        }
        return ChatResult(reply=reply, trace=trace)

    async def learn(self, session_id: str, message: str, reply: str, *,
                    shares_new_info: bool | None = None) -> dict[str, Any]:
        """Store the turn and write MTM. `shares_new_info` is the Controller's gate; None
        (no Controller ran, e.g. a RAG-mode answer) leaves the decision to the Writer's own
        `has_new_facts`."""
        s = self.settings
        t0 = time.perf_counter()
        user_vec = await self.embedder.embed_one(message) if self.index_turns else None
        user_turn = Turn(session_id=session_id, role="user", content=message,
                         embedding=to_bytes(user_vec) if user_vec is not None else None)
        self.store.add_turn(user_turn)
        self.store.add_turn(Turn(session_id=session_id, role="assistant", content=reply))

        written, llm_calls, fallbacks = None, 0, []
        if shares_new_info is not False:
            written = await run_writer(self.llm, message, grounding=s.writer_grounding)
            llm_calls += len(written.result.attempts)
            if written.result.fallback_used:
                fallbacks.append("writer")
        new_entries: list[MemoryEntry] = []
        reinforced: list[dict] = []
        if written and written.entries:
            vecs = await self.embedder.embed([e.text for e in written.entries])
            with self.store.transaction():
                for e, v in zip(written.entries, vecs, strict=True):
                    hit = self.store.search(v, ["MTM"], k=1)
                    if hit and hit[0][1] >= s.writer_dedup_threshold:
                        existing, score = hit[0]
                        if user_turn.id not in existing.source_turn_ids:
                            existing.source_turn_ids.append(user_turn.id)
                            self.store.update_entry(existing)
                        reinforced.append({"id": existing.id, "text": existing.text,
                                           "new_text": e.text, "score": round(score, 3)})
                        continue
                    entry = MemoryEntry(
                        tier="MTM", text=e.text, kind=e.kind, entities=e.entities,
                        topics=list(e.topics), source_turn_ids=[user_turn.id],
                        embedding=to_bytes(v),
                    )
                    self.store.add_entry(entry)
                    new_entries.append(entry)
        return {
            "turn_id": user_turn.id,
            "writer": {
                "skipped": written is None,
                "entries": [e.model_dump(include={"id", "text", "kind", "entities", "topics"})
                            for e in new_entries],
                "reinforced": reinforced,
                "dropped": written.dropped if written else 0,
                **(_llm_info(written.result) if written else {}),
            },
            "llm_calls": llm_calls,
            "fallbacks": fallbacks,
            "writer_ms": round((time.perf_counter() - t0) * 1000, 1) if written else None,
        }


def merge_learn_trace(trace: dict[str, Any], learned: dict[str, Any]) -> None:
    """Fold a `learn` result into an `answer` trace (the shape the benchmark and UI read)."""
    trace["turn_id"] = learned["turn_id"]
    trace["writer"] = learned["writer"]
    trace["llm_calls"] += learned["llm_calls"]
    trace["fallbacks"] += learned["fallbacks"]
    lat = trace["latency_ms"]
    if learned["writer_ms"] is not None:
        lat["writer"] = learned["writer_ms"]
    lat["total"] = round(lat["reply"] + (learned["writer_ms"] or 0), 1)
