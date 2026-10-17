"""The four benchmark conditions behind one interface.

A: STM only · B: naive RAG over past user turns · C: LightMem-style pipeline · D: C + Dream
after every session. All four share the same Responder prompt, so the only difference is what
memory reaches it.
"""

import time
from typing import Literal, Protocol

from app.config import Settings
from app.dream.dreamer import run_dream
from app.llm import OllamaClient
from app.memory.embed import Embedder, to_bytes
from app.memory.models import Turn
from app.memory.store import MemoryStore
from app.pipeline.agent import Agent, ChatResult
from app.pipeline.responder import run_responder

Condition = Literal["A", "B", "C", "D"]


class Strategy(Protocol):
    async def chat(self, session_id: str, message: str, *, learn: bool = True) -> ChatResult: ...

    async def end_session(self, session_id: str) -> dict | None:
        """Hook run after each scripted session (the Dream, for D)."""


class _Base:
    def __init__(self, llm: OllamaClient, embedder: Embedder, store: MemoryStore,
                 settings: Settings) -> None:
        self.llm, self.embedder, self.store, self.settings = llm, embedder, store, settings

    async def end_session(self, session_id: str) -> dict | None:
        return None

    async def _answer(self, session_id: str, message: str, context: str, learn: bool,
                      lat: dict, user_vec=None) -> ChatResult:
        t0 = time.perf_counter()
        stm = self.store.recent_turns(session_id, self.settings.stm_turns)
        reply = await run_responder(self.llm, context, stm, message)
        lat["responder"] = round((time.perf_counter() - t0) * 1000, 1)
        if learn:
            emb = to_bytes(user_vec) if user_vec is not None else None
            self.store.add_turn(Turn(session_id=session_id, role="user", content=message,
                                     embedding=emb))
            self.store.add_turn(Turn(session_id=session_id, role="assistant", content=reply))
        lat["reply"] = lat["total"] = round(sum(v for k, v in lat.items()
                                                if k in ("retriever", "responder")), 1)
        return ChatResult(reply=reply, trace={
            "model": self.llm.model, "llm_calls": 1, "fallbacks": [], "repairs": {},
            "latency_ms": lat, "memory_context": context})


class NoMemory(_Base):
    """A: the Responder only sees the current session's last turns."""

    async def chat(self, session_id: str, message: str, *, learn: bool = True) -> ChatResult:
        return await self._answer(session_id, message, "", learn, {})


class NaiveRAG(_Base):
    """B: every past user message is embedded; the top-k go straight to the Responder,
    in chronological order, with no Controller, Selector, Writer or consolidation."""

    async def chat(self, session_id: str, message: str, *, learn: bool = True) -> ChatResult:
        t0 = time.perf_counter()
        vec = await self.embedder.embed_one(message)
        hits = self.store.search_turns(vec, self.settings.top_k)
        hits.sort(key=lambda h: h[0].created_at)
        context = "\n".join(f"- ({t.created_at:%Y-%m-%d %H:%M}) The user said: {t.content}"
                            for t, _ in hits)
        lat = {"retriever": round((time.perf_counter() - t0) * 1000, 1)}
        res = await self._answer(session_id, message, context, learn, lat, user_vec=vec)
        res.trace["retrieved"] = len(hits)
        return res


class Pipeline(_Base):
    """C: the full online pipeline (the app's Agent)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.agent = Agent(self.llm, self.embedder, self.store, self.settings)

    async def chat(self, session_id: str, message: str, *, learn: bool = True) -> ChatResult:
        return await self.agent.chat(session_id, message, learn=learn)


class PipelineWithDream(Pipeline):
    """D: C + a Dream after every session."""

    async def end_session(self, session_id: str) -> dict | None:
        return await run_dream(self.store, self.embedder, self.llm, self.settings)


STRATEGIES: dict[Condition, type] = {
    "A": NoMemory, "B": NaiveRAG, "C": Pipeline, "D": PipelineWithDream,
}


def make_strategy(condition: Condition, llm: OllamaClient, embedder: Embedder,
                  store: MemoryStore, settings: Settings) -> Strategy:
    return STRATEGIES[condition](llm, embedder, store, settings)
