"""A scripted stand-in for OllamaClient: no network, deterministic embeddings."""

import hashlib
from collections import defaultdict, deque

import numpy as np

from app.llm import LLMResult, LLMStats


def fake_vector(text: str, dim: int = 16) -> list[float]:
    """Deterministic pseudo-embedding. Texts sharing a `#tag` share a direction, so tests
    can control similarity: "#dog x" and "#dog y" are near-identical, "#cat z" is not."""
    tag = text.split()[0] if text.startswith("#") else text
    seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


class FakeLLM:
    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.stats = LLMStats()
        self.json_replies: dict[str, deque] = defaultdict(deque)  # module -> values or "FAIL"
        self.text_replies: deque[str] = deque()
        self.calls: list[tuple[str, list[dict]]] = []

    def script(self, module: str, *values) -> "FakeLLM":
        self.json_replies[module].extend(values)
        return self

    async def chat_json(self, module, messages, schema, fallback, *, timeout=None):
        self.calls.append((module, messages))
        value = self.json_replies[module].popleft() if self.json_replies[module] else "FAIL"
        if value == "FAIL":
            self.stats.record(module, ["invalid_json"], fallback=True, ms=0)
            return LLMResult(value=fallback(), ok=False, attempts=["invalid_json"], latency_ms=0,
                             error="scripted failure")
        self.stats.record(module, ["ok"], fallback=False, ms=0)
        return LLMResult(value=schema.model_validate(value), ok=True, attempts=["ok"],
                         latency_ms=0)

    async def chat_text(self, module, messages, *, timeout=None):
        self.calls.append((module, messages))
        self.stats.record(module, ["ok"], fallback=False, ms=0)
        return self.text_replies.popleft() if self.text_replies else "ok"

    async def embed(self, texts, model, *, timeout=30.0):
        return [fake_vector(t) for t in texts]

    async def list_models(self, *, timeout=5.0):
        return [self.model, "bge-m3:latest"]

    async def aclose(self):
        pass
