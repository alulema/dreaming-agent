"""Ollama client: JSON output validated with Pydantic, timeouts, and deterministic fallbacks.

Every LLM call is attributed to a pipeline module ("controller", "writer", ...) so we can count,
per module, how often the model produced valid output, invalid JSON, or timed out, and how
often a fallback had to be used. Those counters are data for the write-up.
"""

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Outcome = Literal["ok", "invalid_json", "timeout", "http_error"]


@dataclass
class LLMResult(Generic[T]):
    value: T
    ok: bool  # False => `value` comes from the fallback
    attempts: list[Outcome]
    latency_ms: float
    raw: str | None = None  # last raw model output, for traces
    error: str | None = None

    @property
    def fallback_used(self) -> bool:
        return not self.ok


@dataclass
class ModuleStats:
    calls: int = 0
    ok: int = 0
    invalid_json: int = 0
    timeout: int = 0
    http_error: int = 0
    fallback: int = 0
    total_ms: float = 0.0


@dataclass
class LLMStats:
    modules: dict[str, ModuleStats] = field(default_factory=lambda: defaultdict(ModuleStats))

    def record(self, module: str, attempts: list[Outcome], fallback: bool, ms: float) -> None:
        s = self.modules[module]
        s.calls += len(attempts)
        for a in attempts:
            setattr(s, a, getattr(s, a) + 1)
        s.fallback += int(fallback)
        s.total_ms += ms

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {m: vars(s).copy() for m, s in self.modules.items()}

    def reset(self) -> None:
        self.modules.clear()


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        format_mode: Literal["schema", "json"] = "schema",
        temperature: float = 0.0,
        seed: int | None = 42,
        timeout_s: float = 60.0,
        json_retries: int = 1,
        num_thread: int | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.format_mode = format_mode
        self.temperature = temperature
        self.seed = seed
        self.timeout_s = timeout_s
        self.json_retries = json_retries
        # Under a CPU quota (container limits) llama.cpp still sees every host core and starts one
        # thread per core; the quota then throttles them all. Pinning it to the quota is ~15x faster.
        self.num_thread = num_thread
        self.http = http or httpx.AsyncClient()
        self.stats = LLMStats()

    async def aclose(self) -> None:
        await self.http.aclose()

    def _options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {"temperature": self.temperature}
        if self.seed is not None:
            opts["seed"] = self.seed
        if self.num_thread:
            opts["num_thread"] = self.num_thread
        return opts

    async def _chat(self, messages: list[dict], fmt: Any, timeout: float) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(),
        }
        if fmt is not None:
            payload["format"] = fmt
        r = await self.http.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    async def chat_json(
        self,
        module: str,
        messages: list[dict],
        schema: type[T],
        fallback: Callable[[], T],
        *,
        timeout: float | None = None,
    ) -> LLMResult[T]:
        """Ask for JSON matching `schema`. Invalid JSON is retried; timeouts are not
        (a retry would double an already-blown latency budget). Never raises on model
        misbehaviour: returns the fallback value instead."""
        fmt = schema.model_json_schema() if self.format_mode == "schema" else "json"
        timeout = timeout or self.timeout_s
        attempts: list[Outcome] = []
        raw: str | None = None
        error: str | None = None
        t0 = time.perf_counter()

        for _ in range(1 + self.json_retries):
            try:
                raw = await self._chat(messages, fmt, timeout)
            except httpx.TimeoutException as e:
                attempts.append("timeout")
                error = f"timeout: {e!r}"
                break
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                attempts.append("http_error")
                error = f"http_error: {e!r}"
                break
            try:
                value = schema.model_validate_json(raw)
            except ValidationError as e:
                attempts.append("invalid_json")
                error = f"invalid_json: {e.error_count()} error(s): {e.errors()[0]['msg']}"
                continue
            attempts.append("ok")
            ms = (time.perf_counter() - t0) * 1000
            self.stats.record(module, attempts, fallback=False, ms=ms)
            return LLMResult(value=value, ok=True, attempts=attempts, latency_ms=ms, raw=raw)

        ms = (time.perf_counter() - t0) * 1000
        log.warning("llm fallback module=%s attempts=%s error=%s", module, attempts, error)
        self.stats.record(module, attempts, fallback=True, ms=ms)
        return LLMResult(
            value=fallback(), ok=False, attempts=attempts, latency_ms=ms, raw=raw, error=error
        )

    async def chat_text(
        self, module: str, messages: list[dict], *, timeout: float | None = None
    ) -> str:
        """Free-text generation (the Responder). Errors propagate: there is no sensible
        deterministic fallback for an answer."""
        t0 = time.perf_counter()
        try:
            text = await self._chat(messages, None, timeout or self.timeout_s)
        except httpx.TimeoutException:
            self.stats.record(module, ["timeout"], fallback=False, ms=(time.perf_counter() - t0) * 1000)
            raise
        self.stats.record(module, ["ok"], fallback=False, ms=(time.perf_counter() - t0) * 1000)
        return text

    async def embed(self, texts: list[str], model: str, *, timeout: float = 30.0) -> list[list[float]]:
        if not texts:
            return []
        r = await self.http.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts,
                  "options": {"num_thread": self.num_thread} if self.num_thread else {}},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    async def list_models(self, *, timeout: float = 5.0) -> list[str]:
        r = await self.http.get(f"{self.base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
