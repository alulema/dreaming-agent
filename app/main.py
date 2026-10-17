"""FastAPI entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.config import get_settings
from app.llm import OllamaClient
from app.memory.embed import Embedder
from app.memory.store import SQLiteStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    llm = OllamaClient(
        s.ollama_url, s.llm_model, format_mode=s.llm_format, temperature=s.llm_temperature,
        seed=s.llm_seed, timeout_s=s.llm_timeout_s, json_retries=s.llm_json_retries,
    )
    app.state.settings = s
    app.state.llm = llm
    app.state.embedder = Embedder(llm, s.embed_model, s.embed_timeout_s)
    app.state.store = SQLiteStore(s.db_path)
    yield
    app.state.store.close()
    await llm.aclose()


app = FastAPI(title="Dreaming Agent", lifespan=lifespan)


def _has_model(available: list[str], wanted: str) -> bool:
    # Ollama reports "bge-m3:latest" for a model pulled as "bge-m3".
    return wanted in available or f"{wanted}:latest" in available


@app.get("/health")
async def health(request: Request) -> dict:
    s = request.app.state.settings
    try:
        available = await request.app.state.llm.list_models()
    except Exception as e:  # noqa: BLE001 - health must never raise
        return {"status": "degraded", "ollama": False, "error": repr(e),
                "llm_model": s.llm_model, "embed_model": s.embed_model}
    llm_ok, embed_ok = _has_model(available, s.llm_model), _has_model(available, s.embed_model)
    return {
        "status": "ok" if llm_ok and embed_ok else "degraded",
        "ollama": True,
        "llm_model": {"name": s.llm_model, "available": llm_ok},
        "embed_model": {"name": s.embed_model, "available": embed_ok},
        "models": sorted(available),
        "memory": request.app.state.store.stats(),
        "llm_stats": request.app.state.llm.stats.as_dict(),
    }
