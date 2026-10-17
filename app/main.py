"""FastAPI entrypoint."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import OllamaClient
from app.memory.embed import Embedder
from app.memory.models import new_id
from app.memory.store import SQLiteStore
from app.pipeline.agent import Agent

STATIC = Path(__file__).resolve().parent / "static"
# Embedding-only models are hidden from the chat-model selector.
EMBED_HINTS = ("embed", "bge-", "minilm", "e5-")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    llm = OllamaClient(
        s.ollama_url, s.llm_model, format_mode=s.llm_format, temperature=s.llm_temperature,
        seed=s.llm_seed, timeout_s=s.llm_timeout_s, json_retries=s.llm_json_retries,
    )
    store = SQLiteStore(s.db_path)
    embedder = Embedder(llm, s.embed_model, s.embed_timeout_s)
    app.state.settings = s
    app.state.llm = llm
    app.state.embedder = embedder
    app.state.store = store
    app.state.agent = Agent(llm, embedder, store, s)
    # Single-user demo: one turn at a time keeps memory writes ordered.
    app.state.lock = asyncio.Lock()
    yield
    store.close()
    await llm.aclose()


app = FastAPI(title="Dreaming Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)


class ModelRequest(BaseModel):
    name: str


def _has_model(available: list[str], wanted: str) -> bool:
    # Ollama reports "bge-m3:latest" for a model pulled as "bge-m3".
    return wanted in available or f"{wanted}:latest" in available


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health(request: Request) -> dict:
    s = request.app.state.settings
    llm: OllamaClient = request.app.state.llm
    try:
        available = await llm.list_models()
    except Exception as e:  # noqa: BLE001 - health must never raise
        return {"status": "degraded", "ollama": False, "error": repr(e),
                "llm_model": llm.model, "embed_model": s.embed_model}
    llm_ok, embed_ok = _has_model(available, llm.model), _has_model(available, s.embed_model)
    return {
        "status": "ok" if llm_ok and embed_ok else "degraded",
        "ollama": True,
        "llm_model": {"name": llm.model, "available": llm_ok},
        "embed_model": {"name": s.embed_model, "available": embed_ok},
        "models": sorted(available),
        "memory": request.app.state.store.stats(),
        "llm_stats": llm.stats.as_dict(),
    }


@app.post("/session/new")
async def new_session() -> dict:
    # STM is per session; MTM/LTM are kept across sessions.
    return {"session_id": new_id()}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> dict:
    async with request.app.state.lock:
        try:
            res = await request.app.state.agent.chat(req.session_id, req.message.strip())
        except httpx.TimeoutException as e:
            raise HTTPException(504, "The local model took too long to answer.") from e
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Model server error: {e!r}") from e
    return {"reply": res.reply, "trace": res.trace}


@app.get("/memory")
async def memory(request: Request, session_id: str | None = None) -> dict:
    s, store = request.app.state.settings, request.app.state.store
    stm = store.recent_turns(session_id, s.stm_turns) if session_id else []
    return {
        "stm": [t.model_dump(mode="json") for t in stm],
        "mtm": [e.model_dump(mode="json") for e in store.list_entries(["MTM"])],
        "ltm": [e.model_dump(mode="json") for e in store.list_entries(["LTM"])],
        "stats": store.stats(),
    }


@app.post("/reset")
async def reset(request: Request) -> dict:
    async with request.app.state.lock:
        request.app.state.store.reset()
        request.app.state.llm.stats.reset()
    return {"ok": True}


@app.get("/models")
async def models(request: Request) -> dict:
    llm: OllamaClient = request.app.state.llm
    try:
        available = await llm.list_models()
    except httpx.HTTPError:
        available = []
    chat_models = [m for m in available if not any(h in m.lower() for h in EMBED_HINTS)]
    return {"current": llm.model, "available": sorted(chat_models)}


@app.post("/model")
async def set_model(req: ModelRequest, request: Request) -> dict:
    llm: OllamaClient = request.app.state.llm
    available = await llm.list_models()
    if not _has_model(available, req.name):
        raise HTTPException(404, f"Model {req.name!r} is not available in Ollama.")
    async with request.app.state.lock:
        llm.model = req.name
    return {"current": llm.model}
