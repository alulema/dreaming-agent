import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.test")
    monkeypatch.setenv("DB_PATH", ":memory:")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("EMBED_MODEL", "bge-m3")
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@respx.mock
def test_health_ok(client, respx_mock):
    respx_mock.get("http://ollama.test/api/tags").mock(
        return_value=httpx.Response(
            200, json={"models": [{"name": "qwen2.5:7b-instruct"}, {"name": "bge-m3:latest"}]}
        )
    )
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_model"]["available"] and body["embed_model"]["available"]
    assert body["memory"]["mtm"] == 0


@respx.mock
def test_health_missing_model_is_degraded(client, respx_mock):
    respx_mock.get("http://ollama.test/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "bge-m3:latest"}]})
    )
    body = client.get("/health").json()
    assert body["status"] == "degraded" and not body["llm_model"]["available"]


@respx.mock
def test_health_ollama_down(client, respx_mock):
    respx_mock.get("http://ollama.test/api/tags").mock(side_effect=httpx.ConnectError("down"))
    body = client.get("/health").json()
    assert body["status"] == "degraded" and body["ollama"] is False


def test_chat_memory_session_reset_endpoints(client):
    from app.memory.embed import Embedder
    from app.pipeline.agent import Agent
    from tests.fakes import FakeLLM

    state = client.app.state
    fake = FakeLLM().script(
        "controller",
        {"shares_new_info": True, "needs_memory": False, "queries": [], "tiers": ["MTM"]},
    ).script("writer", {"has_new_facts": True, "entries": [
        {"text": "The user lives in Lima", "kind": "fact", "entities": ["Lima"],
         "topics": ["user.location"]}]})
    fake.text_replies.append("¡Qué bien!")
    state.agent = Agent(fake, Embedder(fake, "fake"), state.store, state.settings)

    sid = client.post("/session/new").json()["session_id"]
    r = client.post("/chat", json={"session_id": sid, "message": "Vivo en Lima"})
    assert r.status_code == 200 and r.json()["reply"] == "¡Qué bien!"

    mem = client.get("/memory", params={"session_id": sid}).json()
    assert len(mem["stm"]) == 2 and [e["text"] for e in mem["mtm"]] == ["The user lives in Lima"]
    assert "embedding" not in mem["mtm"][0]

    assert client.post("/chat", json={"session_id": sid, "message": ""}).status_code == 422
    assert client.post("/reset").json() == {"ok": True}
    assert client.get("/memory").json()["stats"]["mtm"] == 0


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "El agente que sueña" in r.text
