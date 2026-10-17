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


def test_dream_endpoints(client):
    import numpy as np

    from app.memory.embed import Embedder, to_bytes
    from app.memory.models import MemoryEntry
    from tests.fakes import FakeLLM, fake_vector

    state = client.app.state
    for t, turn in (("The user has a cat named Mia", "t1"), ("The user's cat is Mia", "t2")):
        state.store.add_entry(MemoryEntry(text=t, entities=["Mia"], source_turn_ids=[turn],
                                          embedding=to_bytes(np.array(fake_vector("#cat")))))
    fake = FakeLLM().script("consolidator", {"ops": [
        {"op": "MERGE", "targets": ["M1", "M2"], "new_text": "The user has a cat named Mia",
         "reason": "same"}]})
    state.llm, state.embedder = fake, Embedder(fake, "fake")

    diff = client.post("/dream").json()
    assert [a["op"] for a in diff["ops_applied"]] == ["MERGE"]
    assert client.get("/dreams").json()["dreams"][0]["dream_id"] == diff["dream_id"]
    assert client.get("/memory").json()["stats"]["ltm"] == 1


def _fake_app(client):
    from app.memory.embed import Embedder
    from app.pipeline.agent import Agent
    from app.pipeline.strategies import NaiveRAG
    from tests.fakes import FakeLLM

    state = client.app.state
    fake = FakeLLM()
    emb = Embedder(fake, "fake")
    state.llm, state.embedder = fake, emb
    state.agent = Agent(fake, emb, state.store, state.settings, index_turns=True)
    state.rag = NaiveRAG(fake, emb, state.store, state.settings)
    return fake


def test_modes_share_memory_and_learning_runs_in_background(client):
    fake = _fake_app(client)
    writer_out = {"has_new_facts": True, "entries": [
        {"text": "The user lives in Lima", "kind": "fact", "entities": ["Lima"],
         "topics": ["user.location"]}]}

    # 1) RAG mode: one Responder call; the Writer still runs afterwards (its own gate).
    fake.script("writer", writer_out)
    fake.text_replies.append("Anotado.")
    r = client.post("/chat", json={"session_id": "s", "message": "Vivo en Lima", "mode": "rag"})
    body = r.json()
    assert body["reply"] == "Anotado." and body["trace"]["mode"] == "rag"
    learned = client.get(f"/learned/{body['learn_id']}").json()
    assert learned["done"] and [e["text"] for e in learned["writer"]["entries"]] == [
        "The user lives in Lima"]
    # the raw turn was indexed for RAG
    assert client.app.state.store._query(
        "SELECT COUNT(*) FROM turns WHERE embedding IS NOT NULL")[0][0] == 1

    # 2) Compare mode: RAG sees the indexed turn; LightMem sees the MTM fact; one stored turn.
    fake.script("controller", {"shares_new_info": False, "needs_memory": True,
                               "queries": ["where the user lives"], "tiers": ["MTM"]})
    fake.script("selector", {"keep": [1], "context": "x"})
    fake.text_replies.extend(["RAG: Lima", "LightMem: Lima"])
    body = client.post("/chat", json={"session_id": "s", "message": "¿Dónde vivo?",
                                      "mode": "compare"}).json()
    assert body["rag"]["reply"] == "RAG: Lima" and body["reply"] == "LightMem: Lima"
    assert "Vivo en Lima" in body["rag"]["trace"]["memory_context"]
    assert "The user lives in Lima" in body["trace"]["selector"]["context"]
    assert client.get(f"/learned/{body['learn_id']}").json()["writer"]["skipped"]  # gate=False
    assert len(client.get("/memory", params={"session_id": "s"}).json()["stm"]) == 4

    assert client.get("/learned/nope").status_code == 404
    assert client.post("/chat", json={"session_id": "s", "message": "x",
                                      "mode": "other"}).status_code == 422
