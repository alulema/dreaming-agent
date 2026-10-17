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
