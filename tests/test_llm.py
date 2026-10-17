import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from app.llm import OllamaClient

BASE = "http://ollama.test"


class Out(BaseModel):
    needs_memory: bool
    queries: list[str]


def fallback() -> Out:
    return Out(needs_memory=True, queries=["FALLBACK"])


def chat_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


@pytest.fixture
def client():
    return OllamaClient(BASE, "test-model", json_retries=1, timeout_s=1.0)


@respx.mock
async def test_valid_json_is_parsed(client):
    respx.post(f"{BASE}/api/chat").mock(
        return_value=chat_reply('{"needs_memory": false, "queries": []}')
    )
    res = await client.chat_json("controller", [], Out, fallback)
    assert res.ok and not res.fallback_used
    assert res.value == Out(needs_memory=False, queries=[])
    assert res.attempts == ["ok"]
    assert client.stats.as_dict()["controller"]["ok"] == 1


@respx.mock
async def test_schema_is_sent_as_format(client):
    route = respx.post(f"{BASE}/api/chat").mock(
        return_value=chat_reply('{"needs_memory": true, "queries": ["x"]}')
    )
    await client.chat_json("controller", [{"role": "user", "content": "hi"}], Out, fallback)
    body = json.loads(route.calls.last.request.content)
    assert body["format"] == Out.model_json_schema()
    assert body["options"] == {"temperature": 0.0, "seed": 42}
    assert body["stream"] is False


@respx.mock
async def test_json_mode_sends_plain_json_format():
    c = OllamaClient(BASE, "m", format_mode="json")
    route = respx.post(f"{BASE}/api/chat").mock(
        return_value=chat_reply('{"needs_memory": true, "queries": []}')
    )
    await c.chat_json("controller", [], Out, fallback)
    assert json.loads(route.calls.last.request.content)["format"] == "json"


@respx.mock
async def test_invalid_json_is_retried_then_succeeds(client):
    respx.post(f"{BASE}/api/chat").mock(
        side_effect=[
            chat_reply('{"needs_memory": true, "queries": "not-a-list"}'),
            chat_reply('{"needs_memory": true, "queries": ["dog name"]}'),
        ]
    )
    res = await client.chat_json("controller", [], Out, fallback)
    assert res.ok
    assert res.attempts == ["invalid_json", "ok"]
    assert res.value.queries == ["dog name"]


@respx.mock
async def test_invalid_json_exhausts_retries_and_falls_back(client):
    route = respx.post(f"{BASE}/api/chat").mock(return_value=chat_reply("not json at all"))
    res = await client.chat_json("selector", [], Out, fallback)
    assert res.fallback_used
    assert res.value.queries == ["FALLBACK"]
    assert res.attempts == ["invalid_json", "invalid_json"]
    assert route.call_count == 2
    s = client.stats.as_dict()["selector"]
    assert s["invalid_json"] == 2 and s["fallback"] == 1 and s["calls"] == 2


@respx.mock
async def test_timeout_falls_back_without_retry(client):
    route = respx.post(f"{BASE}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    res = await client.chat_json("writer", [], Out, fallback)
    assert res.fallback_used
    assert res.attempts == ["timeout"]
    assert route.call_count == 1
    assert client.stats.as_dict()["writer"]["timeout"] == 1


@respx.mock
async def test_http_error_falls_back(client):
    respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(500, text="boom"))
    res = await client.chat_json("writer", [], Out, fallback)
    assert res.fallback_used and res.attempts == ["http_error"]


@respx.mock
async def test_chat_text_and_embed_and_list_models(client):
    respx.post(f"{BASE}/api/chat").mock(return_value=chat_reply("Hola!"))
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})
    )
    respx.get(f"{BASE}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "bge-m3:latest"}]})
    )
    assert await client.chat_text("responder", []) == "Hola!"
    assert await client.embed(["a", "b"], "bge-m3") == [[1.0, 0.0], [0.0, 1.0]]
    assert await client.embed([], "bge-m3") == []
    assert await client.list_models() == ["bge-m3:latest"]
