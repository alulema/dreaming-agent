"""Phase 0 smoke test: Ollama chat with JSON-schema structured output, and embeddings.

Usage: uv run python scripts/smoke_ollama.py [llm_model] [embed_model]
"""

import json
import sys
import time

import httpx
from pydantic import BaseModel, ValidationError

OLLAMA_URL = "http://localhost:11434"


class ControllerOut(BaseModel):
    needs_memory: bool
    queries: list[str]
    tiers: list[str]


PROMPT = (
    "You decide whether answering the user's message requires long-term memory about the user. "
    "Reply with JSON: needs_memory (bool), queries (list of search queries), "
    "tiers (subset of STM, MTM, LTM).\n\nUser message: ¿Te acuerdas cómo se llama mi perro?"
)


def chat(model: str, fmt) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "format": fmt,
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"], (time.perf_counter() - t0) * 1000


def main() -> None:
    llm = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b-instruct"
    emb = sys.argv[2] if len(sys.argv) > 2 else None
    for label, fmt in (("format=json", "json"), ("format=schema", ControllerOut.model_json_schema())):
        content, ms = chat(llm, fmt)
        try:
            out = ControllerOut.model_validate_json(content)
            print(f"[{llm} {label}] OK {ms:.0f}ms -> {out.model_dump()}")
        except ValidationError as e:
            print(f"[{llm} {label}] INVALID {ms:.0f}ms -> {content!r} ({e.error_count()} errors)")
    if emb:
        t0 = time.perf_counter()
        r = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": emb, "input": ["Mi perro se llama Toby", "My dog is called Toby"]},
            timeout=300,
        )
        r.raise_for_status()
        vecs = r.json()["embeddings"]
        import numpy as np

        a, b = (np.array(v) / np.linalg.norm(v) for v in vecs)
        print(f"[{emb}] dim={len(vecs[0])} cos(ES,EN)={float(a @ b):.3f} "
              f"{(time.perf_counter() - t0) * 1000:.0f}ms")


if __name__ == "__main__":
    main()
