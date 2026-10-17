"""Probe embedding cosine similarities for paraphrases, updates and unrelated facts.

Informs the Dream clustering threshold (spec default 0.80).
Usage: uv run python -m scripts.probe_similarity [embed_model]
"""

import asyncio
import sys

from app.config import get_settings
from app.llm import OllamaClient
from app.memory.embed import Embedder, to_bytes
from app.memory.models import MemoryEntry
from app.memory.store import SQLiteStore

PAIRS = {
    "paraphrase (ES/ES)": ("Mi perro se llama Toby", "Tengo un perro llamado Toby"),
    "paraphrase (ES/EN)": ("Mi perro se llama Toby", "My dog's name is Toby"),
    "paraphrase (fact form)": ("The user's dog is named Toby", "User has a dog called Toby"),
    "update (city)": ("The user lives in Quito", "The user moved to Madrid"),
    "update (job)": ("The user works as a nurse", "The user now works as a data analyst"),
    "same topic, diff fact": ("The user's dog is named Toby", "The user's dog is a beagle"),
    "unrelated": ("The user's dog is named Toby", "The user prefers tea over coffee"),
    "unrelated (ES/EN)": ("Me gusta correr los domingos", "The user works as a nurse"),
}

MEMORY = [
    "The user's dog is named Toby",
    "The user lives in Quito",
    "The user works as a nurse",
    "The user prefers tea over coffee",
    "The user plays football on weekends",
    "La hermana del usuario se llama Ana",
]
QUERIES = ["¿Cómo se llama mi perro?", "Where do I live?", "¿Qué deportes practico?",
           "What's my sister's name?"]


async def main() -> None:
    s = get_settings()
    model = sys.argv[1] if len(sys.argv) > 1 else s.embed_model
    llm = OllamaClient(s.ollama_url, s.llm_model)
    emb = Embedder(llm, model)
    print(f"== {model}")
    for label, (a, b) in PAIRS.items():
        va, vb = await emb.embed([a, b])
        print(f"  {float(va @ vb):.3f}  {label}")

    store = SQLiteStore(":memory:")
    for text, v in zip(MEMORY, await emb.embed(MEMORY), strict=True):
        store.add_entry(MemoryEntry(text=text, embedding=to_bytes(v)))
    for q in QUERIES:
        (top, score), (second, s2) = store.search(await emb.embed_one(q), ["MTM"], k=2)
        print(f"  {q!r:32} -> {top.text!r} ({score:.3f}) | 2nd {s2:.3f}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
