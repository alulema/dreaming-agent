"""Phase 2 acceptance: run a scripted 10-turn conversation through the real pipeline.

Prints each reply with per-stage latency, what the Writer stored, and the final MTM.
Usage: uv run python -m scripts.demo_conversation [llm_model] [--json out.json]
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.llm import OllamaClient
from app.memory.embed import Embedder
from app.memory.store import SQLiteStore
from app.pipeline.agent import Agent

# Fictional user. Mixes ES/EN, a paraphrase, an update, small talk and memory questions.
TURNS = [
    "Hola! Me llamo Lucía y vivo en Quito.",
    "I have a dog called Toby, he's a beagle.",
    "¿Qué tiempo suele hacer en Quito en julio?",
    "Trabajo como enfermera en un hospital, turnos de noche.",
    "By the way, my beagle Toby turned 5 last week.",
    "No me gusta nada el café, prefiero el té verde.",
    "¿Te acuerdas de cómo se llama mi perro?",
    "Big news: I just moved to Madrid for a new job as a data analyst.",
    "¿Dónde vivo ahora y en qué trabajo?",
    "Gracias, eso es todo por hoy.",
]


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out_path in args:
        args.remove(out_path)
    settings = Settings(db_path=str(Path(tempfile.mkdtemp()) / "demo.db"))
    if args:
        settings.llm_model = args[0]
    llm = OllamaClient(settings.ollama_url, settings.llm_model, format_mode=settings.llm_format,
                       seed=settings.llm_seed, timeout_s=settings.llm_timeout_s)
    store = SQLiteStore(settings.db_path)
    agent = Agent(llm, Embedder(llm, settings.embed_model), store, settings)
    traces = []

    print(f"== model={settings.llm_model} embed={settings.embed_model}\n")
    for i, msg in enumerate(TURNS, 1):
        res = await agent.chat("demo-session", msg)
        t = res.trace
        traces.append({"message": msg, "reply": res.reply, "trace": t})
        lat = " ".join(f"{k}={v / 1000:.1f}s" for k, v in t["latency_ms"].items())
        print(f"[{i}] USER: {msg}")
        print(f"    BOT:  {res.reply}")
        c = t["controller"]
        print(f"    controller: needs_memory={c['needs_memory']} queries={c['queries']}")
        if t["selector"]:
            print(f"    selector context: {t['selector']['context']!r}")
        if t["selector"]:
            print(f"    retrieved={len(t['retrieval']['candidates'])} "
                  f"kept={len(t['selector']['kept_ids'])}")
        for e in t["writer"]["entries"]:
            print(f"    + MTM: {e['text']}  {e['topics']} {e['entities']}")
        for r in t["writer"]["reinforced"]:
            print(f"    = reinforced ({r['score']}): {r['text']!r} <- {r['new_text']!r}")
        extra = []
        if t["fallbacks"]:
            extra.append(f"FALLBACKS={t['fallbacks']}")
        if t["repairs"]:
            extra.append(f"repairs={t['repairs']}")
        print(f"    calls={t['llm_calls']} {lat} {' '.join(extra)}\n")

    print("== Final MTM")
    for e in store.list_entries(["MTM"]):
        print(f"  - {e.text}  topics={e.topics} access={e.access_count} "
              f"turns={len(e.source_turn_ids)}")
    print(f"\n== stats {store.stats()}")
    print(f"== llm stats {json.dumps(llm.stats.as_dict(), indent=1)}")
    if out_path:
        Path(out_path).write_text(json.dumps(traces, indent=2, ensure_ascii=False, default=str))
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
