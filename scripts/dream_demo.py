"""Phase 3 acceptance: 3 scripted sessions, then two Dreams in a row.

Checks that the Dream proposes sensible ops and that the second Dream is (almost) idempotent.
Usage: uv run python -m scripts.dream_demo [llm_model] [--json out.json]
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.dream.dreamer import run_dream
from app.llm import OllamaClient
from app.memory.embed import Embedder
from app.memory.store import SQLiteStore
from app.pipeline.agent import Agent

# Fictional user. Paraphrases (MERGE), changes (UPDATE), trivia (FORGET), repeated facts (PROMOTE).
SESSIONS = [
    [
        "Hola! Me llamo Lucía y vivo en Quito.",
        "I have a dog called Toby, he's a beagle.",
        "Trabajo como enfermera en un hospital, turnos de noche.",
        "No me gusta nada el café, prefiero el té verde.",
    ],
    [
        "Hoy saqué a pasear a Toby, mi beagle, por el parque.",
        "Me encanta correr los domingos por la mañana.",
        "My dog Toby turned 5 last week.",
        "Hoy almorcé un sándwich de pollo.",
        "¿Te acuerdas de cómo se llama mi perro?",
    ],
    [
        "Big news: I just moved to Madrid for a new job as a data analyst.",
        "Ya no trabajo de enfermera, ahora soy analista de datos.",
        "Últimamente tomo café todas las mañanas, ya me gusta.",
        "Vivo en Madrid desde hace un mes, en el barrio de Malasaña.",
        "¿Dónde vivo ahora y en qué trabajo?",
    ],
]


def show_memory(store: SQLiteStore, title: str) -> None:
    print(f"\n== {title}  {store.stats()}")
    for e in store.list_entries():
        print(f"  [{e.tier}] {e.text}  topics={e.topics} turns={len(e.source_turn_ids)} "
              f"access={e.access_count}")


def show_diff(d: dict) -> None:
    print(f"\n== DREAM {d['dream_id'][:8]}: {d['duration_ms'] / 1000:.1f}s, "
          f"{len(d['clusters'])} clusters, {d['llm_calls']} LLM calls, "
          f"{d['ops_proposed']} proposed, {len(d['ops_applied'])} applied, "
          f"{len(d['ops_rejected'])} rejected")
    for c in d["clusters"]:
        print(f"   cluster: {len(c['mtm'])} MTM + {len(c['ltm'])} LTM linked={c['linked']}")
    for a in d["ops_applied"]:
        olds = " | ".join(t["text"] for t in a["targets"])
        arrow = f" -> [{a['new_tier']}] {a['new_text']}" if a["new_text"] else ""
        print(f"  ✓ {a['op']}: {olds}{arrow}\n      reason: {a['reason']}")
    for r in d["ops_rejected"]:
        print(f"  ✗ {r['op']}: {' | '.join(r['targets'])} new_text={r['new_text']!r}\n"
              f"      error: {r['error']}  (reason: {r['reason']})")
    for w in d["warnings"]:
        print(f"  ⚠ possibly stale: {w['text']!r} (shares {w['shared_topics']} with UPDATE "
              f"{w['update_text']!r})")
    print(f"  before={d['stats_before']}\n  after ={d['stats_after']}")


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    for flag in ("--json", "--dreams"):
        if flag in sys.argv and sys.argv[sys.argv.index(flag) + 1] in args:
            args.remove(sys.argv[sys.argv.index(flag) + 1])
    settings = Settings(db_path=str(Path(tempfile.mkdtemp()) / "dream.db"))
    if args:
        settings.llm_model = args[0]
    llm = OllamaClient(settings.ollama_url, settings.llm_model, format_mode=settings.llm_format,
                       seed=settings.llm_seed, timeout_s=settings.llm_timeout_s)
    store = SQLiteStore(settings.db_path)
    embedder = Embedder(llm, settings.embed_model)
    agent = Agent(llm, embedder, store, settings)

    print(f"== model={settings.llm_model} embed={settings.embed_model} "
          f"cluster_threshold={settings.cluster_threshold}")
    for s, turns in enumerate(SESSIONS, 1):
        for msg in turns:
            res = await agent.chat(f"session-{s}", msg)
            print(f"[s{s}] {msg}\n      -> {res.reply}")
    show_memory(store, "Memory before Dream")

    n_dreams = int(sys.argv[sys.argv.index("--dreams") + 1]) if "--dreams" in sys.argv else 2
    diffs = []
    for k in range(n_dreams):
        diffs.append(await run_dream(store, embedder, llm, settings))
        show_diff(diffs[-1])
        show_memory(store, f"Memory after Dream {k + 1}")
    applied = [len(d["ops_applied"]) for d in diffs]
    proposed = [d["ops_proposed"] for d in diffs]
    ratio = applied[1] / applied[0] if len(applied) > 1 and applied[0] else 0.0
    print(f"\n== Idempotency: applied per Dream {applied}, proposed per Dream {proposed}; "
          f"dream2/dream1 applied = {ratio:.0%} ({'OK' if ratio <= 0.2 else 'ABOVE 20%'})")

    q = await agent.chat("session-4", "¿Dónde vivo, en qué trabajo y qué bebo por las mañanas?")
    print(f"\n[s4] post-dream question -> {q.reply}\n     context: "
          f"{(q.trace['selector'] or {}).get('context')!r}")
    if out_path:
        Path(out_path).write_text(json.dumps(diffs, indent=2, ensure_ascii=False, default=str))
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
