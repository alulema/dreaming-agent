import re

import numpy as np
import pytest

from app.config import Settings
from app.memory.embed import Embedder, to_bytes
from app.memory.models import MemoryEntry
from app.memory.store import SQLiteStore
from app.pipeline.agent import Agent
from app.pipeline.controller import ControllerOut, repair
from app.pipeline.prompts import PROMPTS_DIR, load_prompt
from app.pipeline.retriever import Candidate, retrieve
from app.pipeline.selector import run_selector
from app.pipeline.writer import WriterEntry, WriterOut, clean
from tests.fakes import FakeLLM, fake_vector


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


def add(store, text, tier="MTM", **kw) -> MemoryEntry:
    e = MemoryEntry(text=text, tier=tier, embedding=to_bytes(np.array(fake_vector(text))), **kw)
    store.add_entry(e)
    return e


# --- prompts ---

def test_all_prompts_render_with_their_placeholders():
    for path in PROMPTS_DIR.glob("*.md"):
        p = load_prompt(path.stem)
        names = set(re.findall(r"\{\{(\w+)\}\}", p.template))
        out = p.render(**{n: f"<{n}>" for n in names})
        assert "{{" not in out and len(p.version) == 8


def test_render_missing_value_raises():
    with pytest.raises(KeyError):
        load_prompt("writer").render()


# --- controller ---

def test_repair_fills_empty_queries_and_tiers():
    out, repairs = repair(
        ControllerOut(shares_new_info=False, needs_memory=True, queries=[" "], tiers=[]), "msg"
    )
    assert out.queries == ["msg"] and out.tiers == ["MTM", "LTM"]
    assert repairs == ["empty_queries", "empty_tiers"]


def test_repair_leaves_good_output_alone():
    src = ControllerOut(shares_new_info=True, needs_memory=False, queries=[], tiers=["MTM"])
    out, repairs = repair(src, "msg")
    assert out == src and repairs == []


# --- retriever ---

async def test_retrieve_merges_queries_keeping_best_score(store):
    dog = add(store, "#dog name Toby")
    add(store, "#cat name Nube")
    emb = Embedder(FakeLLM(), "fake")
    # each query hits its own entry exactly; the other entry is seen at a lower score too
    cands = await retrieve(store, emb, ["#dog name?", "#cat what?"], ["MTM"], k=5)
    assert len({c.entry.id for c in cands}) == len(cands) == 2
    assert all(c.score == pytest.approx(1.0) for c in cands)
    only_dog = await retrieve(store, emb, ["#dog name?"], ["MTM"], k=1)
    assert [c.entry.id for c in only_dog] == [dog.id]
    assert await retrieve(store, emb, [], ["MTM"], k=5) == []


# --- selector ---

def cands(store, *texts):
    return [Candidate(add(store, t), 0.9 - i / 10) for i, t in enumerate(texts)]


async def test_selector_maps_positions_and_repairs_out_of_range(store):
    cs = cands(store, "#a", "#b", "#c")
    llm = FakeLLM().script("selector", {"keep": [3, 3, 9, 1], "context": "ctx"})
    sel = await run_selector(llm, "q", cs, fallback_k=2)
    assert [c.entry.text for c in sel.kept] == ["#c", "#a"]
    assert sel.context == "ctx" and sel.repairs == ["out_of_range_keep"]


async def test_selector_empty_context_is_rebuilt(store):
    cs = cands(store, "#a", "#b")
    llm = FakeLLM().script("selector", {"keep": [2], "context": "  "})
    sel = await run_selector(llm, "q", cs, fallback_k=2)
    assert sel.context == "- #b" and sel.repairs == ["empty_context"]


async def test_selector_fallback_keeps_top_k(store):
    cs = cands(store, "#a", "#b", "#c", "#d")
    sel = await run_selector(FakeLLM(), "q", cs, fallback_k=3)  # no script -> fallback
    assert sel.result.fallback_used
    assert [c.entry.text for c in sel.kept] == ["#a", "#b", "#c"]


async def test_selector_skipped_without_candidates():
    llm = FakeLLM()
    sel = await run_selector(llm, "q", [], fallback_k=3)
    assert sel.result is None and sel.kept == [] and llm.calls == []


# --- writer ---

def w(text, **kw) -> WriterEntry:
    return WriterEntry(text=text, kind=kw.get("kind", "fact"), entities=kw.get("entities", []),
                       topics=kw.get("topics", []))


def test_writer_clean_gate_meta_dupes_whitespace():
    out = WriterOut(has_new_facts=True, entries=[
        w("The user  lives in   Lima."),
        w("the user lives in lima"),               # duplicate (case / trailing dot)
        w("The user asked about the weather"),     # meta
        w("   "),                                  # empty
        w("The user has a cat", entities=["cat", " cat ", ""], topics=["user.pets", "user.pets"]),
    ])
    kept, dropped = clean(out)
    assert [e.text for e in kept] == ["The user lives in Lima.", "The user has a cat"]
    assert kept[1].entities == ["cat"] and kept[1].topics == ["user.pets"]
    assert dropped == 3


def test_writer_clean_respects_gate():
    kept, dropped = clean(WriterOut(has_new_facts=False, entries=[w("The user has a cat")]))
    assert kept == [] and dropped == 1


# --- agent ---

def make_agent(store, llm, **overrides) -> Agent:
    settings = Settings(db_path=":memory:", **overrides)
    return Agent(llm, Embedder(llm, "fake"), store, settings)


def ctrl(new: bool, mem: bool, queries=()):
    return {"shares_new_info": new, "needs_memory": mem, "queries": list(queries),
            "tiers": ["MTM", "LTM"]}


async def test_agent_statement_turn_writes_mtm(store):
    llm = FakeLLM().script("controller", ctrl(True, False)).script("writer", {
        "has_new_facts": True,
        "entries": [{"text": "#dog The user's dog is Toby", "kind": "fact",
                     "entities": ["Toby"], "topics": ["user.pets"]}],
    })
    llm.text_replies.append("Nice!")
    res = await make_agent(store, llm, retrieval_mode="controller").chat("s1", "My dog is Toby")
    assert res.reply == "Nice!"
    t = res.trace
    assert t["selector"] is None and t["llm_calls"] == 3 and t["fallbacks"] == []
    assert [e["text"] for e in t["writer"]["entries"]] == ["#dog The user's dog is Toby"]
    mtm = store.list_entries(["MTM"])
    assert len(mtm) == 1 and mtm[0].topics == ["user.pets"]
    assert mtm[0].source_turn_ids == [t["turn_id"]]
    assert [x.role for x in store.recent_turns("s1", 10)] == ["user", "assistant"]
    assert {"controller", "responder", "writer", "reply", "total"} <= t["latency_ms"].keys()


async def test_agent_question_turn_retrieves_and_skips_writer(store):
    dog = add(store, "#dog The user's dog is Toby")
    llm = FakeLLM().script("controller", ctrl(False, True, ["#dog name"])).script(
        "selector", {"keep": [1], "context": "The dog is Toby."}
    )
    res = await make_agent(store, llm, selector_mode="compress").chat("s1", "What's my dog called?")
    t = res.trace
    assert t["writer"]["skipped"] and t["llm_calls"] == 3
    assert t["selector"]["kept_ids"] == [dog.id]
    # the Responder's system prompt carries the Selector's context
    responder_msgs = next(m for mod, m in llm.calls if mod == "responder")
    assert "The dog is Toby." in responder_msgs[0]["content"]
    assert store.get_entry(dog.id).access_count == 1


async def test_agent_dedup_reinforces_instead_of_duplicating(store):
    old = add(store, "#dog The user's dog is Toby")
    llm = FakeLLM().script("controller", ctrl(True, False)).script("writer", {
        "has_new_facts": True,
        "entries": [{"text": "#dog Toby is the user's dog", "kind": "fact",
                     "entities": [], "topics": []}],
    })
    res = await make_agent(store, llm).chat("s1", "Toby is my dog")
    assert res.trace["writer"]["entries"] == []
    assert res.trace["writer"]["reinforced"][0]["id"] == old.id
    assert len(store.list_entries()) == 1
    assert store.get_entry(old.id).source_turn_ids == [res.trace["turn_id"]]


async def test_agent_fallbacks_are_reported(store):
    llm = FakeLLM()  # nothing scripted: controller falls back (needs memory, writes)
    res = await make_agent(store, llm).chat("s1", "hello")
    t = res.trace
    assert t["fallbacks"] == ["controller", "writer"]  # selector skipped: empty memory
    assert t["controller"]["queries"] == ["hello"]


async def test_agent_stm_window(store):
    llm = FakeLLM()
    agent = make_agent(store, llm, stm_turns=2)
    for i in range(3):
        llm.script("controller", ctrl(False, False))
        await agent.chat("s1", f"m{i}")
    responder_msgs = [m for mod, m in llm.calls if mod == "responder"][-1]
    # system + 2 STM turns + current message
    assert len(responder_msgs) == 4 and responder_msgs[-1]["content"] == "m2"


# --- v2 options ---

def test_writer_grounding_drops_leaked_names_but_keeps_translations():
    out = WriterOut(has_new_facts=True, entries=[
        w("The user lives in Bilbao"),                       # leaked from a prompt example
        w("The user has a parrot named Kiwi"),
        w("The user's sister lives in Lisbon"),              # Lisboa in the message
        w("The user is 34 years old"),                       # invented number
    ])
    msg = "Tengo un loro que se llama Kiwi y mi hermana vive en Lisboa"
    kept, dropped = clean(out, msg)
    assert [e.text for e in kept] == ["The user has a parrot named Kiwi",
                                      "The user's sister lives in Lisbon"]
    assert dropped == 2
    assert len(clean(out)[0]) == 4                           # grounding off (v1)


async def test_selector_filter_mode_tops_up_and_orders(store):
    cs = cands(store, "#a", "#b", "#c", "#d")
    for i, c in enumerate(cs):
        c.entry.created_at = c.entry.created_at.replace(year=2026 - i)  # #d oldest
    llm = FakeLLM().script("selector", {"keep": [2], "context": "compressed"})
    sel = await run_selector(llm, "q", cs, fallback_k=3, mode="filter", min_keep=3)
    assert [c.entry.text for c in sel.kept] == ["#c", "#b", "#a"]   # top-up, chronological
    assert "compressed" not in sel.context and sel.context.count("\n") == 2


async def test_agent_always_retrieves_in_v2(store):
    dog = add(store, "#dog The user's dog is Toby")
    llm = FakeLLM().script("controller", ctrl(False, False))
    res = await make_agent(store, llm, retrieval_mode="always").chat("s1", "#dog what?")
    assert res.trace["controller"]["forced_retrieval"]
    assert res.trace["selector"]["kept_ids"] == [dog.id]
    assert res.trace["selector"]["skipped"]                  # no Selector LLM call when forced
    assert [m for m, _ in llm.calls] == ["controller", "responder"]
    llm2 = FakeLLM().script("controller", ctrl(False, False))
    res2 = await make_agent(store, llm2, retrieval_mode="controller").chat("s1", "#dog what?")
    assert res2.trace["selector"] is None
