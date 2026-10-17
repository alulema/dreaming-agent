from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.config import Settings
from app.dream.applier import commit, prepare, result_tier
from app.dream.cluster import build_clusters
from app.dream.consolidator import format_entries, label_entries, normalize_label
from app.dream.dreamer import run_dream
from app.dream.validator import (
    Rules,
    dropped_current_values,
    stale_warnings,
    ungrounded_terms,
    validate_ops,
)
from app.memory.embed import Embedder, to_bytes
from app.memory.models import MemoryEntry, Op
from app.memory.store import SQLiteStore
from tests.fakes import FakeLLM, fake_vector

T0 = datetime(2026, 1, 1, tzinfo=UTC)
RULES = Rules(promote_min_turns=2, promote_min_access=3, max_entry_chars=300)


def at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def mk(text, tier="MTM", h=0, vec_tag=None, **kw) -> MemoryEntry:
    kw.setdefault("source_turn_ids", [f"turn-{text}"])
    v = np.array(fake_vector(vec_tag or text))
    return MemoryEntry(text=text, tier=tier, created_at=at(h), embedding=to_bytes(v), **kw)


def by_id(*entries):
    return {e.id: e for e in entries}


def check(op, *entries, allowed=None):
    _, rejected = validate_ops([op], by_id(*entries), RULES,
                                   {0: allowed} if allowed is not None else None,
                                   [0] if allowed is not None else None)
    return rejected[0].error if rejected else None


# --- validator: rejections the spec asks for ---

def test_rejects_unknown_ids():
    a = mk("The user has a cat")
    assert "unknown id" in check(Op(op="FORGET", target_ids=["nope"]), a)


def test_rejects_ids_outside_the_cluster_shown_to_the_model():
    a, b = mk("a"), mk("b")
    err = check(Op(op="MERGE", target_ids=[a.id, b.id], new_text="The user a b"), a, b,
                allowed={a.id})
    assert "unknown id" in err


def test_rejects_forget_on_ltm():
    e = mk("The user lives in Lima", tier="LTM")
    assert check(Op(op="FORGET", target_ids=[e.id]), e) == "FORGET over LTM is not allowed"


def test_rejects_merge_of_single_id():
    e = mk("The user has a cat")
    assert "at least 2" in check(Op(op="MERGE", target_ids=[e.id], new_text="x"), e)


def test_rejects_merge_without_text_and_across_tiers():
    a, b = mk("The user has a cat"), mk("The user's cat", tier="LTM")
    assert "needs new_text" in check(Op(op="MERGE", target_ids=[a.id, b.id], new_text=" "), a, b)
    assert check(Op(op="MERGE", target_ids=[a.id, b.id], new_text="The user has a cat"),
                 a, b) == "MERGE across tiers"


def test_rejects_duplicates_empty_inactive_and_too_long():
    a = mk("The user has a cat")
    gone = mk("The user had a hamster", status="forgotten")
    assert check(Op(op="FORGET", target_ids=[]), a) == "no targets"
    assert check(Op(op="FORGET", target_ids=[a.id, a.id]), a) == "duplicate target"
    assert "inactive" in check(Op(op="FORGET", target_ids=[gone.id]), gone)
    b = mk("The user's cat is grey")
    assert check(Op(op="MERGE", target_ids=[a.id, b.id], new_text="x" * 301), a, b) == \
        "new_text too long"


def test_conflict_second_op_on_same_id_is_rejected():
    a, b, c = mk("The user has a cat", h=1), mk("The user's cat", h=2), mk("x", h=3)
    ops = [Op(op="MERGE", target_ids=[a.id, b.id], new_text="The user has a cat"),
           Op(op="FORGET", target_ids=[b.id, c.id])]
    valid, rejected = validate_ops(ops, by_id(a, b, c), RULES)
    assert valid == ops[:1] and "conflict" in rejected[0].error


def test_promote_needs_evidence_and_mtm():
    once = mk("The user has a cat")
    twice = mk("The user has a dog", source_turn_ids=["t1", "t2"])
    used = mk("The user is vegan", access_count=3)
    ltm = mk("The user lives in Lima", tier="LTM", source_turn_ids=["t1", "t2"])
    assert "without evidence" in check(Op(op="PROMOTE", target_ids=[once.id]), once)
    assert check(Op(op="PROMOTE", target_ids=[twice.id]), twice) is None
    assert check(Op(op="PROMOTE", target_ids=[used.id]), used) is None
    assert "not MTM" in check(Op(op="PROMOTE", target_ids=[ltm.id]), ltm)
    assert "exactly 1" in check(Op(op="PROMOTE", target_ids=[twice.id, used.id]), twice, used)


def test_forget_of_reinforced_entry_is_rejected():
    e = mk("The user runs", source_turn_ids=["t1", "t2"])
    assert "reinforced" in check(Op(op="FORGET", target_ids=[e.id]), e)


def test_update_needs_two_entries_and_one_mtm():
    a = mk("The user lives in Lima", tier="LTM", h=1)
    b = mk("The user lives in Cusco", tier="LTM", h=2)
    assert "at least 2" in check(Op(op="UPDATE", target_ids=[a.id], new_text="x"), a)
    assert "at least one MTM" in check(
        Op(op="UPDATE", target_ids=[a.id, b.id], new_text="The user lives in Cusco"), a, b)


def test_update_old_mtm_with_newer_ltm_is_valid():
    # Regression (Phase 3): "newest must be MTM" rejected this legitimate cleanup.
    old = mk("The user works as a nurse", h=1, entities=["nurse"], topics=["user.job"])
    cur = mk("The user is a data analyst", tier="LTM", h=2, entities=["data analyst"],
             topics=["user.job"])
    op = Op(op="UPDATE", target_ids=[old.id, cur.id],
            new_text="The user is a data analyst (previously a nurse)")
    assert check(op, old, cur) is None


# --- validator: grounding and information loss (Phase 3 findings) ---

def test_grounding_catches_prompt_example_leak():
    a, b = mk("The user drinks coffee", h=1), mk("The user likes tea", h=2)
    err = check(Op(op="UPDATE", target_ids=[a.id, b.id],
                   new_text="The user lives in Bilbao (previously lived in Lima)"), a, b)
    assert "not grounded" in err and "Bilbao" in err
    assert ungrounded_terms("The user's cat Nube is 3", [mk("The user's cat Nube")]) == ["3"]
    assert ungrounded_terms("The user lives in Malaga", [mk("The user lives in Málaga")]) == []


def test_update_that_drops_current_value_of_another_topic_is_rejected():
    quito = mk("The user lives in Quito", h=1, entities=["Quito"], topics=["user.location"])
    both = mk("The user is a data analyst in Madrid", tier="LTM", h=2,
              entities=["Madrid", "data analyst"], topics=["user.location", "user.job"])
    mal = mk("The user lives in Malasaña, Madrid", h=3, entities=["Madrid", "Malasaña"],
             topics=["user.location"])
    bad = Op(op="UPDATE", target_ids=[quito.id, both.id, mal.id],
             new_text="The user lives in Madrid, Malasaña (previously Quito)")
    assert "drops current value" in check(bad, quito, both, mal)
    good = bad.model_copy(update={
        "new_text": "The user lives in Malasaña, Madrid and works as a data analyst"})
    assert check(good, quito, both, mal) is None
    assert dropped_current_values("Toby", [mk("x", entities=["Toby"])]) == []


def test_merge_that_loses_a_number_or_name_is_rejected():
    a, b = mk("The user has a dog named Toby"), mk("The user's dog Toby is 5 years old")
    err = check(Op(op="MERGE", target_ids=[a.id, b.id], new_text="The user has a dog named Toby"),
                a, b)
    assert "loses details" in err and "5" in err


def test_stale_warning_for_older_entry_sharing_topic():
    quito = mk("The user lives in Quito", h=1, topics=["user.location"])
    a = mk("The user moved to Madrid", h=2, topics=["user.location"], entities=["Madrid"])
    b = mk("The user lives in Madrid", h=3, topics=["user.location"], entities=["Madrid"])
    cat = mk("The user has a cat", h=0, topics=["user.pets"])
    op = Op(op="UPDATE", target_ids=[a.id, b.id], new_text="The user lives in Madrid")
    w = stale_warnings([op], by_id(quito, a, b, cat), [{quito.id, a.id, b.id, cat.id}])
    assert [x["entry_id"] for x in w] == [quito.id]


# --- clustering ---

def test_cluster_links_by_similarity_topic_and_entity():
    a = mk("#pets a", topics=["user.pets"])
    b = mk("#other-vec b", topics=["user.pets"])               # topic link to a
    c = mk("#x c", entities=["Toby"], vec_tag="#c")
    d = mk("#y d", entities=["toby "], vec_tag="#d")            # entity link (normalised)
    e = mk("#pets e")                                           # cosine link to a
    f = mk("#lonely f", topics=["other"])                       # "other" never links
    g = mk("#lonely2 g", topics=["other"])
    cl = build_clusters([a, b, c, d, e, f, g], [], threshold=0.75, max_size=8,
                        max_ltm_context=5)
    groups = sorted(sorted(x.text for x in c.mtm) for c in cl)
    assert ["#lonely f", "#lonely2 g"] in groups               # singletons batched together
    assert sorted(["#pets a", "#other-vec b", "#pets e"]) in groups
    assert sorted(["#x c", "#y d"]) in groups
    assert next(c for c in cl if "#lonely f" in [x.text for x in c.mtm]).linked is False


def test_cluster_splits_large_components_and_attaches_ltm_context():
    mtm = [mk(f"#job {i}", h=i, topics=["user.job"]) for i in range(5)]
    ltm = [mk("#zzz", tier="LTM", topics=["user.job"]), mk("#unrelated", tier="LTM")]
    cl = build_clusters(mtm, ltm, threshold=0.75, max_size=2, max_ltm_context=5)
    assert [len(c.mtm) for c in cl] == [2, 2, 1]
    assert all([x.text for x in c.ltm] == ["#zzz"] for c in cl)
    assert build_clusters([], ltm, threshold=0.75, max_size=2, max_ltm_context=5) == []


# --- consolidator labels ---

def test_labels_and_promotable_flag():
    c = build_clusters([mk("#a", h=1, topics=["user.pets"], source_turn_ids=["t1", "t2"]),
                        mk("#b", h=2, topics=["user.pets"])],
                       [mk("#c", tier="LTM", topics=["user.pets"])],
                       threshold=0.75, max_size=8, max_ltm_context=5)[0]
    labeled = label_entries(c)
    assert list(labeled) == ["M1", "M2", "L1"]
    lines = format_entries(labeled, RULES).splitlines()
    assert "promotable" in lines[0] and "promotable" not in lines[1] + lines[2]
    assert normalize_label(" [m2] ") == "M2"


# --- applier ---

@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


async def test_applier_merge_update_promote_forget(store):
    cat1 = mk("The user has a cat named Mia", h=1, entities=["Mia"], topics=["user.pets"],
              access_count=1)
    cat2 = mk("The user's cat Mia is grey", h=2, entities=["Mia", "grey"], topics=["user.pets"],
              access_count=1)
    lima = mk("The user lives in Lima", h=3, topics=["user.location"])
    cusco = mk("The user moved to Cusco", h=4, topics=["user.location"], access_count=2)
    vegan = mk("The user is vegan", h=5, source_turn_ids=["t1", "t2"])
    lunch = mk("The user had soup today", h=6)
    for e in (cat1, cat2, lima, cusco, vegan, lunch):
        store.add_entry(e)
    ops = [
        Op(op="MERGE", target_ids=[cat1.id, cat2.id], new_text="The user has a grey cat Mia",
           reason="same cat"),
        Op(op="UPDATE", target_ids=[lima.id, cusco.id], new_text="The user lives in Cusco"),
        Op(op="PROMOTE", target_ids=[vegan.id]),
        Op(op="FORGET", target_ids=[lunch.id]),
    ]
    llm = FakeLLM()
    prepared = await prepare(store, Embedder(llm, "fake"), ops)
    with store.transaction():
        applied = commit(store, prepared, RULES, merge_promotes=True)

    merge, update, promote, forget = applied
    new_cat = store.get_entry(merge["new_id"])
    assert new_cat.text == "The user has a grey cat Mia"
    assert new_cat.tier == "LTM"                     # 2 source turns => promoted on merge
    assert new_cat.entities == ["Mia", "grey"] and new_cat.access_count == 2
    assert new_cat.created_at == cat2.created_at      # fact time, not Dream time
    assert set(new_cat.source_turn_ids) == {f"turn-{cat1.text}", f"turn-{cat2.text}"}
    assert store.get_entry(cat1.id).superseded_by == new_cat.id
    assert store.get_entry(cat1.id).status == "superseded"

    new_city = store.get_entry(update["new_id"])
    assert new_city.access_count == 2 and update["new_tier"] == "LTM"
    assert store.get_entry(promote["new_id"]).text == "The user is vegan"
    assert promote["new_tier"] == "LTM"
    assert store.get_entry(lunch.id).status == "forgotten" and forget["new_id"] is None
    assert [e.text for e in store.list_entries()] == [
        "The user has a grey cat Mia", "The user lives in Cusco", "The user is vegan"]


def test_result_tier_rules():
    once = mk("a")
    ltm = mk("b", tier="LTM")
    merge = Op(op="MERGE", target_ids=[], new_text="x")
    assert result_tier(merge, [once, mk("c")], RULES, merge_promotes=True) == "LTM"  # 2 turns
    assert result_tier(merge, [once, once], RULES, merge_promotes=True) == "MTM"     # 1 turn
    assert result_tier(merge, [once, mk("c")], RULES, merge_promotes=False) == "MTM"
    assert result_tier(Op(op="UPDATE", target_ids=[], new_text="x"), [once, ltm], RULES,
                       merge_promotes=False) == "LTM"


# --- dreamer (end to end with a scripted model) ---

async def test_dream_end_to_end_and_idempotent(store):
    a = mk("#cat The user has a cat named Mia", h=1, entities=["Mia"], topics=["user.pets"])
    b = mk("#cat The user's cat is called Mia", h=2, entities=["Mia"], topics=["user.pets"])
    for e in (a, b):
        store.add_entry(e)
    llm = FakeLLM().script("consolidator", {"ops": [
        {"op": "MERGE", "targets": ["M1", "[m2]"], "new_text": "The user has a cat named Mia",
         "reason": "same fact"},
        {"op": "FORGET", "targets": ["M9"], "reason": "hallucinated label"},
    ]})
    settings = Settings(db_path=":memory:")
    diff = await run_dream(store, Embedder(llm, "fake"), llm, settings)

    assert diff["ops_proposed"] == 2 and diff["llm_calls"] == 1
    assert [x["op"] for x in diff["ops_applied"]] == ["MERGE"]
    assert "unknown id" in diff["ops_rejected"][0]["error"]
    # two distinct source turns => the merged fact is born in LTM
    assert (diff["stats_before"]["mtm"], diff["stats_after"]["mtm"]) == (2, 0)
    assert diff["stats_after"]["ltm"] == 1
    assert store.list_dreams()[0]["dream_id"] == diff["dream_id"]

    # Second Dream: no MTM left => no clusters, no LLM calls, nothing changes.
    diff2 = await run_dream(store, Embedder(llm, "fake"), llm, settings)
    assert diff2["ops_applied"] == [] and diff2["llm_calls"] == 0
    assert diff2["stats_after"] == diff["stats_after"]


async def test_dream_rolls_back_if_apply_fails(store, monkeypatch):
    a = mk("#cat The user has a cat named Mia", h=1, entities=["Mia"])
    b = mk("#cat The user's cat is called Mia", h=2, entities=["Mia"])
    for e in (a, b):
        store.add_entry(e)
    llm = FakeLLM().script("consolidator", {"ops": [
        {"op": "MERGE", "targets": ["M1", "M2"], "new_text": "The user has a cat named Mia",
         "reason": "same"}]})

    def boom(*args, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "save_dream", boom)
    with pytest.raises(RuntimeError):
        await run_dream(store, Embedder(llm, "fake"), llm, Settings(db_path=":memory:"))
    assert sorted(e.text for e in store.list_entries()) == sorted([a.text, b.text])
