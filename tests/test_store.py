import numpy as np
import pytest

from app.memory.embed import from_bytes, normalize, to_bytes
from app.memory.models import MemoryEntry, Turn
from app.memory.store import SQLiteStore


def vec(*xs: float) -> np.ndarray:
    return normalize(np.array(xs))


def entry(text: str, v: np.ndarray, tier: str = "MTM", **kw) -> MemoryEntry:
    return MemoryEntry(text=text, tier=tier, embedding=to_bytes(v), **kw)


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


def test_embedding_roundtrip():
    v = vec(3, 4)
    assert np.allclose(from_bytes(to_bytes(v)), [0.6, 0.8])
    assert np.allclose(normalize(np.zeros((1, 2))), 0)  # no division by zero


def test_entry_roundtrip(store):
    e = entry("User's dog is called Toby", vec(1, 0), entities=["Toby", "dog"],
              kind="fact", source_turn_ids=["t1"])
    store.add_entry(e)
    got = store.get_entry(e.id)
    assert got.text == e.text and got.entities == ["Toby", "dog"]
    assert got.source_turn_ids == ["t1"] and got.embedding == e.embedding
    assert got.created_at == e.created_at
    assert store.get_entry("missing") is None


def test_add_entry_requires_embedding(store):
    with pytest.raises(ValueError):
        store.add_entry(MemoryEntry(text="no vector"))


def test_update_entry_and_missing(store):
    e = entry("lives in Quito", vec(1, 0))
    store.add_entry(e)
    e.status, e.superseded_by = "superseded", "other"
    store.update_entry(e)
    got = store.get_entry(e.id)
    assert got.status == "superseded" and got.superseded_by == "other"
    with pytest.raises(KeyError):
        store.update_entry(entry("ghost", vec(1, 0)))


def test_search_ranks_by_cosine_and_filters_tier_and_status(store):
    a = entry("dog Toby", vec(1, 0))
    b = entry("likes coffee", vec(0, 1))
    c = entry("dog breed beagle", vec(0.9, 0.1), tier="LTM")
    d = entry("old dog fact", vec(1, 0), status="forgotten")
    for e in (a, b, c, d):
        store.add_entry(e)

    hits = store.search(vec(1, 0), ["MTM", "LTM"], k=2)
    assert [h[0].id for h in hits] == [a.id, c.id]
    assert hits[0][1] == pytest.approx(1.0)

    assert [h[0].id for h in store.search(vec(1, 0), ["LTM"], k=5)] == [c.id]
    assert store.search(vec(1, 0), [], k=5) == []
    assert store.search(vec(1, 0), ["MTM"], k=0) == []


def test_list_entries_filters(store):
    store.add_entry(entry("a", vec(1, 0)))
    store.add_entry(entry("b", vec(1, 0), tier="LTM"))
    store.add_entry(entry("c", vec(1, 0), status="forgotten"))
    assert len(store.list_entries()) == 2
    assert len(store.list_entries(include_inactive=True)) == 3
    assert [e.text for e in store.list_entries(["LTM"])] == ["b"]


def test_touch_increments_access(store):
    e = entry("a", vec(1, 0))
    store.add_entry(e)
    store.touch([e.id])
    store.touch([e.id])
    store.touch([])
    got = store.get_entry(e.id)
    assert got.access_count == 2 and got.last_accessed_at is not None


def test_recent_turns_window_and_order(store):
    for i in range(10):
        store.add_turn(Turn(session_id="s1", role="user", content=f"m{i}"))
    store.add_turn(Turn(session_id="s2", role="user", content="other"))
    assert [t.content for t in store.recent_turns("s1", 3)] == ["m7", "m8", "m9"]
    assert store.recent_turns("s1", 0) == []
    assert [t.content for t in store.recent_turns("s2", 6)] == ["other"]


def test_search_turns_only_embedded(store):
    store.add_turn(Turn(session_id="s", role="user", content="dog", embedding=to_bytes(vec(1, 0))))
    store.add_turn(Turn(session_id="s", role="user", content="no vec"))
    hits = store.search_turns(vec(1, 0), 5)
    assert [h[0].content for h in hits] == ["dog"]


def test_transaction_rolls_back_on_error(store):
    store.add_entry(entry("kept", vec(1, 0)))
    with pytest.raises(RuntimeError), store.transaction():
        store.add_entry(entry("lost-1", vec(1, 0)))
        with store.transaction():  # nested: inner exit must not commit
            store.add_entry(entry("lost-2", vec(1, 0)))
        raise RuntimeError("boom")
    assert [e.text for e in store.list_entries()] == ["kept"]


def test_transaction_commits(store):
    with store.transaction():
        store.add_entry(entry("a", vec(1, 0)))
        store.add_entry(entry("b", vec(1, 0)))
    assert len(store.list_entries()) == 2


def test_dreams_snapshot_stats_reset(store):
    store.add_entry(entry("abcdefgh", vec(1, 0)))
    store.add_entry(entry("x", vec(1, 0), tier="LTM"))
    store.add_entry(entry("gone", vec(1, 0), status="forgotten"))
    snap = store.snapshot()
    assert len(snap["entries"]) == 3 and "embedding" not in snap["entries"][0]

    st = store.stats()
    assert (st["mtm"], st["ltm"], st["inactive"], st["tokens"]) == (1, 1, 1, 3)

    store.save_dream("d1", {"ops_applied": [], "duration_ms": 5}, snap, snap)
    dreams = store.list_dreams()
    assert dreams[0]["dream_id"] == "d1" and dreams[0]["duration_ms"] == 5

    store.add_turn(Turn(session_id="s", role="user", content="hi"))
    store.reset()
    assert store.stats()["mtm"] == 0 and store.list_dreams() == []
    assert store.stats()["turns"] == 0


def test_file_backed_store_persists(tmp_path):
    path = str(tmp_path / "sub" / "m.db")
    s1 = SQLiteStore(path)
    s1.add_entry(entry("persisted", vec(1, 0)))
    s1.close()
    s2 = SQLiteStore(path)
    assert [e.text for e in s2.list_entries()] == ["persisted"]
    s2.close()
