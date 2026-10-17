"""Memory storage: a `MemoryStore` interface and its SQLite implementation.

Embeddings live as float32 BLOBs; similarity search is brute-force cosine in numpy, which is
plenty for a single user's memory (hundreds to a few thousand entries). The interface exists so
the backend can later be swapped for pgvector without touching the pipeline.
"""

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.memory.embed import from_bytes
from app.memory.models import MemoryEntry, StoredTier, Turn, new_id, utcnow


class MemoryStore(ABC):
    # --- turns (STM source + naive RAG baseline) ---
    @abstractmethod
    def add_turn(self, turn: Turn) -> None: ...

    @abstractmethod
    def recent_turns(self, session_id: str, n: int) -> list[Turn]:
        """Last `n` turns of a session, oldest first."""

    @abstractmethod
    def search_turns(self, query: np.ndarray, k: int) -> list[tuple[Turn, float]]:
        """Cosine top-k over all embedded turns (any session)."""

    # --- entries (MTM / LTM) ---
    @abstractmethod
    def add_entry(self, entry: MemoryEntry) -> None: ...

    @abstractmethod
    def update_entry(self, entry: MemoryEntry) -> None: ...

    @abstractmethod
    def get_entry(self, entry_id: str) -> MemoryEntry | None: ...

    @abstractmethod
    def list_entries(
        self, tiers: list[StoredTier] | None = None, include_inactive: bool = False
    ) -> list[MemoryEntry]: ...

    @abstractmethod
    def search(
        self, query: np.ndarray, tiers: list[StoredTier], k: int
    ) -> list[tuple[MemoryEntry, float]]:
        """Cosine top-k over *active* entries of the given tiers, best first."""

    @abstractmethod
    def touch(self, entry_ids: list[str]) -> None:
        """Record a retrieval hit: bump access_count and last_accessed_at."""

    # --- dreams ---
    @abstractmethod
    def save_dream(self, dream_id: str, diff: dict, before: dict, after: dict) -> None: ...

    @abstractmethod
    def list_dreams(self) -> list[dict]: ...

    # --- misc ---
    @abstractmethod
    def snapshot(self) -> dict:
        """JSON-serialisable view of all entries (embeddings excluded)."""

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def transaction(self) -> Any:
        """Context manager: all writes inside commit together or not at all."""

    @abstractmethod
    def reset(self) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    seq INTEGER NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, seq);
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    text TEXT NOT NULL,
    kind TEXT NOT NULL,
    entities TEXT NOT NULL,
    topics TEXT NOT NULL,
    source_turn_ids TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    access_count INTEGER NOT NULL,
    last_accessed_at TEXT,
    status TEXT NOT NULL,
    superseded_by TEXT,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_tier ON entries(tier, status);
CREATE TABLE IF NOT EXISTS dreams (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    diff TEXT NOT NULL,
    before_snapshot TEXT NOT NULL,
    after_snapshot TEXT NOT NULL
);
"""

_ENTRY_COLS = (
    "id, tier, text, kind, entities, topics, source_turn_ids, created_at, updated_at, "
    "access_count, last_accessed_at, status, superseded_by, embedding"
)


_N_COLS = len(_ENTRY_COLS.split(","))


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SQLiteStore(MemoryStore):
    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._tx_depth = 0
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- transactions ---
    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.rollback()
                raise
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self._conn.commit()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            if self._tx_depth == 0:
                self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # --- turns ---
    def add_turn(self, turn: Turn) -> None:
        with self._lock:
            seq = self._query("SELECT COALESCE(MAX(seq), 0) + 1 FROM turns")[0][0]
            self._exec(
                "INSERT INTO turns (id, session_id, role, content, created_at, seq, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (turn.id, turn.session_id, turn.role, turn.content,
                 turn.created_at.isoformat(), seq, turn.embedding),
            )

    @staticmethod
    def _row_to_turn(r: sqlite3.Row) -> Turn:
        return Turn(
            id=r["id"], session_id=r["session_id"], role=r["role"], content=r["content"],
            created_at=_dt(r["created_at"]), embedding=r["embedding"],
        )

    def recent_turns(self, session_id: str, n: int) -> list[Turn]:
        if n <= 0:
            return []
        rows = self._query(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY seq DESC LIMIT ?", (session_id, n)
        )
        return [self._row_to_turn(r) for r in reversed(rows)]

    def search_turns(self, query: np.ndarray, k: int) -> list[tuple[Turn, float]]:
        turns = [self._row_to_turn(r) for r in
                 self._query("SELECT * FROM turns WHERE embedding IS NOT NULL")]
        return _top_k(turns, [t.embedding for t in turns], query, k)

    # --- entries ---
    @staticmethod
    def _entry_params(e: MemoryEntry) -> tuple:
        return (
            e.id, e.tier, e.text, e.kind, json.dumps(e.entities, ensure_ascii=False),
            json.dumps(e.topics), json.dumps(e.source_turn_ids),
            e.created_at.isoformat(), e.updated_at.isoformat(),
            e.access_count, e.last_accessed_at.isoformat() if e.last_accessed_at else None,
            e.status, e.superseded_by, e.embedding,
        )

    @staticmethod
    def _row_to_entry(r: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=r["id"], tier=r["tier"], text=r["text"], kind=r["kind"],
            entities=json.loads(r["entities"]), topics=json.loads(r["topics"]),
            source_turn_ids=json.loads(r["source_turn_ids"]),
            created_at=_dt(r["created_at"]), updated_at=_dt(r["updated_at"]),
            access_count=r["access_count"], last_accessed_at=_dt(r["last_accessed_at"]),
            status=r["status"], superseded_by=r["superseded_by"], embedding=r["embedding"],
        )

    def add_entry(self, entry: MemoryEntry) -> None:
        if not entry.embedding:
            raise ValueError("entry has no embedding")
        self._exec(f"INSERT INTO entries ({_ENTRY_COLS}) VALUES ({', '.join('?' * _N_COLS)})",
                   self._entry_params(entry))

    def update_entry(self, entry: MemoryEntry) -> None:
        entry.updated_at = utcnow()
        id_, *rest = self._entry_params(entry)
        cols = [c.strip() for c in _ENTRY_COLS.split(",")][1:]
        cur = self._exec(
            f"UPDATE entries SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
            (*rest, id_),
        )
        if cur.rowcount == 0:
            raise KeyError(entry.id)

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        rows = self._query("SELECT * FROM entries WHERE id = ?", (entry_id,))
        return self._row_to_entry(rows[0]) if rows else None

    def list_entries(
        self, tiers: list[StoredTier] | None = None, include_inactive: bool = False
    ) -> list[MemoryEntry]:
        sql, params = "SELECT * FROM entries WHERE 1=1", []
        if tiers is not None:
            if not tiers:
                return []
            sql += f" AND tier IN ({', '.join('?' * len(tiers))})"
            params += list(tiers)
        if not include_inactive:
            sql += " AND status = 'active'"
        sql += " ORDER BY created_at, id"
        return [self._row_to_entry(r) for r in self._query(sql, tuple(params))]

    def search(
        self, query: np.ndarray, tiers: list[StoredTier], k: int
    ) -> list[tuple[MemoryEntry, float]]:
        entries = self.list_entries(tiers)
        return _top_k(entries, [e.embedding for e in entries], query, k)

    def touch(self, entry_ids: list[str]) -> None:
        if not entry_ids:
            return
        self._exec(
            "UPDATE entries SET access_count = access_count + 1, last_accessed_at = ? "
            f"WHERE id IN ({', '.join('?' * len(entry_ids))})",
            (utcnow().isoformat(), *entry_ids),
        )

    # --- dreams ---
    def save_dream(self, dream_id: str, diff: dict, before: dict, after: dict) -> None:
        self._exec(
            "INSERT INTO dreams (id, created_at, diff, before_snapshot, after_snapshot) "
            "VALUES (?, ?, ?, ?, ?)",
            (dream_id, utcnow().isoformat(), json.dumps(diff, ensure_ascii=False, default=str),
             json.dumps(before, ensure_ascii=False, default=str),
             json.dumps(after, ensure_ascii=False, default=str)),
        )

    def list_dreams(self) -> list[dict]:
        rows = self._query("SELECT id, created_at, diff FROM dreams ORDER BY created_at")
        return [{"dream_id": r["id"], "created_at": r["created_at"], **json.loads(r["diff"])}
                for r in rows]

    # --- misc ---
    def snapshot(self) -> dict:
        return {
            "taken_at": utcnow().isoformat(),
            "entries": [e.model_dump(mode="json") for e in self.list_entries(include_inactive=True)],
        }

    def stats(self) -> dict:
        active = self.list_entries()
        by_tier = {t: [e for e in active if e.tier == t] for t in ("MTM", "LTM")}
        return {
            "mtm": len(by_tier["MTM"]),
            "ltm": len(by_tier["LTM"]),
            "tokens": sum(approx_tokens(e.text) for e in active),
            "inactive": len(self.list_entries(include_inactive=True)) - len(active),
            "turns": self._query("SELECT COUNT(*) FROM turns")[0][0],
        }

    def reset(self) -> None:
        with self.transaction():
            for table in ("turns", "entries", "dreams"):
                self._exec(f"DELETE FROM {table}")


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token); only used for relative memory-size comparisons."""
    return max(1, round(len(text) / 4))


def _top_k(items: list, blobs: list[bytes], query: np.ndarray, k: int) -> list[tuple[Any, float]]:
    if not items or k <= 0:
        return []
    mat = np.stack([from_bytes(b) for b in blobs])
    scores = mat @ np.asarray(query, dtype=mat.dtype)
    order = np.argsort(-scores, kind="stable")[:k]
    return [(items[i], float(scores[i])) for i in order]


__all__ = ["MemoryStore", "SQLiteStore", "approx_tokens", "new_id"]
