"""Core data models: conversation turns, memory entries and dream operations."""

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["STM", "MTM", "LTM"]
StoredTier = Literal["MTM", "LTM"]
EntryKind = Literal["fact", "preference", "event", "summary"]
EntryStatus = Literal["active", "superseded", "forgotten"]
OpType = Literal["MERGE", "UPDATE", "PROMOTE", "FORGET"]


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Turn(BaseModel):
    """One raw message of a session. STM is the sliding window over these."""

    id: str = Field(default_factory=new_id)
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=utcnow)
    # Only used by the naive-RAG baseline (condition B).
    embedding: bytes | None = Field(default=None, exclude=True, repr=False)


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=new_id)
    tier: StoredTier = "MTM"
    text: str
    kind: EntryKind = "fact"
    entities: list[str] = Field(default_factory=list)
    source_turn_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    access_count: int = 0
    last_accessed_at: datetime | None = None
    # History is never physically deleted: replaced entries point to their successor,
    # forgotten entries are soft-deleted.
    status: EntryStatus = "active"
    superseded_by: str | None = None
    embedding: bytes = Field(default=b"", exclude=True, repr=False)

    @property
    def is_active(self) -> bool:
        return self.status == "active"


class Op(BaseModel):
    """A typed consolidation operation proposed by the model and validated by code."""

    op: OpType
    target_ids: list[str]
    new_text: str | None = None
    reason: str = ""
