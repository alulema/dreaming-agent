"""Writer: after each reply, extracts 0..n new MTM entries from the exchange."""

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm import LLMResult, OllamaClient
from app.memory.grounding import ungrounded
from app.memory.models import EntryKind, Topic
from app.pipeline.prompts import load_prompt


class WriterEntry(BaseModel):
    text: str
    kind: EntryKind
    entities: list[str]
    topics: list[Topic] = Field(max_length=2)


class WriterOut(BaseModel):
    # Decided first (field order is generation order): an explicit gate made the 7B stop
    # emitting meta-entries such as "The user asked about..." for questions. See LOG.md.
    has_new_facts: bool
    entries: list[WriterEntry]


@dataclass
class WriterResult:
    entries: list[WriterEntry]
    result: LLMResult[WriterOut]
    dropped: int  # entries discarded by `clean` (empty, duplicated, meta, gated)


# "The user asked/is asking/said/greeted..." describes the conversation, not the user's life.
_META = re.compile(r"^the user (asked|asks|is asking|said|says|greeted|greets|thanked|thanks)\b",
                   re.IGNORECASE)


def clean(out: WriterOut, source: str | None = None) -> tuple[list[WriterEntry], int]:
    """`source`: the user's message; when given, entries with names or numbers that are not
    in it are dropped (v1/3B: "The user lives in Bilbao" from "Tengo un loro…")."""
    if not out.has_new_facts:
        return [], len(out.entries)
    seen: set[str] = set()
    kept: list[WriterEntry] = []
    for e in out.entries:
        text = " ".join(e.text.split())
        key = text.lower().rstrip(".")
        if not text or key in seen or _META.match(text):
            continue
        if source is not None and ungrounded(text, source, fuzzy=True):
            continue
        seen.add(key)
        kept.append(e.model_copy(update={
            "text": text,
            "entities": list(dict.fromkeys(x.strip() for x in e.entities if x.strip())),
            "topics": list(dict.fromkeys(e.topics)),
        }))
    return kept, len(out.entries) - len(kept)


async def run_writer(llm: OllamaClient, message: str, *, grounding: bool = False) -> WriterResult:
    """Sees only the user's message. With the assistant's reply (run 1) or the previous
    assistant message (run 2) in view, the 7B kept re-extracting facts the assistant had just
    recited from memory ("echo" duplicates). See LOG.md, Phase 2."""
    prompt = load_prompt("writer").render(message=message)
    result = await llm.chat_json(
        "writer",
        [{"role": "user", "content": prompt}],
        WriterOut,
        fallback=lambda: WriterOut(has_new_facts=False, entries=[]),
    )
    entries, dropped = clean(result.value, message if grounding else None)
    return WriterResult(entries=entries, result=result, dropped=dropped)
