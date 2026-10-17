"""Selector: verifies retrieved candidates against the question and compresses them."""

from dataclasses import dataclass

from pydantic import BaseModel

from app.llm import LLMResult, OllamaClient
from app.pipeline.prompts import load_prompt
from app.pipeline.retriever import Candidate


class SelectorOut(BaseModel):
    # 1-based positions in the candidate list. Small models copy short integers far more
    # reliably than 32-char uuids; they are mapped back to entry ids in code.
    keep: list[int]
    context: str


@dataclass
class Selection:
    kept: list[Candidate]
    context: str
    result: LLMResult[SelectorOut] | None  # None => skipped (no candidates)
    repairs: list[str]


def format_candidates(candidates: list[Candidate]) -> str:
    return "\n".join(
        f"[{i}] ({c.entry.updated_at:%Y-%m-%d %H:%M}, {c.entry.tier}) {c.entry.text}"
        for i, c in enumerate(candidates, start=1)
    )


def fallback_context(kept: list[Candidate]) -> str:
    return "\n".join(f"- {c.entry.text}" for c in kept)


async def run_selector(
    llm: OllamaClient, message: str, candidates: list[Candidate], fallback_k: int
) -> Selection:
    if not candidates:
        return Selection(kept=[], context="", result=None, repairs=[])
    prompt = load_prompt("selector").render(
        message=message, candidates=format_candidates(candidates)
    )
    n_fb = min(fallback_k, len(candidates))
    result = await llm.chat_json(
        "selector",
        [{"role": "user", "content": prompt}],
        SelectorOut,
        fallback=lambda: SelectorOut(
            keep=list(range(1, n_fb + 1)), context=fallback_context(candidates[:n_fb])
        ),
    )
    repairs: list[str] = []
    positions = list(dict.fromkeys(result.value.keep))
    valid = [p for p in positions if 1 <= p <= len(candidates)]
    if len(valid) != len(positions):
        repairs.append("out_of_range_keep")
    kept = [candidates[p - 1] for p in valid]
    context = result.value.context.strip()
    if kept and not context:
        context = fallback_context(kept)
        repairs.append("empty_context")
    return Selection(kept=kept, context=context, result=result, repairs=repairs)
