"""Controller: decides whether memory is needed and plans the retrieval queries."""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm import LLMResult, OllamaClient
from app.memory.models import StoredTier
from app.pipeline.prompts import load_prompt


class ControllerOut(BaseModel):
    # Gates the Writer: the 7B Writer alone kept extracting junk from questions
    # ("The user's dog's name is not stated"). See LOG.md, Phase 2.
    shares_new_info: bool
    needs_memory: bool
    queries: list[str] = Field(max_length=3)
    tiers: list[StoredTier]


@dataclass
class ControllerDecision:
    out: ControllerOut
    result: LLMResult[ControllerOut]
    repairs: list[str]


def repair(out: ControllerOut, message: str) -> tuple[ControllerOut, list[str]]:
    """Schema-valid output can still be useless (Phase 0 finding: needs_memory=true with no
    queries). Fix it deterministically and report what was fixed."""
    repairs: list[str] = []
    queries = [q.strip() for q in out.queries if q.strip()]
    if out.needs_memory and not queries:
        queries = [message]
        repairs.append("empty_queries")
    tiers = list(dict.fromkeys(out.tiers))
    if out.needs_memory and not tiers:
        tiers = ["MTM", "LTM"]
        repairs.append("empty_tiers")
    fixed = out.model_copy(update={"queries": queries, "tiers": tiers})
    return fixed, repairs


async def run_controller(llm: OllamaClient, message: str, stm_text: str) -> ControllerDecision:
    prompt = load_prompt("controller").render(stm=stm_text or "(none)", message=message)
    result = await llm.chat_json(
        "controller",
        [{"role": "user", "content": prompt}],
        ControllerOut,
        fallback=lambda: ControllerOut(
            shares_new_info=True, needs_memory=True, queries=[message], tiers=["MTM", "LTM"]
        ),
    )
    out, repairs = repair(result.value, message)
    return ControllerDecision(out=out, result=result, repairs=repairs)
