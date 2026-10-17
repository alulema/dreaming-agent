"""Consolidator: asks the model for typed ops over one cluster.

Entries are shown with short labels (M1.., L1..) instead of uuids; labels are mapped back to
entry ids here. Unknown labels are kept verbatim so the Validator rejects them as unknown ids.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from app.dream.cluster import Cluster
from app.dream.validator import Rules, has_evidence
from app.llm import LLMResult, OllamaClient
from app.memory.models import MemoryEntry, Op, OpType
from app.pipeline.prompts import load_prompt


class LLMOp(BaseModel):
    op: OpType
    targets: list[str]
    new_text: str | None = None
    reason: str


class ConsolidatorOut(BaseModel):
    ops: list[LLMOp]


@dataclass
class Proposal:
    ops: list[Op]
    labels: dict[str, str]  # label -> entry id
    result: LLMResult[ConsolidatorOut]


def label_entries(cluster: Cluster) -> dict[str, MemoryEntry]:
    labels = {f"M{i}": e for i, e in enumerate(cluster.mtm, start=1)}
    labels.update({f"L{i}": e for i, e in enumerate(cluster.ltm, start=1)})
    return labels


def format_entries(labeled: dict[str, MemoryEntry], rules: Rules) -> str:
    lines = []
    for label, e in labeled.items():
        topics = ", ".join(e.topics) or "-"
        n = len(e.source_turn_ids)
        # The model ignored the turn count ("stated in 2 turns" for 1-turn entries), so the
        # PROMOTE eligibility computed by code is spelled out.
        promotable = ", promotable" if e.tier == "MTM" and has_evidence([e], rules) else ""
        lines.append(f"[{label}] ({e.created_at:%Y-%m-%d %H:%M}, {e.tier}, {topics}, "
                     f"{n} turn{'s' if n != 1 else ''}{promotable}) {e.text}")
    return "\n".join(lines)


def normalize_label(raw: str) -> str:
    return raw.strip().strip("[]").upper()


async def propose_ops(llm: OllamaClient, cluster: Cluster, rules: Rules) -> Proposal:
    labeled = label_entries(cluster)
    prompt = load_prompt("consolidator").render(entries=format_entries(labeled, rules))
    result = await llm.chat_json(
        "consolidator",
        [{"role": "user", "content": prompt}],
        ConsolidatorOut,
        fallback=lambda: ConsolidatorOut(ops=[]),
    )
    ops = []
    for o in result.value.ops:
        ids = [labeled[lbl].id if (lbl := normalize_label(t)) in labeled else t for t in o.targets]
        ops.append(Op(op=o.op, target_ids=ids, new_text=o.new_text, reason=o.reason))
    return Proposal(ops=ops, labels={k: e.id for k, e in labeled.items()}, result=result)
