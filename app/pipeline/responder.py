"""Responder: generates the reply from STM + the Selector's compressed memory context."""

from app.llm import OllamaClient
from app.memory.models import Turn
from app.pipeline.prompts import load_prompt


def build_messages(memory_context: str, stm: list[Turn], message: str) -> list[dict]:
    system = load_prompt("responder").render(memory=memory_context or "(nothing yet)")
    return [
        {"role": "system", "content": system},
        *({"role": t.role, "content": t.content} for t in stm),
        {"role": "user", "content": message},
    ]


async def run_responder(
    llm: OllamaClient, memory_context: str, stm: list[Turn], message: str
) -> str:
    return (await llm.chat_text("responder", build_messages(memory_context, stm, message))).strip()
