"""Prompt templates live in app/prompts/<name>.md and use {{placeholders}}.

Each prompt's short content hash goes into the trace, so every benchmark result can be tied to
the exact prompt version that produced it.
"""

import hashlib
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class Prompt:
    name: str
    template: str
    version: str  # first 8 hex chars of sha256(template)

    def render(self, **values: str) -> str:
        missing = set(_PLACEHOLDER.findall(self.template)) - values.keys()
        if missing:
            raise KeyError(f"prompt {self.name!r} missing values: {sorted(missing)}")
        return _PLACEHOLDER.sub(lambda m: values[m.group(1)], self.template)


@cache
def load_prompt(name: str) -> Prompt:
    template = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    return Prompt(name, template, hashlib.sha256(template.encode()).hexdigest()[:8])
