"""Deterministic keyword scoring for the benchmark.

Rules (documented in bench/README.md):
- Text is normalised: lowercase, accents removed, whitespace collapsed.
- `expected` is a list of groups; every group must match at least one of its alternatives.
- Alternatives of 3 characters or fewer ("si", "ux", "12", "dos") must match as whole words;
  longer ones match as substrings, so stems work ("escal" matches escalada / escalar).
- `forbidden` alternatives are substrings; any match makes the answer wrong.
- Abstention questions are correct when the answer contains an abstention phrase; on every
  other question an abstaining answer is wrong even if it happens to contain a keyword
  ("No tengo información sobre si eras vegetariano" must not count as "eras vegetariano").
  Only the first sentence is checked for that penalty, so a correct answer with a trailing
  caveat ("...Montevideo. No mencionaste otras ciudades.") still counts.

Keyword scoring is knowingly imperfect: every answer is saved for manual review, and the
cases where it gets things wrong are part of the write-up.
"""

import re
import unicodedata
from dataclasses import dataclass

ABSTENTION_PHRASES = [
    "no lo se", "no se ", "no lo recuerdo", "no recuerdo", "no me lo has dicho",
    "no me has dicho", "no me has contado", "no me lo has contado", "no lo has mencionado",
    "no has mencionado", "no me has mencionado", "no mencionaste", "no tengo esa informacion",
    "no tengo informacion", "no tengo ese dato", "no tengo registro", "no dispongo",
    "desconozco", "no estoy seguro", "no tengo constancia",
    "i don't know", "i do not know", "you haven't told me", "you have not told me",
    "you haven't mentioned", "you never mentioned", "you didn't mention",
    "i don't have that information", "i don't have any information", "i'm not sure",
    "i don't remember", "i do not have", "don't have information", "do not have information",
    "don't have any details", "no tengo datos", "no tengo detalles",
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(text.split())


def matches(text: str, keyword: str) -> bool:
    """`text` must already be normalised."""
    kw = normalize(keyword)
    if len(kw) <= 3:
        return re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text) is not None
    return kw in text


def is_abstention(answer: str) -> bool:
    text = normalize(answer) + " "
    return any(normalize(p) in text or p in text for p in ABSTENTION_PHRASES)


def first_sentence(answer: str) -> str:
    return re.split(r"(?<=[.!?])\s", answer.strip(), maxsplit=1)[0]


@dataclass
class Score:
    correct: bool
    abstained: bool
    missing: list[list[str]]   # expected groups with no match
    forbidden_hits: list[str]


def score(question: dict, answer: str) -> Score:
    text = normalize(answer)
    abstained = is_abstention(answer)
    if question["category"] == "abstention":
        return Score(correct=abstained, abstained=abstained, missing=[], forbidden_hits=[])
    missing = [g for g in question["expected"] if not any(matches(text, k) for k in g)]
    hits = [f for f in question["forbidden"] if normalize(f) in text]
    leads_with_abstention = is_abstention(first_sentence(answer))
    return Score(correct=not missing and not hits and not leads_with_abstention,
                 abstained=abstained, missing=missing, forbidden_hits=hits)
