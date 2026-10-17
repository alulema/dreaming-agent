import json
import re
from pathlib import Path

import pytest

from app.pipeline.prompts import PROMPTS_DIR
from bench.scoring import is_abstention, matches, normalize, score

BENCH = Path(__file__).resolve().parent.parent / "bench"
SESSIONS = sorted((BENCH / "sessions").glob("*.json"))
QUESTIONS = json.loads((BENCH / "questions.json").read_text())["questions"]

# Persona used while tuning the prompts in Phases 2-3; reusing it would overfit the benchmark.
TUNING_PERSONA = {"lucia", "quito", "toby", "madrid", "malasana", "marco", "pelusa", "simba",
                  "valencia", "sevilla", "triana"}


def bench_text() -> str:
    parts = [json.dumps(json.loads(p.read_text())["turns"], ensure_ascii=False) for p in SESSIONS]
    parts += [q["question"] for q in QUESTIONS]
    return normalize(" ".join(parts))


def prompt_example_names() -> set[str]:
    """Capitalised words inside the prompts' example lines (lines starting with '-' or '[A')."""
    names = set()
    for p in PROMPTS_DIR.glob("*.md"):
        for line in p.read_text().splitlines():
            if line.lstrip().startswith(("- \"", "[A", "{\"ops\"")):
                names |= {normalize(w) for w in re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}", line)}
    return names - {"the", "user", "thanks", "recommend", "what", "do", "mi", "me", "soy",
                    "no", "big", "answer", "memories", "example", "como", "cuantos", "gracias"}


# --- data integrity ---

def test_sessions_shape():
    assert [p.stem for p in SESSIONS] == [f"s{i}" for i in range(1, 9)]
    for p in SESSIONS:
        s = json.loads(p.read_text())
        assert s["id"] == p.stem and 5 <= len(s["turns"]) <= 7
        assert all(isinstance(t, str) and t.strip() for t in s["turns"])


def test_question_counts_and_shape():
    cats = [q["category"] for q in QUESTIONS]
    assert {c: cats.count(c) for c in set(cats)} == {
        "single_fact": 6, "updated_fact": 6, "multi_session": 6, "temporal": 5, "abstention": 5}
    assert len({q["id"] for q in QUESTIONS}) == 28
    for q in QUESTIONS:
        assert (q["category"] == "abstention") == (not q["expected"])
        assert all(g and all(k.strip() for k in g) for g in q["expected"])


def test_no_prompt_example_entities_in_bench():
    # Phase 2-3 finding: models copy prompt examples verbatim when unsure.
    names = prompt_example_names()
    assert {"nube", "bilbao", "lima", "iker"} <= names  # the extractor sees the examples
    text = bench_text()
    leaked = sorted(n for n in names if re.search(rf"\b{re.escape(n)}\b", text))
    assert leaked == []


def test_no_tuning_persona_in_bench():
    text = bench_text()
    assert sorted(n for n in TUNING_PERSONA if re.search(rf"\b{n}\b", text)) == []


def test_abstention_answers_are_not_in_the_sessions():
    text = normalize(" ".join(json.dumps(json.loads(p.read_text())["turns"], ensure_ascii=False)
                              for p in SESSIONS))
    for word in ("madre", "mother", "coche", "car ", "color", "sangre", "blood", "universidad"):
        assert word not in text, word


# --- scoring ---

def test_normalize_and_matching():
    assert normalize("  Maní  y   VALPARAÍSO ") == "mani y valparaiso"
    t = normalize("Sí, trabajas en UX y naciste el 12 de marzo")
    assert matches(t, "si") and matches(t, "ux") and matches(t, "12")
    assert not matches(normalize("situación difícil"), "si")      # short => whole word
    assert matches(normalize("Te gusta escalar"), "escal")        # long => stem/substring


def q(id_):
    return next(x for x in QUESTIONS if x["id"] == id_)


@pytest.mark.parametrize("qid,answer,correct", [
    ("q01", "Te llamas Mateo.", True),
    ("q03", "Eres alérgico al maní.", True),
    ("q03", "You're allergic to peanuts.", True),
    ("q05", "Tu cumpleaños es el 12 de marzo.", True),
    ("q05", "Tu cumpleaños es en marzo.", False),                  # both groups needed
    ("q07", "Montevideo.", True),
    ("q07", "Montevideo (antes Valparaíso).", False),              # forbidden old value
    ("q08", "Eres diseñador UX en Pagora.", True),
    ("q08", "Eres arquitecto.", False),
    ("q10", "Ya no eres vegetariano: ahora comes pescado.", True),
    ("q10", "Eres vegetariano.", False),
    ("q12", "Tienes dos mascotas: Kiwi y Ramón.", True),
    ("q13", "Natación, escalada y ciclismo.", True),
    ("q13", "Nadas y escalas.", False),
    ("q19", "Hiciste un curso online de diseño UX.", True),
    ("q21", "Antes: viajaste a la Patagonia y después te mudaste a Montevideo.", True),
    ("q21", "Fue después de mudarte a Montevideo.", False),
    ("q23", "Sí, eras vegetariano en esa época.", True),
    ("q24", "No me lo has dicho.", True),
    ("q24", "Tu madre se llama Rosa.", False),
    ("q27", "I don't know, you haven't told me.", True),
])
def test_reference_answers(qid, answer, correct):
    assert score(q(qid), answer).correct is correct


def test_abstention_flag_on_regular_questions():
    s = score(q("q01"), "No lo sé, no me lo has dicho.")
    assert s.abstained and not s.correct and s.missing == [["mateo"]]
    assert is_abstention("Lo siento, no tengo esa información.")
    assert not is_abstention("Te llamas Mateo.")
