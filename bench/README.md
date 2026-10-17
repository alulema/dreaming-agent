# Benchmark

Compares 4 memory conditions with the same model and the same scripted data.

| Condition | Memory |
|---|---|
| A | STM only (last turns of the current session) |
| B | Naive RAG: every past user turn embedded; top-k straight to the Responder |
| C | Full online pipeline (Controller → Retriever → Selector → Responder → Writer), no Dream |
| D | C + a Dream at the end of every session |

## Data

A fictional user ("Mateo") across 8 sessions (`sessions/s1..s8.json`, 45 user turns). The
assistant's replies are whatever the model says, so they are not part of the data. The sessions
deliberately include:

- **Paraphrases** for MERGE: the parrot Kiwi is mentioned in 5 sessions, in Spanish and English.
- **Changes** for UPDATE: job (architect → UX designer), city (Valparaíso → Montevideo), diet
  (vegetarian → eats fish), the sister's city (Lisbon → Porto), sport preference.
- **Trivial noise** for FORGET: weather, meals, traffic, one per session.
- **Mixed languages**: s3 and s6 are in English, s5 mixes both within single sentences.
- **Questions to the assistant** inside the sessions, which should not produce memories.

`questions.json` holds 28 questions, asked in a new session after the 8 sessions:
single_fact (6), updated_fact (6), multi_session (6), temporal (5) and abstention (5, facts
never stated).

Guard rails (enforced by `tests/test_bench.py`): no entity from the prompts' examples appears in
the data (models copy examples verbatim when unsure), and none from the persona used while
tuning the prompts.

## Scoring (`scoring.py`)

- Normalisation: lowercase, no accents.
- `expected`: a list of groups, and every group must match one of its alternatives. Alternatives
  of 3 characters or fewer match as whole words; longer ones match as substrings (stems).
- `forbidden`: substrings that make an answer wrong (typically the old value of an updated fact).
  The updated_fact questions ask for short answers, to reduce correct answers that mention the
  old value "for context".
- Abstention: correct if the answer contains an abstention phrase ("no lo sé", "no me lo has
  dicho", "I don't know"...).

Keyword scoring is imperfect by design: every answer is saved to `results/<run_id>.jsonl` for
manual review.
