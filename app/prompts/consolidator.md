You are consolidating the memory of a personal assistant while it is offline ("dreaming"). Below are related memories about the user. Propose operations that make this memory cleaner and more accurate.

Memories (label, date of the fact, tier, topics, number of turns where it was stated, and "promotable" when it may be promoted):
{{entries}}

Tiers: MTM = recent facts written during conversations. LTM = consolidated long-term knowledge.

Allowed operations:
- MERGE: two or more MTM memories say the same thing, or are details of one thing (the same pet, the same person). targets = their labels; new_text = one sentence that keeps every detail of all of them.
- UPDATE: memories that contradict each other because something changed over time (the user moved, changed job, changed a preference). The memory with the LATER date is the current truth. targets = ALL the memories that hold an old or new value of that attribute, including LTM ones; new_text = the current fact, and if useful the previous value in parentheses.
- PROMOTE: one MTM memory marked "promotable" that is a stable fact about the user. targets = its label; new_text = the fact, cleaned up. Never PROMOTE a memory that is not marked "promotable".
- FORGET: an MTM memory about a one-off moment with no lasting value (what the user ate one day, today's weather, a single errand). Names, family, pets, home, job, health, hobbies and tastes are NEVER trivial: do not forget them.

Rules:
- Be conservative: if memories are about different things, do NOT merge them. If nothing needs to change, return {"ops": []}.
- Use only the facts in the memories above. Never add names, places or numbers that are not written in the targets.
- Write new_text in English, in third person, starting with "The user".
- Use each label in at most one operation, and only labels from the list above.
- reason: a short explanation (one sentence).

Example of the format only (the labels A1-A4 and these facts do not exist in the list above; never copy them):
[A1] (2026-01-03 10:00, MTM, user.pets, 1 turn) The user has a cat named Nube
[A2] (2026-01-03 10:05, MTM, user.pets, 1 turn) The user's cat is called Nube
[A3] (2026-01-05 18:00, MTM, user.location, 1 turn) The user lives in Lima
[A4] (2026-02-10 09:00, MTM, user.location, 1 turn) The user moved to Bilbao
{"ops": [{"op": "MERGE", "targets": ["A1", "A2"], "new_text": "The user has a cat named Nube", "reason": "Same fact stated twice"}, {"op": "UPDATE", "targets": ["A3", "A4"], "new_text": "The user lives in Bilbao (previously lived in Lima)", "reason": "The user moved; the later memory is current"}]}

Answer with JSON only.
