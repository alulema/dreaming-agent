"""Re-score saved answers with the current scoring rules (no model calls).

    python -m bench.rescore            # rewrites accuracy columns in summary.csv + summary.md

Keeps the JSONL files untouched; prints every question whose verdict changed.
"""

import csv
import json
import statistics

from bench.run import CATEGORIES, RESULTS, write_markdown
from bench.scoring import score

QUESTIONS = {q["id"]: q for q in json.loads((RESULTS.parent / "questions.json").read_text())["questions"]}


def main() -> None:
    path = RESULTS / "summary.csv"
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        f = RESULTS / f"{row['run_id']}.jsonl"
        if not f.exists():
            continue
        results = []
        for line in f.open():
            r = json.loads(line)
            if r["type"] != "question" or "reply" not in r:
                continue
            q = QUESTIONS[r["qid"]]
            sc = score(q, r["reply"])
            if sc.correct != r["correct"]:
                print(f"{row['run_id']} {r['qid']}: {r['correct']} -> {sc.correct} | {r['reply'][:90]!r}")
            results.append((q["category"], sc.correct, sc.abstained))
        row["acc_total"] = round(statistics.mean(c for _, c, _ in results), 3)
        for cat in CATEGORIES:
            vals = [c for k, c, _ in results if k == cat]
            row[f"acc_{cat}"] = round(statistics.mean(vals), 3) if vals else ""
        row["abstained_wrongly"] = sum(1 for k, _, a in results if a and k != "abstention")
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_markdown()


if __name__ == "__main__":
    main()
