"""Benchmark runner.

    python -m bench.run --condition D --model qwen2.5:7b-instruct --runs 3
    python -m bench.run --condition all --model qwen2.5:3b-instruct --runs 1 \
        --sessions s1,s2 --questions q01,q07 --tag smoke

Each (condition, run) starts from an empty memory, replays the 8 scripted sessions, then asks
every question in its own fresh session in read-only mode. Writes:
  bench/results/<run_id>.jsonl   every turn, Dream and answer (for manual review)
  bench/results/summary.csv      one row per (condition, run), appended
  bench/results/summary.md       aggregated table (mean ± sd over runs), regenerated
"""

import argparse
import asyncio
import csv
import json
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

from app.config import Settings
from app.llm import OllamaClient
from app.memory.embed import Embedder
from app.memory.store import SQLiteStore, approx_tokens
from app.pipeline.agent import PROMPT_NAMES
from app.pipeline.prompts import load_prompt
from app.pipeline.strategies import make_strategy
from bench.scoring import score

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"
CATEGORIES = ["single_fact", "updated_fact", "multi_session", "temporal", "abstention"]
STAGES = ["controller", "retriever", "selector", "responder", "writer"]


def git_rev() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=BENCH, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "app"], capture_output=True,
                               text=True, cwd=BENCH, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prompt_versions() -> str:
    names = (*PROMPT_NAMES, "consolidator")
    return " ".join(f"{n}@{load_prompt(n).version}" for n in names)


def load_sessions(only: list[str] | None) -> list[dict]:
    sessions = [json.loads(p.read_text()) for p in sorted((BENCH / "sessions").glob("s*.json"),
                                                          key=lambda p: int(p.stem[1:]))]
    return [s for s in sessions if not only or s["id"] in only]


def load_questions(only: list[str] | None) -> list[dict]:
    qs = json.loads((BENCH / "questions.json").read_text())["questions"]
    return [q for q in qs if not only or q["id"] in only]


def pct(values: list[float], p: float) -> float | None:
    return round(float(np.percentile(values, p)), 1) if values else None


def memory_size(store: SQLiteStore, condition: str) -> dict:
    if condition == "A":
        return {"mem_entries": 0, "mem_tokens": 0, "mtm": 0, "ltm": 0}
    if condition == "B":
        rows = store._query("SELECT content FROM turns WHERE embedding IS NOT NULL")
        return {"mem_entries": len(rows), "mem_tokens": sum(approx_tokens(r[0]) for r in rows),
                "mtm": 0, "ltm": 0}
    st = store.stats()
    return {"mem_entries": st["mtm"] + st["ltm"], "mem_tokens": st["tokens"],
            "mtm": st["mtm"], "ltm": st["ltm"]}


async def run_one(condition: str, model: str, run: int, args, sessions, questions) -> dict:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{model.replace(':', '-').replace('/', '-')}_{condition}_r{run}"
    if args.tag:
        run_id += f"_{args.tag}"
    settings = Settings(db_path=str(Path(tempfile.mkdtemp()) / "bench.db"), llm_model=model,
                        embed_model=args.embed_model)
    llm = OllamaClient(settings.ollama_url, model, format_mode=args.llm_format,
                       temperature=settings.llm_temperature, seed=settings.llm_seed,
                       timeout_s=settings.llm_timeout_s, json_retries=settings.llm_json_retries)
    store = SQLiteStore(settings.db_path)
    strategy = make_strategy(condition, llm, Embedder(llm, settings.embed_model), store, settings)
    out = (RESULTS / f"{run_id}.jsonl").open("w", encoding="utf-8")

    def emit(record: dict) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **record}
        out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        out.flush()

    t_run = time.perf_counter()
    turn_lat, reply_lat, turn_calls, errors = [], [], [], 0
    stage_lat: dict[str, list[float]] = {s: [] for s in STAGES}
    repairs = written = 0
    dreams: list[dict] = []

    for s in sessions:
        for i, msg in enumerate(s["turns"], 1):
            try:
                res = await strategy.chat(s["id"], msg)
            except httpx.HTTPError as e:
                errors += 1
                emit({"type": "turn", "session": s["id"], "idx": i, "message": msg, "error": repr(e)})
                continue
            t = res.trace
            turn_lat.append(t["latency_ms"]["total"])
            reply_lat.append(t["latency_ms"]["reply"])
            turn_calls.append(t["llm_calls"])
            for st in STAGES:
                if st in t["latency_ms"]:
                    stage_lat[st].append(t["latency_ms"][st])
            repairs += sum(len(v) for v in t["repairs"].values())
            entries = (t.get("writer") or {}).get("entries", [])
            written += len(entries)
            emit({"type": "turn", "session": s["id"], "idx": i, "message": msg, "reply": res.reply,
                  "latency_ms": t["latency_ms"], "llm_calls": t["llm_calls"],
                  "fallbacks": t["fallbacks"], "repairs": t["repairs"],
                  "controller": t.get("controller"), "written": [e["text"] for e in entries],
                  "reinforced": len((t.get("writer") or {}).get("reinforced", []))})
        diff = await strategy.end_session(s["id"])
        if diff:
            dreams.append(diff)
            emit({"type": "dream", "session": s["id"], "duration_ms": diff["duration_ms"],
                  "llm_calls": diff["llm_calls"], "ops_proposed": diff["ops_proposed"],
                  "applied": [{k: a[k] for k in ("op", "new_tier", "new_text", "reason")}
                              | {"targets": [x["text"] for x in a["targets"]]}
                              for a in diff["ops_applied"]],
                  "rejected": [{k: r[k] for k in ("op", "targets", "new_text", "error")}
                               for r in diff["ops_rejected"]],
                  "warnings": [w["text"] for w in diff["warnings"]],
                  "stats_after": diff["stats_after"]})
        print(f"  [{run_id}] {s['id']} done ({time.perf_counter() - t_run:.0f}s)", flush=True)

    mem = memory_size(store, condition)
    q_lat, q_calls, results = [], [], []
    for q in questions:
        try:
            res = await strategy.chat(f"question-{q['id']}", q["question"], learn=False)
        except httpx.HTTPError as e:
            errors += 1
            results.append((q, False, False))
            emit({"type": "question", "qid": q["id"], "category": q["category"],
                  "question": q["question"], "error": repr(e), "correct": False})
            continue
        t = res.trace
        sc = score(q, res.reply)
        q_lat.append(t["latency_ms"]["reply"])
        q_calls.append(t["llm_calls"])
        results.append((q, sc.correct, sc.abstained))
        ctx = (t.get("selector") or {}).get("context") if "selector" in t else t.get("memory_context")
        emit({"type": "question", "qid": q["id"], "category": q["category"],
              "question": q["question"], "reply": res.reply, "correct": sc.correct,
              "abstained": sc.abstained, "missing": sc.missing, "forbidden_hits": sc.forbidden_hits,
              "context": ctx, "latency_ms": t["latency_ms"], "llm_calls": t["llm_calls"],
              "fallbacks": t["fallbacks"]})

    llm_stats = llm.stats.as_dict()
    row = {
        "run_id": run_id, "model": model, "condition": condition, "run": run,
        "llm_format": args.llm_format, "tag": args.tag or "",
        "git_rev": git_rev(), "prompts": prompt_versions(),
        "n_questions": len(results),
        "acc_total": round(statistics.mean(c for _, c, _ in results), 3) if results else None,
        **{f"acc_{cat}": (round(statistics.mean(c for q, c, _ in results if q["category"] == cat), 3)
                          if any(q["category"] == cat for q, _, _ in results) else None)
           for cat in CATEGORIES},
        "abstained_wrongly": sum(1 for q, _, a in results if a and q["category"] != "abstention"),
        "session_turns": len(turn_lat),
        "turn_p50_ms": pct(turn_lat, 50), "turn_p95_ms": pct(turn_lat, 95),
        "reply_p50_ms": pct(reply_lat, 50), "reply_p95_ms": pct(reply_lat, 95),
        "question_p50_ms": pct(q_lat, 50), "question_p95_ms": pct(q_lat, 95),
        **{f"{st}_p50_ms": pct(stage_lat[st], 50) for st in STAGES},
        "llm_calls_per_turn": round(statistics.mean(turn_calls), 2) if turn_calls else None,
        "llm_calls_per_question": round(statistics.mean(q_calls), 2) if q_calls else None,
        **mem,
        "written_entries": written,
        "fallbacks": sum(m["fallback"] for m in llm_stats.values()),
        "invalid_json": sum(m["invalid_json"] for m in llm_stats.values()),
        "timeouts": sum(m["timeout"] for m in llm_stats.values()),
        "repairs": repairs,
        "dreams": len(dreams),
        "dream_ms_total": round(sum(d["duration_ms"] for d in dreams), 1),
        "dream_llm_calls": sum(d["llm_calls"] for d in dreams),
        "ops_applied": sum(len(d["ops_applied"]) for d in dreams),
        "ops_rejected": sum(len(d["ops_rejected"]) for d in dreams),
        "stale_warnings": sum(len(d["warnings"]) for d in dreams),
        "errors": errors,
        "wall_s": round(time.perf_counter() - t_run, 1),
    }
    emit({"type": "summary", **row, "llm_stats": llm_stats})
    out.close()
    store.close()
    await llm.aclose()
    return row


def append_csv(row: dict) -> None:
    path = RESULTS / "summary.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    rows.append({k: ("" if v is None else v) for k, v in row.items()})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _agg(rows: list[dict], key: str, scale: float = 1.0, digits: int = 0) -> str:
    vals = [x * scale for x in (_num(r.get(key)) for r in rows) if x is not None]
    if not vals:
        return "–"
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {statistics.stdev(vals):.{digits}f}"


def write_markdown() -> None:
    path = RESULTS / "summary.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r.get("llm_format", ""), r.get("tag", ""),
                           r["condition"]), []).append(r)
    lines = ["# Benchmark results", "",
             ("Mean ± sd over runs. Accuracy in %, latency in seconds (p50 / p95 per user turn "
             "while chatting; questions are answered read-only)."), ""]
    current = None
    for (model, fmt, tag, cond), rs in sorted(groups.items()):
        if (model, fmt, tag) != current:
            current = (model, fmt, tag)
            lines += ["", f"## {model} · format={fmt or 'schema'}{' · ' + tag if tag else ''}", "",
                      ("| Cond. | Runs | Total | Single | Updated | Multi | Temporal | Abstain | "
                      "Turn p50 | Turn p95 | Question p50 | LLM calls/turn | Memory (entries / tokens) "
                      "| Fallbacks | Dream ops ✓/✗ |"),
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        lines.append(
            f"| {cond} | {len(rs)} | {_agg(rs, 'acc_total', 100)} | {_agg(rs, 'acc_single_fact', 100)} "
            f"| {_agg(rs, 'acc_updated_fact', 100)} | {_agg(rs, 'acc_multi_session', 100)} "
            f"| {_agg(rs, 'acc_temporal', 100)} | {_agg(rs, 'acc_abstention', 100)} "
            f"| {_agg(rs, 'turn_p50_ms', 0.001, 1)} | {_agg(rs, 'turn_p95_ms', 0.001, 1)} "
            f"| {_agg(rs, 'question_p50_ms', 0.001, 1)} | {_agg(rs, 'llm_calls_per_turn', 1, 2)} "
            f"| {_agg(rs, 'mem_entries')} / {_agg(rs, 'mem_tokens')} | {_agg(rs, 'fallbacks')} "
            f"| {_agg(rs, 'ops_applied')} / {_agg(rs, 'ops_rejected')} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--condition", default="all", help="A, B, C, D, a comma list, or 'all'")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--llm-format", choices=["schema", "json"], default="schema")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--sessions", help="comma list, e.g. s1,s2 (default: all)")
    ap.add_argument("--questions", help="comma list, e.g. q01,q07 (default: all)")
    ap.add_argument("--tag", default="", help="label stored with the results (e.g. smoke)")
    args = ap.parse_args()

    conditions = ["A", "B", "C", "D"] if args.condition == "all" else args.condition.split(",")
    sessions = load_sessions(args.sessions.split(",") if args.sessions else None)
    questions = load_questions(args.questions.split(",") if args.questions else None)
    RESULTS.mkdir(exist_ok=True)
    print(f"== {args.model}: conditions={conditions} runs={args.runs} sessions={len(sessions)} "
          f"questions={len(questions)}", flush=True)
    # Runs are interleaved (A1 B1 C1 D1 A2 ...) so drift in machine load spreads across conditions.
    for run in range(1, args.runs + 1):
        for cond in conditions:
            row = await run_one(cond, args.model, run, args, sessions, questions)
            append_csv(row)
            write_markdown()
            print(f"== {row['run_id']}: acc={row['acc_total']} turn_p50={row['turn_p50_ms']}ms "
                  f"calls/turn={row['llm_calls_per_turn']} mem={row['mem_entries']} "
                  f"wall={row['wall_s']}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
