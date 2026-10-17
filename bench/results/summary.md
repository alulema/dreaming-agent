# Benchmark results

Mean ± sd over runs. Accuracy in %, latency in seconds (p50 / p95 per user turn while chatting; questions are answered read-only).


## qwen2.5:7b-instruct · format=schema · v1

| Cond. | Runs | Total | Single | Updated | Multi | Temporal | Abstain | Turn p50 | Turn p95 | Question p50 | LLM calls/turn | Memory (entries / tokens) | Fallbacks | Dream ops ✓/✗ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3 | 18 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 100 ± 0 | 5.6 ± 3.7 | 13.0 ± 8.4 | 6.0 ± 4.1 | 1.00 ± 0.00 | 0 ± 0 / 0 ± 0 | 0 ± 0 | 0 ± 0 / 0 ± 0 |
| B | 3 | 89 ± 0 | 100 ± 0 | 100 ± 0 | 67 ± 0 | 80 ± 0 | 100 ± 0 | 15.8 ± 10.3 | 27.0 ± 15.0 | 12.1 ± 8.6 | 1.00 ± 0.00 | 45 ± 0 / 584 ± 0 | 0 ± 0 | 0 ± 0 / 0 ± 0 |
| C | 3 | 69 ± 2 | 83 ± 0 | 83 ± 0 | 44 ± 10 | 33 ± 12 | 100 ± 0 | 30.5 ± 17.6 | 51.5 ± 25.2 | 33.5 ± 21.7 | 2.92 ± 0.01 | 45 ± 0 / 476 ± 0 | 0 ± 0 | 0 ± 0 / 0 ± 0 |
| D | 3 | 66 ± 4 | 67 ± 0 | 83 ± 0 | 44 ± 10 | 33 ± 12 | 100 ± 0 | 29.2 ± 19.5 | 46.9 ± 31.3 | 41.3 ± 14.5 | 2.92 ± 0.01 | 30 ± 5 / 418 ± 47 | 4 ± 5 | 13 ± 5 / 17 ± 6 |
