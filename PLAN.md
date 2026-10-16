# Plan de desarrollo — El agente que sueña

> Plan de ejecución de [`agente-que-suena-SPEC.md`](agente-que-suena-SPEC.md) (qué construir) bajo el contrato de [`DEMO_INTEGRATION.md`](DEMO_INTEGRATION.md) (cómo publicarlo).
> Notas del paper: [`docs/paper-notes.md`](docs/paper-notes.md) · Bitácora: [`LOG.md`](LOG.md).

Regla: **no se avanza de fase sin cumplir sus criterios de aceptación.** Cada fase se cierra con un commit y una entrada en `LOG.md`.

## Decisiones transversales

| Tema | Decisión | Razón |
|---|---|---|
| Modelos | **Benchmark local** (M1 Pro, 16 GB): `qwen2.5:7b-instruct` y `qwen2.5:3b-instruct`, más `llama3.2:1b` / `qwen2.5:1.5b-instruct` como "tamaño del paper". **Demo público**: 1.5B o 3B, elegido con latencia medida | El pod del demo tiene 2 vCPU / 4 GiB sin GPU; el pipeline hace 4–5 llamadas por turno. El paper usa SLM de 1–1.5B |
| Embeddings | `bge-m3` en el benchmark; para el demo, evaluar `nomic-embed-text` o un MiniLM multilingüe por memoria | bge-m3 ocupa ~1.2 GB |
| Salida JSON | `format` de Ollama con **JSON Schema generado desde Pydantic**; validación, 1 reintento y fallback | Ollama ≥0.5 acepta schemas; medir la tasa de inválidos con y sin schema |
| Determinismo | `temperature=0`, `seed` fijo y configurable | 3 corridas por condición; reportar varianza igual |
| Paquetes | `uv` + `pyproject.toml`; `requirements.txt` exportado para Docker | |
| Raíz | La carpeta `dreaming-agent/` (equivale a `agente-que-suena/` de la spec) | |
| Bitácora | `LOG.md` = log de la spec + devlog del contrato. Registra decisiones, **desafíos superados** y **hallazgos** | Materia prima del post |
| Prompts | `app/prompts/<module>.md` versionados; el hash del prompt va en la traza | Reproducibilidad |

---

## Fase 0 — Preparación
- [x] Leer la spec, el contrato y el paper (resumen en `docs/paper-notes.md`)
- [x] `PLAN.md`, `LOG.md` y estructura de carpetas
- [ ] `pyproject.toml` (uv), `.gitignore`, venv y dependencias
- [ ] Modelos: `qwen2.5:7b-instruct`, `qwen2.5:3b-instruct`, `bge-m3`, `nomic-embed-text`, `llama3.2:1b`
- [ ] Smoke test: chat y embeddings vía la API de Ollama, más structured output con schema
- [ ] Primer commit
- [ ] **Alexis:** confirmar `docs/paper-notes.md` contra el paper

## Fase 1 — Esqueleto
- `app/config.py`: settings desde el entorno (`LLM_MODEL`, `EMBED_MODEL`, `OLLAMA_URL`, `DB_PATH`, `STM_TURNS=6`, `TOP_K=8`, `CLUSTER_THRESHOLD=0.80`, `PROMOTE_MIN_ACCESS`, timeouts y seed)
- `app/llm.py`: `OllamaClient.chat_json(module, messages, schema, fallback, timeout)`. Usa `format=<schema>`, valida con Pydantic, hace 1 reintento y luego el fallback. Contadores por módulo (`ok`, `invalid_json`, `timeout`, `fallback`) y latencia
- `app/memory/models.py`: `MemoryEntry`, `Op`, `Turn`, DTOs de la API
- `app/memory/embed.py`: embeddings vía Ollama `/api/embed`, normalizados, serialización a bytes
- `app/memory/store.py`: interfaz `MemoryStore` (ABC) y `SQLiteStore`. Tablas `turns`, `entries`, `dreams` y `snapshots`; búsqueda coseno con numpy que excluye superseded/forgotten; `transaction()`
- `app/main.py`: FastAPI con `GET /health` (Ollama arriba, modelos presentes)
- **✅ Aceptación:** `pytest` pasa los tests de `llm.py` (mock de httpx: válido, inválido, timeout → fallback) y de `store.py`; `/health` lista los modelos

## Fase 2 — Pipeline online
- Prompts: `controller.md`, `selector.md`, `responder.md`, `writer.md`
- `pipeline/`: Controller (si `needs_memory=false` no recupera), Retriever (código: top-k por consulta, dedupe, sube `access_count`), Selector, Responder y Writer, cada uno con el fallback de la spec §3.2
- `pipeline/agent.py`: orquesta todo y arma el `trace` (salida por módulo, ids, fallbacks y ms por etapa)
- Endpoints `/chat`, `/session/new`, `/memory` y `/reset`
- `static/index.html`: chat + "ver traza" + pestañas STM/MTM/LTM; `demo-theme.css` con fallback de tokens; banner
- **✅ Aceptación:** `scripts/demo_conversation.py` (10 turnos) genera entradas MTM razonables (revisión manual en LOG) y la traza muestra la latencia por etapa

## Fase 3 — Dream
- `dream/cluster.py`: grafo con aristas por coseno > umbral o entidades compartidas; componentes conexos; partir los que superen `MAX_CLUSTER_SIZE`; incluir las LTM vecinas como contexto (para detectar UPDATE contra LTM)
- `dream/consolidator.py`: prompt `consolidator.md` → `list[Op]`
- `dream/validator.py` (**puro**): ids existentes y activos; MERGE con ≥2 ids del mismo tier; `new_text` obligatorio en MERGE/UPDATE/PROMOTE; PROMOTE solo desde MTM con ≥2 turnos de origen o `access_count ≥ umbral`; FORGET nunca sobre LTM; conflictos entre ops del mismo Dream (un id tocado dos veces → se rechaza la segunda)
- `dream/applier.py`: transacción, snapshot antes y después, soft-delete y `superseded_by`; devuelve el diff de §3.4
- `/dream`, `/dreams` y panel de diff en la UI (verde / rojo tachado / amarillo con originales colapsados)
- **✅ Aceptación:** tests del validator (ids inexistentes, FORGET sobre LTM, MERGE de un id, etc.) y del applier; un segundo Dream seguido produce ≤ 20 % de las ops del primero

## Fase 4 — Benchmark
- `app/pipeline/strategies.py`: A (solo STM), B (RAG ingenuo sobre turnos crudos), C (pipeline), D (C + Dream al cerrar cada sesión). **Mismo código que la app**
- `bench/sessions/01..08.json`: usuario ficticio. Incluye paráfrasis (MERGE), cambios (UPDATE: mudanza, trabajo, preferencia invertida), ruido (FORGET) y mezcla ES/EN. **Las revisa Alexis**
- `bench/questions.json`: 28 preguntas (6 single, 6 updated, 6 multi, 5 temporal, 5 abstention) con `expected_keywords`, `forbidden_keywords` y categoría
- `bench/scoring.py` (puro, con tests): normalización sin acentos y detección de abstenciones
- `bench/run.py --condition {A,B,C,D,all} --model ... --runs 3`: `results/<run_id>.jsonl`, `summary.csv` y tabla markdown (exactitud total y por categoría, p50/p95 total y por etapa, llamadas al LLM por turno, tamaño de memoria, fallbacks y ops rechazadas)
- Orden: smoke con 1.5B → corrida completa con 7B (de noche) → 3B
- Revisión manual de los jsonl: dónde se equivoca el scoring
- **✅ Aceptación:** `summary.csv` con 4 condiciones × 3 corridas para al menos un modelo

## Fase 5 — Publicación
- `Dockerfile` (app, puerto 8080, SQLite efímero en `/tmp`) y `Dockerfile.ollama` (modelo horneado), más `docker-compose.yml` para correr local
- Medir con `--cpus=2 --memory=4g` → elegir el modelo del demo
- `DEMO_INFO` bilingüe + `demo-panel.js`; manejo elegante del teardown
- `README.md` **portable** (sin mencionar alexisalulema.com ni la nube): qué es, arquitectura, prerrequisitos, build, run, env, uso, resultados, limitaciones y cita
- Hand-off manifest (`shareable: false`, sin secretos) y tabla de glosario para el post
- **Alexis:** política de open source, publicar la imagen en GHCR como Public, post EN/ES y aviso a los autores

## Riesgos y cómo los medimos
| Riesgo | Señal |
|---|---|
| JSON inválido en modelos pequeños | contador `invalid_json` por módulo y modelo |
| Writer infla o se queda corto | entradas MTM por turno |
| Falso MERGE / olvido de algo importante | revisión manual de diffs + preguntas `single_fact` en D vs C |
| UPDATE mal resuelto | `forbidden_keywords` en `updated_fact` |
| C/D más lentas que B | p50/p95 por etapa (puede ser *el hallazgo*) |
| Clusters grandes | tamaño de cluster vs. ops rechazadas |
| Responder local vs. GPT-4o-mini del paper | limitación declarada en el post |
