# LOG — El agente que sueña

Bitácora cronológica e **interna**: decisiones, desafíos superados, hallazgos, bugs, resultados y todo lo que falla.
Es la materia prima del post. Formato de cada entrada: `## YYYY-MM-DD HH:MM — título`, con etiquetas
**[DECISIÓN]**, **[DESAFÍO]** (problema y cómo se resolvió), **[HALLAZGO]**, **[FALLO]** y **[RESULTADO]**.

---

## 2026-10-16 18:50 — Fase 0: arranque

- **[DECISIÓN]** Raíz del proyecto: `dreaming-agent/` (la spec la llama `agente-que-suena/`). Un único `LOG.md` sirve de log de la spec y de devlog del contrato de demos.
- **[DECISIÓN]** Separar el modelo del benchmark del modelo del demo público. Entorno local: M1 Pro con 16 GB, Ollama 0.40.2. El pod del demo tiene 2 vCPU / 4 GiB sin GPU (rag-blogposts necesitó Qwen 0.5B para quedarse en ~6–7 s al primer token), y este pipeline hace 4–5 llamadas al LLM por turno → 7B en ese pod es inviable. Benchmark: 7B y 3B en el Mac. Demo: 1.5B/3B según la latencia medida en la Fase 5.
- **[DECISIÓN]** JSON estructurado con `format=<JSON Schema>` de Ollama (generado desde Pydantic) en vez de `format: "json"`. Se medirá la tasa de JSON inválido con y sin schema.
- **[HALLAZGO]** Lectura (asistida) del paper, detalle en `docs/paper-notes.md`; pendiente de confirmar por Alexis:
  - Sus SLM son **más pequeños** que el default de la spec: Llama-3.2-1B por defecto y Qwen2.5-1.5B. Usar 1–1.5B en el demo público no es un recorte: **es fiel al paper**. Por eso se añade `llama3.2:1b` al benchmark.
  - El Selector del paper está **afinado con LoRA**; el nuestro no (el fine-tuning es un no-objetivo).
  - La **respuesta final la genera GPT-4o-mini**, y la consolidación un "large-context LLM". En el MVP todo es local, incluido el Responder y el Consolidator: es la divergencia más grande y debe ir explícita en el post.
  - La LTM del paper es un **grafo** (Entity/Concept, aristas IsA/HasProperty/…), global y des-identificada. La del MVP es una lista plana.
  - La consolidación es automática cada 10–15 turnos y lleva decaimiento de confianza. Nosotros usamos un botón manual (y, en el bench, Dream al final de cada sesión).
  - Las secciones §3.4 y §4.1.4 del paper son inconsistentes sobre el top-k de la etapa 1 (2K/n frente a top-10).
  - No se encontró código publicado.

## 2026-10-16 19:05 — Fase 0: entorno y smoke test de Ollama

- **[DECISIÓN]** `uv` + `pyproject.toml` (Python 3.11). Deps: fastapi, uvicorn, httpx, pydantic(-settings), numpy. Dev: pytest, pytest-asyncio, respx, ruff.
- **[DESAFÍO]** `uv sync` superó el timeout de 2 min porque competía por ancho de banda con la descarga de los modelos (~8 GB). Se dejó terminar en segundo plano; sin errores. Lección: descargar los modelos antes que nada, o en paralelo pero sin bloquear.
- **[HALLAZGO]** Primer smoke test (`scripts/smoke_ollama.py`, prompt tipo Controller, `qwen2.5:1.5b-instruct`, temp 0, seed 42):
  - `format: "json"` → JSON sintácticamente válido pero **schema inválido**: `"tiers": "STM"` (string en vez de lista). Pydantic lo rechaza.
  - `format: <JSON Schema>` → válido a la primera, y además **más rápido** (786 ms vs 3427 ms; la primera llamada incluye la carga del modelo, así que la comparación de latencia no es justa todavía).
  - Conclusión provisional: "JSON válido" ≠ "JSON que cumple el contrato". El structured output con schema debería bajar mucho la tasa de fallbacks. **Medirlo en el bench** (flag `LLM_FORMAT=schema|json`).
  - Detalle curioso: el modelo reescribió la consulta como "¿Cómo se llama **tu** perro?" (cambió la persona gramatical). Para la búsqueda por embeddings da casi igual, pero es una señal de que el Controller puede distorsionar las consultas.

## 2026-10-16 19:20 — Fase 0: smoke test con todos los modelos · cierre de fase

Modelos descargados: `qwen2.5:7b-instruct` (4.7 GB), `qwen2.5:3b-instruct` (1.9 GB), `qwen2.5:1.5b-instruct`, `llama3.2:1b` (1.3 GB), `bge-m3` (1.2 GB), `nomic-embed-text` (274 MB).

Prompt tipo Controller ("¿Te acuerdas cómo se llama mi perro?"), temperatura 0, seed 42, en frío y en caliente:

| Modelo | json | schema | Latencia en caliente | Salida |
|---|---|---|---|---|
| qwen2.5:7b | ✅ | ✅ | ~1.1–1.3 s | `needs_memory=true`, **`queries=[]`**, tiers `[LTM]` |
| qwen2.5:3b | ✅ | ✅ | ~0.55 s | `needs_memory=true`, **`queries=[]`**, tiers `[STM, LTM]` |
| qwen2.5:1.5b | ❌ (`tiers` como string) | ✅ | ~0.8 s | correcto |
| llama3.2:1b | ✅ | ✅ | ~0.7 s | **`needs_memory=false`** pero emite 3 queries (una inútil: "¿Qué es un perro?") |

- **[FALLO] / corrección:** en la entrada anterior atribuí a `format=schema` una mejora de latencia (786 vs 3427 ms). **Era falso**: la primera llamada pagaba la carga del modelo. En caliente, json y schema tardan lo mismo. La ventaja del schema es solo la validez.
- **[HALLAZGO]** Un JSON válido según el schema puede ser **semánticamente inútil**: 7B y 3B devolvieron `queries=[]` con `needs_memory=true`. **El schema no reemplaza las reglas de negocio.** → Regla del Controller en la Fase 2: si `needs_memory=true` y `queries` viene vacío, usar el mensaje del usuario como query (contar el evento como "repair", no como fallback).
- **[HALLAZGO]** llama3.2:1b es **internamente incoherente** (dice que no necesita memoria y aun así genera consultas). Anticipa el riesgo de los modelos del tamaño del paper sin fine-tuning; recordar que el Selector del paper sí está afinado con LoRA.
- **[HALLAZGO] Embeddings multilingües.** Coseno entre "Mi perro se llama Toby" y "My dog is called Toby":
  - `bge-m3` (1024-d): **0.974**
  - `nomic-embed-text` (768-d): **0.589**

  **nomic-embed-text no sirve** para sesiones que mezclan ES/EN: queda descartado como alternativa. Para el demo público (con poca RAM) hay que buscar otro embedding multilingüe más liviano (candidatos: `paraphrase-multilingual-MiniLM-L12-v2`, ya usado en rag-blogposts, o algún embedding multilingüe pequeño de Ollama). Se decide en la Fase 5 midiendo.
- **[RESULTADO]** ✅ Fase 0 completa. Ollama 0.40.2 sirve chat con JSON Schema y embeddings; el entorno Python queda listo.

## 2026-10-16 19:45 — Fase 1: esqueleto (llm.py, store, /health)

- **[DECISIÓN]** `llm.py` → `OllamaClient.chat_json(module, messages, schema, fallback)`:
  - **Nunca lanza** si el modelo se porta mal: devuelve el fallback con la causa.
  - Reintenta **solo** el JSON inválido (`json_retries=1`). Un timeout **no** se reintenta, porque duplicaría una latencia que ya se pasó del presupuesto.
  - Contadores por módulo: `ok / invalid_json / timeout / http_error / fallback` y ms. Son los números del post.
  - `chat_text` (Responder) sí propaga errores: no existe un fallback determinista sensato para una respuesta.
- **[DECISIÓN]** `MemoryEntry` añade `status: active|superseded|forgotten` a lo que pide la spec. `superseded_by` dice *quién* reemplaza; `status` cubre el soft-delete de FORGET, que no tiene sucesor.
- **[DECISIÓN]** `SQLiteStore`: una conexión con `RLock` y transacciones anidables (solo la más externa hace commit o rollback, cubierto por test). Búsqueda coseno por fuerza bruta en numpy. Las tablas `turns` guardan embedding opcional para el baseline B (RAG ingenuo).
- **[DESAFÍO]** Los tests de `/health` fallaban con `respx`: el `TestClient` de Starlette usa su propio transporte ASGI, así que una ruta de "pass-through" para `testserver` nunca se llamaba y `assert_all_called` fallaba. Solución: mockear solo la ruta de Ollama; el TestClient no pasa por respx.
- **[DESAFÍO]** `scripts/*.py` no podía importar `app` (el proyecto no se instala como paquete). Solución: ejecutarlos como módulo desde la raíz: `uv run python -m scripts.<nombre>`.
- **[RESULTADO]** 24 tests en verde (`llm`: válido, retry→ok, inválido→fallback, timeout sin retry, http error, schema enviado como `format`; `store`: CRUD, coseno con filtros de tier/estado, `touch`, ventana STM, rollback anidado, snapshot/stats/reset, persistencia en disco; `/health`: ok, degradado y Ollama caído). `ruff` limpio. `/health` contra Ollama real → `status: ok` con qwen2.5:7b + bge-m3.

### [HALLAZGO] El umbral de clustering 0.80 de la spec dejaría fuera justo los UPDATE

`scripts/probe_similarity.py`, coseno con bge-m3:

| Par | Coseno |
|---|---|
| Paráfrasis ES/ES · ES/EN · forma "the user…" | 0.977 · 0.970 · 0.978 |
| Mismo tema, hecho distinto ("dog named Toby" vs "dog is a beagle") | **0.810** |
| **Actualización** de ciudad ("lives in Quito" vs "moved to Madrid") | **0.742** |
| **Actualización** de trabajo ("nurse" vs "now data analyst") | **0.757** |
| No relacionados (EN/EN · ES/EN) | 0.576 · 0.484 |

- Con el umbral de 0.80 se agruparían las paráfrasis (MERGE ✅) y "mismo tema, otro hecho" (riesgo de **falso MERGE**), pero **no** las contradicciones (0.74–0.76). El Consolidator nunca vería juntos "vivo en Quito" y "me mudé a Madrid" → **ningún UPDATE**.
- Implicaciones para la Fase 3:
  1. Las aristas por **entidades compartidas** (spec §3.3) no son opcionales: son el único camino para que un UPDATE llegue al Consolidator. Ojo: "Quito" y "Madrid" no comparten entidad; lo compartido es el *atributo* (residencia). Habrá que pedir al Writer entidades/temas normalizados (p. ej. `user.location`), o bajar el umbral.
  2. Probar umbrales de ~0.70 a 0.72: atrapan los UPDATE y dejan fuera lo no relacionado (<0.58), a cambio de clusters más grandes (vigilar `MAX_CLUSTER_SIZE`).
  3. Estas son frases sintéticas; hay que **re-medir con lo que realmente escriba el Writer** en la Fase 2.
- La recuperación cruzada ES→EN funciona: las 4 consultas encuentran el hecho correcto en el top-1, con 0.59–0.71 frente a 0.46–0.51 del segundo. Ojo con cualquier umbral absoluto de recuperación: un acierto real puede puntuar solo ~0.6.
