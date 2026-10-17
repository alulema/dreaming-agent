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

## 2026-10-16 20:30 — Fase 2: primera conversación de 10 turnos (qwen2.5:7b, prompts v1)

`scripts/demo_conversation.py`: usuaria ficticia "Lucía" (ES/EN, una paráfrasis, una mudanza con cambio de trabajo, small talk, dos preguntas de memoria).

- **[RESULTADO]** El pipeline funciona de punta a punta. **37 llamadas JSON: 37 ok, 0 inválidas, 0 fallbacks** (con schema). Responde bien las preguntas de memoria: "Se llama Toby" y "Vives en Madrid y trabajas como analista de datos" (UPDATE resuelto en línea gracias a la fecha que el Selector ve en cada candidato).
- **[HALLAZGO] Latencia (M1 Pro, 7B, en caliente):** respuesta ~7–8 s con memoria y ~3 s sin ella; ~12–14 s en total con el Writer. Por etapa: controller ~2 s, selector ~3 s, responder ~2 s, **writer ~4–6 s (la etapa más cara)**. El primer turno tardó 20 s por la carga en frío del LLM (9.9 s) y de bge-m3 (4.9 s).
  → El Writer corre *después* de la respuesta; pasarlo a segundo plano bajaría la latencia percibida a la de la respuesta (anotado para la Fase 5).
- **[FALLO] Controller:** `needs_memory=true` en **8/10 turnos**, incluidos los que solo cuentan algo nuevo ("Me llamo Lucía…", "Tengo un perro…"). El 7B ignora esa regla del prompt, y cada uno de esos turnos paga ~3–4 s extra de retriever+selector.
- **[FALLO] Selector:** inyecta meta-comentarios inventados: "The name and age of the dog have **not changed**", "This preference has not changed". **Lo provocó nuestro prompt** ("when facts changed, say it changed"): el modelo lo aplica siempre. También conservó hechos irrelevantes (la enfermería para una pregunta sobre té, el café para la mudanza) y en un turno devolvió un índice fuera de rango (lo reparó el código).
- **[FALLO] Responder:** "I **remember** you mentioning that you have a dog named Toby" en el mismo turno en que el usuario lo contaba por primera vez: finge recordar. Lo induce "who remembers past conversations" del system prompt.
- **[FALLO] Writer — eco de recuerdos:** en las *preguntas* ("¿Cómo se llama mi perro?", "¿Dónde vivo ahora?") volvió a escribir los hechos que el asistente acababa de recitar → duplicados (Toby ×3, Madrid ×2, analista ×2, incluido "analyst **of data**"). El prompt le pide ignorar la respuesta del asistente; el 7B no lo hace. **La memoria se infla consigo misma.**
- **[FALLO] Writer — omisiones y temas:** no guardó el nombre (Lucía), la raza (beagle, en el turno 2) ni la edad de Toby (5). Asignó temas erróneos: `user.sports` a la enfermería, `other` al beagle y al té.
- **[HALLAZGO] Los `topics` funcionan para lo que se crearon:** "lives in Quito" y "moved to Madrid" comparten `user.location`; "nurse" y "data analyst" comparten `user.job`. El Dream podrá ver esas contradicciones aunque el coseno no las una. La MTM final (10 entradas) es un buen caso de prueba para la Fase 3: 3 MERGE posibles (Toby) y 2 UPDATE (ciudad, trabajo).

## 2026-10-16 21:30 — Fase 2: cuatro iteraciones de prompts y una compuerta del Writer

Misma conversación de 10 turnos, qwen2.5:7b. Resumen:

| Iteración | Cambio | Entradas MTM | Entradas basura | Llamadas LLM | Tiempo medio por turno |
|---|---|---|---|---|---|
| v1 | prompts iniciales; el Writer ve mensaje + respuesta | 10 | ~4 (duplicados "eco") | 37 | ~12 s |
| v2 | few-shot en el Controller; Selector sin "say it changed"; Responder "no finjas recordar"; el Writer ve el mensaje previo del asistente | 15 | ~6 | 37 | ~13 s |
| v3 | el Writer ve **solo** el mensaje del usuario + dedup al escribir (≥0.95 → refuerzo) | 12 | ~4 | 37 | ~12 s |
| v4 | ejemplos neutros en el Writer + `has_new_facts` + filtro de metadatos + **compuerta `shares_new_info` en el Controller** | **9** | **0** | **26** | **~7 s** |

- **[HALLAZGO] ⭐ Fuga de los ejemplos del prompt.** Aislando el Writer, el 7B respondió a "Gracias, eso es todo por hoy." con `"The user moved to Madrid."`, y a "¿Te acuerdas de cómo se llama mi perro?" con `"The user's dog is named Toby."`. **Esos textos eran los ejemplos del propio prompt.** Cuando no hay nada que extraer, el modelo copia el ejemplo.
  - Mi usuaria de prueba usaba los mismos nombres que los ejemplos (Toby, Madrid), así que la fuga **se hacía pasar por un "eco" de la conversación**. El diagnóstico "eco" de v1/v2 fue **al menos en parte equivocado**: algunas de esas entradas eran copia literal del ejemplo, no de la respuesta.
  - Lecciones: (1) los ejemplos de los prompts deben usar entidades que **nunca** aparezcan en los datos de evaluación; (2) incluir ejemplos explícitos de salida vacía. → En la Fase 4: un test que verifique que ninguna entidad de los ejemplos de los prompts aparece en `bench/`.
- **[HALLAZGO] Ajustar el prompt del Writer no bastó:** con ejemplos neutros dejó de inventar, pero ante preguntas personales seguía escribiendo meta-entradas ("The user asked about their current location and job", "The user's dog's name is not stated") aun con un campo `has_new_facts` previo (2/2 preguntas → `true`).
  - **[DECISIÓN] Compuerta en el Controller:** nuevo campo `shares_new_info`; si es `false`, el Writer **no se ejecuta**. Separar "¿esto cuenta algo?" (clasificar intención, lo que ya hace el Controller) de "extrae los hechos" funcionó donde el prompt del Writer fallaba: 10/11 bien en aislamiento (falló una preferencia; se añadió un ejemplo de gustos) y 10/10 en la corrida completa.
  - Efecto colateral bueno: el mismo prompt reescrito hizo que `needs_memory` dejara de dispararse con los enunciados (de 8/10 a 2/10, justo las dos preguntas). **Llamadas por turno: 3.7 → 2.6; latencia media: ~12 s → ~7 s.**
  - Divergencia frente al paper/spec: el Controller decide dos cosas (recuperar y escribir). Queda como defensa en profundidad: `has_new_facts` + regex de metadatos en el Writer.
- **[DECISIÓN] Dedup al escribir (≥0.95 con una MTM activa → no se duplica; se añade el turno a `source_turn_ids`).** Es fiel al paper (Alg. 2: el Writer fusiona o reescribe entradas repetitivas) y aporta la evidencia "visto en ≥2 turnos" que necesita PROMOTE. Las paráfrasis por debajo de 0.95 siguen siendo trabajo del Dream.
- **[DECISIÓN]** El Selector recibe candidatos numerados `[1]..[k]` con fecha y devuelve índices (no uuids). Con las fechas visibles resolvió bien "¿Dónde vivo ahora?" → Madrid, sin necesidad de Dream.
- **[FALLO] pendiente** (no bloquea):
  - "Moved to Madrid **for a new job as a data analyst**" quedó como una sola entrada con dos temas, a pesar del ejemplo multi-hecho. El Dream tendrá que manejar una entrada que actualiza a dos (Quito→Madrid y enfermera→analista).
  - El Controller a veces emite `queries` con `needs_memory=false` (incoherente, sin efecto).
  - Español del 7B: "me acuero" (v1) y "hace calor en Quito en julio" (Quito ronda 10–20 °C). No es memoria, pero el lector lo notará.
- **[RESULTADO] ✅ Criterio de la Fase 2 (parte pipeline):** una conversación de 10 turnos genera entradas MTM razonables (9: nombre, ciudad, perro, raza, edad, trabajo, turnos, preferencia, mudanza+trabajo) y la traza tiene la latencia por etapa. Prompts finales: controller/selector/responder/writer, con su hash en cada traza.

## 2026-10-16 22:15 — Fase 2: API, UI, tests · cierre de fase

- **[DECISIÓN] API:** `/chat`, `/session/new`, `/memory`, `/reset`, `/health`, `/models` y `/model` (selector de modelo en caliente). Un `asyncio.Lock` serializa los turnos: es un demo de un solo usuario y así las escrituras de memoria quedan ordenadas. Un timeout de Ollama → 504 con un mensaje legible.
- **[DECISIÓN] UI** (`app/static/index.html`, vanilla, sin build): chat + "ver traza" (Controller, candidatos con su score y los que conservó el Selector en verde, contexto comprimido, Writer, barras de latencia por etapa, versión de cada prompt) + pestañas STM/MTM/LTM con contadores; las entradas nuevas se resaltan. Usa los tokens de `demo-theme.css` con fallbacks propios; textos ES/EN según el navegador. Botón Dream deshabilitado hasta la Fase 3.
- **[DESAFÍO]** Las barras de latencia salían como `[object HTMLSpanElement]`: el helper `el()` hacía `kids.flat()`, que aplana un solo nivel, y `stages.map(...)` devolvía arrays de arrays. Solución: `flat(Infinity)`. Se detectó probando la UI en Chrome, no en los tests de API.
- **[RESULTADO] Prueba en navegador** (7B, BD nueva): "Hola, soy Marco y tengo una gata llamada Pelusa" → 2 entradas MTM (nombre y gata), 3.2 s. "¿Cómo se llama mi mascota?" → "Tu mascota se llama Pelusa, una gata." (6.8 s, Writer omitido). "¿Te acuerdas de cómo me llamo?" → correcto (7.2 s: controller 2.9 s, retriever 0.15 s, selector 1.8 s, responder 2.3 s).
- **[FALLO]** El 7B volvió a escribir "me **acuero**" (por "me acuerdo"): segunda vez en el día. Parece un error sistemático de qwen2.5:7b en español con ese verbo. Anotar para el post.
- **[HALLAZGO] UX:** al recargar la página se pierde el historial visible del chat, pero la STM sigue en el servidor (la sesión vive en `sessionStorage`). Aceptable para el MVP.
- **[RESULTADO]** 42 tests en verde (+18: render de prompts, reparaciones del Controller, mapeo/fallback del Selector, `clean` del Writer, merge del Retriever, Agent con un LLM falso con guion para turnos de enunciado/pregunta/dedup/fallback/ventana STM, endpoints). `ruff` limpio.
- **[RESULTADO] ✅ Fase 2 completa.** Criterio: la conversación de 10 turnos genera entradas MTM razonables (9 de 9 reales) y la traza muestra la latencia por etapa.
