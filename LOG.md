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

## 2026-10-17 00:30 — Fase 3: el Dream (diseño y primeras corridas)

- **[DECISIÓN] Semántica precisa de las ops** (la spec dejaba dos huecos):
  - **UPDATE**: `target_ids` = *todas* las entradas sobre el atributo (viejas y nuevas); `new_text` = valor vigente. Todas quedan `superseded_by` la nueva. Debe incluir ≥1 MTM: reescribir conocimiento consolidado exige evidencia nueva de la conversación.
  - **MERGE/UPDATE → LTM directo** si el resultado cumple la evidencia de PROMOTE (≥2 turnos o ≥3 accesos; `MERGE_PROMOTES=true`). Con "un id por op por Dream", un MERGE no podía ir seguido de su PROMOTE en el mismo Dream, y eso garantizaba trabajo pendiente para el siguiente (anti-idempotente). El paper describe así la consolidación: lo que se consolida *es* LTM.
  - **FORGET** rechazado también sobre entradas "reforzadas" (dichas en ≥2 turnos).
  - El modelo ve etiquetas `M1..`/`L1..` (no uuids); el código las mapea. Una etiqueta inventada llega como id desconocido y el Validator la rechaza.
- **[DECISIÓN] Clustering:** aristas por coseno ≥ **0.75** (no 0.80, por el hallazgo de la Fase 1) **o** tema compartido (excepto `other`) **o** entidad compartida (normalizada). Componentes conexos con un tope de 8; los sueltos van en lotes para poder PROMOTE/FORGET; cada cluster lleva hasta 5 LTM relacionadas como contexto.
- Escenario `scripts/dream_demo.py`: 3 sesiones, 14 turnos, 16 entradas MTM con paráfrasis (Toby ×4), cambios (Quito→Madrid, enfermera→analista, té→café) y ruido (sándwich). 7B. Cinco corridas con correcciones entre ellas:

### Corrida 1 — prompts v1, Validator básico
- ✅ UPDATE gigante (6 entradas de trabajo+ciudad → "data analyst living in Madrid… (previously nurse in Quito)") y MERGE del perro.
- **[FALLO] ⭐ Otra vez la fuga del ejemplo:** `UPDATE ["…coffee…", "M4"] → "The user lives in Bilbao (previously lived in Lima)"`, **copia literal del ejemplo del prompt del Consolidator**. Se rechazó *por casualidad*: ese cluster no tenía M4. **Con ≥4 entradas habría sido válida y habría reemplazado un hecho real por uno inventado.**
- **[FALLO] Falso FORGET:** olvidó "enjoys running on Sundays" (un hobby) y conservó "had a chicken sandwich for lunch".
- **[FALLO]** El UPDATE dejó activa "lives in Quito", contradiciendo la LTM nueva.

### Corrida 2 — Validator con grounding + prompt v2
- **[DECISIÓN] Grounding determinista:** nombres propios (palabras en mayúscula que no inician frase) y números de `new_text` deben aparecer en los *targets*. Atrapa la fuga (Bilbao, Lima), detalles inventados ("moved in 2025") y tolera acentos (Málaga/Malaga). Etiquetas del ejemplo cambiadas a `A1..A4`, que nunca coinciden con las reales; FORGET y PROMOTE con criterios explícitos.
- ✅ Sin fugas ni falsos FORGET.
- **[FALLO] Bug mío:** las entradas creadas por el Dream tenían `created_at = ahora`. En el Dream 2 la LTM parecía *más nueva* que hechos aprendidos después, y la regla "lo más nuevo debe ser MTM" rechazó UPDATE legítimos. **Corregido:** el resultado hereda el `created_at` del target más reciente (la hora del *hecho*, no la del sueño); `updated_at` sí es ahora.
- **[FALLO]** La respuesta después del Dream **empeoró**: "trabajas como enfermera", porque las MTM obsoletas de enfermera seguían activas y el Selector las prefirió.

### Corrida 3 — fix de created_at, regla "≥1 MTM", marca `promotable`
- **[FALLO] ⭐ El Dream borró un hecho vigente.** El UPDATE de *ubicación* del Dream 2 tomó como target la LTM "data analyst (previously nurse)", que tenía temas de ubicación y de trabajo heredados de la entrada compuesta "moved to Madrid **for a new job as a data analyst**", y la reemplazó por "lives in Madrid, Malasaña (previously Quito)". **El trabajo actual desapareció de la memoria activa** y el agente respondió "trabajas como enfermera en un hospital". Es el riesgo "el Consolidator borra algo importante" de la spec, materializado a través de una entrada con dos temas.
- **[DECISIÓN] Preservación del valor vigente:** en un UPDATE, para cada tema, las entidades del target más reciente con ese tema deben sobrevivir en `new_text`. En un MERGE, ningún nombre ni número de los targets puede perderse. Probado sobre los casos reales: atrapa el UPDATE de ubicación del Dream 2 (pierde "data analyst"), el UPDATE de trabajo del Dream 1 (pierde "Madrid") y un MERGE que perdía la edad "5".
- **[DECISIÓN] Avisos "possibly stale":** tras un UPDATE, una MTM activa, más antigua y del mismo cluster que comparte tema se reporta en el diff, **pero no se aplica**: compartir tema no prueba contradicción ("tiene una hermana" vs "se casó", ambos `user.family`). En la corrida 2 marcó exactamente las 3 obsoletas (Quito, enfermera, turnos de noche).
- **[HALLAZGO]** El modelo ignora el número de turnos que se le muestra: dice "stated in 2 turns" en entradas de 1 turno para justificar un PROMOTE. Ahora el código calcula y escribe `promotable` en cada línea, y aun así el 7B sigue proponiendo PROMOTE de "Lucía" (que no lo es), diciendo "marked promotable". Lo frena el Validator.

### Corrida 4 — con las reglas de preservación
- ✅ Dream 1: UPDATE limpio de ubicación ("Madrid, Malasaña (previously Quito)") y MERGE del perro. El UPDATE de trabajo que perdía "Madrid" fue **rechazado** (la memoria queda sin consolidar pero sin pérdida).
- ✅ Pregunta después de los Dreams: "Vives en Malasaña, Madrid, trabajas como analista de datos" y, sobre la bebida, **se abstiene** en lugar de inventar.
- **[FALLO] Falso rechazo:** "UPDATE drops current value: ['dog']", porque el texto decía "beagle". Las entidades genéricas hacen la regla demasiado estricta. El Dream siguiente aplicó el mismo cambio como MERGE.
- **[HALLAZGO] Café/té:** las dos entradas *sí* compartían la entidad `coffee` y cayeron en el mismo cluster; el 7B propuso un UPDATE con **un solo target** y además **invirtió la cronología** ("drinks green tea… previously coffee", cuando lo último fue el café). Rechazado por la regla de ≥2 targets. El razonamiento temporal es un punto débil claro del 7B.
- **[HALLAZGO] Idempotencia:** ops aplicadas 2 → 1 (50%; la spec pide ≤20%). Propuestas 6 → 6: el modelo **repite cada sueño las mismas propuestas inválidas** (PROMOTE de Lucía, UPDATE de un solo target con `new_text=null` para decir "no cambies nada"). Con números tan pequeños, una sola op legítima pendiente rompe el umbral.

### Corrida 5 — 3 Dreams seguidos (`--dreams 3`)
- **[RESULTADO] ✅ Idempotencia:** ops aplicadas por Dream **[3, 0, 0]** (Dream 2 / Dream 1 = 0 % ≤ 20 %); propuestas [8, 4, 4]. La memoria converge en un Dream: 16 MTM → 10 MTM + 3 LTM, 157 → 151 tokens.
- **[HALLAZGO] No hay determinismo completo:** con `temperature=0` y `seed=42`, la corrida 4 dio [2, 1] y la 5 dio [3, 0, 0]. Algunas salidas (y por tanto la memoria escrita en las sesiones) cambian entre corridas. → En la Fase 4 son obligatorias las 3 corridas por condición, reportando la varianza.
- **[HALLAZGO] Punto fijo con conflictos sin resolver:** las mismas 4 propuestas inválidas se repiten en cada Dream (el UPDATE del trabajo que pierde "Madrid", el café con un solo target y la cronología invertida, el PROMOTE de "Lucía" sin evidencia, un UPDATE "no-op"). La memoria es estable, pero "trabaja de enfermera" sigue activa junto a "ahora es analista": el Validator evita el daño, no lo arregla. Las preguntas sobre el trabajo dependen entonces de que el Selector use las fechas (y en esta corrida lo hizo bien).
- Coste del Dream con 7B: 30–36 s y 4 llamadas para 16 entradas; ~10–13 s cuando no hay nada nuevo.

## 2026-10-17 01:15 — Fase 3: API y UI del Dream · cierre de fase

- **[DECISIÓN]** `POST /dream` (bajo el mismo lock que `/chat`) y `GET /dreams` (el más reciente primero). La UI tiene el botón 🌙 Dream con un contador "soñando… N s", una pestaña **Dreams** con el diff de cada sueño (UPDATE: viejas tachadas en rojo + nueva en verde; MERGE en amarillo con los originales colapsados; PROMOTE en verde; FORGET tachado), los avisos "posiblemente obsoleto", las rechazadas colapsadas con su error y las stats antes→después.
- **[DESAFÍO]** El diff mostraba `[object HTMLDivElement],…`: `Element.append(array)` convierte el array a string. Solución: `append(...array)`. Es el mismo tipo de bug que el de las barras de latencia de la Fase 2. Otra vez lo detectó la prueba visual en Chrome, no los tests.
- **[RESULTADO] Prueba en navegador** (usuaria "Ana": Valencia → Sevilla/Triana, gato Simba de 3 años, profesora de música): el Dream en 18.6 s aplicó UPDATE → LTM "lives in Triana, Sevilla (previously lived in Valencia)" y MERGE → LTM del gato, y avisó de que "The user's name is Ana **and they live in Valencia**" quedó posiblemente obsoleta. Esa entrada compuesta no podía entrar en el UPDATE sin perder el nombre, y la regla de preservación lo impidió. Las **entradas compuestas del Writer** (dos hechos en una frase) son la causa raíz de varios fallos del Dream.
- **[FALLO] menor** El Writer escribió "The user's cat Simba **has been named**" (frase rara), y la usuaria de prueba se llamaba "Ana", nombre que aparece en un ejemplo del prompt del Controller. Para el bench (Fase 4) hay que validar automáticamente que los datos no compartan entidades con los ejemplos de los prompts.
- **[RESULTADO]** 65 tests en verde (+23 del Dream: cada rechazo que pide la spec —ids inexistentes, FORGET sobre LTM, MERGE de un id— más fuera de scope, inactivos, duplicados, conflictos, evidencia de PROMOTE, FORGET de reforzadas, grounding/fuga del ejemplo, pérdida de valor vigente, pérdida de detalles en MERGE, avisos de obsolescencia, clustering por coseno/tema/entidad, partición y contexto LTM, etiquetas/promotable, applier completo con `created_at` del hecho, reglas de tier, Dream de punta a punta con un modelo con guion, idempotencia y **rollback** si falla el guardado). `ruff` limpio.
- **[RESULTADO] ✅ Fase 3 completa.** Criterios: (1) los tests del Validator rechazan las ops inválidas; (2) un segundo Dream seguido produce ≤ 20 % de las ops aplicadas del primero (corrida 5: 0 %; la corrida 4 dio 50 % → el criterio se cumple, pero no en todas las corridas).

## 2026-10-17 02:00 — Fase 4: datos del benchmark (borrador para revisión de Alexis)

- **[DECISIÓN] Persona nueva ("Mateo")**, distinta de la usada para ajustar los prompts (Lucía/Quito/Toby, Marco, Ana/Valencia). Reusarla habría medido el sobreajuste de los prompts, no el sistema. Hay un test que lo garantiza.
- **[DECISIÓN] Ningún nombre de los ejemplos de los prompts en los datos** (Ana, Lima, Perú, Nube, Iker, Bilbao): test automático que extrae los nombres de las líneas de ejemplo de cada prompt. Su primera versión dio un falso positivo ("Cómo" de "¿Cómo se llama…?").
- **[DECISIÓN] Formato del scoring:** `expected` como **grupos de alternativas** (no una lista plana de keywords) porque el agente responde en ES o EN ("maní"/"peanut", "marzo"/"March"). Las keywords de ≤3 caracteres van con límites de palabra ("si", "ux", "12"); las más largas, como substring/raíz ("escal" → escalada/escalar).
- **[HALLAZGO] Diseñar `forbidden` es delicado:** la LTM consolidada guarda "(previously lived in X)" y el Responder tiende a decir "vives en Montevideo (antes en Valparaíso)", lo que es *correcto* pero contiene el valor viejo. Mitigación: las preguntas `updated_fact` piden respuesta corta ("responde solo con la ciudad"), y en dieta y temporal el `forbidden` se acota o queda vacío (anotado en cada pregunta). Hay que revisar a mano estos casos en los resultados.
- 8 sesiones / 45 turnos (s3 y s6 en inglés, s5 mezcla idiomas dentro de la misma frase), 28 preguntas (6/6/6/5/5). 28 tests nuevos (integridad de datos + respuestas de referencia del scoring). **93 tests en verde.**
