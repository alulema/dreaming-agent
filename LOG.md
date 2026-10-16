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
