# El agente que sueña — Especificación del MVP

> Demo #1 de la serie **"Paper → Demo en vivo → Post honesto"** de alexisalulema.com.
> Paper base: **LightMem: Lightweight LLM Agent Memory with Small Language Models** — arXiv:2604.07798 (ACL 2026 main).
> ⚠️ Existe otro paper de 2025 también llamado "LightMem" (arXiv:2510.18866). Citar siempre por ID: **2604.07798**.

---

## 0. Instrucciones para Claude Code

- Lee este documento completo antes de escribir código. Trabaja **por fases** (sección 9) y no avances a la siguiente fase hasta que la anterior pase sus criterios de aceptación.
- **Esto NO es una réplica completa del paper.** Es un MVP inspirado en su arquitectura, con una medición honesta. Ante la duda, elige lo más simple.
- Todo corre **local**: Ollama + modelos abiertos. **Cero APIs externas, cero API keys.**
- Los puntos marcados **[VERIFICAR EN PAPER]** son supuestos míos sobre el diseño original. No los inventes: déjalos configurables y anota en `LOG.md` lo que confirme Alexis al leer el paper.
- Mantén un `LOG.md` con fecha y hora: decisiones, bugs, resultados y **todo lo que falle**. Ese archivo es la materia prima del post.
- Identificadores, código y comentarios en inglés. Los textos de la UI pueden ser bilingües (EN/ES).
- Tests con `pytest` para la lógica determinista (validación de operaciones, aplicación de diffs y scoring).

---

## 1. Objetivo

Construir un agente conversacional con **memoria en tres niveles** y una **fase de consolidación offline ("sueño")** que el usuario dispara con un botón **Dream**, viendo un **diff** de lo que el agente fusionó, actualizó, promovió u olvidó.

Y medir algo concreto: **¿consolidar la memoria mejora las respuestas, la latencia o el tamaño de la memoria con un modelo pequeño local?**

### Pregunta de investigación del post
> "Implementé LightMem con un modelo de ~3–7B en mi PC: ¿la consolidación offline realmente ayuda a un modelo pequeño, y dónde se rompe?"

### No-objetivos (MVP)
- Multiusuario (el paper lo soporta; aquí, un solo usuario).
- Replicar los benchmarks del paper o sus números.
- Fine-tuning de cualquier tipo.
- UI pulida. Basta con que sea funcional y clara.

---

## 2. Stack

| Componente | Elección | Nota |
|---|---|---|
| Lenguaje | Python 3.11+ | |
| API | FastAPI | Igual que los otros demos del sitio |
| LLM | Ollama, `qwen2.5:7b-instruct` por defecto | Configurable; probar también `qwen2.5:3b` y `llama3.2:3b` |
| Embeddings | Ollama, `bge-m3` (multilingüe) | Configurable; alternativa: `nomic-embed-text` |
| Almacenamiento | SQLite + embeddings como BLOB, similitud coseno en numpy | Detrás de una interfaz `MemoryStore`, para cambiar a pgvector después |
| Frontend | Una sola página HTML + JS vanilla servida por FastAPI | Sin build step |
| Contenedor | Dockerfile con modelos descargados en build | Fase 5, para la infraestructura efímera |

Configuración por variables de entorno en `config.py` (`LLM_MODEL`, `EMBED_MODEL`, `OLLAMA_URL`, `DB_PATH`, umbrales).

---

## 3. Arquitectura

```
                 ┌──────────── ONLINE (cada turno) ────────────┐
 user msg ──► Controller ──► Retriever ──► Selector ──► Responder ──► reply
                (plan)        (STM+MTM+LTM)  (verifica/   (genera)
                                             comprime)
                                                         └──► Writer ──► MTM
                 └──────────────────────────────────────────────┘

                 ┌──────────── OFFLINE (botón Dream) ──────────┐
   MTM + LTM ──► Consolidator (propone ops tipadas) ──► Validator (código)
                                                    ──► Applier ──► diff
                 └──────────────────────────────────────────────┘
```

### 3.1 Niveles de memoria

| Nivel | Contenido | Ciclo de vida |
|---|---|---|
| **STM** (corto plazo) | Últimos N turnos de la sesión actual (texto crudo) | Ventana deslizante; N configurable (default 6) |
| **MTM** (mediano plazo) | Resúmenes/hechos extraídos por el Writer en línea | Crece en cada turno; es lo que se consolida |
| **LTM** (largo plazo) | Conocimiento consolidado: hechos estables, preferencias, entidades | Solo cambia en el Dream |

Esquema mínimo de una entrada MTM/LTM:

```python
class MemoryEntry(BaseModel):
    id: str                 # uuid
    tier: Literal["MTM", "LTM"]
    text: str               # el hecho o resumen, en una frase
    kind: Literal["fact", "preference", "event", "summary"]
    entities: list[str]     # nombres propios o temas
    source_turn_ids: list[str]
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed_at: datetime | None = None
    superseded_by: str | None = None   # historial: no se borra, se marca
    embedding: bytes
```

### 3.2 Módulos online

El paper usa tres SLM especializados en línea: **Controller** (intención y planificación de la consulta), **Selector** (verificación y compresión de candidatos) y **Writer** (escritura incremental de memoria). **[VERIFICAR EN PAPER]** prompts exactos, formato de salida y si los tres son el mismo modelo con distinto prompt.

En el MVP, **mismo modelo con tres prompts distintos**, cada uno con **salida JSON validada con Pydantic**:

1. **Controller** → `{needs_memory: bool, queries: list[str], tiers: list["STM"|"MTM"|"LTM"]}`
   - Si `needs_memory=false`, se salta la recuperación (ahorra latencia; medirlo).
2. **Retriever** (código, no LLM) → top-k por coseno sobre los niveles pedidos; k configurable (default 8).
3. **Selector** → recibe candidatos y pregunta; devuelve `{keep_ids: list[str], compressed_context: str}`.
4. **Responder** → genera la respuesta con STM + `compressed_context`.
5. **Writer** → tras responder, extrae 0..n entradas MTM nuevas del turno: `{entries: [{text, kind, entities}]}`.

**Regla de robustez (estilo Agentic Racing):** si un módulo devuelve JSON inválido o tarda más de un timeout, se usa un **fallback determinista** y se registra el evento.
- Controller → `needs_memory=true`, query = mensaje del usuario, todos los niveles.
- Selector → quedarse con los top-3 tal cual.
- Writer → no escribir nada en ese turno.

Contar cuántas veces se activa cada fallback: **es un dato del post**.

### 3.3 Consolidación offline (el "sueño")

**Principio de diseño:** *el modelo propone, el código dispone.* El LLM nunca edita la memoria directamente; emite **operaciones tipadas** que el código valida y aplica.

Operaciones permitidas:

```python
class Op(BaseModel):
    op: Literal["MERGE", "UPDATE", "PROMOTE", "FORGET"]
    target_ids: list[str]        # entradas existentes afectadas
    new_text: str | None = None  # para MERGE / UPDATE / PROMOTE
    reason: str                  # explicación corta (se muestra en el diff)
```

| Op | Significado | Validación en código |
|---|---|---|
| `MERGE` | Fusiona 2+ entradas MTM duplicadas o redundantes en una | ≥2 ids existentes, mismo tier; `new_text` no vacío |
| `UPDATE` | Un hecho más nuevo reemplaza a uno viejo (contradicción) | La vieja queda `superseded_by`; nunca se borra físicamente |
| `PROMOTE` | Un hecho MTM estable pasa a LTM | Debe haberse visto en ≥2 turnos o tener `access_count ≥ umbral` |
| `FORGET` | Marca como olvidada una entrada trivial u obsoleta | Nunca sobre LTM en el MVP; soft-delete |

Algoritmo del Dream:
1. Agrupar entradas MTM por similitud (coseno > umbral, default 0.80) y por entidades compartidas. **No mandar toda la memoria al modelo de una vez.**
2. Por cada grupo, pedir al Consolidator una lista de `Op`.
3. Validar cada op; las inválidas se **rechazan y se registran** (contarlas también).
4. Aplicar en una transacción, guardar un snapshot **antes/después** y generar el diff.
5. **Idempotencia:** correr Dream dos veces seguidas no debería generar casi ninguna op nueva. Medirlo.

**[VERIFICAR EN PAPER]** criterios reales de consolidación de MTM→LTM, frecuencia y si hay decaimiento o olvido.

### 3.4 Diff del Dream

Estructura que devuelve `POST /dream`:

```json
{
  "dream_id": "...",
  "duration_ms": 0,
  "llm_calls": 0,
  "ops_applied": [{"op": "MERGE", "target_ids": [], "new_id": "...", "reason": "..."}],
  "ops_rejected": [{"op": "...", "error": "..."}],
  "stats_before": {"mtm": 0, "ltm": 0, "tokens": 0},
  "stats_after":  {"mtm": 0, "ltm": 0, "tokens": 0}
}
```

La UI lo muestra como un diff legible: verde para entradas nuevas o promovidas, rojo tachado para olvidadas o reemplazadas, amarillo para fusionadas (con las originales colapsadas debajo).

---

## 4. API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/chat` | `{session_id, message}` → `{reply, trace}`; `trace` incluye la salida de cada módulo, los ids recuperados, los fallbacks y la latencia por etapa |
| `POST` | `/session/new` | Inicia sesión (vacía STM, conserva MTM/LTM) |
| `GET` | `/memory` | Estado actual de STM/MTM/LTM |
| `POST` | `/dream` | Ejecuta la consolidación y devuelve el diff |
| `GET` | `/dreams` | Historial de diffs |
| `POST` | `/reset` | Borra toda la memoria (solo en el demo) |
| `GET` | `/health` | Estado de Ollama y modelos |

---

## 5. UI (una página)

- **Izquierda:** chat. Cada respuesta tiene un botón "ver traza" que despliega qué recuperó, qué conservó el Selector y cuánto tardó cada etapa.
- **Derecha:** tres pestañas STM / MTM / LTM con contadores.
- **Arriba:** botón **🌙 Dream**, botón "Nueva sesión" y selector de modelo (si hay varios descargados).
- **Panel de diff** tras cada Dream, con historial.
- Banner fijo: "Modelo local vía Ollama · sin APIs externas · inspirado en arXiv:2604.07798".

---

## 6. Benchmark casero

Objetivo: comparar **4 condiciones** con el mismo modelo y los mismos datos.

| Condición | Descripción |
|---|---|
| **A. Sin memoria** | Solo STM |
| **B. RAG ingenuo** | Todos los turnos pasados embebidos; top-k directo al Responder |
| **C. LightMem sin Dream** | Pipeline online completo, sin consolidar |
| **D. LightMem con Dream** | C + Dream al final de cada sesión |

### 6.1 Datos
`bench/sessions/*.json`: **8 sesiones** guionizadas de un usuario ficticio (no usar datos personales reales), con turnos del usuario fijos (el agente responde lo que responda). Deben incluir a propósito:
- Hechos que se repiten con distinta redacción (para probar MERGE).
- Hechos que cambian: mudanza, cambio de trabajo, preferencia que se invierte (para UPDATE).
- Ruido trivial (para FORGET).
- Mezcla de español e inglés en algunas sesiones (los embeddings son multilingües; ver si se rompe).

### 6.2 Preguntas
`bench/questions.json`: **25–30 preguntas** hechas en una sesión final nueva, cada una con `expected_keywords`, `forbidden_keywords` (por ejemplo, el valor viejo de un hecho actualizado) y una categoría:

| Categoría | Ejemplo | Cuántas |
|---|---|---|
| `single_fact` | "¿Cómo se llama mi perro?" | 6 |
| `updated_fact` | "¿Dónde vivo ahora?" (cambió en la sesión 5) | 6 |
| `multi_session` | "¿Qué deportes he mencionado?" | 6 |
| `temporal` | "¿Qué hice antes de cambiar de trabajo?" | 5 |
| `abstention` | Algo que nunca se dijo → debe responder que no lo sabe | 5 |

### 6.3 Scoring (determinista)
- **Correcta:** contiene todos los `expected_keywords` (normalizados: minúsculas, sin acentos) y ningún `forbidden_keyword`.
- **Abstención correcta:** detectada por una lista de frases ("no lo sé", "no me lo has dicho", "I don't know"...) y sin inventar.
- Guardar todas las respuestas en `bench/results/<run_id>.jsonl` para **revisión manual**: el scoring por keywords se equivoca, y anotar dónde se equivoca también va al post.

### 6.4 Métricas por condición
- Exactitud total y por categoría.
- Latencia por turno: p50 y p95, total y por etapa.
- Llamadas al LLM por turno.
- Tamaño de memoria (entradas y tokens aproximados) al final.
- Activaciones de fallback y ops rechazadas.
- **3 corridas por condición** (temperatura 0 si es posible; reportar varianza igual).

Script: `python -m bench.run --condition D --model qwen2.5:7b-instruct --runs 3` → genera `bench/results/summary.csv` y una tabla en markdown lista para el post.

---

## 7. Estructura del repo

```
agente-que-suena/
├── app/
│   ├── main.py            # FastAPI
│   ├── config.py
│   ├── llm.py             # cliente Ollama + JSON validado + timeouts + fallback
│   ├── memory/
│   │   ├── store.py       # interfaz MemoryStore + SQLiteStore
│   │   ├── models.py      # MemoryEntry, Op, etc.
│   │   └── embed.py
│   ├── pipeline/
│   │   ├── controller.py
│   │   ├── retriever.py
│   │   ├── selector.py
│   │   ├── responder.py
│   │   └── writer.py
│   ├── dream/
│   │   ├── cluster.py
│   │   ├── consolidator.py
│   │   ├── validator.py
│   │   └── applier.py     # transacción + snapshot + diff
│   ├── prompts/           # un .md por módulo, versionados
│   └── static/index.html
├── bench/
│   ├── sessions/
│   ├── questions.json
│   ├── run.py
│   └── results/
├── tests/
├── Dockerfile
├── LOG.md
└── README.md
```

---

## 8. Riesgos conocidos (anotar en LOG.md si ocurren)

- El modelo pequeño no respeta el JSON → medir la tasa; probar `format: json` de Ollama.
- El Writer extrae demasiado (memoria inflada) o demasiado poco.
- El Consolidator fusiona hechos distintos (falso MERGE) o borra algo importante.
- UPDATE mal resuelto: el agente responde con el valor viejo (cubierto por `forbidden_keywords`).
- La latencia del pipeline de 4 llamadas hace que C/D sean más lentas que B: puede ser *el hallazgo*, no un fracaso.
- Clusters demasiado grandes → prompts largos → el modelo pierde el hilo.

---

## 9. Fases y criterios de aceptación

### Fase 1 — Esqueleto (sábado, mañana)
- Cliente Ollama con salida JSON validada, timeout y fallback; `MemoryStore` en SQLite; `/health`.
- ✅ Tests de `llm.py` (con mock) y de `store.py` pasan; `/health` reporta los modelos.

### Fase 2 — Pipeline online (sábado, tarde)
- Controller, Retriever, Selector, Responder, Writer; `/chat` con `trace`.
- UI básica: chat + paneles de memoria.
- ✅ Una conversación de 10 turnos genera entradas MTM razonables, y la traza muestra la latencia por etapa.

### Fase 3 — Dream (domingo, mañana)
- Clustering, Consolidator, Validator, Applier, diff en la UI.
- ✅ Tests del Validator rechazan ops inválidas (ids inexistentes, FORGET sobre LTM, MERGE de un solo id).
- ✅ Un segundo Dream seguido produce ≤ 20% de las ops del primero.

### Fase 4 — Benchmark (domingo, tarde)
- Sesiones, preguntas, runner, scoring y resumen.
- ✅ `summary.csv` con las 4 condiciones × 3 corridas para al menos un modelo.

### Fase 5 — Publicación (semana siguiente)
- Dockerfile con modelos en la imagen; integración con la infraestructura de demos efímeros.
- README con resultados, limitaciones y cita al paper.
- Post EN/ES: qué propone el paper, qué implementé, la tabla de resultados, **qué falló** y qué haría distinto.
- Antes de hacer público el repo: confirmar internamente la política sobre proyectos personales/open source.
- Avisar a los autores del paper con el enlace al demo.

---

## 10. Para leer en el paper antes de la Fase 2

- [ ] Prompts o formato de salida de Controller, Selector y Writer.
- [ ] Cómo y cuándo se consolida MTM → LTM; si hay olvido o decaimiento.
- [ ] Diseño de la consulta en dos etapas.
- [ ] Métricas y benchmarks usados (para comparar el espíritu, no los números).
- [ ] Si hay código publicado y con qué licencia (si lo hay, **no copiar**; implementar desde el paper).
