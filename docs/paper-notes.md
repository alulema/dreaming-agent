# Notas del paper — LightMem (arXiv:2604.07798)

> **Lightweight LLM Agent Memory with Small Language Models** — Zhang et al., ACL 2026 (main). v3, 22-abr-2026.
> HTML: https://arxiv.org/html/2604.07798v3 · Licencia del artículo: CC BY-NC-SA 4.0.
> ⚠️ No confundir con el otro "LightMem" de 2025 (arXiv:2510.18866). Citar siempre por ID.

**Procedencia:** estas notas salen de una extracción automática del HTML v3 (2026-10-16), **no de una lectura humana**.
Las secciones citadas (§) deben ser **confirmadas por Alexis** al leer el paper; marcar cada punto con ✅ al confirmarlo.

---

## Checklist §10 de la spec

### 1. Módulos online (Controller / Selector / Writer)
- **SLM-1 Controller** (§3.3, §10.1.1): descompone la consulta en *hypothetical queries* (HQs), enruta cada una a MTM o LTM y emite restricciones de metadatos (ϕt) y un presupuesto. Salida: "JSON format only". El prompt completo **no** está publicado; §10 lo describe en prosa.
- **SLM-2 Selector** (§3.4, §10.2.1): filtro de consistencia semántica = re-ranking de la etapa 2. Salida JSON. **Está afinado con LoRA** (2.000 muestras construidas; §4.1.4).
- **SLM-3 Writer** (§3.5, §10.3.1): comprime el turno en "one or more short semantic memory summaries" para MTM.
- **¿Mismo modelo?** No queda claro. Modelo por defecto: **Llama-3.2-1B-Instruct cuantizado vía Ollama**; Qwen2.5-1.5B-Instruct como prueba de robustez (§4.1.4).
- **Responder:** **GPT-4o-mini** fijo (§4.1.4), es decir, la respuesta final la genera un modelo grande por API. Las tablas también reportan backbones Qwen2.5/Llama-3.2 (ambiguo).
- **Consolidación offline:** la hace "a large-context LLM" (no especificado).

### 2. Recuperación en dos etapas
- Etapa 1: búsqueda vectorial por HQ con filtros de metadatos. Embeddings **all-MiniLM-L6-v2** (384-d, solo inglés). Top-10 según §4.1.4, aunque §3.4 habla de 2K candidatos repartidos entre las HQs (inconsistente).
- Etapa 2: el Selector conserva "the most relevant half" de los candidatos. Presupuesto final K: no especificado.
- Capacidad MTM B = 10⁴.

### 3. Consolidación MTM → LTM
- Frecuencia: **cada 10–15 turnos**, o cuando MTM acumula suficientes entradas o hay presión de capacidad (Tabla 10, §11.2).
- Candidatos: entradas nuevas, entradas reactivadas por recuperación y entradas de baja utilidad bajo presión de capacidad (Alg. 3).
- Operaciones: abstracción y **des-identificación**; búsqueda de anclas en LTM; insertar nodos y aristas; **merge/update si el solapamiento supera un umbral** (valor no especificado); **decaimiento de confianza** para candidatos con poca evidencia; borrado o pérdida de peso de lo obsoleto.
- Desalojo de MTM por recencia y estadísticas de acceso (Alg. 2).

### 4. Esquema
- MTM: resumen semántico, información temporal, estadísticas de acceso, embedding y user id.
- **LTM es un grafo**: nodos Entity/Concept, aristas IsA/HasProperty/RelatedTo/Implies, sin identificadores de usuario. Es global y compartida entre usuarios.
- STM: no se persiste ni se recupera.

### 5. Evaluación
- LoCoMo y DialSim; F1, BLEU-1, ROUGE-L/2, METEOR y SBERT. Baselines: LoCoMo, ReadAgent, MemoryBank, MemGPT y A-MEM.
- Resultado principal: +2.5 F1 sobre A-MEM; latencia de recuperación p50 de 83 ms y 581 ms end-to-end.
- Significancia estadística: bootstrap pareado con 1.000 remuestreos y 3 semillas.

### 6. Código
- No se encontró repositorio enlazado. (Si aparece uno, **no copiar**: implementar desde el paper.)

---

## Divergencias deliberadas del MVP (para el post)

| Paper | MVP | Motivo |
|---|---|---|
| SLM de 1–1.5B, Selector afinado con LoRA | Qwen2.5 3B/7B sin fine-tuning, mismo modelo con prompts distintos | El fine-tuning está fuera de alcance |
| Responder GPT-4o-mini (API) | Responder local (el mismo SLM) | Regla de cero APIs: es la diferencia más grande frente al paper |
| Consolidador: LLM de contexto largo | El mismo SLM local | Ídem; es un riesgo para la calidad del Dream |
| LTM como grafo de conocimiento | LTM como lista plana de entradas | Simplicidad |
| Consolidación automática cada 10–15 turnos | Botón Dream manual (más al final de cada sesión en el bench) | Visibilidad del demo |
| Multiusuario, LTM global des-identificada | Un solo usuario | No-objetivo |
| all-MiniLM-L6-v2 (inglés) | bge-m3 (multilingüe) | Las sesiones mezclan ES/EN |
| Etapa 2: conservar la mitad | Selector con `keep_ids` + contexto comprimido | Spec original; se puede probar "keep half" como variante |
