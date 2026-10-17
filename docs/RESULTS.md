# Resultados del benchmark

> Materia prima para el post. Datos crudos: `bench/results/summary.csv` (una fila por corrida) y los JSONL por corrida (cada turno, Dream y respuesta). Diseño del benchmark: [`bench/README.md`](../bench/README.md). Bitácora completa: [`LOG.md`](../LOG.md).

## v1: línea base (2026-10-17)

**Setup.** `qwen2.5:7b-instruct` (Q4, Ollama 0.40.2) para todos los módulos, `bge-m3` para embeddings, temperatura 0, seed 42, JSON Schema en `format`. MacBook Pro M1 Pro, 16 GB. 8 sesiones guionizadas (45 turnos de usuario, ES/EN) de un usuario ficticio ("Mateo") y 28 preguntas en sesiones nuevas, en modo solo lectura. 3 corridas por condición, intercaladas. Commit `29d427a`.

### Exactitud (%; media ± desviación sobre 3 corridas)

| Condición | Total | Hecho simple (6) | Hecho actualizado (6) | Multi-sesión (6) | Temporal (5) | Abstención (5) |
|---|---|---|---|---|---|---|
| A — sin memoria (solo STM) | 18 ± 0 | 0 | 0 | 0 | 0 | 100 |
| **B — RAG ingenuo** | **89 ± 0** | **100** | **100** | **67** | **80** | **100** |
| C — pipeline LightMem (sin Dream) | 69 ± 2 | 83 | 83 | 44 ± 10 | 33 ± 12 | 100 |
| D — pipeline + Dream por sesión | 66 ± 4 | 67 | 83 | 44 ± 10 | 33 ± 12 | 100 |

### Memoria y coste

| Condición | Memoria final (entradas / ~tokens) | Llamadas LLM por turno | Dream (por corrida) |
|---|---|---|---|
| A | 0 | 1 | — |
| B | 45 frases del usuario / 584 | 1 | — |
| C | 45 hechos / 476 | 2.9 | — |
| D | 30 ± 5 hechos / 418 ± 47 | 2.9 | 8 sueños, ~30 llamadas, 13 ± 5 ops aplicadas, 17 ± 6 rechazadas |

**Latencias: no válidas en v1.** El Mac entró en reposo durante la corrida y solo avanzaba en "DarkWakes" con estrangulamiento térmico (ver LOG). Se repiten en v2.

### Lectura

1. **Con un 7B local, el RAG ingenuo le gana por 20 puntos al pipeline estilo LightMem, y el Dream resta 3 más.** El Dream sí cumple como compresor: −33 % de entradas y −12 % de tokens frente a C.
2. **Por qué pierde C** (diagnóstico reconstruyendo la memoria de una corrida):
   - *El Controller decide no buscar*: "¿Qué idioma estoy aprendiendo?" → `needs_memory=false`. Sin contexto, el modelo **inventa** ("Estás aprendiendo español").
   - *El Selector poda de más*: la búsqueda trae "architect at Estudio Faro" (primer candidato, 0.71) y "took an online UX design course" (0.65), pero el Selector se queda con un solo hecho y el Responder contesta "no tengo información". Las preguntas temporales necesitan **unir** hechos, y comprimir los separa.
   - *El ruido compite*: para "¿qué deportes te he mencionado?", "had medialunas for breakfast" puntúa igual que los deportes reales.
3. **Lo que hizo el Dream:**
   - Convirtió un error del Writer ("The user's name is Kiwi", sacado de "mi loro Kiwi…") en un **UPDATE del nombre**: en 2 de 3 corridas el agente termina diciendo **"Te llamas Kiwi"**. Las reglas deterministas validan forma (nombres inventados, valores perdidos), **no verdad**.
   - Olvidó (FORGET) **la alergia al maní**, el curso de UX y el viaje a la Patagonia: hechos dichos una sola vez.
   - Limpió ruido real (supermercado, película), unificó las menciones de las mascotas, y las reglas bloquearon 18 UPDATE que perdían valores vigentes.
4. **Por qué gana B:** guarda las frases literales del usuario. No hay ningún paso donde un modelo pequeño pueda reescribir, perder o inventar un hecho, y el 7B razona bien sobre 8 frases crudas.

### Advertencias
- Un solo usuario ficticio, 28 preguntas: es una medición de caso, no un benchmark general.
- Scoring por palabras clave con revisión manual: se corrigieron 2 errores del scoring (un falso positivo de abstención y un falso negativo) y se re-puntuó sin volver a correr.
- q22 ("¿Qué mascota tenía antes de adoptar a Ramón?") falla en todas las condiciones porque la pregunta es ambigua: está mal diseñada.
- El paper usa SLM de 1–1.5B afinados (el Selector con LoRA) y **GPT-4o-mini como Responder**. Aquí todo es un 7B local sin fine-tuning. **Esto no evalúa el paper**: mide si su arquitectura ayuda a un modelo pequeño, sin las piezas grandes.
