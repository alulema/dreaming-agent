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

## v2: correcciones dirigidas por el diagnóstico (2026-10-17)

**Cambios** (flags de configuración; v1 sigue siendo reproducible): (1) recuperar siempre, aunque el Controller diga que no hace falta; (2) el Responder recibe los textos originales de las entradas elegidas (≥3, con fecha), no el resumen del Selector; (3) grounding del Writer (nombres y números deben estar en el mensaje del usuario); (4) el Dream solo puede olvidar eventos puntuales sin temas protegidos. Validado con los datos de desarrollo, no con los del bench. Mismo modelo, datos y protocolo que v1, con el Mac despierto. Commit `629bda1`.

### Exactitud (%; media ± desviación sobre 3 corridas)

| Condición | v1 | **v2** | Hecho simple | Actualizado | Multi-sesión | Temporal | Abstención |
|---|---|---|---|---|---|---|---|
| A — sin memoria | 18 | 18 ± 0 | 0 | 0 | 0 | 0 | 100 |
| B — RAG ingenuo | 89 | **88 ± 2** | 100 | 100 | 67 | 73 ± 12 | 100 |
| C — pipeline | 69 ± 2 | **79 ± 0** | 83 | 83 | 50 | 80 | 100 |
| D — pipeline + Dream | 66 ± 4 | **79 ± 9** (71 / 75 / 89) | 89 ± 19 | 83 ± 17 | 67 ± 29 | 53 ± 12 | 100 |

### Latencia (válida en v2; M1 Pro, 7B Q4)

| Condición | Turno de chat p50 / p95 | Pregunta p50 | Llamadas LLM por turno | Coste del Dream |
|---|---|---|---|---|
| A | 1.3 s / 3.0 s | 1.3 s | 1 | — |
| B | 3.4 s / 4.3 s | 2.6 s | 1 | — |
| C | 7.9 s / 11.5 s | 7.2 s | 2.9 | — |
| D | 7.0 s / 10.4 s | 7.4 s | 2.9 | ~25 s por sueño (8 sueños: ~200 s y ~28 llamadas por corrida) |

Tiempo de respuesta percibido (sin el Writer, que corre después): C/D ~4.5 s frente a 3.4 s de B.

### Memoria

| Condición | Entradas / ~tokens al final | Dream: ops aplicadas / rechazadas por corrida |
|---|---|---|
| B | 45 frases / 584 | — |
| C | 45 hechos / 476 | — |
| D | 32 ± 3 / 423 ± 11 | 17 ± 4 / 27 ± 3 |

### Lectura

1. **Las correcciones cierran la mitad de la brecha:** C pasa de 69 a 79 (+10, muy estable: 79/79/79) y D de 66 a 79 (+13). Pero **B sigue arriba (88)**, con la mitad de latencia y un tercio de llamadas al LLM.
2. **El Dream ya no resta en promedio, pero añade varianza:** D va de 71 a 89 según la corrida. Su mejor corrida empata con B; la peor queda por debajo de C. El Dream comprime la memoria un 29 % (45 → 32 entradas).
3. **Arreglar una pieza expone el error de otra.** Al pasar los textos originales (modo `filter`), el Responder ve a la vez "name is Mateo" y "name is **Kiwi**" (error del Writer). Como el prompt le dice que confíe en lo más reciente, C responde **"Te llamas Kiwi"** en las 3 corridas (en v1, el resumen del Selector lo ocultaba). En D, el Dream *a veces* repara ese error (MERGE "parrot Kiwi" + "name is Kiwi" → "has a parrot named Kiwi"; UPDATE → "Mateo (previously named Kiwi)") y a veces no.
4. **Corrupciones semánticas que ninguna regla atrapa** (todo está *grounded*): el Dream escribió **"The user lives in Pagora"** (Pagora es la empresa) y **"The user lives in Valparaíso (previously allergic to peanuts)"**. Las reglas deterministas evitan la invención; no evitan la *mezcla*.
5. **El Responder de 7B también falla con el contexto correcto:** en D2/q19 recibió "architect at Estudio Faro" + "took an online UX design course" y aun así contestó "no tengo información". En C, el hecho explícito "left the Estudio Faro" lo ayudaba a unir los dos.

### Conclusión para el post (7B local, sin fine-tuning)
- La arquitectura LightMem (Controller/Selector/Writer + consolidación) **no le gana a un RAG ingenuo** con un modelo pequeño en este caso: 79 vs 88 %, con 2.3× la latencia y 2.9× las llamadas.
- Cada llamada extra al LLM es **otra oportunidad de reescribir mal un hecho**. B nunca reescribe nada: guarda lo que dijo el usuario.
- El Dream es un buen **compresor** (−29 % de entradas) y a veces **repara** errores del Writer, pero de forma **no fiable**: misma configuración, de 71 a 89 % según la corrida.
