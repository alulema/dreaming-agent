# Hand-off: manifest para la infra y glosario para el post

> Documento interno de entrega (ver `DEMO_INTEGRATION.md`, secciones "Lo que entregas a la infra" y "Términos técnicos para el post"). No es parte del manual de réplica (`README.md`).

## Manifest

| Campo | Valor |
|---|---|
| `projectId` | `dreaming-agent` |
| Nombre legible | ES: "El agente que sueña" · EN: "The dreaming agent" |
| Repo GitHub (público) | `https://github.com/alulema/dreaming-agent` *(pendiente de hacerlo público)* |
| Imagen principal + puerto | `ghcr.io/alulema/dreaming-agent-app:latest`, puerto `8080` (ingress, sirve `/`) |
| `shareable` | `false`: la memoria es estado por usuario. Un entorno por request |
| Secretos a inyectar | **Ninguno** (sin APIs externas) |
| Recursos extra | Sidecar **Ollama**: `ghcr.io/alulema/dreaming-agent-ollama:latest`, puerto `11434` (solo interno; modelos horneados, sin red de salida). La app lo encuentra en `OLLAMA_URL` (por defecto `http://localhost:11434`, que sirve si ambos contenedores comparten pod) |
| Recursos | **Se piden 4 vCPU / 6 GiB** (CPU only), más que el estándar de 2 vCPU / 4 GiB. Reparto sugerido: Ollama 4 vCPU / 5 GiB de límite, app 0.25 vCPU / 512 MiB. Con 2 vCPU una respuesta tarda ~75 s; con 4, ~30 s. Con 4 GiB la memoria de Ollama llega a 3.8 GiB (justo). Si solo hay 2 vCPU, funciona igual: poner `LLM_NUM_THREAD=2` |
| Variables de la app | `LLM_NUM_THREAD=4` (**importante**, ver abajo; la imagen trae 2 por defecto); `LLM_MODEL`, `EMBED_MODEL` y `LLM_TIMEOUT_S` traen sus valores por defecto en la imagen |
| Health | `GET /health` → 200 siempre; `"status": "ok"` cuando el sidecar responde y tiene los dos modelos (si no, `"degraded"`) |
| Arranque | App < 2 s. Ollama: ~10–20 s en cargar los modelos en RAM al primer uso (el primer turno es más lento) |
| Estado | SQLite efímero en `/tmp`; un teardown abrupto no pierde nada que deba sobrevivir. La UI muestra "la sesión expiró, recarga" ante errores de red o 502/503/504 del gateway |

**Sobre `LLM_NUM_THREAD`.** Dentro de un contenedor, Ollama ve todos los núcleos del nodo y lanza un hilo por núcleo. Con un límite de 2 vCPU eso es **~15× más lento** (medido: 91 s frente a 4 s por llamada). La app pasa `num_thread` en cada petición; debe coincidir con las vCPU del sidecar (redondeando hacia abajo, mínimo 1).

## Glosario para el post

| Término, tal como aparece en el post | Definición corta | Wikipedia |
|---|---|---|
| LightMem | Arquitectura de memoria para agentes con modelos de lenguaje pequeños (arXiv:2604.07798): módulos ligeros deciden qué recordar, qué recuperar y cómo consolidarlo. | — (paper: https://arxiv.org/abs/2604.07798) |
| STM (memoria de corto plazo) | Los últimos turnos de la conversación actual, que el modelo ve tal cual. | https://es.wikipedia.org/wiki/Memoria_a_corto_plazo |
| MTM (memoria de mediano plazo) | Hechos atómicos extraídos de la conversación ("el usuario vive en Quito"), guardados con su embedding para buscarlos después. | — |
| LTM (memoria de largo plazo) | Conocimiento consolidado: hechos fusionados o confirmados varias veces que el Dream promueve desde la MTM. | https://es.wikipedia.org/wiki/Memoria_a_largo_plazo |
| RAG (Retrieval-Augmented Generation) | Técnica que busca fragmentos de texto relevantes y los pasa al modelo junto con la pregunta, para que responda con esa información. | https://es.wikipedia.org/wiki/Generaci%C3%B3n_aumentada_por_recuperaci%C3%B3n |
| embedding | Vector de números que representa el significado de un texto; textos parecidos tienen vectores cercanos. | https://en.wikipedia.org/wiki/Word_embedding |
| similitud coseno | Medida de cuán parecidas son las direcciones de dos vectores (1 = idénticas, 0 = sin relación); se usa para comparar embeddings. | https://es.wikipedia.org/wiki/Similitud_coseno |
| SLM (Small Language Model) | Modelo de lenguaje pequeño (de cientos de millones a unos pocos miles de millones de parámetros) que puede correr en hardware modesto. | https://en.wikipedia.org/wiki/Small_language_model |
| LoRA (Low-Rank Adaptation) | Técnica de ajuste fino que entrena solo unas matrices pequeñas añadidas al modelo, en vez de todos sus pesos. | https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)#Low-rank_adaptation |
| consolidación de la memoria | En neurociencia, el proceso (en parte durante el sueño) que estabiliza los recuerdos; aquí, la fase offline que fusiona, actualiza u olvida entradas. | https://es.wikipedia.org/wiki/Consolidaci%C3%B3n_de_la_memoria |
| structured output / JSON Schema | Restringir la salida del modelo a un JSON que cumple un esquema, para que el código pueda leerla sin adivinar. | https://en.wikipedia.org/wiki/JSON#Metadata_and_schema |
| fallback | Valor por defecto que usa el sistema cuando el modelo falla (JSON inválido o timeout), para no romper la conversación. | — |
| soft-delete | Marcar un registro como borrado o reemplazado sin eliminarlo, para poder rastrear su historia. | — |
| grounding | Comprobar que lo que escribe el modelo está respaldado por el texto de origen (aquí: nombres y números presentes en el mensaje). | — |
| CFS / cuota de CPU | Mecanismo del kernel Linux que limita cuánto CPU usa un contenedor; si un programa lanza más hilos que la cuota, los estrangula a todos. | https://en.wikipedia.org/wiki/Completely_Fair_Scheduler |
| Ollama | Servidor local que descarga y ejecuta modelos de lenguaje abiertos mediante una API HTTP. | — |
| cuantización (Q4) | Guardar los pesos del modelo con menos bits (4 en vez de 16) para que ocupe y tarde menos, a costa de algo de precisión. | https://en.wikipedia.org/wiki/Quantization_(signal_processing) |

*Antes de entregarla: quitar los términos que el glosario del sitio ya tenga.*
