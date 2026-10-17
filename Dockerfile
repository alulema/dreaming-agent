# App image: FastAPI + the memory pipeline. Talks to an Ollama server over HTTP (OLLAMA_URL).
# Memory lives in an ephemeral SQLite file: every container starts with an empty memory.
# LLM_NUM_THREAD must match the CPUs of the Ollama container (see README).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=/tmp/dreaming-agent/memory.db \
    OLLAMA_URL=http://localhost:11434 \
    LLM_MODEL=qwen2.5:3b-instruct \
    EMBED_MODEL=bge-m3 \
    LLM_TIMEOUT_S=180 \
    LLM_NUM_THREAD=2

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ ./app/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
