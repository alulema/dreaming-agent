"""Embeddings via Ollama, L2-normalised and stored as float32 bytes."""

import numpy as np

from app.llm import OllamaClient

DTYPE = np.float32


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=DTYPE).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=DTYPE)


def normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=DTYPE)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.where(norms == 0, 1, norms)


class Embedder:
    def __init__(self, client: OllamaClient, model: str, timeout_s: float = 30.0) -> None:
        self.client = client
        self.model = model
        self.timeout_s = timeout_s

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Returns an (n, dim) matrix of unit vectors."""
        vecs = await self.client.embed(texts, self.model, timeout=self.timeout_s)
        return normalize(np.array(vecs, dtype=DTYPE))

    async def embed_one(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]
