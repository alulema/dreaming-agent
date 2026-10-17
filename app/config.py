"""Runtime configuration, read from environment variables (or a local .env file)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Models / Ollama
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "bge-m3"
    # "schema": pass the Pydantic JSON Schema as Ollama `format`; "json": plain JSON mode.
    llm_format: Literal["schema", "json"] = "schema"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_timeout_s: float = 60.0
    llm_json_retries: int = 1
    embed_timeout_s: float = 30.0

    # Storage
    db_path: str = "data/memory.db"

    # Online pipeline
    stm_turns: int = 6
    top_k: int = 8
    selector_fallback_k: int = 3

    # Dream
    cluster_threshold: float = 0.80
    max_cluster_size: int = 8
    promote_min_access: int = 3
    promote_min_turns: int = 2

    # Demo metadata (injected by the hosting environment, optional)
    project_id: str = ""
    demo_slot: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
