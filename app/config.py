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
    llm_num_thread: int = 0  # 0 = Ollama default; set to the CPU quota when containerised
    embed_timeout_s: float = 30.0

    # Storage
    db_path: str = "data/memory.db"

    # Online pipeline
    stm_turns: int = 6
    top_k: int = 8
    selector_fallback_k: int = 3
    # v1 behaviour: "controller" + "compress" + writer_grounding=False. v2 defaults below
    # come from the v1 benchmark diagnosis (docs/RESULTS.md):
    # "always": retrieve on every turn, even when the Controller says memory isn't needed
    # (it skipped "what language am I learning?"); the Controller still writes the queries.
    retrieval_mode: Literal["controller", "always"] = "always"
    # "compress": Responder gets the Selector's summary. "filter": Responder gets the original
    # texts of the kept entries (dated, chronological), topped up to `selector_min_keep` with
    # the best-scored candidates (the Selector kept one fact where two were needed).
    selector_mode: Literal["compress", "filter"] = "filter"
    selector_min_keep: int = 3
    # Drop Writer entries with names/numbers absent from the user's message (prompt leakage).
    writer_grounding: bool = True
    # A new Writer entry this similar to an active MTM entry is not stored again: its turn is
    # added to the existing entry's sources instead (reinforcement). > 1.0 disables it.
    writer_dedup_threshold: float = 0.95

    # Dream
    # Spec default was 0.80; with bge-m3, contradictions ("lives in X" vs "moved to Y") score
    # 0.74-0.76, so they would never meet. Shared topics also link entries. See LOG.md.
    cluster_threshold: float = 0.75
    max_cluster_size: int = 8
    max_ltm_context: int = 5
    promote_min_access: int = 3
    promote_min_turns: int = 2
    # MERGE/UPDATE results that meet the PROMOTE evidence rule are born in LTM.
    merge_promotes: bool = True
    # FORGET only one-off events whose topics are all outside this set (v1 forgot an allergy).
    protect_from_forget: bool = True
    max_entry_chars: int = 300

    # Demo metadata (injected by the hosting environment, optional)
    project_id: str = ""
    demo_slot: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
