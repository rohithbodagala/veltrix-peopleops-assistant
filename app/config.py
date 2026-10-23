"""Runtime settings read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


def _env(env: dict, name: str, default: str = "") -> str:
    value = env.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    manifest_path: Path
    db_path: Path
    app_today: date | None
    llm_provider: str
    llm_model: str
    openai_api_key: str
    google_api_key: str
    embeddings: bool
    embedding_cache: Path
    llm_timeout: float
    request_deadline: float
    log_level: str
    log_file: str

    def today(self) -> date:
        """APP_TODAY if set, otherwise the system date."""
        return self.app_today or date.today()


def load_settings(env: dict | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    today_raw = _env(env, "APP_TODAY")
    return Settings(
        corpus_dir=Path(_env(env, "CORPUS_DIR", "./data/corpus")),
        manifest_path=Path(_env(env, "MANIFEST_PATH", "./data/manifest.json")),
        db_path=Path(_env(env, "DB_PATH", "./data/veltrix.db")),
        app_today=date.fromisoformat(today_raw) if today_raw else None,
        llm_provider=_env(env, "LLM_PROVIDER", "auto").lower(),
        llm_model=_env(env, "LLM_MODEL"),
        openai_api_key=_env(env, "OPENAI_API_KEY"),
        google_api_key=_env(env, "GOOGLE_API_KEY") or _env(env, "GEMINI_API_KEY"),
        embeddings=_env(env, "EMBEDDINGS", "on").lower() not in ("off", "0", "false", "no"),
        embedding_cache=Path(_env(env, "FASTEMBED_CACHE_PATH", "./models/fastembed")),
        llm_timeout=float(_env(env, "LLM_TIMEOUT_SECONDS", "20")),
        request_deadline=float(_env(env, "REQUEST_DEADLINE_SECONDS", "27")),
        log_level=_env(env, "LOG_LEVEL", "INFO").upper(),
        log_file=_env(env, "LOG_FILE"),
    )
