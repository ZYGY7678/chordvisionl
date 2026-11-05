"""Runtime configuration, read from the environment.

Nothing secret has a default value that would work in production.  The
encryption key in particular is generated per-process if it is missing,
which makes local development painless and makes a misconfigured
deployment obvious: stored API keys stop decrypting after a restart, and
the server says so instead of failing silently.
"""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="CHORDVISION_", extra="ignore",
    )

    app_name: str = "ChordVision"
    environment: str = "development"
    debug: bool = False

    # --- storage -------------------------------------------------------
    database_url: str = "sqlite:///./chordvision.db"
    """Postgres in production, e.g.
    ``postgresql+psycopg://user:pass@host/chordvision``.  SQLite is the
    default so the project runs with no external services."""

    storage_dir: str = "./storage"
    max_upload_bytes: int = 120 * 1024 * 1024      # 120 MB
    max_audio_seconds: float = 20 * 60             # 20 minutes

    # --- security --------------------------------------------------------
    secret_key: Optional[str] = None
    """Used to derive the key that encrypts stored provider API keys.
    Must be set in production; see .env.example."""

    cors_origins: List[str] = Field(default_factory=lambda: [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
    ])

    rate_limit_per_minute: int = 60
    upload_rate_limit_per_hour: int = 30

    # --- analysis ---------------------------------------------------------
    analysis_workers: int = 2
    quality_set: str = "sevenths"
    job_retention_hours: int = 72

    # --- transcription ------------------------------------------------------
    transcription_timeout_s: float = 600.0
    allow_audio_upload_to_provider: bool = False
    """Transcription means sending the audio to a third party.  That is a
    decision for the person who owns the recording, so it is off until
    they turn it on per project."""

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")

    def resolved_secret(self) -> str:
        if self.secret_key:
            return self.secret_key
        if self.is_production:
            raise RuntimeError(
                "CHORDVISION_SECRET_KEY must be set in production. "
                "Generate one with: python -c \"import secrets;print(secrets.token_urlsafe(48))\""
            )
        # Development only: ephemeral, so stored keys do not survive a
        # restart. That is intentional - it keeps a dev box from quietly
        # behaving like a production one.
        global _EPHEMERAL
        if _EPHEMERAL is None:
            _EPHEMERAL = secrets.token_urlsafe(48)
        return _EPHEMERAL


_EPHEMERAL: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
