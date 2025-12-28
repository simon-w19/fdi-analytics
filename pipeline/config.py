"""Configuration helpers for ETL jobs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = DEFAULT_PROJECT_ROOT / "data" / "processed" / "player_stats_all.csv"


def _sanitize_path(path_str: str | None) -> Path:
    if not path_str:
        return DEFAULT_DATA_PATH
    path = Path(path_str).expanduser().resolve()
    return path


@dataclass(slots=True)
class Settings:
    """Runtime configuration sourced from environment variables."""

    csv_path: Path = _sanitize_path(os.getenv("FDI_CSV_PATH"))
    table_name: str = os.getenv("FDI_DB_TABLE", "player_stats")
    db_schema: str = os.getenv("FDI_DB_SCHEMA", "public")
    db_host: str = os.getenv("POSTGRES_HOST", "localhost")
    db_port: str = os.getenv("POSTGRES_PORT", "5432")
    db_user: str = os.getenv("POSTGRES_USER", "fdi_admin")
    db_password: str = os.getenv("POSTGRES_PASSWORD", "fdi_password")
    db_name: str = os.getenv("POSTGRES_DB", "fdi")
    database_url_env: str | None = os.getenv("DATABASE_URL")
    refresh_minutes: int = int(os.getenv("FDI_REFRESH_MINUTES", "10080"))

    @property
    def database_url(self) -> str:
        if self.database_url_env:
            return self.database_url_env
        return (
            "postgresql+psycopg://"
            f"{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
