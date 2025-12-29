"""Configuration helpers for ETL jobs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = DEFAULT_PROJECT_ROOT / "data" / "raw" / "player_stats_raw.csv"
DEFAULT_PROCESSED_PATH = DEFAULT_PROJECT_ROOT / "data" / "processed" / "player_stats_all.csv"
DEFAULT_METRICS_PATH = DEFAULT_PROJECT_ROOT / "reports" / "metrics" / "data_quality.json"


def _sanitize_path(path_str: str | None, default: Path) -> Path:
    if not path_str:
        return default
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (DEFAULT_PROJECT_ROOT / path).resolve()
    return path


def _optional_int(raw_value: str | None) -> int | None:
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _as_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Runtime configuration sourced from environment variables."""

    raw_csv_path: Path = _sanitize_path(os.getenv("FDI_RAW_CSV_PATH"), DEFAULT_RAW_PATH)
    processed_csv_path: Path = _sanitize_path(
        os.getenv("FDI_PROCESSED_CSV_PATH") or os.getenv("FDI_CSV_PATH"),
        DEFAULT_PROCESSED_PATH,
    )
    metrics_path: Path = _sanitize_path(os.getenv("FDI_METRICS_PATH"), DEFAULT_METRICS_PATH)
    table_name: str = os.getenv("FDI_DB_TABLE", "player_stats")
    db_schema: str = os.getenv("FDI_DB_SCHEMA", "public")
    db_host: str = os.getenv("POSTGRES_HOST", "localhost")
    db_port: str = os.getenv("POSTGRES_PORT", "5432")
    db_user: str = os.getenv("POSTGRES_USER", "fdi_admin")
    db_password: str = os.getenv("POSTGRES_PASSWORD", "fdi_password")
    db_name: str = os.getenv("POSTGRES_DB", "fdi")
    database_url_env: str | None = os.getenv("DATABASE_URL")
    refresh_minutes: int = int(os.getenv("FDI_REFRESH_MINUTES", "10080"))
    ingest_chunksize: int = int(os.getenv("FDI_DB_CHUNKSIZE", "5000"))
    scrape_delay_seconds: float = float(os.getenv("FDI_SCRAPE_DELAY_SECONDS", "0"))
    scrape_max_players: int | None = _optional_int(os.getenv("FDI_SCRAPE_MAX_PLAYERS"))
    skip_scrape: bool = _as_bool(os.getenv("FDI_SKIP_SCRAPE"))
    scrape_driver: str = os.getenv("FDI_SCRAPE_DRIVER", "requests").lower()
    scrape_retry_attempts: int = int(os.getenv("FDI_SCRAPE_RETRIES", "3"))
    scrape_retry_delay: float = float(os.getenv("FDI_SCRAPE_RETRY_DELAY", "1.5"))
    playwright_browser: str = os.getenv("FDI_PLAYWRIGHT_BROWSER", "chromium")
    playwright_headless: bool = _as_bool(os.getenv("FDI_PLAYWRIGHT_HEADLESS", "true"), True)
    playwright_timeout_ms: int = int(float(os.getenv("FDI_PLAYWRIGHT_TIMEOUT", "30")) * 1000)
    mlflow_enabled: bool = _as_bool(os.getenv("MLFLOW_ENABLED"))
    mlflow_tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "FDI Rating")

    @property
    def database_url(self) -> str:
        if self.database_url_env:
            return self.database_url_env
        return (
            "postgresql+psycopg://"
            f"{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
