"""Load processed CSV data into PostgreSQL."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import settings

LOGGER = logging.getLogger("pipeline.ingest")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
POSTGRES_MAX_BIND_PARAMS = 65_535


def _read_dataset(csv_path: Path | str | bytes | "os.PathLike[str]" | "os.PathLike[bytes]") -> pd.DataFrame:
    path = Path(csv_path)
    LOGGER.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    if df.empty:
        LOGGER.warning("Dataset at %s is empty.", csv_path)
    return df


def _get_engine() -> Engine:
    LOGGER.info("Connecting to database %s", settings.database_url)
    return create_engine(settings.database_url, pool_pre_ping=True)


def _determine_chunk_size(num_columns: int) -> int:
    """Respect PostgreSQL's bind parameter limit to avoid exit code 1 during bulk loads."""
    if num_columns <= 0:
        return settings.ingest_chunksize
    safe_upper_bound = max(1, (POSTGRES_MAX_BIND_PARAMS // num_columns) - 1)
    chunk_size = max(1, min(settings.ingest_chunksize, safe_upper_bound))
    if chunk_size < settings.ingest_chunksize:
        LOGGER.info(
            "Reducing chunk size from %s to %s to stay below PostgreSQL bind limit",
            settings.ingest_chunksize,
            chunk_size,
        )
    return chunk_size


def load_dataframe_into_db(df: pd.DataFrame, engine: Optional[Engine] = None) -> None:
    if engine is None:
        engine = _get_engine()
    LOGGER.info(
        "Writing %s rows into %s.%s",
        len(df),
        settings.db_schema,
        settings.table_name,
    )
    chunk_size = _determine_chunk_size(len(df.columns))
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.db_schema}"))
        df.to_sql(
            name=settings.table_name,
            con=conn,
            schema=settings.db_schema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=chunk_size,
        )
    LOGGER.info("Finished loading dataset into PostgreSQL.")


def main(csv_path: Path | str | None = None) -> None:
    target_path = Path(csv_path) if csv_path else settings.processed_csv_path
    df = _read_dataset(target_path)
    if df.empty:
        LOGGER.warning("Skipping ingestion because the dataframe is empty.")
        return
    engine = _get_engine()
    load_dataframe_into_db(df, engine=engine)


if __name__ == "__main__":
    main()
