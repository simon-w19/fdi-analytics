"""Load processed CSV data into PostgreSQL."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import settings

LOGGER = logging.getLogger("pipeline.ingest")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def _read_dataset(csv_path: str | bytes | "os.PathLike[str]" | "os.PathLike[bytes]") -> pd.DataFrame:
    LOGGER.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        LOGGER.warning("Dataset at %s is empty.", csv_path)
    return df


def _get_engine() -> Engine:
    LOGGER.info("Connecting to database %s", settings.database_url)
    return create_engine(settings.database_url, pool_pre_ping=True)


def load_dataframe_into_db(df: pd.DataFrame, engine: Optional[Engine] = None) -> None:
    if engine is None:
        engine = _get_engine()
    LOGGER.info(
        "Writing %s rows into %s.%s",
        len(df),
        settings.db_schema,
        settings.table_name,
    )
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.db_schema}"))
        df.to_sql(
            name=settings.table_name,
            con=conn,
            schema=settings.db_schema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5_000,
        )
    LOGGER.info("Finished loading dataset into PostgreSQL.")


def main() -> None:
    df = _read_dataset(settings.csv_path)
    if df.empty:
        LOGGER.warning("Skipping ingestion because the dataframe is empty.")
        return
    engine = _get_engine()
    load_dataframe_into_db(df, engine=engine)


if __name__ == "__main__":
    main()
