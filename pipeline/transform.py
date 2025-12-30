"""Transform raw scraping output into the processed modeling dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import settings
from .features import FEATURE_COLUMNS, MODEL_METADATA_COLUMNS, engineer_features

LOGGER = logging.getLogger("pipeline.transform")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
DEFAULT_EXTRA_COLUMNS = ["api_rank", "api_sum_field1", "api_sum_field2", "api_overall_stat"]


def transform_dataset(
    raw_csv_path: Path | str | None = None,
    output_csv_path: Path | str | None = None,
) -> pd.DataFrame:
    source = Path(raw_csv_path) if raw_csv_path else settings.raw_csv_path
    target = Path(output_csv_path) if output_csv_path else settings.processed_csv_path
    LOGGER.info("Transforming dataset %s -> %s", source, target)
    df = pd.read_csv(source)
    df = engineer_features(df)
    df["scraped_at"] = pd.Timestamp.now(tz="UTC")
    ordered_columns: list[str] = []
    seen: set[str] = set()
    for column in [*MODEL_METADATA_COLUMNS, *FEATURE_COLUMNS, *DEFAULT_EXTRA_COLUMNS]:
        if column in df.columns and column not in seen:
            ordered_columns.append(column)
            seen.add(column)
    transformed = df.loc[:, ordered_columns]
    target.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(target, index=False)
    LOGGER.info("Wrote %s rows to %s", len(transformed), target)
    return transformed


__all__ = ["transform_dataset"]
