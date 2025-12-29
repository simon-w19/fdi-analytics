"""Monitoring utilities for dataset metrics."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import settings

LOGGER = logging.getLogger("pipeline.monitoring")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

NUMERIC_SUMMARY_COLUMNS = [
    "profile_fdi_rating",
    "last_12_months_averages",
    "last_12_months_first_9_averages",
    "last_12_months_checkout_pcnt",
    "last_12_months_functional_doubles_pcnt",
    "last_12_months_180_s",
]


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """Create a lightweight JSON-serializable metric summary."""

    null_share = {
        column: float(df[column].isna().mean())
        for column in df.columns
        if df[column].isna().any()
    }
    numeric_ranges = {}
    for column in NUMERIC_SUMMARY_COLUMNS:
        if column in df.columns and not df[column].dropna().empty:
            numeric_ranges[column] = {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "mean": float(df[column].mean()),
            }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_share": null_share,
        "numeric_ranges": numeric_ranges,
    }


def persist_dataset_metrics(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist dataset metrics to disk for monitoring purposes."""

    summary = summarize_dataframe(df)
    target = path or settings.metrics_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote dataset metrics to %s", target)
    return target


__all__ = ["summarize_dataframe", "persist_dataset_metrics"]
