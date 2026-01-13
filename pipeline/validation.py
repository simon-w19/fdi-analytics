"""Data validation helpers using Pandera."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

from .config import settings

LOGGER = logging.getLogger("pipeline.validation")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

RATE_COLUMNS = [
    "last_12_months_checkout_pcnt",
    "last_12_months_functional_doubles_pcnt",
    "last_12_months_pcnt_legs_won",
    "last_12_months_pcnt_legs_won_throwing_first",
    "last_12_months_pcnt_legs_won_throwing_second",
]

NUMERIC_NON_NEGATIVE = [
    "profile_total_earnings",
    "profile_9_darters",
    "profile_tour_card_years",
    "profile_order_of_merit",
    "last_12_months_180_s",
    "last_12_months_171_180_s",
    "last_12_months_140_s",
    "last_12_months_131_140_s",
    "api_sum_field2",
]

SCHEMA = DataFrameSchema(
    {
        "player_id": Column(int, checks=pa.Check.ge(0)),
        "player_name": Column(str, nullable=True),
        "country": Column(str, nullable=True),
        "profile_fdi_rating": Column(float, nullable=True),
        "profile_total_earnings": Column(float, checks=pa.Check.ge(0), nullable=True),
        "profile_order_of_merit": Column(float, checks=pa.Check.ge(0), nullable=True),
        "last_12_months_checkout_pcnt": Column(
            float,
            checks=pa.Check.between(0, 100),
            nullable=True,
        ),
        "last_12_months_functional_doubles_pcnt": Column(
            float,
            checks=pa.Check.between(0, 100),
            nullable=True,
        ),
        # Allow any positive scoring average; low values surfaced in production scrapes
        "last_12_months_averages": Column(float, checks=pa.Check.gt(0), nullable=True),
        "scraped_at": Column(pa.DateTime, nullable=False),
    },
    coerce=True,
)


class DataQualityError(RuntimeError):
    """Raised when validation fails."""


@dataclass(slots=True)
class ValidationReport:
    """Structured outcome of the validation step."""

    rows: int
    issues: Dict[str, Any]


def validate_processed_dataset(df: pd.DataFrame, *, strict: bool = True) -> ValidationReport:
    """Validate the processed dataset using Pandera and domain checks."""

    rows = len(df)
    issues: Dict[str, Any] = {}
    try:
        SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:  # pragma: no cover - exercised in tests
        issues["schema"] = exc.failure_cases.to_dict()
        if strict:
            raise DataQualityError("Schema validation failed") from exc
        LOGGER.warning("Schema validation reported issues: %s", exc.failure_cases.head())

    duplicate_ids = df[df.duplicated(subset=["player_id"], keep=False)]
    if not duplicate_ids.empty:
        sample = duplicate_ids["player_id"].unique().tolist()[:5]
        issues["duplicate_player_id"] = sample
        if strict:
            raise DataQualityError(f"Duplicate player_id detected: {sample}")
        LOGGER.warning("Duplicate player ids detected: %s", sample)

    for column in RATE_COLUMNS:
        if column in df.columns:
            invalid_mask = (df[column] < 0) | (df[column] > 100)
            if invalid_mask.any():
                offenders = df.loc[invalid_mask, column].head().tolist()
                issues.setdefault("rate_out_of_bounds", {})[column] = offenders
                if strict:
                    raise DataQualityError(f"Column {column} has values outside 0-100")
                LOGGER.warning("Column %s has out-of-bounds values: %s", column, offenders)

    for column in NUMERIC_NON_NEGATIVE:
        if column in df.columns:
            invalid_mask = df[column] < 0
            if invalid_mask.any():
                offenders = df.loc[invalid_mask, column].head().tolist()
                issues.setdefault("negative_values", {})[column] = offenders
                if strict:
                    raise DataQualityError(f"Column {column} contains negative values")
                LOGGER.warning("Column %s contains negative values: %s", column, offenders)

    LOGGER.info("Validation finished with %s rows checked.", rows)
    return ValidationReport(rows=rows, issues=issues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed FDI dataset")
    parser.add_argument("--csv", type=Path, default=settings.processed_csv_path)
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Log validation issues instead of raising",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    validate_processed_dataset(df, strict=not args.allow_warnings)
    LOGGER.info("Validation passed for %s", args.csv)


if __name__ == "__main__":
    main()
