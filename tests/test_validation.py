"""Tests for the validation helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline import validation


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "A",
                "country": "ENG",
                "profile_fdi_rating": 1200,
                "profile_total_earnings": 1000,
                "profile_order_of_merit": 10,
                "last_12_months_checkout_pcnt": 42,
                "last_12_months_functional_doubles_pcnt": 50,
                "last_12_months_averages": 98,
                "last_12_months_pcnt_legs_won": 60,
            }
        ]
    )


def test_validation_passes_on_clean_dataset():
    df = _sample_df()
    report = validation.validate_processed_dataset(df)
    assert report.rows == 1
    assert report.issues == {}


def test_validation_rejects_duplicate_ids():
    df = pd.concat([_sample_df(), _sample_df()], ignore_index=True)
    with pytest.raises(validation.DataQualityError):
        validation.validate_processed_dataset(df)


def test_validation_rejects_negative_values():
    df = _sample_df().copy()
    df.loc[0, "profile_total_earnings"] = -5
    with pytest.raises(validation.DataQualityError):
        validation.validate_processed_dataset(df)
