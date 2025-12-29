"""Tests for feature engineering helpers."""
from __future__ import annotations

import pandas as pd

from pipeline.features import FEATURE_COLUMNS, engineer_features


def test_engineer_features_adds_derived_columns():
    sample = {
        "player_id": 1,
        "player_name": "Test Player",
        "profile_fdi_rating": 1200,
        "profile_total_earnings": 1_000_000,
        "profile_9_darters": 5,
        "profile_season_win_pct": 65,
        "profile_tour_card_years": 4,
        "profile_highest_average": 101.0,
        "profile_highest_tv_average": 104.0,
        "profile_order_of_merit": 10,
        "last_12_months_averages": 98.0,
        "last_12_months_first_9_averages": 102.0,
        "last_12_months_first_3_averages": 110.0,
        "last_12_months_with_throw_averages": 100.0,
        "last_12_months_against_throw_averages": 97.0,
        "last_12_months_highest_checkout": 170.0,
        "last_12_months_checkout_pcnt": 42.0,
        "last_12_months_functional_doubles_pcnt": 48.0,
        "last_12_months_pcnt_legs_won": 60.0,
        "last_12_months_pcnt_legs_won_throwing_first": 64.0,
        "last_12_months_pcnt_legs_won_throwing_second": 55.0,
        "last_12_months_180_s": 200,
        "last_12_months_171_180_s": 50,
        "last_12_months_140_s": 300,
        "last_12_months_131_140_s": 80,
        "age": 27,
        "api_sum_field1": 1000,
        "api_sum_field2": 800,
    }
    df = pd.DataFrame([sample])
    engineered = engineer_features(df)
    required = {
        "log_total_earnings",
        "checkout_combo",
        "experience_intensity",
        "power_scoring_ratio",
        "country",
    }
    missing = required - set(engineered.columns)
    assert not missing, f"Missing engineered columns: {missing}"
    assert engineered["country"].iloc[0] == "UNK"
    assert engineered["power_scoring_ratio"].iloc[0] > 0
    assert set(FEATURE_COLUMNS).issubset(engineered.columns)
