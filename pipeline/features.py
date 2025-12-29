"""Shared feature definitions and helpers for the FDI pipeline."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

TARGET_COL = "profile_fdi_rating"
COUNTRY_FEATURE = "country"
FEATURE_COLUMNS: List[str] = [
    "age",
    "profile_total_earnings",
    "log_total_earnings",
    "profile_9_darters",
    "profile_season_win_pct",
    "season_win_rate",
    "profile_tour_card_years",
    "profile_highest_average",
    "profile_highest_tv_average",
    "profile_order_of_merit",
    "last_12_months_averages",
    "last_12_months_first_9_averages",
    "first9_delta",
    "last_12_months_first_3_averages",
    "last_12_months_with_throw_averages",
    "last_12_months_against_throw_averages",
    "momentum_gap",
    "last_12_months_highest_checkout",
    "last_12_months_checkout_pcnt",
    "last_12_months_functional_doubles_pcnt",
    "checkout_combo",
    "last_12_months_pcnt_legs_won",
    "last_12_months_pcnt_legs_won_throwing_first",
    "last_12_months_pcnt_legs_won_throwing_second",
    "last_12_months_180_s",
    "last_12_months_171_180_s",
    "last_12_months_140_s",
    "last_12_months_131_140_s",
    "api_sum_field1",
    "api_sum_field2",
    "experience_intensity",
    "earnings_per_year",
    "first9_ratio",
    "break_efficiency",
    "hold_break_spread",
    "power_scoring_ratio",
    "tv_stage_delta",
    COUNTRY_FEATURE,
]
NUMERIC_INPUT_COLUMNS: List[str] = [
    "age",
    "profile_total_earnings",
    "profile_9_darters",
    "profile_season_win_pct",
    "profile_tour_card_years",
    "profile_highest_average",
    "profile_highest_tv_average",
    "profile_order_of_merit",
    "last_12_months_averages",
    "last_12_months_first_9_averages",
    "last_12_months_first_3_averages",
    "last_12_months_with_throw_averages",
    "last_12_months_against_throw_averages",
    "last_12_months_highest_checkout",
    "last_12_months_checkout_pcnt",
    "last_12_months_functional_doubles_pcnt",
    "last_12_months_pcnt_legs_won",
    "last_12_months_pcnt_legs_won_throwing_first",
    "last_12_months_pcnt_legs_won_throwing_second",
    "last_12_months_180_s",
    "last_12_months_171_180_s",
    "last_12_months_140_s",
    "last_12_months_131_140_s",
    "api_sum_field1",
    "api_sum_field2",
]
DERIVED_FEATURES: List[str] = [
    "log_total_earnings",
    "season_win_rate",
    "checkout_combo",
    "first9_delta",
    "momentum_gap",
    "experience_intensity",
    "earnings_per_year",
    "first9_ratio",
    "break_efficiency",
    "hold_break_spread",
    "power_scoring_ratio",
    "tv_stage_delta",
]
BASE_NUMERIC_COLS: List[str] = sorted(set(NUMERIC_INPUT_COLUMNS + [TARGET_COL]))
NUMERIC_FEATURES: List[str] = [col for col in FEATURE_COLUMNS if col != COUNTRY_FEATURE]
CATEGORICAL_FEATURES: List[str] = [COUNTRY_FEATURE]
MODEL_METADATA_COLUMNS: List[str] = [
    "player_id",
    "player_name",
    "country_code",
    COUNTRY_FEATURE,
    TARGET_COL,
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: np.nan})
    ratio = numerator / denom
    return ratio.replace([np.inf, -np.inf], np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    if "country_code" in engineered.columns:
        country_code = engineered["country_code"].fillna("UNK")
    else:
        country_code = pd.Series(["UNK"] * len(engineered), index=engineered.index)
    if COUNTRY_FEATURE in engineered.columns:
        engineered[COUNTRY_FEATURE] = engineered[COUNTRY_FEATURE].fillna("UNK")
    else:
        engineered[COUNTRY_FEATURE] = country_code
    engineered["country_code"] = country_code
    for col in BASE_NUMERIC_COLS:
        if col in engineered.columns:
            engineered[col] = pd.to_numeric(engineered[col], errors="coerce")
    engineered["log_total_earnings"] = np.log1p(engineered["profile_total_earnings"].clip(lower=0))
    engineered["season_win_rate"] = engineered["profile_season_win_pct"] / 100.0
    engineered["checkout_combo"] = (
        engineered["last_12_months_checkout_pcnt"]
        * engineered["last_12_months_functional_doubles_pcnt"]
        / 100.0
    )
    engineered["first9_delta"] = (
        engineered["last_12_months_first_9_averages"] - engineered["last_12_months_averages"]
    )
    engineered["momentum_gap"] = (
        engineered["last_12_months_with_throw_averages"]
        - engineered["last_12_months_against_throw_averages"]
    )
    experience_denom = engineered["age"].replace({0: np.nan})
    engineered["experience_intensity"] = _safe_ratio(
        engineered["profile_tour_card_years"], experience_denom
    )
    engineered["earnings_per_year"] = _safe_ratio(
        engineered["profile_total_earnings"], engineered["profile_tour_card_years"]
    )
    engineered["first9_ratio"] = _safe_ratio(
        engineered["last_12_months_first_9_averages"], engineered["last_12_months_averages"]
    )
    engineered["break_efficiency"] = _safe_ratio(
        engineered["last_12_months_pcnt_legs_won_throwing_second"],
        engineered["last_12_months_pcnt_legs_won_throwing_first"],
    )
    engineered["hold_break_spread"] = (
        engineered["last_12_months_pcnt_legs_won_throwing_first"]
        - engineered["last_12_months_pcnt_legs_won_throwing_second"]
    )
    power_numerator = engineered["last_12_months_180_s"] + engineered["last_12_months_171_180_s"]
    power_denominator = engineered["last_12_months_140_s"] + engineered["last_12_months_131_140_s"]
    engineered["power_scoring_ratio"] = _safe_ratio(power_numerator, power_denominator)
    engineered["tv_stage_delta"] = (
        engineered["profile_highest_tv_average"] - engineered["profile_highest_average"]
    )
    for col in DERIVED_FEATURES:
        if col in engineered.columns:
            engineered[col] = pd.to_numeric(engineered[col], errors="coerce")
    return engineered


def default_input_vector(df: pd.DataFrame) -> Dict[str, float]:
    medians = df[NUMERIC_INPUT_COLUMNS].median(numeric_only=True)
    defaults = medians.to_dict()
    defaults[COUNTRY_FEATURE] = "UNK"
    defaults["player_name"] = "Custom Input"
    return defaults


def build_prediction_frame(raw_inputs: Dict[str, float]) -> pd.DataFrame:
    payload = raw_inputs.copy()
    payload.setdefault(COUNTRY_FEATURE, "UNK")
    payload["country_code"] = payload.get(COUNTRY_FEATURE, "UNK")
    df = pd.DataFrame([payload])
    engineered = engineer_features(df)
    missing = sorted(set(FEATURE_COLUMNS) - set(engineered.columns))
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")
    return engineered[FEATURE_COLUMNS]


__all__ = [
    "TARGET_COL",
    "COUNTRY_FEATURE",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "NUMERIC_INPUT_COLUMNS",
    "CATEGORICAL_FEATURES",
    "DERIVED_FEATURES",
    "MODEL_METADATA_COLUMNS",
    "default_input_vector",
    "build_prediction_frame",
    "engineer_features",
]
