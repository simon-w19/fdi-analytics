"""Shared feature definitions and helpers for the FDI pipeline.

Multikollinearitäts-Analyse (Stand: Januar 2026):
==================================================
Ursprünglich 38 Features, davon 36 mit VIF > 5 (problematisch).
64 Feature-Paare mit |r| > 0.8 identifiziert.

KRITISCHE FUNDE:
1. Perfekte Kollinearität (r=1.0): profile_season_win_pct ↔ season_win_rate
2. Sehr hohe Korrelation (r>0.99):
   - last_12_months_180_s ↔ last_12_months_171_180_s
   - last_12_months_averages ↔ with/against_throw_averages
   - checkout_pcnt ↔ functional_doubles_pcnt
   
LÖSUNG: Reduziertes Feature-Set (REDUCED_FEATURE_COLUMNS) für stabile Koeffizienten.
Das vollständige Set (FEATURE_COLUMNS) bleibt für Regularisierung (Lasso/Ridge) verfügbar.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

TARGET_COL = "profile_fdi_rating"
COUNTRY_FEATURE = "country"

# =============================================================================
# VOLLSTÄNDIGES FEATURE-SET (Original, 38 Features)
# Enthält redundante Features - nur für Regularisierungsmodelle (Lasso/Ridge)
# =============================================================================
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

# =============================================================================
# REDUZIERTES FEATURE-SET (11 Features, VIF-optimiert)
# Entfernt redundante Features nach Multikollinearitäts-Analyse
# Empfohlen für: Lineare Regression, Interpretation, stabile Koeffizienten
# =============================================================================
REDUCED_FEATURE_COLUMNS: List[str] = [
    # === Grundlegende Spielermerkmale ===
    "age",                           # Alter (Erfahrung & Physis)
    # ENTFERNT: profile_tour_card_years (r=0.87 mit experience_intensity)
    "profile_order_of_merit",        # Offizielles Ranking (externe Validierung)
    
    # === Finanzieller Erfolg (nur log-transformiert, da nicht-linear) ===
    "log_total_earnings",            # BEHALTEN: Logarithmiert für Normalität
    # ENTFERNT: profile_total_earnings (redundant mit log-Version, r=0.96)
    
    # === Gewinnquote (nur Original, nicht skaliert) ===
    "profile_season_win_pct",        # BEHALTEN: Season-Gewinnquote (%)
    # ENTFERNT: season_win_rate (identisch, nur /100, r=1.0)
    # ENTFERNT: last_12_months_pcnt_legs_won (r=0.95 mit season_win_pct)
    
    # === Leistungsmetriken - Kern ===
    "last_12_months_averages",       # BEHALTEN: 3-Dart-Average (zentral!)
    # ENTFERNT: with/against_throw_averages (subsumt, r>0.99)
    # ENTFERNT: last_12_months_first_9_averages (r=0.99 mit averages)
    # ENTFERNT: last_12_months_first_3_averages (r=0.97 mit averages)
    
    # === Checkout-Qualität (nur eine Variante) ===
    "last_12_months_checkout_pcnt",  # BEHALTEN: Checkout-% (Finish-Qualität)
    # ENTFERNT: functional_doubles_pcnt (nahezu identisch, r=0.99)
    # ENTFERNT: checkout_combo (engineered aus den obigen, redundant)
    # ENTFERNT: profile_highest_average (r=0.93 mit averages, r=0.86 mit checkout)
    
    # === Power Scoring (nur aggregiert) ===
    "power_scoring_ratio",           # BEHALTEN: Engineered (180s+171s)/(140s+131s)
    # ENTFERNT: last_12_months_180_s, 171_180_s, 140_s, 131_140_s (Einzelkomponenten)
    
    # === Erfahrungs-/Erfolgsmetriken ===
    "profile_9_darters",             # BEHALTEN: Anzahl perfekter Legs (niedrig korreliert)
    # ENTFERNT: profile_highest_tv_average (korreliert hoch mit highest_average)
    # ENTFERNT: tv_stage_delta (abgeleitet)
    
    # === API-Daten ===
    # ENTFERNT: api_sum_field1 (r=0.9999 mit last_12_months_averages, kein echter Stat)
    # ENTFERNT: api_sum_field2 (hoch korreliert mit field1)
    
    # === Engineered Features (nieder-korreliert, informativ) ===
    "momentum_gap",                  # BEHALTEN: With-Against Throw Delta
    "experience_intensity",          # BEHALTEN: Tour-Card-Years / Age
    # ENTFERNT: earnings_per_year (korreliert mit log_earnings + years)
    
    # === Kategorisch ===
    COUNTRY_FEATURE,                 # Nationalität
]

# Abgeleitete Listen für reduziertes Set
REDUCED_NUMERIC_FEATURES: List[str] = [
    col for col in REDUCED_FEATURE_COLUMNS if col != COUNTRY_FEATURE
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
    "scraped_at",
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
    "REDUCED_FEATURE_COLUMNS",
    "REDUCED_NUMERIC_FEATURES",
    "NUMERIC_FEATURES",
    "NUMERIC_INPUT_COLUMNS",
    "CATEGORICAL_FEATURES",
    "DERIVED_FEATURES",
    "MODEL_METADATA_COLUMNS",
    "default_input_vector",
    "build_prediction_frame",
    "engineer_features",
]
