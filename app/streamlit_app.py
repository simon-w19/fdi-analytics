"""Streamlit UI for serving the tuned FDI rating pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "player_stats_all.csv"
PIPELINE_PATH = PROJECT_ROOT / "models" / "best_fdi_pipeline.joblib"
TARGET_COL = "profile_fdi_rating"

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
    "country",
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

PLAYER_META_COLUMNS: List[str] = [
    "player_id",
    "player_name",
    "country_code",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: np.nan})
    ratio = numerator / denom
    return ratio.replace([np.inf, -np.inf], np.nan)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    df[TARGET_COL] = pd.to_numeric(df.get(TARGET_COL), errors="coerce")
    df["player_name"] = df["player_name"].fillna("Unbekannter Spieler")
    return df


@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not PIPELINE_PATH.exists():
        return None
    return joblib.load(PIPELINE_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    if "country" not in engineered.columns:
        engineered["country"] = engineered.get("country_code", "UNK").fillna("UNK")
    else:
        engineered["country"] = engineered["country"].fillna("UNK")
    engineered["country_code"] = engineered.get("country_code", engineered["country"])
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
    defaults["country"] = "UNK"
    defaults["player_name"] = "Custom Input"
    return defaults


def render_input_form(row: pd.Series, countries: List[str]) -> Dict[str, float]:
    inputs: Dict[str, float] = {}
    st.subheader("Feature-Eingabe")
    inputs["player_name"] = st.text_input(
        "Spielername (optional)", value=str(row.get("player_name", "FDI Prospect"))
    )
    default_country = row.get("country", "UNK")
    country_options = countries
    if default_country not in country_options:
        country_options = sorted({default_country} | set(country_options))
    inputs["country"] = st.selectbox(
        "Land", options=country_options, index=country_options.index(default_country)
    )
    chunk_size = 3
    for idx in range(0, len(NUMERIC_INPUT_COLUMNS), chunk_size):
        cols = st.columns(chunk_size)
        for offset, column in enumerate(NUMERIC_INPUT_COLUMNS[idx : idx + chunk_size]):
            value = float(row.get(column, 0.0))
            step = 0.01 if "pct" in column or "rate" in column else 1.0
            with cols[min(offset, len(cols) - 1)]:
                inputs[column] = st.number_input(
                    column,
                    value=value,
                    step=step,
                    format="%.4f" if step < 1 else "%.2f",
                )
    return inputs


def build_prediction_frame(raw_inputs: Dict[str, float]) -> pd.DataFrame:
    payload = raw_inputs.copy()
    payload["country_code"] = payload.get("country", "UNK")
    df = pd.DataFrame([payload])
    engineered = engineer_features(df)
    missing = sorted(set(FEATURE_COLUMNS) - set(engineered.columns))
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")
    return engineered[FEATURE_COLUMNS]


def predict_rating(pipeline, features: pd.DataFrame) -> float:
    prediction = float(pipeline.predict(features)[0])
    return prediction


st.set_page_config(page_title="FDI Rating Predictor", layout="wide")
st.title("FDI Rating Predictor")
st.caption(
    "Online-Scoring App, die das in den Notebooks trainierte Pipeline-Modell (Random Forest GridSearch) verwendet."
)

try:
    DATA_CACHE = load_dataset()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

PIPELINE = load_pipeline()
if PIPELINE is None:
    st.warning(
        "Kein Modellartefakt gefunden. Führe die Notebook-Zellen zum Export der Pipeline aus (models/best_fdi_pipeline.joblib)."
    )
else:
    st.success("Pipeline geladen und einsatzbereit.")

player_names = ["Freie Eingabe"] + sorted(DATA_CACHE["player_name"].unique())
selection = st.sidebar.selectbox("Spielerprofil laden", options=player_names)
if selection != "Freie Eingabe":
    base_row = DATA_CACHE.loc[DATA_CACHE["player_name"] == selection].iloc[0]
else:
    defaults = default_input_vector(DATA_CACHE)
    base_row = pd.Series(defaults)

country_choices = sorted({"UNK"} | set(DATA_CACHE["country"].unique()))

with st.form("prediction_form"):
    user_inputs = render_input_form(base_row, country_choices)
    submitted = st.form_submit_button("FDI Rating vorhersagen")

if submitted:
    if PIPELINE is None:
        st.error("Pipeline konnte nicht geladen werden.")
    else:
        try:
            feature_frame = build_prediction_frame(user_inputs)
            prediction = predict_rating(PIPELINE, feature_frame)
        except Exception as exc:
            st.exception(exc)
        else:
            st.metric(label="Vorhergesagtes FDI-Rating", value=f"{prediction:.2f}")
            with st.expander("Verwendete Features"):
                st.dataframe(feature_frame.transpose(), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Modellquelle: notebooks/fdi_rating_modeling.ipynb – Random Forest (GridSearchCV) Pipeline."
)
