"""FDI Rating service combining Gradio UI and FastAPI endpoints."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pipeline.config import settings

LOGGER = logging.getLogger("app.gradio")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "player_stats_all.csv"
PIPELINE_PATH = PROJECT_ROOT / "models" / "best_fdi_pipeline.joblib"
TARGET_COL = "profile_fdi_rating"

ENGINE: Engine | None = None
DATA_CACHE: pd.DataFrame | None = None
PLAYER_CHOICES: List[str] = ["Freie Eingabe"]
COUNTRY_CHOICES: List[str] = ["UNK"]

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


class PredictionRequest(BaseModel):
    """Schema für die API-Vorhersage."""

    player_name: str | None = None
    country: str | None = None
    features: Dict[str, float]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: np.nan})
    ratio = numerator / denom
    return ratio.replace([np.inf, -np.inf], np.nan)


def get_engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        LOGGER.info("Connecting to %s", settings.database_url)
        ENGINE = create_engine(settings.database_url, pool_pre_ping=True)
    return ENGINE


def load_dataset_from_db() -> pd.DataFrame:
    engine = get_engine()
    query = text(
        f'SELECT * FROM "{settings.db_schema}"."{settings.table_name}"'
    )
    df = pd.read_sql_query(query, engine)
    if df.empty:
        raise ValueError("Die Tabelle player_stats ist leer.")
    return engineer_features(df)


def load_dataset_from_csv() -> pd.DataFrame:
    if not CSV_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {CSV_DATA_PATH}")
    LOGGER.warning("Loading dataset from CSV fallback: %s", CSV_DATA_PATH)
    df = pd.read_csv(CSV_DATA_PATH)
    return engineer_features(df)


def load_dataset() -> pd.DataFrame:
    try:
        return load_dataset_from_db()
    except Exception as exc:
        LOGGER.warning("Falling back to CSV because DB load failed: %s", exc)
    return load_dataset_from_csv()


def refresh_data_cache(force: bool = False) -> pd.DataFrame:
    global DATA_CACHE, PLAYER_CHOICES, COUNTRY_CHOICES
    if force or DATA_CACHE is None:
        DATA_CACHE = load_dataset()
        player_names = sorted(DATA_CACHE["player_name"].dropna().unique().tolist())
        PLAYER_CHOICES = ["Freie Eingabe"] + player_names
        COUNTRY_CHOICES = sorted({"UNK"} | set(DATA_CACHE["country"].dropna().tolist()))
        LOGGER.info("Dataset cache primed with %s rows.", len(DATA_CACHE))
    return DATA_CACHE


try:
    refresh_data_cache(force=True)
except Exception as exc:
    LOGGER.error("Unable to prime dataset cache at startup: %s", exc)


def load_pipeline():
    if not PIPELINE_PATH.exists():
        LOGGER.warning("Pipeline artefact fehlt unter %s", PIPELINE_PATH)
        return None
    LOGGER.info("Loading pipeline from %s", PIPELINE_PATH)
    return joblib.load(PIPELINE_PATH)


PIPELINE = load_pipeline()


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


def build_prediction_frame(raw_inputs: Dict[str, float]) -> pd.DataFrame:
    payload = raw_inputs.copy()
    payload.setdefault("country", "UNK")
    payload["country_code"] = payload.get("country", "UNK")
    df = pd.DataFrame([payload])
    engineered = engineer_features(df)
    missing = sorted(set(FEATURE_COLUMNS) - set(engineered.columns))
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")
    return engineered[FEATURE_COLUMNS]


def predict_rating(pipeline, features: pd.DataFrame) -> float:
    return float(pipeline.predict(features)[0])


def _get_prefilled_row(selection: str, dataset: pd.DataFrame) -> pd.Series:
    if selection != "Freie Eingabe" and selection in set(dataset["player_name"].tolist()):
        row = dataset.loc[dataset["player_name"] == selection].iloc[0]
    else:
        row = pd.Series(default_input_vector(dataset))
    row["country"] = row.get("country", row.get("country_code", "UNK"))
    if pd.isna(row.get("player_name", np.nan)):
        fallback = selection if selection != "Freie Eingabe" else "FDI Prospect"
        row["player_name"] = fallback
    return row


def _prepare_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    series = features.iloc[0]
    values = ["nan" if pd.isna(series[col]) else str(series[col]) for col in FEATURE_COLUMNS]
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "value": values})


def handle_prefill(selection: str) -> List[float]:
    dataset = refresh_data_cache()
    row = _get_prefilled_row(selection, dataset)
    numeric_values = [float(row.get(col, 0.0)) for col in NUMERIC_INPUT_COLUMNS]
    return [str(row.get("player_name", "FDI Prospect")), row.get("country", "UNK")] + numeric_values


def handle_prediction(player_name: str, country: str, *numeric_values: Sequence[float]):
    if PIPELINE is None:
        raise gr.Error("Kein Modellartefakt gefunden. Bitte exportiere best_fdi_pipeline.joblib im Notebook.")
    payload = {}
    for col, val in zip(NUMERIC_INPUT_COLUMNS, numeric_values):
        payload[col] = float(val if val is not None else 0.0)
    payload["player_name"] = player_name or "FDI Prospect"
    payload["country"] = country or "UNK"
    feature_frame = build_prediction_frame(payload)
    rating = predict_rating(PIPELINE, feature_frame)
    preview_table = _prepare_feature_table(feature_frame)
    return round(rating, 3), preview_table


def build_interface():
    with gr.Blocks(title="FDI Rating Predictor") as demo:
        gr.Markdown("""# FDI Rating Predictor\nLive-Predictions direkt aus der PostgreSQL-Datenbank.""")
        try:
            dataset = refresh_data_cache()
        except Exception as exc:
            gr.Markdown(
                f"❗ **Datensatz fehlt.** Verbindung zur Datenbank/CSV fehlgeschlagen: {exc}."
            )
            return demo

        initial_row = _get_prefilled_row(PLAYER_CHOICES[0], dataset)
        default_country = initial_row.get("country", COUNTRY_CHOICES[0])

        status_lines = []
        if PIPELINE is None:
            status_lines.append(
                "⚠️ Kein Modellartefakt gefunden. Führe die Export-Zelle im Notebook aus (models/best_fdi_pipeline.joblib)."
            )
        else:
            status_lines.append("✅ Modellpipeline geladen.")
        status_lines.append(f"📦 Spieler im Cache: {len(dataset):,}")
        gr.Markdown("\n".join(status_lines))

        player_selector = gr.Dropdown(
            choices=PLAYER_CHOICES,
            value=PLAYER_CHOICES[0],
            label="Spielerprofil laden",
        )
        player_name_box = gr.Textbox(
            label="Spielername (optional)",
            value=str(initial_row.get("player_name", "FDI Prospect")),
        )
        country_dropdown = gr.Dropdown(
            choices=COUNTRY_CHOICES,
            value=default_country,
            label="Land",
        )

        numeric_components: List[gr.Number] = []
        chunk_size = 3
        for idx in range(0, len(NUMERIC_INPUT_COLUMNS), chunk_size):
            with gr.Row():
                for column in NUMERIC_INPUT_COLUMNS[idx : idx + chunk_size]:
                    comp = gr.Number(
                        label=column,
                        value=float(initial_row.get(column, 0.0)),
                    )
                    numeric_components.append(comp)

        predict_button = gr.Button("FDI Rating vorhersagen", variant="primary")
        prediction_output = gr.Number(label="Vorhergesagtes FDI-Rating", interactive=False)
        feature_table = gr.Dataframe(
            headers=["feature", "value"],
            datatype=["str", "str"],
            row_count=(len(FEATURE_COLUMNS), "dynamic"),
            label="Genutzter Feature-Vektor",
            interactive=False,
        )

        player_selector.change(
            fn=handle_prefill,
            inputs=player_selector,
            outputs=[player_name_box, country_dropdown] + numeric_components,
        )

        predict_button.click(
            fn=handle_prediction,
            inputs=[player_name_box, country_dropdown] + numeric_components,
            outputs=[prediction_output, feature_table],
        )

    return demo


def register_api_routes(api: FastAPI) -> None:
    @api.get("/api/health")
    def healthcheck():
        rows = len(DATA_CACHE) if DATA_CACHE is not None else 0
        return {
            "status": "ok",
            "has_model": PIPELINE is not None,
            "rows_cached": rows,
            "schema": settings.db_schema,
            "table": settings.table_name,
        }

    @api.get("/api/players")
    def list_players(limit: int = 50):
        dataset = refresh_data_cache()
        limited = dataset[["player_name", "country", TARGET_COL]].head(limit).fillna("UNK")
        return limited.to_dict(orient="records")

    @api.post("/api/predict")
    def predict_api(request: PredictionRequest):
        if PIPELINE is None:
            raise HTTPException(status_code=503, detail="Pipeline nicht geladen")
        numeric_map = {col: float(request.features.get(col, 0.0)) for col in NUMERIC_INPUT_COLUMNS}
        payload = {
            **numeric_map,
            "player_name": request.player_name or "FDI Prospect",
            "country": request.country or "UNK",
        }
        feature_frame = build_prediction_frame(payload)
        rating = predict_rating(PIPELINE, feature_frame)
        return {
            "prediction": rating,
            "player_name": payload["player_name"],
            "country": payload["country"],
        }

    @api.post("/api/refresh-cache")
    def refresh_cache():
        dataset = refresh_data_cache(force=True)
        return JSONResponse({"rows": len(dataset)})


def create_service() -> FastAPI:
    demo = build_interface()
    api = FastAPI(title="FDI Rating Service")
    register_api_routes(api)
    return gr.mount_gradio_app(api, demo, path="/")


def main() -> None:
    app = create_service()
    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
