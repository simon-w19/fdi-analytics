"""FDI Rating service combining Gradio UI and FastAPI endpoints."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import altair as alt
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import uvicorn

alt.data_transformers.disable_max_rows()
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pipeline.config import settings
from pipeline.features import (
    FEATURE_COLUMNS,
    NUMERIC_INPUT_COLUMNS,
    TARGET_COL,
    build_prediction_frame,
    default_input_vector,
    engineer_features,
)

LOGGER = logging.getLogger("app.gradio")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "player_stats_all.csv"
PIPELINE_PATH = PROJECT_ROOT / "models" / "best_fdi_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "latest_metrics.json"
ENGINE: Engine | None = None
DATA_CACHE: pd.DataFrame | None = None
PLAYER_CHOICES: List[str] = ["Freie Eingabe"]
COUNTRY_CHOICES: List[str] = ["UNK"]
INSIGHT_FEATURE_CHOICES: List[str] = [
    "last_12_months_averages",
    "last_12_months_first_9_averages",
    "profile_highest_tv_average",
    "profile_highest_average",
    "profile_total_earnings",
    "season_win_rate",
    "earnings_per_year",
    "break_efficiency",
    "power_scoring_ratio",
]


class PredictionRequest(BaseModel):
    """Schema für die API-Vorhersage."""

    player_name: str | None = None
    country: str | None = None
    features: Dict[str, float]


def load_metrics_payload() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        LOGGER.warning("Metrics file not found at %s", METRICS_PATH)
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.error("Unable to parse metrics JSON: %s", exc)
        return {}


def _format_best_params(best_params: Dict[str, Any] | None) -> str:
    if not best_params:
        return "—"
    pairs = []
    for key, value in best_params.items():
        clean_key = key.replace("model__", "")
        pairs.append(f"{clean_key}={value}")
    return ", ".join(pairs)


def build_metrics_dataframe() -> pd.DataFrame | None:
    payload = load_metrics_payload()
    metrics = payload.get("metrics", {})
    if not metrics:
        return None
    rows = []
    for model, stats in metrics.items():
        rows.append(
            {
                "Model": model.replace("_", " ").title(),
                "MAE": round(stats.get("mae", float("nan")), 3),
                "RMSE": round(stats.get("rmse", float("nan")), 3),
                "R2": round(stats.get("r2", float("nan")), 3),
                "CV MAE": round(stats.get("cv_mae_mean", float("nan")), 3),
                "CV Std": round(stats.get("cv_mae_std", float("nan")), 3),
                "Best Params": _format_best_params(stats.get("best_params")),
                "Tuning CV MAE": round(stats.get("tuning_cv_mae", float("nan")), 3)
                if stats.get("tuning_cv_mae")
                else None,
            }
        )
    return pd.DataFrame(rows)


def build_metric_story(metrics_df: pd.DataFrame | None) -> str:
    if metrics_df is None or metrics_df.empty:
        return "Noch keine Trainingsmetriken gefunden."
    ordered = metrics_df.sort_values("MAE")
    best = ordered.iloc[0]
    story_lines = [
        f"- **{best['Model']}** erreicht den niedrigsten MAE ({best['MAE']:.2f}) bei $R^2={best['R2']:.3f}$ und dient als Production-Referenz.",
    ]
    if len(ordered) > 1:
        second = ordered.iloc[1]
        delta = second["MAE"] - best["MAE"]
        story_lines.append(
            f"- **{second['Model']}** liegt nur {delta:.2f} Punkte zurück und nutzt {second['Best Params']} als beste Hyperparameter."
        )
    tail = ordered.nlargest(1, "MAE").iloc[0]
    if tail["Model"] != best["Model"]:
        story_lines.append(
            f"- **{tail['Model']}** liefert mehr Flexibilität, zeigt aber höheren MAE ({tail['MAE']:.2f}); bei Bedarf könnten zusätzliche Features helfen."
        )
    return "\n".join(story_lines)


def build_top_players_table(dataset: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    if dataset is None or dataset.empty:
        return pd.DataFrame()
    columns = [
        "player_name",
        "country",
        TARGET_COL,
        "last_12_months_averages",
        "profile_highest_average",
        "profile_total_earnings",
        "season_win_rate",
    ]
    available_columns = [col for col in columns if col in dataset.columns]
    return (
        dataset.sort_values(TARGET_COL, ascending=False)
        .loc[:, available_columns]
        .head(limit)
        .reset_index(drop=True)
    )


def render_scatter_plot(feature: str):
    dataset = refresh_data_cache()
    if dataset is None or feature not in dataset.columns:
        fallback = pd.DataFrame({"Hinweis": ["Feature nicht verfügbar."]})
        return alt.Chart(fallback).mark_text(size=14).encode(text="Hinweis:N")
    df = dataset[[feature, TARGET_COL, "player_name", "country"]].dropna()
    if df.empty:
        return alt.Chart(pd.DataFrame({"Hinweis": ["Keine Daten für diesen Plot"]})).mark_text()
    return (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X(feature, title=feature),
            y=alt.Y(TARGET_COL, title="FDI Rating"),
            color=alt.Color("country", legend=None),
            tooltip=["player_name", "country", feature, TARGET_COL],
        )
        .properties(height=380)
        .interactive()
    )


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


def render_prediction_tab(dataset: pd.DataFrame | None, dataset_error: str | None) -> None:
    gr.Markdown("## Prediction Studio")
    if dataset_error or dataset is None:
        gr.Markdown(
            f"❗ **Datenquelle fehlt.** Verbindung zur Datenbank/CSV fehlgeschlagen: {dataset_error or 'unbekannter Fehler'}."
        )
        return

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


def render_insights_tab(dataset: pd.DataFrame | None, dataset_error: str | None) -> None:
    gr.Markdown("## Insights & EDA Board")
    if dataset_error or dataset is None:
        gr.Markdown(
            f"❗ **Keine Datenbasis für die Visualisierung.** Bitte ETL laufen lassen: {dataset_error or 'unbekannter Fehler'}."
        )
        return

    metrics_df = build_metrics_dataframe()
    with gr.Accordion("Modell-Leaderboard", open=True):
        if metrics_df is None:
            gr.Markdown("Noch keine Trainingsmetriken gefunden. Führe `pipeline.train` aus.")
        else:
            gr.Dataframe(value=metrics_df, interactive=False, label="Trainingsmetriken")
            gr.Markdown(build_metric_story(metrics_df))

    top_players = build_top_players_table(dataset)
    with gr.Accordion("Top 15 Spieler nach FDI", open=True):
        if top_players.empty:
            gr.Markdown("Keine Spieler im Cache gefunden.")
        else:
            gr.Dataframe(value=top_players, interactive=False, label="Top-Spieler")

    gr.Markdown("### Feature Explorer")
    feature_dropdown = gr.Dropdown(
        choices=INSIGHT_FEATURE_CHOICES,
        value=INSIGHT_FEATURE_CHOICES[0],
        label="Feature gegen FDI plotten",
    )
    scatter_plot = gr.Plot(value=render_scatter_plot(INSIGHT_FEATURE_CHOICES[0]))
    feature_dropdown.change(fn=render_scatter_plot, inputs=feature_dropdown, outputs=scatter_plot)


def build_interface():
    with gr.Blocks(title="FDI Analytics Command Center") as demo:
        try:
            dataset = refresh_data_cache()
            dataset_error = None
        except Exception as exc:
            dataset = None
            dataset_error = str(exc)

        with gr.Tabs():
            with gr.Tab("Prediction Studio"):
                render_prediction_tab(dataset, dataset_error)
            with gr.Tab("Insights & EDA"):
                render_insights_tab(dataset, dataset_error)

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
