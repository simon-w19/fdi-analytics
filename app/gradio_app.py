"""FDI Rating service combining Gradio UI and FastAPI endpoints.

Features:
- Prediction Studio: Einzelvorhersagen mit Spieler-Presets und FDI-Vergleich
- What-If Comparison: Zwei Spieler nebeneinander vergleichen
- Feature Contributions: Visualisierung der Feature-Beiträge
- Model Info Panel: Aktive Modelldetails
- Export: Vorhersagen als JSON herunterladen
- Insights & EDA: Interaktive Datenexploration
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import altair as alt
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
from pipeline.features import (
    FEATURE_COLUMNS,
    NUMERIC_INPUT_COLUMNS,
    TARGET_COL,
    build_prediction_frame,
    default_input_vector,
    engineer_features,
)

alt.data_transformers.disable_max_rows()

LOGGER = logging.getLogger("app.gradio")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "player_stats_all.csv"
PIPELINE_PATH = PROJECT_ROOT / "models" / "best_fdi_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "latest_metrics.json"
ENGINE: Engine | None = None
DATA_CACHE: pd.DataFrame | None = None
PLAYER_NAME_SET: set[str] = set()  # Fast lookup cache
PLAYER_CHOICES: List[str] = ["Freie Eingabe"]
TOP_PLAYER_CHOICES: List[str] = []  # Top 100 players for quick dropdown
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


# =============================================================================
# MODEL INFO & METRICS HELPERS
# =============================================================================

def get_model_info() -> Dict[str, Any]:
    """Extract model metadata for the info panel."""
    info = {
        "model_type": "Unbekannt",
        "n_features": len(FEATURE_COLUMNS),
        "hyperparameters": {},
        "training_date": "Unbekannt",
        "performance": {},
    }
    
    if PIPELINE is not None:
        try:
            model = PIPELINE.named_steps.get("model")
            if model is not None:
                info["model_type"] = type(model).__name__
                params = model.get_params()
                relevant_keys = ["alpha", "max_iter", "n_estimators", "max_depth", "min_samples_leaf"]
                info["hyperparameters"] = {k: v for k, v in params.items() if k in relevant_keys}
        except Exception as e:
            LOGGER.warning("Could not extract model info: %s", e)
    
    payload = load_metrics_payload()
    if payload:
        info["training_date"] = payload.get("timestamp", "Unbekannt")
        best_model = payload.get("best_model", "")
        metrics = payload.get("metrics", {})
        if best_model and best_model in metrics:
            info["performance"] = metrics[best_model]
    
    return info


def build_model_info_markdown() -> str:
    """Generate markdown summary for the model info panel."""
    info = get_model_info()
    
    lines = [
        "### 🤖 Aktives Modell",
        f"**Typ:** {info['model_type']}",
        f"**Features:** {info['n_features']}",
    ]
    
    if info["hyperparameters"]:
        params_str = ", ".join(f"{k}={v}" for k, v in info["hyperparameters"].items())
        lines.append(f"**Hyperparameter:** {params_str}")
    
    perf = info["performance"]
    if perf:
        lines.append("")
        lines.append("### 📊 Performance (Test-Set)")
        if isinstance(perf.get("r2"), (int, float)):
            lines.append(f"- **R²:** {perf['r2']:.4f}")
        if isinstance(perf.get("mae"), (int, float)):
            lines.append(f"- **MAE:** {perf['mae']:.2f} FDI-Punkte")
        if isinstance(perf.get("rmse"), (int, float)):
            lines.append(f"- **RMSE:** {perf['rmse']:.2f}")
        if perf.get("cv_mae_mean"):
            lines.append(f"- **CV MAE:** {perf['cv_mae_mean']:.2f} ± {perf.get('cv_mae_std', 0):.2f}")
    
    if info["training_date"] != "Unbekannt":
        lines.append("")
        lines.append(f"*Trainiert: {str(info['training_date'])[:19]}*")
    
    return "\n".join(lines)


# =============================================================================
# FEATURE CONTRIBUTION ANALYSIS
# =============================================================================

def compute_feature_contributions(pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute approximate feature contributions using coefficient-based attribution.
    For Lasso/Linear models, this shows the scaled contribution of each feature.
    """
    if pipeline is None:
        return pd.DataFrame()
    
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocess"]
        
        feature_names = preprocessor.get_feature_names_out()
        X_transformed = preprocessor.transform(features)
        
        if hasattr(model, "coef_"):
            coefs = np.ravel(model.coef_)
            if hasattr(X_transformed, "toarray"):
                X_vals = X_transformed.toarray().ravel()
            else:
                X_vals = np.ravel(X_transformed)
            
            contributions = coefs * X_vals
            
            df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": coefs,
                "Value": X_vals,
                "Contribution": contributions,
            })
            df["Abs_Contribution"] = df["Contribution"].abs()
            return df.sort_values("Abs_Contribution", ascending=False).head(15)
        
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances,
                "Contribution": importances,
            })
            df["Abs_Contribution"] = df["Contribution"].abs()
            return df.sort_values("Abs_Contribution", ascending=False).head(15)
    
    except Exception as e:
        LOGGER.warning("Could not compute contributions: %s", e)
    
    return pd.DataFrame()


def render_contribution_chart(contrib_df: pd.DataFrame | None):
    """Render a waterfall-style contribution chart."""
    if contrib_df is None or contrib_df.empty:
        fallback = pd.DataFrame({"Hinweis": ["Keine Feature-Beiträge verfügbar."]})
        return alt.Chart(fallback).mark_text(size=14).encode(text="Hinweis:N")
    
    plot_df = contrib_df.head(12).copy()
    plot_df = plot_df.sort_values("Contribution")
    
    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Contribution:Q", title="Beitrag zum FDI-Rating"),
            y=alt.Y("Feature:N", sort=plot_df["Feature"].tolist(), title="Feature"),
            color=alt.condition(
                "datum.Contribution > 0",
                alt.value("#00b894"),
                alt.value("#d63031"),
            ),
            tooltip=[
                "Feature",
                alt.Tooltip("Contribution:Q", format=".2f"),
            ],
        )
        .properties(height=350, title="Feature-Beiträge zur Vorhersage")
    )


# =============================================================================
# EXPORT FUNCTIONALITY
# =============================================================================

def export_prediction_json(
    player_name: str, country: str, predicted_fdi: float, 
    actual_fdi: float | None, features: Dict[str, float]
) -> str:
    """Export prediction as JSON string."""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "player_name": player_name,
        "country": country,
        "predicted_fdi": round(predicted_fdi, 2),
        "actual_fdi_dartsorakel": actual_fdi,
        "delta": round(predicted_fdi - actual_fdi, 2) if actual_fdi else None,
        "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
        "model_info": get_model_info(),
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)


# =============================================================================
# PLAYER DATA HELPERS
# =============================================================================

def _get_player_row(player_name: str, dataset: pd.DataFrame) -> pd.Series | None:
    """Get a player's data row from the dataset."""
    # Use cached set for O(1) lookup
    if player_name and player_name != "Freie Eingabe" and player_name in PLAYER_NAME_SET:
        matches = dataset.loc[dataset["player_name"] == player_name]
        if not matches.empty:
            return matches.iloc[0]
    return None


def get_player_actual_fdi(player_name: str, dataset: pd.DataFrame) -> float | None:
    """Get the actual DartsOrakel FDI rating for a player."""
    row = _get_player_row(player_name, dataset)
    if row is not None and TARGET_COL in row.index:
        val = row[TARGET_COL]
        if pd.notna(val):
            return float(val)
    return None


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================


def build_dataset_snapshot(dataset: pd.DataFrame | None) -> str:
    if dataset is None or dataset.empty:
        return "Keine Daten im Cache – ETL erneut starten."
    target = dataset[TARGET_COL]
    target_iqr = target.quantile(0.75) - target.quantile(0.25)
    scraped_info = "Unbekannt"
    if "scraped_at" in dataset.columns:
        last_scrape = pd.to_datetime(dataset["scraped_at"], errors="coerce").max()
        if pd.notna(last_scrape):
            scraped_info = f"{last_scrape.tz_localize('UTC') if last_scrape.tzinfo is None else last_scrape}"  # best-effort string
    lines = [
        f"- **Beobachtungen**: {len(dataset):,}",
        f"- **Länder**: {dataset['country'].nunique() if 'country' in dataset.columns else 'N/A'}",
        f"- **Features (numeric/kategorial)**: {len(FEATURE_COLUMNS) - 1} / 1",
        f"- **FDI Ø / Median**: {target.mean():.1f} / {target.median():.1f}",
        f"- **FDI IQR**: {target_iqr:.1f}",
        f"- **Letzter Scrape**: {scraped_info}",
    ]
    return "\n".join(lines)


def build_correlation_table(dataset: pd.DataFrame | None, top_n: int = 10) -> pd.DataFrame:
    if dataset is None or dataset.empty:
        return pd.DataFrame()
    numeric_cols = dataset.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != TARGET_COL]
    if not numeric_cols:
        return pd.DataFrame()
    corr_series = dataset[numeric_cols].corrwith(dataset[TARGET_COL]).dropna()
    if corr_series.empty:
        return pd.DataFrame()
    corr_df = (
        pd.DataFrame({"Feature": corr_series.index, "Correlation": corr_series.values})
        .assign(abs_corr=lambda d: d["Correlation"].abs())
        .sort_values("abs_corr", ascending=False)
        .drop(columns="abs_corr")
        .head(top_n)
    )
    return corr_df


def render_correlation_chart(corr_df: pd.DataFrame | None):
    if corr_df is None or corr_df.empty:
        fallback = pd.DataFrame({"Hinweis": ["Keine Korrelationen berechnet."]})
        return alt.Chart(fallback).mark_text(size=14).encode(text="Hinweis:N")
    ordered = corr_df.sort_values("Correlation")
    scale = alt.Scale(domain=[-1, 1])
    return (
        alt.Chart(ordered)
        .mark_bar()
        .encode(
            x=alt.X("Correlation:Q", scale=scale, title="Pearson-Korrelation"),
            y=alt.Y("Feature:N", sort=ordered["Feature"].tolist(), title="Feature"),
            color=alt.condition("datum.Correlation > 0", alt.value("#00b894"), alt.value("#d63031")),
            tooltip=["Feature", alt.Tooltip("Correlation", format=".3f")],
        )
        .properties(height=300)
    )


def build_country_summary(dataset: pd.DataFrame | None, top_n: int = 10) -> pd.DataFrame:
    if dataset is None or dataset.empty or "country" not in dataset.columns:
        return pd.DataFrame()
    grouping = dataset.groupby("country")
    players = grouping.size().rename("players")
    avg_fdi = grouping[TARGET_COL].mean().rename("avg_fdi")
    summary = (
        pd.concat([players, avg_fdi], axis=1)
        .sort_values("avg_fdi", ascending=False)
        .head(top_n)
        .reset_index()
    )
    summary["avg_fdi"] = summary["avg_fdi"].round(1)
    return summary.rename(columns={"country": "Country", "players": "Players", "avg_fdi": "Avg FDI"})


def render_country_chart(country_df: pd.DataFrame | None):
    if country_df is None or country_df.empty:
        fallback = pd.DataFrame({"Hinweis": ["Keine Länderstatistiken verfügbar."]})
        return alt.Chart(fallback).mark_text(size=14).encode(text="Hinweis:N")
    ordered = country_df.sort_values("Avg FDI", ascending=True)
    return (
        alt.Chart(ordered)
        .mark_bar()
        .encode(
            x=alt.X("Avg FDI:Q", title="Ø FDI"),
            y=alt.Y("Country:N", sort=ordered["Country"].tolist()),
            color=alt.Color("Players:Q", title="Spielerzahl", scale=alt.Scale(scheme="blues")),
            tooltip=["Country", "Players", "Avg FDI"],
        )
        .properties(height=300)
    )


def build_category_frequency(dataset: pd.DataFrame | None) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    if dataset is None or dataset.empty:
        return tables
    if "country" in dataset.columns:
        freq = dataset["country"].fillna("UNK").value_counts().reset_index()
        freq.columns = ["Country", "Frequency"]
        tables["country"] = freq
    return tables


def extract_feature_importances(pipeline, top_n: int = 15) -> pd.DataFrame:
    if pipeline is None:
        return pd.DataFrame()
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()
        model = pipeline.named_steps["model"]
    except Exception:
        return pd.DataFrame()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.ravel(model.coef_)
    else:
        return pd.DataFrame()
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.assign(abs_importance=lambda d: d["Importance"].abs())
    return df.sort_values("abs_importance", ascending=False).drop(columns="abs_importance").head(top_n)


def render_importance_chart(importance_df: pd.DataFrame | None):
    if importance_df is None or importance_df.empty:
        fallback = pd.DataFrame({"Hinweis": ["Keine Feature-Importances verfügbar."]})
        return alt.Chart(fallback).mark_text(size=14).encode(text="Hinweis:N")
    ordered = importance_df.sort_values("Importance")
    return (
        alt.Chart(ordered)
        .mark_bar()
        .encode(
            x=alt.X("Importance:Q", title="Modellgewichtung"),
            y=alt.Y("Feature:N", sort=ordered["Feature"].tolist()),
            color=alt.condition("datum.Importance > 0", alt.value("#6c5ce7"), alt.value("#fdcb6e")),
            tooltip=["Feature", alt.Tooltip("Importance", format=".4f")],
        )
        .properties(height=320)
    )


def build_feature_story(importance_df: pd.DataFrame | None) -> str:
    if importance_df is None or importance_df.empty:
        return "Kein Modell geladen – bitte Training abschließen."
    top = importance_df.iloc[0]
    tail = importance_df.iloc[-1]
    lines = [
        f"- **{top['Feature']}** dominiert die Vorhersage (höchste absolute Gewichtung).",
        f"- **{tail['Feature']}** liefert den geringsten Beitrag und kann für Sparmodelle reduziert werden.",
    ]
    return "\n".join(lines)


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
    
    # Sample for performance if dataset is large (>500 points slow down browser)
    max_points = 500
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    
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
    global DATA_CACHE, PLAYER_CHOICES, COUNTRY_CHOICES, PLAYER_NAME_SET, TOP_PLAYER_CHOICES
    if force or DATA_CACHE is None:
        DATA_CACHE = load_dataset()
        player_names = sorted(DATA_CACHE["player_name"].dropna().unique().tolist())
        PLAYER_NAME_SET = set(player_names)  # Fast O(1) lookup
        PLAYER_CHOICES = ["Freie Eingabe"] + player_names
        
        # Top 100 players by FDI for quick dropdown (much faster than 3000 items)
        top_players = (
            DATA_CACHE.nlargest(100, TARGET_COL)["player_name"]
            .dropna()
            .tolist()
        )
        TOP_PLAYER_CHOICES = ["-- Tippe Spielernamen --"] + top_players
        
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
    # Use cached set for O(1) lookup instead of O(n) list conversion
    if selection != "Freie Eingabe" and selection in PLAYER_NAME_SET:
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
    values = ["nan" if pd.isna(series[col]) else f"{series[col]:.4f}" for col in FEATURE_COLUMNS]
    return pd.DataFrame({"Feature": FEATURE_COLUMNS, "Wert": values})


def handle_prefill(selection: str) -> List:
    """Handle player selection and prefill all inputs including actual FDI."""
    # Handle empty or placeholder selections
    if not selection or selection in ("-- Tippe Spielernamen --", "Freie Eingabe"):
        return ["FDI Prospect", "UNK", "N/A"] + [0.0] * len(NUMERIC_INPUT_COLUMNS)
    
    # Use cached dataset - don't reload unless necessary
    dataset = DATA_CACHE if DATA_CACHE is not None else refresh_data_cache()
    row = _get_prefilled_row(selection.strip(), dataset)
    
    # Get actual FDI for comparison
    actual_fdi = get_player_actual_fdi(selection, dataset)
    actual_fdi_display = f"{actual_fdi:.1f}" if actual_fdi else "N/A"
    
    numeric_values = [float(row.get(col, 0.0) or 0.0) for col in NUMERIC_INPUT_COLUMNS]
    
    return [
        str(row.get("player_name", "FDI Prospect")),
        row.get("country", "UNK"),
        actual_fdi_display,
    ] + numeric_values


def handle_prediction(player_name: str, country: str, actual_fdi_str: str, *numeric_values: Sequence[float]):
    """Handle prediction with contribution analysis and comparison."""
    if PIPELINE is None:
        raise gr.Error("Kein Modellartefakt gefunden. Bitte exportiere best_fdi_pipeline.joblib im Notebook.")
    
    # Build feature payload
    payload = {}
    for col, val in zip(NUMERIC_INPUT_COLUMNS, numeric_values):
        payload[col] = float(val if val is not None else 0.0)
    payload["player_name"] = player_name or "FDI Prospect"
    payload["country"] = country or "UNK"
    
    # Get prediction
    feature_frame = build_prediction_frame(payload)
    rating = predict_rating(PIPELINE, feature_frame)
    
    # Parse actual FDI for comparison
    try:
        actual_fdi = float(actual_fdi_str) if actual_fdi_str and actual_fdi_str != "N/A" else None
    except ValueError:
        actual_fdi = None
    
    # Compute delta
    if actual_fdi is not None:
        delta = rating - actual_fdi
        comparison_text = f"**Δ = {delta:+.1f}** (Modell {'über' if delta > 0 else 'unter'}schätzt)"
    else:
        comparison_text = "Kein DartsOrakel-Vergleichswert verfügbar"
    
    # Feature contributions
    contrib_df = compute_feature_contributions(PIPELINE, feature_frame)
    contrib_chart = render_contribution_chart(contrib_df)
    
    # Feature table
    feature_table = _prepare_feature_table(feature_frame)
    
    # Export JSON
    export_json = export_prediction_json(player_name, country, rating, actual_fdi, payload)
    
    return (
        round(rating, 1),
        actual_fdi if actual_fdi else "N/A",
        comparison_text,
        contrib_chart,
        feature_table,
        export_json,
    )


# =============================================================================
# WHAT-IF COMPARISON
# =============================================================================

def handle_whatif_comparison(player1: str, player2: str):
    """Compare two players side by side."""
    if PIPELINE is None:
        raise gr.Error("Kein Modellartefakt geladen.")
    
    # Use cached dataset
    dataset = DATA_CACHE if DATA_CACHE is not None else refresh_data_cache()
    
    results = []
    for player_name in [player1, player2]:
        row = _get_prefilled_row(player_name, dataset)
        
        # Build payload
        payload = {}
        for col in NUMERIC_INPUT_COLUMNS:
            payload[col] = float(row.get(col, 0.0) or 0.0)
        payload["player_name"] = str(row.get("player_name", player_name))
        payload["country"] = row.get("country", "UNK")
        
        # Predict
        feature_frame = build_prediction_frame(payload)
        predicted = predict_rating(PIPELINE, feature_frame)
        actual = get_player_actual_fdi(player_name, dataset)
        
        results.append({
            "Spieler": payload["player_name"],
            "Land": payload["country"],
            "Modell-FDI": round(predicted, 1),
            "DartsOrakel-FDI": actual if actual else "N/A",
            "Delta": round(predicted - actual, 1) if actual else "N/A",
            "3-Dart Avg": round(payload.get("last_12_months_averages", 0), 2),
            "First-9 Avg": round(payload.get("last_12_months_first_9_averages", 0), 2),
            "Checkout %": round(payload.get("last_12_months_checkout_pcnt", 0), 2),
            "Legs Won %": round(payload.get("last_12_months_pcnt_legs_won", 0), 2),
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Create difference visualization
    if len(results) == 2:
        p1, p2 = results
        diff_data = []
        compare_keys = ["3-Dart Avg", "First-9 Avg", "Checkout %", "Legs Won %"]
        for key in compare_keys:
            v1, v2 = p1.get(key, 0), p2.get(key, 0)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                diff_data.append({
                    "Metrik": key,
                    "Spieler 1": v1,
                    "Spieler 2": v2,
                    "Differenz": v1 - v2,
                })
        diff_df = pd.DataFrame(diff_data)
        
        diff_chart = (
            alt.Chart(diff_df)
            .mark_bar()
            .encode(
                x=alt.X("Differenz:Q", title=f"{p1['Spieler']} - {p2['Spieler']}"),
                y=alt.Y("Metrik:N", sort=compare_keys),
                color=alt.condition(
                    "datum.Differenz > 0",
                    alt.value("#0984e3"),
                    alt.value("#e17055"),
                ),
                tooltip=["Metrik", "Spieler 1", "Spieler 2", "Differenz"],
            )
            .properties(height=200, title="Feature-Unterschiede")
        )
    else:
        diff_chart = alt.Chart(pd.DataFrame()).mark_text()
    
    # Summary text
    if len(results) == 2 and isinstance(results[0]["Modell-FDI"], (int, float)) and isinstance(results[1]["Modell-FDI"], (int, float)):
        fdi_diff = results[0]["Modell-FDI"] - results[1]["Modell-FDI"]
        summary = f"**{results[0]['Spieler']}** wird um **{abs(fdi_diff):.1f} FDI-Punkte** {'höher' if fdi_diff > 0 else 'niedriger'} eingeschätzt als **{results[1]['Spieler']}**."
    else:
        summary = "Vergleich nicht möglich."
    
    return comparison_df, diff_chart, summary


def render_prediction_tab(dataset: pd.DataFrame | None, dataset_error: str | None) -> None:
    gr.Markdown("## 🎯 Prediction Studio")
    gr.Markdown("Wähle einen Spieler aus der Top-100 Liste oder tippe einen Namen ein.")
    
    if dataset_error or dataset is None:
        gr.Markdown(f"❗ **Datenquelle fehlt.** {dataset_error or 'unbekannter Fehler'}.")
        return
    
    # Model info panel in sidebar
    with gr.Row():
        with gr.Column(scale=3):
            # Player selection - use Top 100 dropdown (fast!) + search textbox
            with gr.Row():
                player_selector = gr.Dropdown(
                    choices=TOP_PLAYER_CHOICES,
                    value=TOP_PLAYER_CHOICES[0],
                    label="🎯 Top 100 Spieler (schnelle Auswahl)",
                    scale=2,
                )
                player_search = gr.Textbox(
                    label="🔍 Spieler suchen (beliebiger Name)",
                    placeholder="z.B. Michael van Gerwen",
                    scale=2,
                )
            
            with gr.Row():
                player_name_box = gr.Textbox(label="Spielername", value="FDI Prospect")
                country_dropdown = gr.Dropdown(choices=COUNTRY_CHOICES, value="UNK", label="Land")
                actual_fdi_box = gr.Textbox(label="DartsOrakel FDI (Referenz)", value="N/A", interactive=False)
        
        with gr.Column(scale=1):
            gr.Markdown(build_model_info_markdown())
    
    # Numeric inputs in collapsible section
    with gr.Accordion("📊 Feature-Eingaben (werden bei Spielerauswahl automatisch gefüllt)", open=False):
        numeric_components: List[gr.Number] = []
        chunk_size = 4
        for idx in range(0, len(NUMERIC_INPUT_COLUMNS), chunk_size):
            with gr.Row():
                for column in NUMERIC_INPUT_COLUMNS[idx : idx + chunk_size]:
                    comp = gr.Number(label=column, value=0.0)
                    numeric_components.append(comp)
    
    # Prediction button and outputs
    predict_button = gr.Button("🔮 FDI Rating vorhersagen", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            prediction_output = gr.Number(label="📈 Modell-Vorhersage", interactive=False)
        with gr.Column(scale=1):
            actual_display = gr.Textbox(label="📊 DartsOrakel (Referenz)", interactive=False)
        with gr.Column(scale=1):
            comparison_text = gr.Markdown("Warte auf Vorhersage...")
    
    # Contribution chart
    gr.Markdown("### 📈 Feature-Beiträge zur Vorhersage")
    contribution_chart = gr.Plot(label="Feature Contributions")
    
    # Feature table and export
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("📋 Vollständiger Feature-Vektor", open=False):
                feature_table = gr.Dataframe(
                    headers=["Feature", "Wert"],
                    datatype=["str", "str"],
                    label="Genutzter Feature-Vektor",
                    interactive=False,
                )
        with gr.Column(scale=1):
            with gr.Accordion("💾 Export (JSON)", open=False):
                export_json_box = gr.Code(language="json", label="JSON Export")
                gr.Markdown("*Kopiere den JSON-Inhalt für externe Verwendung.*")
    
    # Event handlers
    player_selector.change(
        fn=handle_prefill,
        inputs=player_selector,
        outputs=[player_name_box, country_dropdown, actual_fdi_box] + numeric_components,
    )
    
    # Search box also triggers prefill
    player_search.submit(
        fn=handle_prefill,
        inputs=player_search,
        outputs=[player_name_box, country_dropdown, actual_fdi_box] + numeric_components,
    )
    
    predict_button.click(
        fn=handle_prediction,
        inputs=[player_name_box, country_dropdown, actual_fdi_box] + numeric_components,
        outputs=[prediction_output, actual_display, comparison_text, contribution_chart, feature_table, export_json_box],
    )


def render_whatif_tab(dataset: pd.DataFrame | None, dataset_error: str | None) -> None:
    gr.Markdown("## ⚖️ What-If Vergleich")
    gr.Markdown("Vergleiche zwei Spieler aus der Top-100 oder tippe Namen ein.")
    
    if dataset_error or dataset is None:
        gr.Markdown(f"❗ **Datenquelle fehlt.** {dataset_error or 'unbekannter Fehler'}.")
        return
    
    gr.Markdown("### Schnellauswahl (Top 100)")
    with gr.Row():
        player1_dropdown = gr.Dropdown(
            choices=TOP_PLAYER_CHOICES[1:] if len(TOP_PLAYER_CHOICES) > 1 else TOP_PLAYER_CHOICES,
            value=TOP_PLAYER_CHOICES[1] if len(TOP_PLAYER_CHOICES) > 1 else None,
            label="🎯 Spieler 1",
        )
        player2_dropdown = gr.Dropdown(
            choices=TOP_PLAYER_CHOICES[1:] if len(TOP_PLAYER_CHOICES) > 1 else TOP_PLAYER_CHOICES,
            value=TOP_PLAYER_CHOICES[2] if len(TOP_PLAYER_CHOICES) > 2 else None,
            label="🎯 Spieler 2",
        )
    
    gr.Markdown("### Oder: Beliebigen Spieler suchen")
    with gr.Row():
        player1_search = gr.Textbox(label="Spieler 1 (Name eingeben)", placeholder="z.B. Peter Wright")
        player2_search = gr.Textbox(label="Spieler 2 (Name eingeben)", placeholder="z.B. Gary Anderson")
    
    compare_button = gr.Button("🔄 Vergleichen", variant="primary")
    
    comparison_summary = gr.Markdown("Wähle zwei Spieler und klicke 'Vergleichen'.")
    comparison_table = gr.Dataframe(label="Vergleichstabelle", interactive=False)
    difference_chart = gr.Plot(label="Feature-Unterschiede")
    
    def compare_with_fallback(p1_dropdown, p2_dropdown, p1_search, p2_search):
        """Use search box if filled, otherwise dropdown."""
        player1 = p1_search.strip() if p1_search and p1_search.strip() else p1_dropdown
        player2 = p2_search.strip() if p2_search and p2_search.strip() else p2_dropdown
        return handle_whatif_comparison(player1, player2)
    
    compare_button.click(
        fn=compare_with_fallback,
        inputs=[player1_dropdown, player2_dropdown, player1_search, player2_search],
        outputs=[comparison_table, difference_chart, comparison_summary],
    )


def render_insights_tab(dataset: pd.DataFrame | None, dataset_error: str | None) -> None:
    gr.Markdown("## Insights & EDA Board")
    if dataset_error or dataset is None:
        gr.Markdown(
            f"❗ **Keine Datenbasis für die Visualisierung.** Bitte ETL laufen lassen: {dataset_error or 'unbekannter Fehler'}."
        )
        return

    gr.Markdown("### Daten-Snapshot")
    gr.Markdown(build_dataset_snapshot(dataset))

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

    corr_df = build_correlation_table(dataset)
    with gr.Accordion("Feature-Korrelationen zum FDI", open=False):
        if corr_df.empty:
            gr.Markdown("Keine numerischen Features für die Korrelationsanalyse gefunden.")
        else:
            gr.Plot(value=render_correlation_chart(corr_df))
            gr.Dataframe(value=corr_df, interactive=False, label="Top-Korrelationen")

    country_df = build_country_summary(dataset)
    with gr.Accordion("Country Performance Board", open=False):
        if country_df.empty:
            gr.Markdown("Noch keine Länderkennzahlen verfügbar.")
        else:
            gr.Plot(value=render_country_chart(country_df))
            gr.Dataframe(value=country_df, interactive=False, label="FDI nach Land")

    freq_tables = build_category_frequency(dataset)
    with gr.Accordion("Kategorie-Häufigkeiten", open=False):
        if not freq_tables:
            gr.Markdown("Keine kategorialen Features für eine Häufigkeitsanalyse verfügbar.")
        else:
            for name, table in freq_tables.items():
                gr.Markdown(f"**{name.title()}**")
                gr.Dataframe(value=table, interactive=False, label=f"{name.title()} Frequency")

    importance_df = extract_feature_importances(PIPELINE)
    with gr.Accordion("Model Feature Impact", open=False):
        if importance_df.empty:
            gr.Markdown("Kein Modellartefakt geladen oder das Modell liefert keine Importances.")
        else:
            gr.Plot(value=render_importance_chart(importance_df))
            gr.Dataframe(value=importance_df, interactive=False, label="Feature-Gewichtungen")
            gr.Markdown(build_feature_story(importance_df))

    gr.Markdown("### Feature Explorer")
    feature_dropdown = gr.Dropdown(
        choices=INSIGHT_FEATURE_CHOICES,
        value=INSIGHT_FEATURE_CHOICES[0],
        label="Feature gegen FDI plotten",
    )
    scatter_plot = gr.Plot(value=render_scatter_plot(INSIGHT_FEATURE_CHOICES[0]))
    feature_dropdown.change(fn=render_scatter_plot, inputs=feature_dropdown, outputs=scatter_plot)


def build_interface():
    with gr.Blocks(title="FDI Analytics Command Center", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 FDI Analytics Command Center")
        gr.Markdown("*Vorhersage und Analyse von Darts-Spieler-Ratings im Vergleich zu DartsOrakel*")
        
        try:
            dataset = refresh_data_cache()
            dataset_error = None
        except Exception as exc:
            dataset = None
            dataset_error = str(exc)

        with gr.Tabs():
            with gr.Tab("🎯 Prediction Studio"):
                render_prediction_tab(dataset, dataset_error)
            with gr.Tab("⚖️ What-If Vergleich"):
                render_whatif_tab(dataset, dataset_error)
            with gr.Tab("📊 Insights & EDA"):
                render_insights_tab(dataset, dataset_error)

    return demo


def register_api_routes(api: FastAPI) -> None:
    @api.get("/api/health")
    def healthcheck():
        rows = len(DATA_CACHE) if DATA_CACHE is not None else 0
        model_info = get_model_info()
        return {
            "status": "ok",
            "has_model": PIPELINE is not None,
            "model_type": model_info["model_type"],
            "rows_cached": rows,
            "schema": settings.db_schema,
            "table": settings.table_name,
        }

    @api.get("/api/model-info")
    def model_info_endpoint():
        return get_model_info()

    @api.get("/api/players")
    def list_players(limit: int = 50):
        dataset = refresh_data_cache()
        limited = dataset[["player_name", "country", TARGET_COL]].head(limit).fillna("UNK")
        return limited.to_dict(orient="records")

    @api.get("/api/player/{player_name}")
    def get_player(player_name: str):
        dataset = refresh_data_cache()
        row = _get_player_row(player_name, dataset)
        if row is None:
            raise HTTPException(status_code=404, detail="Player not found")
        return row.to_dict()

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
        
        # Get actual FDI if player exists
        dataset = refresh_data_cache()
        actual_fdi = get_player_actual_fdi(request.player_name, dataset) if request.player_name else None
        
        return {
            "prediction": rating,
            "actual_fdi_dartsorakel": actual_fdi,
            "delta": round(rating - actual_fdi, 2) if actual_fdi else None,
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
