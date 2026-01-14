"""Train and evaluate the FDI rating prediction pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import settings
from .diagnostics import run_full_diagnostics
from .features import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COL,
    engineer_features,
)

try:  # pragma: no cover - optional in testing
    import mlflow
except ImportError:  # pragma: no cover - handled by guards
    mlflow = None

LOGGER = logging.getLogger("pipeline.train")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
PARAM_GRIDS: Dict[str, Dict[str, list[Any]]] = {
    "lasso": {
        "model__alpha": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "model__max_iter": [20000, 40000],
    },
}


def _mlflow_enabled() -> bool:
    return bool(settings.mlflow_tracking_uri) and settings.mlflow_enabled and mlflow is not None


def _mlflow_run(run_name: str):
    if not _mlflow_enabled():
        return nullcontext()
    assert mlflow is not None  # for type checking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return mlflow.start_run(run_name=run_name)


def _log_mlflow_payload(params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    if not _mlflow_enabled():
        return
    assert mlflow is not None
    mlflow.log_params({key: str(value) for key, value in params.items()})
    numeric_metrics = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)


def tune_model(name: str, model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, Dict[str, Any]]:
    grid = PARAM_GRIDS.get(name)
    if not grid:
        model.fit(x_train, y_train)
        return model, {"best_params": None}
    search = GridSearchCV(
        model,
        param_grid=grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    best_mae = float(-search.best_score_)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "tuning_cv_mae": best_mae,
    }


def load_dataset(csv_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else settings.processed_csv_path
    LOGGER.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    df = engineer_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL])
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_features = [col for col in FEATURE_COLUMNS if col not in NUMERIC_FEATURES]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "lasso": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Lasso(alpha=0.001, max_iter=20000, random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> Tuple[float, float, float]:
    predictions = model.predict(x_valid)
    mae = mean_absolute_error(y_valid, predictions)
    mse = mean_squared_error(y_valid, predictions)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_valid, predictions)
    return mae, rmse, r2


def persist_artifacts(
    model: Pipeline,
    metrics: Dict[str, Dict[str, Any]],
    model_path: Path,
    metrics_path: Path,
    diagnostics: Dict[str, Any] | None = None,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    LOGGER.info("Persisted pipeline to %s", model_path)

    metrics_payload = {
        "metrics": metrics,
        "diagnostics": diagnostics or {},
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote metrics to %s", metrics_path)


def train(csv_path: Path | str | None = None, model_path: Path | None = None, metrics_path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    df = load_dataset(csv_path)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    leaderboard: Dict[str, Dict[str, Any]] = {}
    best_name: str | None = None
    best_model: Pipeline | None = None
    best_score = float("inf")

    for name, model in models.items():
        LOGGER.info("Training %s", name)
        with _mlflow_run(run_name=name):
            tuned_model, tuning_meta = tune_model(name, model, x_train, y_train)
            mae, rmse, r2 = evaluate_model(tuned_model, x_valid, y_valid)
            cv_scores = cross_val_score(tuned_model, x_train, y_train, cv=5, scoring="neg_mean_absolute_error")
            metrics_payload = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "cv_mae_mean": float(-cv_scores.mean()),
                "cv_mae_std": float(cv_scores.std()),
            }
            if tuning_meta.get("best_params"):
                metrics_payload["best_params"] = tuning_meta["best_params"]
            if "tuning_cv_mae" in tuning_meta:
                metrics_payload["tuning_cv_mae"] = tuning_meta["tuning_cv_mae"]
            leaderboard[name] = metrics_payload
            params_payload: Dict[str, Any] = {"model": name}
            best_params = tuning_meta.get("best_params") or {}
            for key, value in best_params.items():
                params_payload[f"best_{key}"] = str(value)
            _log_mlflow_payload(params_payload, metrics_payload)
        LOGGER.info("%s -> MAE %.3f | RMSE %.3f | R2 %.3f", name, mae, rmse, r2)
        if mae < best_score:
            best_score = mae
            best_model = tuned_model
            best_name = name

    if best_model is None or best_name is None:
        raise RuntimeError("Model training failed; no candidate produced a score.")

    LOGGER.info("Best model: %s (MAE %.3f)", best_name, best_score)
    
    # Run comprehensive diagnostics on best model
    LOGGER.info("Running diagnostics on best model...")
    try:
        # Count features: Get number of features after preprocessing
        preprocessor = best_model.named_steps["preprocessor"]
        preprocessor.fit(x_train)
        feature_names = preprocessor.get_feature_names_out()
        n_features = len(feature_names)
        
        diagnostics = run_full_diagnostics(
            best_model,
            x_train,
            y_train,
            x_valid,
            y_valid,
            n_features,
        )
        diagnostics["best_model_name"] = best_name
        LOGGER.info("Diagnostics: Adjusted R² = %.4f", diagnostics.get("adjusted_r2", 0))
    except Exception as e:
        LOGGER.warning("Could not run diagnostics: %s", e)
        diagnostics = {}
    
    artefact = model_path or Path("models/best_fdi_pipeline.joblib")
    metrics_file = metrics_path or Path("reports/metrics/latest_metrics.json")
    persist_artifacts(best_model, leaderboard, artefact, metrics_file, diagnostics)
    if _mlflow_enabled():
        assert mlflow is not None
        mlflow.log_artifact(str(artefact))
        mlflow.log_artifact(str(metrics_file))
        mlflow.set_tag("best_model", best_name)
        # Log diagnostics to MLflow
        if diagnostics:
            mlflow.log_metrics({k: v for k, v in diagnostics.items() if isinstance(v, (int, float))})
    return leaderboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FDI pipeline")
    parser.add_argument("--csv", type=Path, default=None, help="Processed CSV path")
    parser.add_argument("--model-path", type=Path, default=None, help="Where to store the trained pipeline")
    parser.add_argument("--metrics-path", type=Path, default=None, help="Where to store evaluation metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leaderboard = train(csv_path=args.csv, model_path=args.model_path, metrics_path=args.metrics_path)
    LOGGER.info("Leaderboard: %s", json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    main()
