# Trained Models

This directory contains serialized model pipelines for FDI rating prediction.

## Files

- `best_fdi_pipeline.joblib` - Best performing sklearn Pipeline (currently Lasso or Linear Regression)

## Model Architecture

The pipeline includes:
1. **Preprocessing**: `ColumnTransformer` with StandardScaler (numeric) + OneHotEncoder (categorical)
2. **Model**: Best model from comparison (Linear Regression / Lasso / Random Forest)

## Current Performance (Holdout Test Set)

| Metric | Value |
|--------|-------|
| R² | 0.929 |
| MAE | 35.3 FDI points |
| RMSE | 46.1 FDI points |
| CV σ | ±0.43 |

## Usage

```python
import joblib
pipeline = joblib.load("models/best_fdi_pipeline.joblib")
predictions = pipeline.predict(X_new)
```

## Retraining

```bash
uv run python -m pipeline.train \
  --csv data/processed/player_stats_all.csv \
  --model-path models/best_fdi_pipeline.joblib \
  --metrics-path reports/metrics/latest_metrics.json
```