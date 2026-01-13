# Reports & Presentation

This directory contains all deliverables for the project submission.

## Main Deliverables

- **[report.ipynb](report.ipynb)** - Final project report as Jupyter Notebook
- **[presentation.md](presentation.md)** - Teaching guide and presentation outline
- **[slides.md](slides.md)** - Slide content for PDF export (6-8 slides)

## Metrics & Artifacts

Located in `metrics/`:
- `latest_metrics.json` - Current model performance metrics (MAE, RMSE, R², CV scores)
- `model_comparison.csv` - Comparison table across all trained models
- `presentation_metrics.json` - Curated metrics for slides

## Key Results

**Best Model**: Lasso Regression (α=0.01)
- **R²**: 0.929 (92.9% variance explained)
- **MAE**: 35.3 FDI points
- **Top Features**: First-9 Average, Checkout %, Legs Won %, log Earnings

## Visualizations

Generated figures from notebooks can be found in:
- `notebooks/eda.ipynb` - Exploratory visualizations
- `notebooks/fdi_rating_modeling.ipynb` - Model diagnostics and feature importance plots