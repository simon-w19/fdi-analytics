# Processed Data

This directory contains cleaned and feature-engineered datasets ready for modeling.

## Files

- `player_stats_all.csv` - Main dataset with all engineered features (used for training)

## Feature Engineering Applied

The transformation pipeline (`pipeline/transform.py` + `pipeline/features.py`) applies:

1. **Log Transformation**: `log_total_earnings` for handling skewed earnings distribution
2. **Ratio Features**: `season_win_rate`, `first9_ratio`, `break_efficiency`, `power_scoring_ratio`
3. **Delta Features**: `first9_delta`, `momentum_gap`, `tv_stage_delta`
4. **Experience Metrics**: `experience_intensity` (tour card years / age)
5. **Country Normalization**: Standardized country codes

## Target Variable

`profile_fdi_rating` - Future Dart Intelligence rating (continuous, ~50-800 range)

## Data Quality

- Missing values: Handled via median imputation (numeric) / mode imputation (categorical)
- Outliers: Retained for analysis; regularization handles extreme values
- Observations: ~2,900 players with ~2,400 having valid FDI ratings