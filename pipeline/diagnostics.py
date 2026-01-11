"""Advanced diagnostics for model evaluation."""
from __future__ import annotations

import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger("pipeline.diagnostics")


def calculate_adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """
    Calculate Adjusted R² with penalty for additional features.
    
    Formula: Adj R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)
    
    Args:
        r2: R² score
        n_samples: Number of samples
        n_features: Number of features used
    
    Returns:
        Adjusted R² score
    """
    if n_samples <= n_features + 1:
        return r2  # Avoid division by zero
    
    adjustment = (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return 1 - adjustment


def calculate_residual_diagnostics(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Calculate comprehensive residual diagnostics.
    
    Args:
        model: Trained sklearn pipeline
        x_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with diagnostic metrics
    """
    predictions = model.predict(x_test)
    residuals = y_test - predictions
    
    diagnostics = {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
    }
    
    # Optional: Try to calculate advanced metrics if statsmodels is available
    try:
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson
        
        # Prepare data for statsmodels
        preprocessed = model.named_steps["preprocessor"].transform(x_test)
        design = sm.add_constant(preprocessed, has_constant="add")
        
        # Fit OLS model
        ols = sm.OLS(y_test.to_numpy(), design).fit()
        
        # Durbin-Watson test (autocorrelation)
        dw_stat = durbin_watson(ols.resid)
        diagnostics["durbin_watson"] = float(dw_stat)
        
        # Breusch-Pagan test (heteroskedasticity)
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(ols.resid, design)
        diagnostics["breusch_pagan_stat"] = float(bp_stat)
        diagnostics["breusch_pagan_pvalue"] = float(bp_pvalue)
        
        # Heteroskedasticity check (correlation between residuals and fitted values)
        abs_residuals = np.abs(residuals)
        corr = np.corrcoef(predictions, abs_residuals)[0, 1]
        diagnostics["heteroskedasticity_corr"] = float(corr)
        
        LOGGER.info("Advanced diagnostics calculated successfully")
        
    except ImportError:
        LOGGER.warning("statsmodels not available, skipping advanced diagnostics")
    except Exception as e:
        LOGGER.warning("Could not calculate advanced diagnostics: %s", e)
    
    return diagnostics


def calculate_influential_points(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    threshold_multiplier: float = 4.0,
) -> Dict[str, Any]:
    """
    Calculate Cook's Distance to identify influential points.
    
    Args:
        model: Trained sklearn pipeline
        x_train: Training features
        y_train: Training target
        threshold_multiplier: Multiplier for Cook's Distance threshold (default: 4/n)
    
    Returns:
        Dictionary with influential points metrics
    """
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import OLSInfluence
        
        preprocessed = model.named_steps["preprocessor"].transform(x_train)
        design = sm.add_constant(preprocessed, has_constant="add")
        ols = sm.OLS(y_train.to_numpy(), design).fit()
        
        influence = OLSInfluence(ols)
        cooks_d = influence.cooks_distance[0]
        
        # Filter out NaN and inf values
        cooks_d_clean = cooks_d[~np.isnan(cooks_d) & ~np.isinf(cooks_d)]
        
        if len(cooks_d_clean) == 0:
            LOGGER.warning("All Cook's Distance values are NaN or inf, skipping")
            return {}
        
        threshold = threshold_multiplier / len(cooks_d)
        n_influential = int((cooks_d_clean > threshold).sum())
        
        return {
            "cooks_distance_max": float(cooks_d_clean.max()),
            "cooks_distance_mean": float(cooks_d_clean.mean()),
            "cooks_distance_95percentile": float(np.percentile(cooks_d_clean, 95)),
            "n_influential_points": n_influential,
            "influential_points_pct": float(n_influential / len(cooks_d_clean) * 100),
            "threshold": float(threshold),
        }
        
    except ImportError:
        LOGGER.warning("statsmodels not available, skipping Cook's Distance")
        return {}
    except Exception as e:
        LOGGER.warning("Could not calculate Cook's Distance: %s", e)
        return {}


def run_full_diagnostics(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_features: int,
) -> Dict[str, Any]:
    """
    Run complete diagnostic suite on a trained model.
    
    Args:
        model: Trained sklearn pipeline
        x_train: Training features
        y_train: Training target
        x_test: Test features
        y_test: Test target
        n_features: Number of features used in the model
    
    Returns:
        Comprehensive diagnostics dictionary
    """
    from sklearn.metrics import r2_score
    
    LOGGER.info("Running full diagnostic suite...")
    
    # Basic metrics
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    
    # Adjusted R²
    adj_r2 = calculate_adjusted_r2(r2, len(x_test), n_features)
    
    # Residual diagnostics
    residual_diag = calculate_residual_diagnostics(model, x_test, y_test)
    
    # Influential points
    influential_diag = calculate_influential_points(model, x_train, y_train)
    
    diagnostics = {
        "adjusted_r2": adj_r2,
        "n_features": n_features,
        "n_samples_train": len(x_train),
        "n_samples_test": len(x_test),
        **residual_diag,
        **influential_diag,
    }
    
    LOGGER.info("Diagnostics complete: Adjusted R² = %.4f", adj_r2)
    
    return diagnostics
