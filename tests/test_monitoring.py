"""Tests for monitoring summaries."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import monitoring


def test_summarize_dataframe_returns_expected_keys(tmp_path: Path):
    df = pd.DataFrame(
        {
            "player_id": [1, 2],
            "profile_fdi_rating": [1200, 1300],
            "last_12_months_averages": [98.0, 99.5],
            "last_12_months_180_s": [100, 150],
            "country": ["ENG", None],
        }
    )
    summary = monitoring.summarize_dataframe(df)
    assert summary["rows"] == 2
    assert "numeric_ranges" in summary
    target = tmp_path / "metrics.json"
    monitoring.persist_dataset_metrics(df, path=target)
    assert target.exists()
