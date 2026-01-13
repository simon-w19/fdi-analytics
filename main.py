"""FDI Analytics - Entry point for the complete ETL + Training pipeline.

This module provides a convenient single-command entry point to run
the full data pipeline including scraping, transformation, ingestion,
and model training.

Usage:
    uv run python main.py              # Full pipeline
    uv run python main.py --skip-scrape # Skip scraping, use existing data
    uv run python main.py --train-only  # Only train model from processed data
"""
from __future__ import annotations

import argparse
import logging

from pipeline import etl, train

LOGGER = logging.getLogger("fdi-analytics")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def main() -> None:
    """Run the full FDI Analytics pipeline."""
    parser = argparse.ArgumentParser(
        description="FDI Analytics - Complete Data Science Pipeline for Darts Player Rating Prediction"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip web scraping and reuse existing raw data",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip ETL and only run model training on existing processed data",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Limit number of players to scrape (for testing)",
    )
    args = parser.parse_args()

    if args.train_only:
        LOGGER.info("Running model training only...")
        train.train()
    else:
        LOGGER.info("Running full ETL pipeline...")
        etl.run_pipeline(skip_scrape=args.skip_scrape, max_players=args.max_players)
        LOGGER.info("Running model training...")
        train.train()

    LOGGER.info("FDI Analytics pipeline completed successfully!")


if __name__ == "__main__":
    main()
