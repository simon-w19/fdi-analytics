"""End-to-end orchestration: scrape -> transform -> load."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import ingest, monitoring, scraper, transform, validation
from .config import settings

LOGGER = logging.getLogger("pipeline.etl")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def run_pipeline(
    *,
    skip_scrape: bool | None = None,
    max_players: int | None = None,
    delay: float | None = None,
    raw_output: Path | None = None,
    processed_output: Path | None = None,
) -> None:
    should_skip = settings.skip_scrape if skip_scrape is None else skip_scrape
    raw_target = raw_output or settings.raw_csv_path
    processed_target = processed_output or settings.processed_csv_path

    if should_skip:
        LOGGER.info("Skipping scraping step (skip_scrape flag is active).")
    else:
        LOGGER.info("Running scraping step...")
        scraper.run_scraper(output_path=raw_target, max_players=max_players, delay=delay)

    LOGGER.info("Running transform step...")
    processed_df = transform.transform_dataset(
        raw_csv_path=raw_target,
        output_csv_path=processed_target,
    )

    LOGGER.info("Running validation step...")
    validation.validate_processed_dataset(processed_df, strict=True)

    LOGGER.info("Recording dataset metrics...")
    monitoring.persist_dataset_metrics(processed_df)

    LOGGER.info("Running ingestion step...")
    ingest.main(csv_path=processed_target)
    LOGGER.info("ETL pipeline finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FDI ETL pipeline")
    parser.add_argument("--skip-scrape", action="store_true", help="Reuse existing raw CSV and skip scraping")
    parser.add_argument("--max-players", type=int, default=None, help="Limit number of players to scrape")
    parser.add_argument("--delay", type=float, default=None, help="Delay between requests in seconds")
    parser.add_argument("--raw", type=Path, default=None, help="Override raw CSV output path")
    parser.add_argument("--processed", type=Path, default=None, help="Override processed CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        skip_scrape=args.skip_scrape,
        max_players=args.max_players,
        delay=args.delay,
        raw_output=args.raw,
        processed_output=args.processed,
    )


if __name__ == "__main__":
    main()
