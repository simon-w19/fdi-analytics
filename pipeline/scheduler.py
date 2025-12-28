"""Simple scheduler to refresh the PostgreSQL table on a fixed cadence."""
from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timezone

from . import ingest
from .config import settings

LOGGER = logging.getLogger("pipeline.scheduler")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

MIN_INTERVAL_SECONDS = 60


def _interval_seconds() -> int:
    minutes = max(1, settings.refresh_minutes)
    return max(MIN_INTERVAL_SECONDS, minutes * 60)


def run_ingestion_once() -> None:
    LOGGER.info("Starting ingestion cycle at %s", datetime.now(timezone.utc).isoformat())
    try:
        ingest.main()
        LOGGER.info("Ingestion cycle finished successfully.")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Ingestion cycle failed: %s", exc)


def main() -> None:
    interval = _interval_seconds()
    LOGGER.info("Scheduler booted with interval=%s seconds (~%s minutes)", interval, interval // 60)

    should_exit = False

    def _handle_signal(signum, frame):  # type: ignore[override]
        nonlocal should_exit
        LOGGER.info("Received signal %s. Finishing current cycle before exit.", signum)
        should_exit = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while True:
        run_ingestion_once()
        if should_exit:
            LOGGER.info("Scheduler stopping on signal request.")
            break
        LOGGER.info("Sleeping for %s seconds before next cycle.", interval)
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Fatal scheduler error: %s", exc)
        sys.exit(1)
