"""Legacy entrypoint that delegates to the reusable pipeline scraper."""
from __future__ import annotations

from pipeline.scraper import main


if __name__ == "__main__":
    main()
