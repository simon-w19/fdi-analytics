"""Browser helpers for scraping fallbacks."""
from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Optional

try:  # pragma: no cover - optional dependency is installed via pyproject
    from playwright.sync_api import (  # type: ignore
        Browser,
        BrowserContext,
        Page,
        TimeoutError as PlaywrightTimeoutError,
        sync_playwright,
    )
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Browser = BrowserContext = Page = None  # type: ignore
    PlaywrightTimeoutError = Exception  # type: ignore
    sync_playwright = None  # type: ignore

LOGGER = logging.getLogger("pipeline.browser")


class PlaywrightFetcher(AbstractContextManager["PlaywrightFetcher"]):
    """Lightweight wrapper that keeps a single Playwright page alive."""

    def __init__(
        self,
        browser: str = "chromium",
        headless: bool = True,
        timeout_ms: int = 30000,
    ) -> None:
        self.browser_name = browser
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    def __enter__(self) -> "PlaywrightFetcher":  # pragma: no cover - thin wrapper
        if sync_playwright is None:  # pragma: no cover
            raise RuntimeError("Playwright is not installed. Run 'playwright install' once.")
        self._playwright = sync_playwright().start()
        try:
            browser_factory = getattr(self._playwright, self.browser_name)
        except AttributeError as exc:  # pragma: no cover - config error
            raise ValueError(f"Unsupported Playwright browser '{self.browser_name}'") from exc
        self._browser = browser_factory.launch(headless=self.headless)
        self._context = self._browser.new_context()
        self._page = self._context.new_page()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - cleanup path
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def fetch(self, url: str) -> str:
        if not self._page:
            raise RuntimeError("Playwright page not initialized.")
        LOGGER.debug("Playwright fetching %s", url)
        try:
            self._page.goto(url, timeout=self.timeout_ms, wait_until="networkidle")
        except PlaywrightTimeoutError as exc:
            raise TimeoutError(f"Playwright timeout for {url}") from exc
        return self._page.content()


__all__ = ["PlaywrightFetcher"]
