"""Tests for scraper driver resolution."""
from __future__ import annotations

import pytest

from pipeline import scraper


def test_resolve_driver_falls_back_to_requests():
    assert scraper._resolve_driver("invalid") == scraper.DRIVER_REQUESTS
    assert scraper._resolve_driver(None) in scraper.VALID_DRIVERS


def test_fetch_player_html_requires_fetcher_for_playwright(monkeypatch):
    with pytest.raises(RuntimeError):
        scraper.fetch_player_html(1, driver=scraper.DRIVER_PLAYWRIGHT)


def test_fetch_player_html_with_requests(monkeypatch):
    captured = {}

    def fake_fetch(url, session=None, headers=None):  # noqa: D401 - test helper
        captured["url"] = url
        return "<html></html>"

    monkeypatch.setattr(scraper, "_requests_fetch", fake_fetch)
    result = scraper.fetch_player_html(99, driver=scraper.DRIVER_REQUESTS)
    assert "player/stats/99" in captured["url"]
    assert result == "<html></html>"
