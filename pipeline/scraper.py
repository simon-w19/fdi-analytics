"""Scrape darts player statistics from Darts Orakel."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests
from bs4 import BeautifulSoup, Tag

from .config import settings
from .browser import PlaywrightFetcher

LOGGER = logging.getLogger("pipeline.scraper")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

PLAYER_STATS_URL = "https://dartsorakel.com/player/stats/{player_id}"
PLAYER_LIST_API_URL = "https://dartsorakel.com/api/stats/player"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}
API_REQUEST_HEADERS = {
    **REQUEST_HEADERS,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://dartsorakel.com/stats/player",
}
HTTP_TIMEOUT = 30
DRIVER_REQUESTS = "requests"
DRIVER_PLAYWRIGHT = "playwright"
DRIVER_AUTO = "auto"
VALID_DRIVERS = {DRIVER_REQUESTS, DRIVER_PLAYWRIGHT, DRIVER_AUTO}


@dataclass(slots=True)
class PlayerStat:
    section: str
    stat: str
    raw_value: str
    value: Optional[float]
    unit: Optional[str]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PlayerProfile:
    player_id: int
    player_name: Optional[str]
    country_code: Optional[str]
    birth_date: Optional[str]
    age: Optional[int]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PlayerStatsPayload:
    profile: PlayerProfile
    stats: Sequence[PlayerStat]

    def as_wide_record(self) -> dict:
        record = self.profile.as_dict()
        seen = set(record.keys())
        for stat in self.stats:
            base_key = f"{stat.section}_{_normalize_column_name(stat.stat)}"
            if not base_key:
                base_key = "stat"
            candidate = base_key
            suffix = 2
            while candidate in seen:
                candidate = f"{base_key}_{suffix}"
                suffix += 1
            value = stat.value if stat.value is not None else stat.raw_value
            record[candidate] = value
            seen.add(candidate)
        return record


def _normalize_column_name(name: str) -> str:
    cleaned = name.strip().lower().replace("%", "pct").replace("+", "plus")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


@dataclass(slots=True)
class ScrapeMetrics:
    processed: int = 0
    succeeded: int = 0
    skipped: int = 0
    fallback_uses: int = 0


def _resolve_driver(driver: Optional[str]) -> str:
    candidate = (driver or settings.scrape_driver or DRIVER_REQUESTS).lower()
    if candidate not in VALID_DRIVERS:
        LOGGER.warning("Unsupported scrape driver '%s'. Falling back to requests.", candidate)
        return DRIVER_REQUESTS
    return candidate


def _requests_fetch(
    url: str,
    session: Optional[requests.Session] = None,
    headers: Optional[dict[str, str]] = None,
) -> str:
    client = session or requests.Session()
    attempts = max(1, settings.scrape_retry_attempts)
    delay = max(0.0, settings.scrape_retry_delay)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            LOGGER.debug("HTTP GET %s (attempt %s/%s)", url, attempt, attempts)
            response = client.get(url, headers=headers or REQUEST_HEADERS, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts:
                break
            LOGGER.warning("Request attempt %s failed for %s: %s", attempt, url, exc)
            if delay > 0:
                time.sleep(delay * attempt)
        else:
            return response.text
    assert last_exc is not None  # for type-checkers
    raise last_exc


def fetch_player_html(
    player_id: int,
    *,
    session: Optional[requests.Session] = None,
    fetcher: Optional[PlaywrightFetcher] = None,
    driver: Optional[str] = None,
    metrics: Optional[ScrapeMetrics] = None,
) -> str:
    target_url = PLAYER_STATS_URL.format(player_id=player_id)
    resolved_driver = _resolve_driver(driver)
    if resolved_driver == DRIVER_REQUESTS:
        return _requests_fetch(target_url, session=session)
    if resolved_driver == DRIVER_PLAYWRIGHT:
        if fetcher is None:
            raise RuntimeError("Playwright driver selected but no fetcher provided.")
        return fetcher.fetch(target_url)
    if resolved_driver == DRIVER_AUTO:
        try:
            return _requests_fetch(target_url, session=session)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Requests fetch failed for %s (%s). Falling back to Playwright.",
                target_url,
                exc,
            )
            if metrics is not None:
                metrics.fallback_uses += 1
            if fetcher is None:
                raise
            return fetcher.fetch(target_url)
    return _requests_fetch(target_url, session=session)


def _coerce_numeric(value: str) -> tuple[Optional[float], Optional[str]]:
    cleaned = value.strip()
    if not cleaned:
        return None, None

    unit = None
    if cleaned.endswith("%"):
        unit = "%"
        cleaned = cleaned[:-1]

    for prefix in ("£", "€", "$"):
        if cleaned.startswith(prefix):
            unit = unit or prefix
            cleaned = cleaned[len(prefix) :]
            break

    cleaned = cleaned.replace(",", "")

    try:
        if "." in cleaned:
            return float(cleaned), unit
        return float(int(cleaned)), unit
    except ValueError:
        LOGGER.debug("Could not convert value '%s'", value)
        return None, unit


def parse_player_name(soup: BeautifulSoup) -> Optional[str]:
    anchor = soup.select_one(".player-data-container a.fw-bolder")
    if anchor:
        return anchor.get_text(strip=True)
    title = soup.select_one("title")
    if title and "Stats" in title.text:
        return title.text.replace("Stats", "").replace("- Darts Orakel", "").strip()
    return None


def parse_player_profile(soup: BeautifulSoup, player_id: int) -> PlayerProfile:
    name = parse_player_name(soup)
    country_code: Optional[str] = None
    birth_date: Optional[str] = None
    age: Optional[int] = None

    container = soup.select_one(".player-data-container")
    if container:
        flag = container.select_one("img[alt]")
        if flag and flag.get("alt"):
            country_code = flag["alt"].strip() or None

        info_links = container.select("a.text-light-gray-400")
        for link in info_links:
            text = " ".join(link.stripped_strings).strip()
            if not text:
                continue
            if country_code is None and text.isalpha() and len(text) <= 4:
                country_code = text
                continue
            if birth_date is None and "/" in text:
                if "(" in text and ")" in text:
                    date_part, _, rest = text.partition("(")
                    birth_date = date_part.strip()
                    age_str = rest.rstrip(") ")
                    try:
                        age = int(age_str)
                    except ValueError:
                        pass
                else:
                    birth_date = text

    return PlayerProfile(
        player_id=player_id,
        player_name=name,
        country_code=country_code,
        birth_date=birth_date,
        age=age,
    )


def _format_countup_value(node: Optional[Tag]) -> str:
    if node is None:
        return ""
    value_str = node.get("data-kt-countup-value")
    prefix = node.get("data-kt-countup-prefix", "")
    suffix = node.get("data-kt-countup-suffix", "")
    separator = node.get("data-kt-countup-separator", ",")
    decimal_places_attr = node.get("data-kt-countup-decimal-places")
    formatted = node.get_text(strip=True)
    if value_str is None:
        return formatted
    try:
        number = float(value_str)
    except ValueError:
        return formatted
    try:
        decimal_places = int(decimal_places_attr) if decimal_places_attr is not None else None
    except ValueError:
        decimal_places = None
    if decimal_places is None:
        decimal_places = 0 if number.is_integer() else 2
    format_spec = f"{{:,.{decimal_places}f}}"
    base = format_spec.format(number)
    if separator != ",":
        base = base.replace(",", separator)
    return f"{prefix}{base}{suffix}"


def parse_headline_stats(soup: BeautifulSoup) -> Sequence[PlayerStat]:
    stats: list[PlayerStat] = []
    for card in soup.select(".stats-container .single-player-stat"):
        label_node = card.select_one(".text-light-gray-500")
        value_node = card.select_one(".stat-value .fw-bolder")
        if not label_node or not value_node:
            continue

        label = label_node.get_text(strip=True)
        display_value = _format_countup_value(value_node)
        numeric_value, unit = _coerce_numeric(display_value)
        stats.append(
            PlayerStat(
                section="profile",
                stat=label,
                raw_value=display_value,
                value=numeric_value,
                unit=unit,
            )
        )
    return stats


def parse_stats_table(soup: BeautifulSoup) -> Sequence[PlayerStat]:
    table = soup.select_one("table#playerStatsTable")
    if not table:
        raise ValueError("Unable to locate 'playerStatsTable'.")

    rows: Iterable[Tag] = table.select("tbody tr")
    parsed: list[PlayerStat] = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        name = cols[0].get_text(strip=True)
        raw_value = cols[1].get_text(strip=True)
        numeric_value, unit = _coerce_numeric(raw_value)
        parsed.append(
            PlayerStat(
                section="last_12_months",
                stat=name,
                raw_value=raw_value,
                value=numeric_value,
                unit=unit,
            )
        )

    if not parsed:
        raise ValueError("No stats found. The site layout may have changed.")
    return parsed


def scrape_player_stats(
    player_id: int,
    *,
    session: Optional[requests.Session] = None,
    fetcher: Optional[PlaywrightFetcher] = None,
    driver: Optional[str] = None,
    metrics: Optional[ScrapeMetrics] = None,
) -> PlayerStatsPayload:
    html = fetch_player_html(
        player_id=player_id,
        session=session,
        fetcher=fetcher,
        driver=driver,
        metrics=metrics,
    )
    soup = BeautifulSoup(html, "html.parser")
    headline_stats = parse_headline_stats(soup)
    last_12_stats = parse_stats_table(soup)
    profile = parse_player_profile(soup, player_id)
    return PlayerStatsPayload(profile=profile, stats=[*headline_stats, *last_12_stats])


def fetch_all_player_entries(session: Optional[requests.Session] = None) -> list[dict]:
    client = session or requests.Session()
    LOGGER.debug("Fetching player listing from %s", PLAYER_LIST_API_URL)
    response_text = _requests_fetch(
        PLAYER_LIST_API_URL,
        session=client,
        headers=API_REQUEST_HEADERS,
    )
    payload = json.loads(response_text)
    if isinstance(payload, dict):
        data = payload.get("data")
    else:
        data = payload
    if not isinstance(data, list):
        raise ValueError("Unexpected response when requesting the player listing.")
    return data


def scrape_all_players_to_csv(
    output_path: Path | None = None,
    max_players: Optional[int] = None,
    delay: float | None = None,
    driver: Optional[str] = None,
) -> Path:
    session = requests.Session()
    entries = fetch_all_player_entries(session=session)
    if max_players is not None:
        entries = entries[: max(0, max_players)]
    LOGGER.info("Preparing to scrape %s players", len(entries))
    target = output_path or settings.raw_csv_path
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metrics = ScrapeMetrics()
    resolved_driver = _resolve_driver(driver)
    needs_playwright = resolved_driver in {DRIVER_PLAYWRIGHT, DRIVER_AUTO}

    def _scrape_loop(active_fetcher: Optional[PlaywrightFetcher], active_driver: str) -> None:
        for index, entry in enumerate(entries, start=1):
            player_id = entry.get("player_key")
            if player_id is None:
                metrics.skipped += 1
                continue
            metrics.processed += 1
            try:
                payload = scrape_player_stats(
                    int(player_id),
                    session=session,
                    fetcher=active_fetcher,
                    driver=active_driver,
                    metrics=metrics,
                )
            except Exception as exc:  # noqa: BLE001
                metrics.skipped += 1
                LOGGER.warning("Skipping player %s due to error: %s", player_id, exc)
                continue
            metrics.succeeded += 1
            if payload.profile.player_name is None and entry.get("player_name"):
                payload.profile.player_name = entry["player_name"]
            row = payload.as_wide_record()
            row.setdefault("player_name", entry.get("player_name"))
            if not row.get("country_code") and entry.get("country"):
                row["country_code"] = entry["country"] or row.get("country_code")
            row["api_rank"] = entry.get("rank")
            row["api_overall_stat"] = entry.get("stat")
            row["api_sum_field1"] = entry.get("sumField1")
            row["api_sum_field2"] = entry.get("sumField2")
            rows.append(row)
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
            if delay and delay > 0 and index < len(entries):
                time.sleep(delay)
    fieldnames: list[str] = [
        "player_id",
        "player_name",
        "country_code",
        "birth_date",
        "age",
    ]
    if needs_playwright:
        try:
            with PlaywrightFetcher(
                browser=settings.playwright_browser,
                headless=settings.playwright_headless,
                timeout_ms=settings.playwright_timeout_ms,
            ) as fetcher:
                _scrape_loop(fetcher, resolved_driver)
        except RuntimeError as exc:
            if resolved_driver == DRIVER_AUTO:
                LOGGER.warning(
                    "Playwright unavailable (%s). Continuing with requests-only scraping.",
                    exc,
                )
                _scrape_loop(None, DRIVER_REQUESTS)
            else:
                raise
    else:
        _scrape_loop(None, resolved_driver)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    LOGGER.info(
        "Scraping summary -> success=%s skipped=%s fallback_uses=%s driver=%s",
        metrics.succeeded,
        metrics.skipped,
        metrics.fallback_uses,
        resolved_driver,
    )
    LOGGER.info("Finished scraping %s/%s players into %s", len(rows), len(entries), target)
    return target


def run_scraper(
    output_path: Path | None = None,
    max_players: Optional[int] = None,
    delay: float | None = None,
    driver: Optional[str] = None,
) -> Path:
    limit = max_players if max_players is not None else settings.scrape_max_players
    pause = delay if delay is not None else settings.scrape_delay_seconds
    return scrape_all_players_to_csv(
        output_path=output_path or settings.raw_csv_path,
        max_players=limit,
        delay=pause,
        driver=driver,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape darts player statistics")
    parser.add_argument("--output", type=Path, default=settings.raw_csv_path)
    parser.add_argument("--max-players", type=int, default=settings.scrape_max_players)
    parser.add_argument("--delay", type=float, default=settings.scrape_delay_seconds)
    parser.add_argument("--driver", type=str, default=settings.scrape_driver)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scraper(
        output_path=args.output,
        max_players=args.max_players,
        delay=args.delay,
        driver=args.driver,
    )


if __name__ == "__main__":
    main()
