"""Utility to scrape Darts Orakel player statistics."""

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


LOGGER = logging.getLogger(__name__)

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


@dataclass(slots=True)
class PlayerStat:
	"""Single stat entry taken from the profile or stats table."""

	section: str
	stat: str
	raw_value: str
	value: Optional[float]
	unit: Optional[str]

	def as_dict(self) -> dict:
		return asdict(self)


@dataclass(slots=True)
class PlayerProfile:
	"""Basic player metadata extracted from the hero card."""

	player_id: int
	player_name: Optional[str]
	country_code: Optional[str]
	birth_date: Optional[str]
	age: Optional[int]

	def as_dict(self) -> dict:
		return asdict(self)


@dataclass(slots=True)
class PlayerStatsPayload:
	"""Container that bundles player metadata and stats."""

	profile: PlayerProfile
	stats: Sequence[PlayerStat]

	def as_records(self) -> list[dict]:
		return [stat.as_dict() for stat in self.stats]

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
	"""Create safe column identifiers from arbitrary stat labels."""

	cleaned = name.strip().lower().replace("%", "pct").replace("+", "plus")
	cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
	cleaned = re.sub(r"_+", "_", cleaned).strip("_")
	return cleaned


def fetch_player_html(player_id: int, session: Optional[requests.Session] = None) -> str:
	"""Fetch raw HTML for the provided player id."""

	target_url = PLAYER_STATS_URL.format(player_id=player_id)
	client = session or requests.Session()
	LOGGER.debug("Fetching %s", target_url)
	response = client.get(target_url, headers=REQUEST_HEADERS, timeout=HTTP_TIMEOUT)
	response.raise_for_status()
	return response.text


def _coerce_numeric(value: str) -> tuple[Optional[float], Optional[str]]:
	"""Convert numbers or percentages to a numeric representation."""

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
			cleaned = cleaned[len(prefix):]
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
	"""Try to extract the player name from the hero section."""

	anchor = soup.select_one(".player-data-container a.fw-bolder")
	if anchor:
		return anchor.get_text(strip=True)
	title = soup.select_one("title")
	if title and "Stats" in title.text:
		return title.text.replace("Stats", "").replace("- Darts Orakel", "").strip()
	return None


def parse_player_profile(soup: BeautifulSoup, player_id: int) -> PlayerProfile:
	"""Extract country, birth date, and age information."""

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
	"""Recreate the formatted text shown on the site for countup widgets."""

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
	"""Scrape the stats cards next to the profile image."""

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
	"""Parse the main stats table for the last 12 months."""

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


def scrape_player_stats(player_id: int, session: Optional[requests.Session] = None) -> PlayerStatsPayload:
	"""Fetch and parse stats for a single player id."""

	html = fetch_player_html(player_id=player_id, session=session)
	soup = BeautifulSoup(html, "html.parser")
	headline_stats = parse_headline_stats(soup)
	last_12_stats = parse_stats_table(soup)
	profile = parse_player_profile(soup, player_id)
	return PlayerStatsPayload(profile=profile, stats=[*headline_stats, *last_12_stats])


def fetch_all_player_entries(session: Optional[requests.Session] = None) -> list[dict]:
	"""Return the player listing exposed by the dartsorakel API."""

	client = session or requests.Session()
	LOGGER.debug("Fetching player listing from %s", PLAYER_LIST_API_URL)
	response = client.get(PLAYER_LIST_API_URL, headers=API_REQUEST_HEADERS, timeout=HTTP_TIMEOUT)
	response.raise_for_status()
	payload = response.json()
	if isinstance(payload, dict):
		data = payload.get("data")
	else:
		data = payload
	if not isinstance(data, list):
		raise ValueError("Unexpected response when requesting the player listing.")
	return data


def scrape_all_players_to_csv(
	output_path: Path,
	max_players: Optional[int] = None,
	delay: float = 0.0,
) -> None:
	"""Scrape every available player and persist one row per player."""

	session = requests.Session()
	entries = fetch_all_player_entries(session=session)
	if max_players is not None:
		entries = entries[: max(0, max_players)]
	LOGGER.info("Preparing to scrape %s players", len(entries))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	processed = 0
	rows: list[dict] = []
	fieldnames: list[str] = ["player_id", "player_name", "country_code", "birth_date", "age"]
	for index, entry in enumerate(entries, start=1):
		player_id = entry.get("player_key")
		if player_id is None:
			continue
		try:
			payload = scrape_player_stats(int(player_id), session=session)
		except Exception as exc:
			LOGGER.warning("Skipping player %s due to error: %s", player_id, exc)
			continue
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
		processed += 1
		if delay > 0 and index < len(entries):
			time.sleep(delay)
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)
	LOGGER.info("Finished scraping %s/%s players into %s", processed, len(entries), output_path)


def _write_json(path: Path, payload: PlayerStatsPayload) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		json.dump(
			{
				"player": payload.profile.as_dict(),
				"stats": payload.as_records(),
			},
			file,
			ensure_ascii=False,
			indent=2,
		)


def _write_csv(path: Path, payload: PlayerStatsPayload) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	rows = payload.as_records()
	with path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=["section", "stat", "raw_value", "value", "unit"])
		writer.writeheader()
		writer.writerows(rows)


def _print_table(payload: PlayerStatsPayload) -> None:
	profile = payload.profile
	name = profile.player_name or f"Player {profile.player_id}"
	print(f"Stats for {name} (ID: {profile.player_id})")
	if profile.country_code:
		print(f"Country: {profile.country_code}")
	if profile.birth_date:
		age_suffix = f" (Age: {profile.age})" if profile.age is not None else ""
		print(f"Birth date: {profile.birth_date}{age_suffix}")
	print("-" * 60)
	headline = [stat for stat in payload.stats if stat.section == "profile"]
	last_12 = [stat for stat in payload.stats if stat.section == "last_12_months"]
	if headline:
		print("Headline stats:")
		for stat in headline:
			suffix = ""
			if stat.unit and stat.unit not in stat.raw_value:
				suffix = f" ({stat.unit})"
			print(f"{stat.stat:<35} {stat.raw_value:>15}{suffix}")
	if last_12:
		if headline:
			print()
		print("Last 12 months:")
		for stat in last_12:
			suffix = ""
			if stat.unit and stat.unit not in stat.raw_value:
				suffix = f" ({stat.unit})"
			print(f"{stat.stat:<35} {stat.raw_value:>15}{suffix}")


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"player_id",
		type=int,
		nargs="?",
		help="Player id taken from the dartsorakel.com URL. Optional when using --all-players.",
	)
	parser.add_argument(
		"--format",
		choices=("json", "csv", "table"),
		default="table",
		help="Output format. JSON/CSV will optionally honor --output.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help="Output file. Ignored when printing tables.",
	)
	parser.add_argument(
		"-v",
		"--verbose",
		action="store_true",
		help="Turn on verbose logging.",
	)
	parser.add_argument(
		"--save-csv",
		action="store_true",
		help="Persist the scraped stats as CSV in addition to the selected output mode.",
	)
	parser.add_argument(
		"--csv-path",
		type=Path,
		help="Optional CSV path when using --save-csv. Defaults to data/raw/player_<id>_stats.csv.",
	)
	parser.add_argument(
		"--all-players",
		action="store_true",
		help="Scrape every player returned by dartsorakel.com and write a combined CSV.",
	)
	parser.add_argument(
		"--all-output",
		type=Path,
		help="Target CSV path when using --all-players. Defaults to data/processed/player_stats_all.csv.",
	)
	parser.add_argument(
		"--max-players",
		type=int,
		help="Optional upper bound for the number of players when using --all-players.",
	)
	parser.add_argument(
		"--sleep",
		type=float,
		default=0.0,
		help="Delay in seconds between player requests when using --all-players.",
	)

	args = parser.parse_args(argv)

	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

	if args.all_players:
		output = args.all_output or Path("data/processed") / "player_stats_all.csv"
		scrape_all_players_to_csv(
			output_path=output,
			max_players=args.max_players,
			delay=max(0.0, args.sleep),
		)
		LOGGER.info("Combined CSV available at %s", output)
		return

	if args.player_id is None:
		parser.error("player_id is required unless --all-players is set.")

	payload = scrape_player_stats(player_id=args.player_id)

	if args.format == "json":
		if not args.output:
			print(json.dumps(
				{
					"player": payload.profile.as_dict(),
					"stats": payload.as_records(),
				},
				ensure_ascii=False,
				indent=2,
			))
		else:
			_write_json(args.output, payload)
			LOGGER.info("Wrote JSON to %s", args.output)
	elif args.format == "csv":
		target = args.output or Path(f"player_{payload.profile.player_id}_stats.csv")
		_write_csv(target, payload)
		LOGGER.info("Wrote CSV to %s", target)
	else:
		_print_table(payload)

	if args.save_csv:
		csv_target = args.csv_path or Path("data/raw") / f"player_{payload.profile.player_id}_stats.csv"
		_write_csv(csv_target, payload)
		LOGGER.info("Persisted CSV to %s", csv_target)


if __name__ == "__main__":
	run_cli()
