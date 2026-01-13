# Raw Data

This directory contains unprocessed data from the web scraper.

## Files

- `player_stats_raw.csv` - Raw player statistics scraped from dartsorakel.com
- `player_*_stats.csv` - Individual player statistics (if running single-player scrapes)

## Data Source

All data is scraped from [Darts Orakel](https://dartsorakel.com), which provides comprehensive statistics for professional darts players including:

- Player profiles (name, country, age)
- Performance metrics (3-dart averages, checkout percentages)
- Tournament results and earnings
- FDI (Future Dart Intelligence) ratings

## Refresh Cycle

The ETL pipeline refreshes this data weekly (configurable via `FDI_REFRESH_MINUTES`). The Docker scheduler handles automatic updates.

## Note

Raw files may contain missing values and require transformation before analysis. See `pipeline/transform.py` for the cleaning logic.