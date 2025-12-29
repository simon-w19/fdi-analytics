# End-to-End Darts Analytics

Predictive Modeling des Future Dart Intelligence (FDI) Ratings mittels einer containerisierten Pipeline: Scraper (BeautifulSoup), Transformation, PostgreSQL, Scheduler, Training und Gradio-App laufen komplett in dieser Codebasis.

```
[ Scraper ] --> [ Transform ] --> [ Postgres DB ]
  |                     |                 |
  v                     v                 v
 data/raw/*.csv     data/processed/*.csv   Gradio API & UI
```

## Quick Start

1. **Dependencies installieren**
   ```bash
   uv sync
   cp .env.example .env
   ```
2. **Lokale Datenbank & App via Docker starten**
   ```bash
   docker compose build
   docker compose up
   ```
   - `db`: PostgreSQL 16 (persistentes Volume)
  - `etl`: führt einmalig `pipeline.etl` aus (Scrape -> Transform -> Load)
   - `scheduler`: wiederholt denselben Job im Intervall `FDI_REFRESH_MINUTES`
   - `app`: Gradio + FastAPI Service
3. **Frontend öffnen**: http://localhost:${APP_PORT} (Standard 7860)
  - Tab "Prediction Studio": manuelle Eingaben oder Spielerprofile auswählen und Ratings in Echtzeit berechnen.
  - Tab "Insights & EDA": Modell-Leaderboard, Top-Spieler und interaktive Scatterplots zur Kommunikation der Analyse-Erkenntnisse.

## Architektur im Detail

| Baustein | Beschreibung |
| --- | --- |
| `pipeline.scraper` | BeautifulSoup-Scraper für dartsorakel.com inkl. `--max-players` und optionalem Delay. |
| `pipeline.transform` | Bereinigt und feature-engineert die Scraper-Ausgabe (siehe `pipeline.features`). |
| `pipeline.ingest` | Lädt die veredelten CSV-Daten per SQLAlchemy in PostgreSQL. |
| `pipeline.etl` | Orchestriert Scrape -> Transform -> Load; wird vom Docker-Job und vom Scheduler verwendet. |
| `pipeline.scheduler` | Führt die ETL-Routine in einem konfigurierbaren Wochen/Minuten-Rhythmus aus und reagiert sauber auf SIGTERM. |
| `app/gradio_app.py` | Kombiniert Gradio-UI (Prediction Tab + Insights/EDA Tab), FastAPI-Endpunkte (`/api/*`) und das trainierte Pipeline-Modell. |

## Verzeichnisüberblick

```
├── app/                 # Gradio + FastAPI Service
├── data/
│   ├── raw/             # Scraper-Output (`FDI_RAW_CSV_PATH`)
│   └── processed/       # Feature-engineerte Daten (`FDI_PROCESSED_CSV_PATH`)
├── pipeline/
│   ├── scraper.py       # BeautifulSoup Scraper
│   ├── transform.py     # Feature Engineering für Persistenz & Training
│   ├── ingest.py        # Load in PostgreSQL
│   ├── etl.py           # Orchestrierung + CLI
│   ├── scheduler.py     # Cron-ähnlicher Loop
│   └── train.py         # Modelltraining (Linear Regression, Lasso, Random Forest)
├── docker/              # App- & ETL-Dockerfiles
├── models/              # Serialisierte Pipelines (`best_fdi_pipeline.joblib`)
├── notebooks/           # EDA/Modellierung (Scraper-Notebook delegiert an pipeline.scraper)
├── reports/             # Notebook-Reports & Metriken
└── tests/               # Pytest-Suite (Feature Engineering & Ingestion)
```

## Datenpipeline bedienen

- **Einmaliger Lauf (z. B. lokal)**
  ```bash
  uv run python -m pipeline.etl --max-players 250 --delay 0.25
  ```
  - `--skip-scrape`: vorhandene `data/raw/*.csv` wiederverwenden
  - `FDI_SCRAPE_MAX_PLAYERS`, `FDI_SCRAPE_DELAY_SECONDS`, `FDI_SKIP_SCRAPE` im `.env` setzbar
- **Nur Scraper anwerfen**
  ```bash
  uv run python -m pipeline.scraper --max-players 100 --output data/raw/custom.csv
  ```
- **Transformation isoliert testen**
  ```bash
  uv run python -m pipeline.transform --help  # oder direkt in Python importieren
  ```

## Training & Modellvergleich

`pipeline/train.py` vergleicht drei Ansätze (Linear Regression als Baseline, Lasso, Random Forest) über einen 80/20-Split und 5-fold Cross-Validation. Für Lasso und Random Forest läuft automatisch ein GridSearchCV-basiertes Hyperparameter-Tuning (ebenfalls 5-fold). Ergebnis:

```bash
uv run python -m pipeline.train \
  --csv data/processed/player_stats_all.csv \
  --model-path models/best_fdi_pipeline.joblib \
  --metrics-path reports/metrics/latest_metrics.json
```

- Das beste Modell wird als vollständige `sklearn`-Pipeline persistiert und direkt von Gradio geladen.
- Die JSON-Metriken enthalten MAE, RMSE, $R^2$ sowie die CV-Ergebnisse aller Modelle und dienen als Audit-Log.

## Qualitätssicherung

```bash
uv run pytest                # führt die Tests im Ordner tests/ aus
uv run ruff check .          # statische Analyse
uv run ruff format .         # optionales Formatting
```

Die Tests prüfen u. a. das Feature-Engineering sowie das Chunk-Size-Handling beim Bulk-Insert.

## Environment-Variablen

| Variable | Zweck |
| --- | --- |
| `FDI_RAW_CSV_PATH` | Speicherort des Scraper-Outputs (Standard `data/raw/player_stats_raw.csv`). |
| `FDI_PROCESSED_CSV_PATH` | Feature-engineerte CSV für Training & Ingestion. |
| `FDI_SCRAPE_MAX_PLAYERS` | Optionales Limit, z. B. für lokale Tests. Leer = alle Spieler. |
| `FDI_SCRAPE_DELAY_SECONDS` | Delay zwischen Requests zur Schonung des Targets. |
| `FDI_SKIP_SCRAPE` | `true`, um nur Transform + Load auszuführen (nützlich bei reproduzierbaren Re-Runs). |
| `FDI_DB_CHUNKSIZE` | Chunk-Größe für `pandas.to_sql`. |
| `FDI_REFRESH_MINUTES` | Intervall des Docker-Schedulers. |
| `APP_HOST` / `APP_PORT` | Netzwerkeinstellungen der Gradio-App. |
| `MLFLOW_ENABLED` | Aktiviert das optionale MLflow-Tracking (`true`/`false`). |
| `MLFLOW_TRACKING_URI` | URI/Backend für MLflow, z. B. `http://localhost:5000`. |
| `MLFLOW_EXPERIMENT_NAME` | Experimentbezeichnung, falls Logging aktiv ist. |

Weitere Postgres-Parameter (`POSTGRES_*`, `DATABASE_URL`) werden direkt von Docker Compose übernommen.

## Deployment mit Docker Compose

1. `.env` bereitstellen.
2. `docker compose build && docker compose up -d`
3. Logs prüfen: `docker compose logs -f scheduler`

Aktualisierte CSV-Dateien kannst du jederzeit mit `docker compose run --rm etl` sofort neu laden; der Scheduler übernimmt ansonsten das wöchentliche Refresh automatisch.

---

**Tipp:** Für lokale Experimente kannst du `uv run python app/gradio_app.py` starten. Die App verbindet sich automatisch mit der Postgres-Instanz (oder fällt auf die CSV zurück) und stellt zusätzlich die API-Endpunkte `/api/health`, `/api/players` und `/api/predict` bereit.
