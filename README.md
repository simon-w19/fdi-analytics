# 🎯 FDI Analytics

> **End-to-End Darts Analytics: Prädiktive Modellierung des FDI-Ratings mittels einer containerisierten Data-Pipeline**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dieses Projekt prognostiziert das **FDI-Rating** (Future Dart Intelligence) professioneller Darts-Spieler mithilfe statistischer Modelle. Die vollständige Pipeline – von Web-Scraping über Feature Engineering bis zum Deployment – läuft containerisiert.

---

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Architektur](#-architektur)
- [Projektstruktur](#-projektstruktur)
- [Datenpipeline](#-datenpipeline)
- [Modellierung](#-modellierung)
- [KPIs & Ergebnisse](#-kpis--ergebnisse)
- [Konfiguration](#-konfiguration)
- [Entwicklung](#-entwicklung)
- [Lessons Learned](#-lessons-learned)
- [Lizenz](#-lizenz)

---

## ✨ Features

- **🕷️ Automatisiertes Web-Scraping** von [DartsOrakel](https://dartsorakel.com)
- **🔄 ETL-Pipeline** mit Feature Engineering und PostgreSQL-Integration
- **📊 Modellvergleich**: Linear Regression und Lasso mit GridSearchCV
- **🌐 Gradio Web-App** für Echtzeit-Vorhersagen
- **🐳 Vollständig containerisiert** mit Docker Compose
- **⏰ Automatische Updates** via Scheduler (wöchentlich konfigurierbar)

---

## 🚀 Quick Start

### Voraussetzungen

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [uv](https://github.com/astral-sh/uv) (Python Package Manager) für lokale Entwicklung

### Installation

```bash
# Repository klonen
git clone https://github.com/yourusername/fdi-analytics.git
cd fdi-analytics

# Umgebungsvariablen konfigurieren
cp .env.example .env

# Container starten (baut Images, führt ETL aus, startet App)
docker compose up -d
```

### Zugriff

| Endpunkt | URL |
|----------|-----|
| **Web-App** | http://localhost:7860 |
| **API Health** | http://localhost:7860/api/health |
| **API Predict** | http://localhost:7860/api/predict |

---

## 🏗️ Architektur

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Scraper   │────▶│  Transform  │────▶│  PostgreSQL │
│ (Beautiful  │     │  (Feature   │     │    (DB)     │
│    Soup)    │     │ Engineering)│     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│   Gradio    │◀────│   Train     │◀───────────┘
│   Web-App   │     │  (sklearn)  │
└─────────────┘     └─────────────┘
```

### Docker Services

| Service | Beschreibung | Port |
|---------|--------------|------|
| `db` | PostgreSQL 16 (Alpine) mit persistentem Volume | 5432 |
| `etl` | Scraping → Transform → Train → Ingest | - |
| `scheduler` | Periodisches ETL-Refresh | - |
| `app` | Gradio + FastAPI Web-Service | 7860 |

---

## 📁 Projektstruktur

```
fdi-analytics/
├── app/                    # Gradio + FastAPI Web-Service
│   └── gradio_app.py       # Prediction Studio & Insights
├── data/
│   ├── raw/                # Scraper-Output (CSV)
│   └── processed/          # Feature-engineerte Daten
├── docker/                 # Dockerfiles
├── models/                 # Trainierte Modelle (.joblib)
├── notebooks/
│   ├── eda.ipynb           # Explorative Datenanalyse
│   └── fdi_rating_modeling.ipynb  # Modellierung & Evaluation
├── pipeline/
│   ├── scraper.py          # BeautifulSoup Web-Scraper
│   ├── transform.py        # Feature Engineering
│   ├── features.py         # Feature-Definitionen
│   ├── train.py            # Modelltraining & Vergleich
│   ├── ingest.py           # PostgreSQL-Import
│   ├── etl.py              # Pipeline-Orchestrierung
│   └── scheduler.py        # Cron-ähnlicher Loop
├── reports/
│   └── metrics/            # Modell-Metriken (JSON)
├── tests/                  # Pytest-Suite
├── docker-compose.yml
└── pyproject.toml
```

---

## 🔄 Datenpipeline

### Vollständiger ETL-Lauf

```bash
# Lokal (mit uv)
uv run python -m pipeline.etl

# Mit Optionen
uv run python -m pipeline.etl --max-players 100 --skip-train
```

### Einzelne Schritte

```bash
# Nur Scraping
uv run python -m pipeline.scraper --max-players 50 --output data/raw/test.csv

# Nur Training
uv run python -m pipeline.train --csv data/processed/player_stats_all.csv

# App starten (lokal)
uv run python -m app.gradio_app
```

### Docker-Workflow

```bash
# Alles neu bauen und starten
docker compose down && docker compose build --no-cache && docker compose up -d

# ETL manuell triggern
docker compose run --rm etl

# Logs verfolgen
docker compose logs -f etl
```

---

## 📈 Modellierung

### Verglichene Modelle

| Modell | R² | MAE | RMSE |
|--------|-----|-----|------|
| Linear Regression | 0.928 | 35.4 | 46.4 |
| **Lasso (α=0.01)** | **0.928** | **35.4** | **46.2** |

### Feature Engineering

**Numerische Features** (38 total):
- **Performance**: 3-Dart Average, First-9 Average, Checkout %
- **Erfolg**: Season Win Rate, Legs Won %, Order of Merit
- **Finanzen**: Log-transformierte Earnings
- **Abgeleitete**: `first9_delta`, `momentum_gap`, `break_efficiency`, `power_scoring_ratio`

**Kategorisch**: Country (One-Hot Encoded, ~30 Länder)

### Top-5 Prädiktoren

1. **last_12_months_first_9_averages** – Early-Game-Dominanz
2. **last_12_months_checkout_pcnt** – Finish-Qualität
3. **last_12_months_pcnt_legs_won** – Gewinneffizienz
4. **log_total_earnings** – Langfristiger Erfolg
5. **profile_season_win_pct** – Aktuelle Form

---

## 🎯 Ergebnisse

| KPI | Wert |
|-----|------|
| Modellgenauigkeit (MAE) | 35.4 FDI-Punkte |
| Erklärte Varianz (R²) | 0.928 |
| Production Readiness | Docker + <100ms Inference |

---

## ⚙️ Konfiguration

### Wichtige Umgebungsvariablen

| Variable | Beschreibung | Default |
|----------|--------------|---------|
| `FDI_SKIP_SCRAPE` | Scraping überspringen | `false` |
| `FDI_SKIP_TRAIN` | Training überspringen | `false` |
| `FDI_SCRAPE_MAX_PLAYERS` | Spieler-Limit (leer = alle) | - |
| `FDI_SCRAPE_DELAY_SECONDS` | Delay zwischen Requests | `0` |
| `FDI_REFRESH_MINUTES` | Scheduler-Intervall | `10080` (7 Tage) |
| `APP_PORT` | Gradio-Port | `7860` |
| `DATABASE_URL` | PostgreSQL-Connection | siehe `.env` |

Vollständige Liste: siehe [.env.example](.env.example)

---

## 🛠️ Entwicklung

### Setup

```bash
# Dependencies installieren
uv sync

# Tests ausführen
uv run pytest

# Linting
uv run ruff check .

# Formatierung
uv run ruff format .
```

### Modell-Artefakte aktualisieren

```bash
# Training lokal ausführen
uv run python -m pipeline.train

# Container neu starten (lädt neues Modell)
docker compose restart app
```

---

## 💡 Lessons Learned

1. **Log-Transformation**: Earnings sind stark rechtsschief – ohne Log dominieren Ausreißer
2. **Regularisierung**: Lasso reduziert Überanpassung und selektiert automatisch Features
3. **Data Leakage vermeiden**: API-Rankings (`api_rank`) korrelieren perfekt mit Target → entfernt
4. **Feature Engineering**: Abgeleitete Features wie `first9_delta` verbessern Interpretierbarkeit

---

## 📚 Referenzen

- [Introduction to Modern Statistics](https://openintro-ims.netlify.app/) – OpenIntro
- [DartsOrakel](https://dartsorakel.com) – Datenquelle

---

## 📄 Lizenz

MIT License – siehe [LICENSE](LICENSE)

---

<p align="center">
  <i>Entwickelt für das Modul "Data Analytics with Statistics" an der HdM Stuttgart</i>
</p>
