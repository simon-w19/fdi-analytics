# Data Science Project Template

This repository serves as a structured template for Data Science projects. It is designed to help you organize your code, data, and reports efficiently.

## 🚀 Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. `uv` is a fast Python package installer and resolver.

### 1. Prerequisites

First, install `uv` on your system. You can find detailed installation instructions here: [https://github.com/kirenz/uv-setup](https://github.com/kirenz/uv-setup).

### 2. Setup

Clone this repository and navigate to the project folder:

```bash
git clone https://github.com/kirenz/template-repo
cd template-repo
```

Install the project dependencies:

```bash
uv sync
```

This command will create a virtual environment and install all necessary packages defined in `pyproject.toml`.


## 📂 Project Structure

```nohighlight
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks. Naming convention: number (for ordering),
│                         the creator's initials, and a short `-` delimited description, 
│                         e.g. `01-jdoe-data-preparation.ipynb`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting.
│
├── pyproject.toml     <- The configuration file for project dependencies (managed by uv).
└── uv.lock            <- The lock file ensuring reproducible environments.
```

## 📦 Managing Dependencies

To add a new package (e.g., `requests`):

```bash
uv add requests
```

To remove a package:

```bash
uv remove requests
```

## 🛰️ Deployment Preview (Gradio)

1. Train & export the Pipeline in `notebooks/fdi_rating_modeling.ipynb` (Cell "Hyperparameter-Tuning & zeitliche Validierung") – this writes `models/best_fdi_pipeline.joblib`.
2. Starte die Gradio-App via `uv run python app/gradio_app.py`.
3. Wähle ein bestehendes Spielerprofil oder passe Features manuell an, um das FDI-Rating live zu prognostizieren.

## 🗂️ Environment Variables

Kopiere `.env.example` nach `.env` und passe die Werte an:

```bash
cp .env.example .env
```

Wichtige Parameter:

- `POSTGRES_*`: Zugangsdaten für die self-hosted PostgreSQL-Instanz (werden von Docker Compose genutzt).
- `DATABASE_URL`: SQLAlchemy-kompatible URL, die auch von der App/ETL verwendet wird.
- `FDI_CSV_PATH`: Pfad zur CSV, die zusätzlich in die Datenbank geladen wird.
- `FDI_REFRESH_MINUTES`: Intervall in Minuten, nach dem der Scheduler die ETL-Route erneut ausführt (Standard 10 080 Minuten ≙ 1 Woche).
- `APP_PORT`: Externer Port der Gradio-App.

## 🐘 Datenbank-Befüllung

Neben den CSV-Dateien können alle aufbereiteten Spieler-Statistiken in PostgreSQL persistiert werden:

```bash
uv run python -m pipeline.ingest
```

Der Job liest `data/processed/player_stats_all.csv` (oder den via `FDI_CSV_PATH` gesetzten Pfad) und schreibt die Daten in die konfigurierte Tabelle/Schemenstrukur.

## 🐳 Docker & Server Deployment

Das Repository enthält eine vollständige Docker-Compose-Umgebung (`docker-compose.yml`). Sie umfasst:

- `db`: PostgreSQL 16 mit persistentem Volume.
- `etl`: Ein Einmal-Container, der den Ingestion-Job `pipeline.ingest` ausführt und die CSV in die Datenbank lädt.
- `scheduler`: Ein Dauerdienst, der dieselbe Ingestion in dem via `FDI_REFRESH_MINUTES` definierten Intervall neu startet.
- `app`: Die Gradio-Anwendung auf Basis des trainierten Pipelines.

### Build & Run

```bash
cp .env.example .env  # falls noch nicht geschehen
docker compose build
docker compose up
```

Die App steht anschließend unter `http://localhost:${APP_PORT}` bereit. Die Datenbank lauscht auf `localhost:${POSTGRES_PORT}` (Standard 5432) und enthält die Tabelle `public.player_stats`.

### Aktualisierte Daten laden

Falls du die CSV aktualisierst, kannst du entweder auf den Scheduler warten oder sofort einen manuellen Lauf starten:

```bash
docker compose run --rm etl
```

Der Scheduler verarbeitet Änderungen automatisch nach Ablauf des Intervalls. Logs siehst du z. B. mit `docker compose logs -f scheduler`.

So bleibt sowohl die Datenbank als auch die App konsistent.



