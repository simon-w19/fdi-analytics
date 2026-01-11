# Präsentation (Kurzfassung)

**Format:** 6–8 Slides, PDF-Export für Abgabe. Folie-zu-Folie-Stichpunkte unten nutzbar als Vorlage in Google Slides / LibreOffice.

---

## 1) Title & Motivation
- End-to-End Darts Analytics – Prognose des FDI-Ratings
- Warum? FDI kombiniert Skill + Momentum besser als Order of Merit
- Stack: Scraper → Transform → PostgreSQL → sklearn → Gradio

## 2) Datenbasis
- Quelle: dartsorakel.com, Snapshot `player_stats_all.csv`
- 2.9k Spieler:innen, ~30 Features; Target: `profile_fdi_rating` (2.4k beobachtet)
- Hard Facts: Averages, Checkout, Legs-Win%; Soft Facts: log Earnings, Tour-Card-Jahre, Country

## 3) EDA Highlights
- Verteilungen: FDI rechtsschief; Earnings nur nach log sinnvoll
- Starke Korrelationen: First-9 Avg, Checkout-% ↔ FDI
- Länder-Cluster: ENG/NED breite Streuung, DACH enger

## 4) Feature Engineering & Pipeline
- Log-Transformation Earnings; Kombifeatures: `checkout_combo`, `first9_delta`, `momentum_gap`, Ratios (`first9_ratio`, `power_scoring_ratio`)
- Preprocessing: Median/Mode-Imputation, StandardScaler, OneHotEncoder (Country)
- Data Leakage entfernt: `api_rank`, `api_overall_stat`

## 5) Modelle & Metriken (Test-Split)
- Linear Regression: R² 0.928, MAE 35.26, RMSE 46.35
- Lasso (α=0.01): **R² 0.929**, MAE 35.27, RMSE 46.15 (Best aktuell)
- Random Forest (tuned): R² 0.923, MAE 37.35, RMSE 47.94
- 5-fold CV + 80/20 Holdout, Artefakte: `models/best_fdi_pipeline.joblib`, `reports/metrics/latest_metrics.json`

## 6) Modell-Insights
- Wichtigste Treiber: First-9 Avg, Checkout-%, Legs-Win-%; log Earnings liefert Langfrist-Signal
- Residuen: kein starkes Muster; optionale Checks via Breusch-Pagan & Cook's Distance

## 7) Conclusion & Next Steps
- Regularisierte Linear-Modelle genügen aktuell; Baummodelle bringen keine Mehrleistung
- Lücken: fehlende Alters-/Erfahrungswerte, keine Zeitreihen-Features
- Ausblick: Rolling-Formfeatures, MLflow-Tracking, Drift-Monitoring im Scheduler

## 8) Backup / Demo-Hinweise
- Gradio Tabs: Prediction Studio (Single Player), Insights (EDA, Leaderboard)
- Aktualisierung: `uv run python -m pipeline.train` → Artefakte → `docker compose restart app`
