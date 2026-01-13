# 🎯 FDI Analytics – Präsentation (20 Minuten)

> **End-to-End Darts Analytics: Prädiktive Modellierung des FDI-Ratings**
> 
> Modul: Data Analytics with Statistics | HdM Stuttgart

---

## 📑 Agenda & Zeitplan

| # | Thema | Zeit | Kumuliert |
|---|-------|------|-----------|
| 1 | Motivation & Problemstellung | 2 min | 2 min |
| 2 | Datenbasis & Quellen | 2 min | 4 min |
| 3 | Data Architecture | 3 min | 7 min |
| 4 | Explorative Datenanalyse | 4 min | 11 min |
| 5 | Feature Engineering & Modellierung | 4 min | 15 min |
| 6 | Ergebnisse & Evaluation | 2 min | 17 min |
| 7 | Live-Demo | 2 min | 19 min |
| 8 | Fazit & Ausblick | 1 min | 20 min |

---

# 1️⃣ Motivation & Problemstellung (2 min)

## Was ist das FDI-Rating?

**FDI = Future Dart Intelligence**

- Von [DartsOrakel](https://dartsorakel.com) entwickeltes Spieler-Rating
- Kombiniert **aktuelle Form** + **historische Performance** + **Momentum**
- Besser als reines Order-of-Merit-Ranking (nur Preisgeld-basiert)

## Forschungsfrage

> *Können wir das FDI-Rating anhand statistischer "Hard Facts" (Averages, Checkout-%) und psychologischer "Soft Facts" (Earnings, Erfahrung) akkurat vorhersagen?*

## Warum ist das relevant?

- **Data-Driven Decision Making** im Sport-Analytics
- Verständnis der **Erfolgsfaktoren** professioneller Darts-Spieler
- **End-to-End Machine Learning Pipeline** als Praxisbeispiel

---

# 2️⃣ Datenbasis & Quellen (2 min)

## Datenquelle

**DartsOrakel.com** – Die umfassendste Darts-Statistik-Plattform

- ~2.900 Spieler weltweit
- ~30+ Raw-Features pro Spieler
- Wöchentliches Update der Statistiken

## Datensatz-Überblick

| Metrik | Wert |
|--------|------|
| **Beobachtungen** | 2.978 Spieler |
| **Mit FDI-Rating** | ~2.500 (83%) |
| **Features (roh)** | 34 |
| **Features (engineered)** | 38 + Country |
| **Zielvariable** | `profile_fdi_rating` (50–800) |

## Feature-Kategorien

### Hard Facts (Performance-Metriken)
- 3-Dart Average, First-9 Average
- Checkout-%, Functional Doubles %
- 180er, 140er pro Leg

### Soft Facts (Erfolgs-Indikatoren)
- Preisgeld (log-transformiert)
- Tour Card Years
- Order of Merit Ranking
- Ländercode

---

# 3️⃣ Data Architecture (3 min)

## Pipeline-Übersicht

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  🕷️ Scraper │────▶│ 🔧 Transform│────▶│ 🗄️ Postgres │
│ BeautifulSoup     │ Feature Eng.│     │   Database  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│ 🌐 Gradio   │◀────│ 🤖 Training │◀───────────┘
│   Web-App   │     │  sklearn    │
└─────────────┘     └─────────────┘
```

## Technologie-Stack

| Komponente | Technologie |
|------------|-------------|
| **Scraping** | BeautifulSoup, Requests, Playwright (fallback) |
| **Datenbank** | PostgreSQL 16 (Docker Container) |
| **Feature Engineering** | pandas, NumPy |
| **Modellierung** | scikit-learn (Pipeline, GridSearchCV) |
| **Web-App** | Gradio + FastAPI |
| **Deployment** | Docker Compose |
| **Package Manager** | uv |

## Docker-Services

```yaml
services:
  db:         # PostgreSQL mit persistentem Volume
  etl:        # Scrape → Transform → Train → Load
  scheduler:  # Wöchentliches Auto-Refresh
  app:        # Gradio UI + REST API
```

## "Extra Meile"

- **Self-hosted PostgreSQL** statt Cloud-Lösung
- **Vollautomatisierte Pipeline** – ein `docker compose up` reicht
- **Scheduler** für regelmäßige Aktualisierung
- **REST API** für programmatischen Zugriff

---

# 4️⃣ Explorative Datenanalyse (4 min)

## Verteilung der Zielvariable

Das FDI-Rating ist **annähernd normalverteilt** mit leichter Rechtsschiefe:

- **Median**: ~180 FDI-Punkte
- **Mean**: ~195 FDI-Punkte
- **Range**: 50–800+

→ Lineare Modelle sind geeignet ✅

## Korrelationsanalyse

**Top-5 Korrelationen mit FDI:**

| Feature | Korrelation |
|---------|-------------|
| last_12_months_first_9_averages | 0.89 |
| last_12_months_averages | 0.88 |
| last_12_months_checkout_pcnt | 0.75 |
| last_12_months_pcnt_legs_won | 0.72 |
| log_total_earnings | 0.68 |

## Key Insights

### 1. Earnings brauchen Log-Transformation

- Rohwerte: Extrem rechtsschief (wenige Top-Verdiener)
- Nach Log: Annähernd normalverteilt

### 2. Multikollinearität

- `first_9_averages` ↔ `averages`: r = 0.99
- `checkout_pcnt` ↔ `functional_doubles_pcnt`: r = 0.99

→ Feature-Reduktion oder Regularisierung nötig

### 3. Länder-Cluster

- **ENG/NED**: Breite Streuung (Amateure bis Elite)
- **DACH**: Engere Verteilung
- **Emerging Markets**: Wenige Spieler, niedrigere FDI

---

# 5️⃣ Feature Engineering & Modellierung (4 min)

## Engineered Features

| Feature | Formel | Intuition |
|---------|--------|-----------|
| `log_total_earnings` | log(earnings + 1) | Normalisiert Ausreißer |
| `first9_delta` | first_9_avg - overall_avg | "Hot Start" Indikator |
| `momentum_gap` | with_throw - against_throw | Anwurf-Vorteil |
| `break_efficiency` | legs_won_2nd / total_legs | "Clutch"-Faktor |
| `power_scoring_ratio` | 180s / (171-180s) | Konsistenz im Power-Scoring |
| `experience_intensity` | tour_years / age | Karriere-Dichte |

## Preprocessing Pipeline

```python
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])
```

## Modellvergleich

Drei Modelle mit **80/20 Holdout** + **5-fold Cross-Validation**:

| Modell | R² | MAE | RMSE | CV MAE |
|--------|-----|-----|------|--------|
| Linear Regression | 0.928 | 35.4 | 46.4 | 38.6 ±0.56 |
| **Lasso (α=0.01)** | **0.928** | **35.4** | **46.2** | **38.5 ±0.47** |
| Random Forest | 0.923 | 37.4 | 48.1 | 40.3 ±1.09 |

## Hyperparameter-Tuning

**Lasso** (GridSearchCV):
- `alpha`: [0.0001, 0.001, 0.01] → **0.01**
- `max_iter`: [20000, 40000] → **40000**

**Random Forest** (GridSearchCV):
- `n_estimators`: [300, 600, 900] → **900**
- `max_depth`: [10, 12, 16, None] → **None**
- `min_samples_leaf`: [1, 2, 4] → **1**

## Ergebnis

> **Lasso gewinnt** – gleiche Performance wie Linear Regression, aber robuster gegen Multikollinearität

---

# 6️⃣ Ergebnisse & Evaluation (2 min)

## KPI-Dashboard

| KPI | Ist | Soll | Status |
|-----|-----|------|--------|
| **MAE** | 35.4 FDI | < 40 | ✅ |
| **R²** | 0.928 | > 0.85 | ✅ |
| **CV-Robustheit** | ±0.47 | < ±5 | ✅ |
| **Interpretierbar** | Top 5 Features | Ja | ✅ |
| **Production Ready** | Docker + <100ms | 24/7 | ✅ |

## Residuen-Diagnostik

| Test | Wert | Interpretation |
|------|------|----------------|
| **Durbin-Watson** | 1.99 | Keine Autokorrelation ✅ |
| **Breusch-Pagan p** | 0.95 | Homoskedastizität ✅ |
| **Cook's Distance** | 99.8% < Threshold | Keine dominanten Outlier ✅ |

## Feature Importance (Lasso Koeffizienten)

1. **First-9 Average** → Starke Früh-Game-Performance
2. **Checkout %** → Finish-Qualität unter Druck
3. **Legs Won %** → Konsistente Gewinnfähigkeit
4. **Log Earnings** → Langfristiger Erfolg
5. **Season Win Rate** → Aktuelle Form

---

# 7️⃣ Live-Demo (2 min)

## Gradio Web-App

**URL**: http://localhost:7860 (oder Server-IP)

### Tab 1: Prediction Studio

1. Spieler aus Dropdown wählen (z.B. "Luke Littler")
2. Features werden automatisch gefüllt
3. "FDI Rating vorhersagen" klicken
4. Vergleich: Vorhersage vs. DartsOrakel-Referenz

### Tab 2: Insights & EDA

- **Modell-Leaderboard**: Vergleich aller trainierten Modelle
- **Feature-Korrelationen**: Interaktiver Balkenplot
- **Top-15 Spieler**: Tabelle mit höchsten FDI-Ratings
- **Country Performance**: Länder-Statistiken

### API-Endpunkte

```bash
# Health Check
curl http://localhost:7860/api/health

# Prediction
curl -X POST http://localhost:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{"last_12_months_averages": 95.5, ...}'
```

---

# 8️⃣ Fazit & Ausblick (1 min)

## Zusammenfassung

✅ **Forschungsfrage beantwortet**: Hard Facts sagen FDI gut vorher (R² = 0.93)

✅ **End-to-End Pipeline**: Vom Scraping bis zum Deployment weitesgehend automatisiert

✅ **Production-Ready**: Containerisiert, API-fähig, Scheduler für Updates

## Limitationen

- **Keine Zeitreihen**: Aktuell nur Snapshot, keine Form-Entwicklung
- **Fehlende Features**: Psychologische Faktoren (Nervenstärke) nicht messbar

## Ausblick

- **Rolling Form-Features**: Gleitender Durchschnitt der letzten N Turniere
- **Turnier-Typ-Encoding**: Major vs. Floor Event Unterscheidung
- **MLflow Integration**: Experiment-Tracking für A/B-Tests
- **Drift-Monitoring**: Automatische Alerts bei Modell-Degradation

---

# 🙏 Vielen Dank!

## Fragen?

**Repository**: github.com/yourusername/fdi-analytics

**Stack**: Python • PostgreSQL • Docker • scikit-learn • Gradio

---

# 📎 Backup-Slides

## Appendix A: Vollständige Feature-Liste

<details>
<summary>38 Features (klicken zum Ausklappen)</summary>

**Basis-Features:**
- age, profile_total_earnings, log_total_earnings
- profile_9_darters, profile_season_win_pct, season_win_rate
- profile_tour_card_years, profile_highest_average
- profile_highest_tv_average, profile_order_of_merit

**Performance (letzte 12 Monate):**
- last_12_months_averages, last_12_months_first_9_averages
- last_12_months_first_3_averages
- last_12_months_with_throw_averages, last_12_months_against_throw_averages
- last_12_months_highest_checkout, last_12_months_checkout_pcnt
- last_12_months_functional_doubles_pcnt
- last_12_months_pcnt_legs_won
- last_12_months_pcnt_legs_won_throwing_first/second
- last_12_months_180_s, last_12_months_171_180_s
- last_12_months_140_s, last_12_months_131_140_s, api_sum_field2

**Engineered:**
- first9_delta, momentum_gap, checkout_combo
- experience_intensity, earnings_per_year, first9_ratio
- break_efficiency, hold_break_spread, power_scoring_ratio, tv_stage_delta

**Kategorisch:** country (One-Hot Encoded)

</details>

## Appendix B: Multikollinearitäts-Matrix

Die höchsten Korrelationen (|r| > 0.95):

| Feature Paar | Korrelation |
|--------------|-------------|
| first_9_averages ↔ averages | 0.99 |
| checkout_pcnt ↔ functional_doubles | 0.99 |
| 180s ↔ 171_180s | 0.98 |
| with_throw ↔ against_throw | 0.97 |

→ Lasso-Regularisierung penalisiert redundante Features automatisch

## Appendix C: Residuen-Plot

Die Residuen zeigen:
- Keine systematischen Muster (keine Heteroskedastizität)
- Annähernde Normalverteilung
- Wenige Ausreißer (< 5% influential points)

## Appendix D: Docker-Befehle

```bash
# Kompletter Neuaufbau
docker compose down -v --rmi all
docker compose build --no-cache
docker compose up -d

# Nur App neu starten (nach Training)
docker compose restart app

# Logs
docker compose logs -f etl
docker compose logs -f app
```
