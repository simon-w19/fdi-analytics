# FDI Analytics – Unterrichtsleitfaden

Kurzskript für das Modul **Data Analytics with Statistics**. Die Struktur folgt den Gradio-Tabs und greift direkt auf die in `app/gradio_app.py` eingebauten Visualisierungen zurück.

## 1. Einstieg (ca. 3 Minuten)
- Zeige den Daten-Snapshot im Insights-Tab.
- Erkläre die Pipeline in drei Sätzen: Scrape → Feature Engineering → PostgreSQL → Gradio.
- Nenne die wichtigsten Kennzahlen: Beobachtungen, Länder, Feature-Typen, Zeitpunkt des letzten Scrapes.

## 2. Modell-Leaderboard & Methodik (ca. 5 Minuten)
- Öffne das Accordion "Modell-Leaderboard".
- Interpretiere MAE/RMSE/$R^2$ und betone den Vergleich zwischen Linear, Lasso und Random Forest.
- Zeige `reports/metrics/latest_metrics.json` bei Bedarf, um zu erklären, wie die Werte gespeichert werden.

## 3. Explorative Analyse (ca. 8 Minuten)
- "Feature-Korrelationen zum FDI": nutze den Balkenplot, um Führungsmetriken zu diskutieren (z. B. `last_12_months_averages`, `season_win_rate`).
- "Country Performance Board": erläutere Cluster (z. B. ENG/NED vs. Emerging Markets) und wie Spieleranzahl und FDI zusammenhängen.
- "Top 15 Spieler": zeige konkrete Beispiele und verbinde sie mit sichtbaren Korrelationen.

## 4. Modellinterpretation (ca. 6 Minuten)
- Im Accordion "Model Feature Impact" zeigst du die Feature-Importances bzw. Koeffizienten.
- Erläutere, wie sich harte Stats (Scoring, Checkout) und Soft-Facts (Log-Earnings, Momentum) gegenseitig ergänzen.
- Betone, dass die Gradio-App die vollständige Feature-Zeile zur Prediction mitliefert (Tab "Prediction Studio").

## 5. Live-Demo Prediction (ca. 5 Minuten)
- Wechsel zurück zum Tab "Prediction Studio".
- Wähle einen Spieler aus oder fülle ein fiktives Profil.
- Zeige, wie der Feature-Vektor visualisiert wird und interpretiere das vorhergesagte Rating.

## 6. Diskussion & Ausblick (ca. 3 Minuten)
- Hinweise auf mögliche Erweiterungen (MLflow, weitere Features, Monitoring).
- Frage die Studierenden nach Ideas für zusätzliche Visualisierungen oder Validierungsmetriken.

### Praktische Hinweise
- Vor Beginn `uv run python -m pipeline.train` ausführen, damit `models/best_fdi_pipeline.joblib` und `reports/metrics/latest_metrics.json` aktuell sind.
- `docker compose up -d` starten und auf das grüne Health-Signal des App-Containers warten.
- Für Screenshots oder Offline-Demos kannst du `uv run python app/gradio_app.py` lokal starten; die gleiche Insights-Ansicht steht dann bereit.
