# Copilot Anweisungen für dieses Projekt

- **Sprache:** Antworte immer auf Deutsch, aber halte Code-Kommentare auf Englisch.
- **Tech Stack:** Wir nutzen vor allem Python und als Paketmanager uv.

Das ist mein grunsätzliches Projektvorhaben:

Projekt-Titel (Arbeitstitel):
"End-to-End Darts Analytics: Prädiktive Modellierung des FDI-Ratings mittels einer containerisierten Data-Pipeline"

Das Ziel
Du entwickelst eine vollständige Data Science Pipeline – von der Datenbeschaffung bis zum Deployment –, um das FDI Rating (Future Dart Intelligence) professioneller Darts-Spieler vorherzusagen. Ziel ist es, herauszufinden, ob statistische "Hard Facts" in Kombination mit psychologischen/exterenen "Soft Facts" das Rating akkurat abbilden können.

Die 3 Säulen der Umsetzung
1. Data Engineering & Infrastruktur (Die "Extra Meile")
Statt einfacher CSV-Dateien baust du eine professionelle Backend-Architektur auf:

Server: Nutzung deines Hetzner Cloud Servers (Linux).

Datenhaltung: Self-hosted PostgreSQL Datenbank in einem Docker Container (statt fertiger Cloud-Lösungen), um technisches Verständnis von Datenbank-Management zu beweisen.

ETL-Prozess: Ein selbst entwickelter Python-Scraper (BeautifulSoup/Playwright) extrahiert Daten von Darts Orakel und PDC/Wikipedia und speichert sie strukturiert in der DB.

2. Statistische Analyse & Feature Engineering
Du analysierst einen Datensatz mit mindestens 10 Features, wobei du über reine Durchschnittswerte hinausgehst:

Hard Stats: 3-Dart Average, Checkout-Quote, 180er pro Leg, First 9 Average.

Enriched Features:

Win %: Als Maß für "Clutch"-Fähigkeit (Effizienz beim Gewinnen).

Log Prize Money: Logarithmiertes Preisgeld als Indikator für langfristigen Erfolg (unter Berücksichtigung der schiefen Einkommensverteilung).

World Ranking: Externe Validierung der Spielstärke.

Methodik: Du vergleichst ein Lineares Regressionsmodell (Baseline) mit komplexeren Ansätzen wie Random Forest oder Regularisierter Regression (Lasso), um Nicht-Linearitäten und Multikollinearität zu untersuchen.

3. Deployment & Visualisierung
Das Ergebnis ist nicht nur ein PDF-Report, sondern eine nutzbare Anwendung:

Frontend: Eine Streamlit Web-App, die ebenfalls via Docker auf dem Server läuft.

Funktion: User können Spieler-Stats eingeben (oder auswählen) und das Modell prognostiziert das FDI-Rating in Echtzeit.