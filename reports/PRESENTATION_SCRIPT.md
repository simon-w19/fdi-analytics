# FDI Analytics – Präsentationsskript (20 Minuten)

> Vollständiger Fließtext zum Vortragen

---

## 1. Motivation & Problemstellung (2 Minuten)

Guten Tag und herzlich willkommen zu meiner Präsentation über End-to-End Darts Analytics.

Ich möchte heute zeigen, wie man mit einer vollständigen Data-Science-Pipeline das sogenannte FDI-Rating von Darts-Spielern vorhersagen kann. FDI steht für "Future Dart Intelligence" und ist ein von der Plattform DartsOrakel entwickeltes Spieler-Rating. Im Gegensatz zum klassischen Order-of-Merit-Ranking, das nur auf Preisgeldern basiert, berücksichtigt das FDI-Rating auch die aktuelle Form, historische Performance und das Momentum eines Spielers.

Die zentrale Forschungsfrage meines Projekts lautet: Können wir dieses FDI-Rating anhand statistischer Hard Facts – also Leistungsmetriken wie Averages und Checkout-Quoten – in Kombination mit Soft Facts wie Preisgeldern und Erfahrung akkurat vorhersagen?

Das Projekt ist aus mehreren Gründen relevant: Erstens demonstriert es Data-Driven Decision Making im Bereich Sport-Analytics. Zweitens hilft es uns zu verstehen, welche Faktoren einen erfolgreichen Darts-Spieler ausmachen. Und drittens dient es als praktisches Beispiel für eine End-to-End Machine Learning Pipeline – vom Web-Scraping bis zum produktionsreifen Deployment.

---

## 2. Datenbasis & Quellen (2 Minuten)

Die Daten für dieses Projekt stammen von DartsOrakel.com, der wohl umfassendsten Statistik-Plattform für professionelles Darts. Die Seite bietet detaillierte Statistiken zu fast 3.000 Spielern weltweit und wird wöchentlich aktualisiert.

Mein Datensatz umfasst knapp 2.980 Spieler, wobei etwa 2.500 davon – also rund 83 Prozent – ein gültiges FDI-Rating haben. Die restlichen Spieler haben zu wenige Spiele absolviert, um ein aussagekräftiges Rating zu erhalten. Pro Spieler habe ich ursprünglich 34 Rohfeatures gesammelt, die ich durch Feature Engineering auf 38 numerische plus ein kategorisches Feature erweitert habe. Die Zielvariable ist das FDI-Rating, das sich typischerweise zwischen 50 und 800 Punkten bewegt.

Die Features lassen sich in zwei Kategorien einteilen: Auf der einen Seite haben wir die Hard Facts, also reine Performance-Metriken. Dazu gehören der 3-Dart Average, der First-9 Average, die Checkout-Quote, die Functional Doubles Quote sowie die Anzahl der 180er und 140er pro Leg. Auf der anderen Seite stehen die Soft Facts, die eher Erfolgs- und Erfahrungsindikatoren darstellen. Hierzu zählen das Gesamtpreisgeld, die Anzahl der Jahre mit Tour Card, das Order-of-Merit-Ranking sowie der Ländercode des Spielers.

---

## 3. Data Architecture (3 Minuten)

Kommen wir zur technischen Architektur des Projekts. Die Pipeline besteht aus fünf Hauptkomponenten, die alle containerisiert mit Docker Compose laufen.

Am Anfang steht der Scraper, der mit BeautifulSoup die Spielerdaten von DartsOrakel extrahiert. Die Rohdaten werden als CSV gespeichert. Anschließend übernimmt die Transform-Komponente das Feature Engineering – also die Erstellung neuer Features, Normalisierung und Bereinigung der Daten. Die transformierten Daten werden dann in eine PostgreSQL-Datenbank geladen, die ebenfalls als Docker Container läuft. Parallel dazu trainiert das Train-Modul die Machine-Learning-Modelle mit scikit-learn und speichert das beste Modell als joblib-Datei. Schließlich lädt die Gradio Web-App dieses Modell und stellt eine Benutzeroberfläche sowie eine REST-API für Vorhersagen bereit.

Was den Technologie-Stack angeht: Für das Scraping nutze ich BeautifulSoup mit Requests als primärem HTTP-Client und Playwright als Fallback für JavaScript-lastige Seiten. Die Datenbank ist ein PostgreSQL 16 in der Alpine-Variante mit einem persistenten Docker Volume. Feature Engineering und Modellierung laufen komplett mit pandas, NumPy und scikit-learn. Die Web-App kombiniert Gradio für die UI mit FastAPI für die API-Endpunkte. Das Ganze wird mit Docker Compose orchestriert, und als Package Manager verwende ich uv.

Was ich als "Extra Meile" bezeichnen würde: Ich habe bewusst eine self-hosted PostgreSQL-Instanz gewählt statt einer fertigen Cloud-Lösung, um technisches Verständnis von Datenbank-Management zu demonstrieren. Die Pipeline ist vollautomatisiert – ein einziges "docker compose up" reicht aus. Außerdem gibt es einen Scheduler für regelmäßige Datenaktualisierungen und eine REST-API für programmatischen Zugriff.

---

## 4. Explorative Datenanalyse (4 Minuten)

Bevor wir Modelle bauen, müssen wir natürlich die Daten verstehen. Beginnen wir mit der Zielvariable, dem FDI-Rating.

Die Verteilung des FDI-Ratings ist annähernd normalverteilt mit einer leichten Rechtsschiefe. Der Median liegt bei etwa 180 Punkten, der Mittelwert etwas höher bei 195 Punkten. Die Spannweite reicht von etwa 50 bis über 800 Punkte für die absoluten Top-Spieler wie Luke Littler oder Michael van Gerwen. Diese annähernde Normalverteilung ist eine gute Nachricht, denn sie bedeutet, dass lineare Modelle grundsätzlich geeignet sein sollten.

In der Korrelationsanalyse habe ich untersucht, welche Features am stärksten mit dem FDI-Rating zusammenhängen. Die Top-5 sind: Erstens der First-9 Average mit einer Korrelation von 0,89 – das zeigt, wie wichtig ein starker Start in die Legs ist. Zweitens der allgemeine 3-Dart Average mit 0,88. Drittens die Checkout-Quote mit 0,75, also die Fähigkeit, unter Druck die Doppelfelder zu treffen. Viertens der Prozentsatz gewonnener Legs mit 0,72. Und fünftens das logarithmierte Preisgeld mit 0,68.

Bei der Analyse sind mir drei wichtige Erkenntnisse aufgefallen: 

Erstens musste ich das Preisgeld log-transformieren. Die Rohwerte sind extrem rechtsschief – wenige Top-Verdiener wie van Gerwen mit über 10 Millionen Pfund Karriere-Preisgeld verzerren sonst die gesamte Analyse. Nach der Logarithmierung ist die Verteilung deutlich normaler und für lineare Modelle besser geeignet.

Zweitens habe ich massive Multikollinearität entdeckt. Der First-9 Average korreliert mit 0,99 mit dem allgemeinen Average – logisch, da der First-9 ein Subset des gesamten Averages ist. Ähnlich verhält es sich bei der Checkout-Quote und den Functional Doubles. Das bedeutet, dass wir entweder Features reduzieren oder Regularisierung einsetzen müssen, um stabile Koeffizienten zu erhalten.

Drittens zeigen sich interessante Länder-Cluster. England und die Niederlande haben eine breite Streuung von Amateuren bis Elite-Spielern. Die DACH-Region zeigt eine engere Verteilung im mittleren Bereich. Und Emerging Markets wie Asien oder Südamerika haben wenige Spieler mit tendenziell niedrigeren FDI-Werten.

---

## 5. Feature Engineering & Modellierung (4 Minuten)

Basierend auf der EDA habe ich mehrere neue Features entwickelt, um zusätzliche Informationen zu extrahieren.

Das wichtigste ist die Log-Transformation des Preisgeldes, die ich bereits erwähnt habe. Darüber hinaus habe ich den "First-9 Delta" berechnet – die Differenz zwischen First-9 Average und Gesamt-Average. Ein positiver Wert zeigt an, dass ein Spieler besonders stark in die Legs startet. Der "Momentum Gap" misst den Unterschied zwischen Average mit Anwurf und gegen Anwurf – quasi den Heimvorteil. Die "Break Efficiency" zeigt, wie oft ein Spieler Legs gewinnt, wenn er nicht den Anwurf hat. Die "Power Scoring Ratio" setzt 180er ins Verhältnis zu allen Würfen im 171-180 Bereich und misst damit die Konsistenz im Power-Scoring. Und die "Experience Intensity" dividiert die Tour Card Jahre durch das Alter, um die Karrieredichte zu messen.

Für das Preprocessing habe ich eine scikit-learn ColumnTransformer Pipeline gebaut. Numerische Features werden zuerst mit dem Median imputiert – das ist robuster gegen Ausreißer als der Mittelwert – und dann mit einem StandardScaler normalisiert. Kategorische Features, also primär das Land, werden mit dem häufigsten Wert imputiert und dann One-Hot encoded. Das ergibt nach dem Encoding etwa 108 Features insgesamt.

Für den Modellvergleich habe ich drei Ansätze getestet: Lineare Regression als Baseline, Lasso für Regularisierung, und Random Forest als nicht-lineares Modell. Die Evaluation erfolgte mit einem 80/20 Train-Test-Split plus 5-fold Cross-Validation.

Die Ergebnisse zeigen: Lineare Regression erreicht ein R² von 0,928 und einen MAE von 35,4 FDI-Punkten. Lasso mit einem Alpha von 0,01 kommt auf praktisch identische Werte, hat aber eine etwas niedrigere Standardabweichung in der Cross-Validation. Random Forest schneidet überraschenderweise schlechter ab mit einem R² von 0,923 und einem MAE von 37,4 Punkten.

Das Hyperparameter-Tuning lief über GridSearchCV. Für Lasso habe ich verschiedene Alpha-Werte getestet – 0,01 hat sich als optimal erwiesen. Für Random Forest habe ich Estimators, Tiefe und Minimum Samples variiert, aber selbst die beste Konfiguration konnte die linearen Modelle nicht schlagen.

Mein Fazit: Lasso ist der Gewinner. Es erreicht die gleiche Performance wie die einfache lineare Regression, ist aber robuster gegen die Multikollinearität in den Daten, weil es automatisch redundante Features penalisiert.

---

## 6. Ergebnisse & Evaluation (2 Minuten)

Schauen wir uns die finalen Ergebnisse anhand der definierten KPIs an.

Die Modellgenauigkeit, gemessen am MAE, liegt bei 35,4 FDI-Punkten. Das bedeutet, dass unsere Vorhersagen im Durchschnitt etwa 35 Punkte vom tatsächlichen FDI-Rating abweichen. Bei einer Skala von 50 bis 800 Punkten ist das ein sehr guter Wert – er liegt deutlich unter dem Zielwert von 40 Punkten.

Das R² von 0,928 bedeutet, dass unser Modell 92,8 Prozent der Varianz im FDI-Rating erklären kann. Auch das übertrifft den Zielwert von 85 Prozent deutlich.

Die Cross-Validation zeigt eine Standardabweichung von nur 0,47 Punkten beim MAE. Das beweist, dass das Modell robust ist und nicht von der zufälligen Aufteilung der Trainingsdaten abhängt. Der Zielwert war weniger als 5 Punkte Abweichung.

Was die Residuen-Diagnostik angeht: Der Durbin-Watson-Test ergibt 1,99 – nahe am idealen Wert von 2, was keine Autokorrelation der Residuen anzeigt. Der Breusch-Pagan-Test hat einen p-Wert von 0,95, was auf Homoskedastizität hindeutet – also konstante Fehlervarianz über alle Vorhersagebereiche. Und bei Cook's Distance liegen über 99 Prozent der Datenpunkte unter dem kritischen Schwellenwert, es gibt also keine dominanten Ausreißer, die das Modell verzerren.

Zusammenfassend: Alle sechs definierten KPIs sind erfüllt oder übertroffen. Das Projekt ist damit aus statistischer Sicht ein Erfolg.

---

## 7. Live-Demo (2 Minuten)

Jetzt möchte ich Ihnen kurz die Web-App zeigen, die ich für dieses Projekt entwickelt habe.

*[Browser öffnen, http://localhost:7860 aufrufen]*

Die App hat zwei Haupttabs. Im ersten Tab, dem "Prediction Studio", können wir Vorhersagen für einzelne Spieler machen. Oben gibt es ein Dropdown mit den Top-100-Spielern für schnellen Zugriff. Wenn ich zum Beispiel "Luke Littler" auswähle, werden automatisch alle Features aus unserer Datenbank geladen. Sie sehen hier den Namen, das Land, und rechts das aktuelle DartsOrakel FDI-Rating als Referenzwert.

Wenn ich jetzt auf "FDI Rating vorhersagen" klicke, berechnet das Modell eine Vorhersage. Wir sehen links das vorhergesagte Rating, in der Mitte das tatsächliche Rating, und rechts den Delta-Wert – also ob unser Modell über- oder unterschätzt. Darunter gibt es eine Visualisierung der Feature-Beiträge, die zeigt, welche Features am meisten zur Vorhersage beigetragen haben.

Im zweiten Tab "Insights & EDA" finden sich verschiedene Analyse-Werkzeuge: Das Modell-Leaderboard zeigt alle trainierten Modelle mit ihren Metriken. Die Feature-Korrelationen visualisieren die Zusammenhänge als Balkendiagramm. Und die Country Performance zeigt Statistiken nach Ländern gruppiert.

Zusätzlich gibt es REST-API-Endpunkte. Unter /api/health kann man den Status prüfen, und /api/predict akzeptiert POST-Requests für programmatische Vorhersagen.

---

## 8. Fazit & Ausblick (1 Minute)

Zum Abschluss möchte ich die wichtigsten Punkte zusammenfassen.

Die Forschungsfrage ist beantwortet: Ja, wir können das FDI-Rating mit hoher Genauigkeit vorhersagen. Hard Facts wie der First-9 Average und die Checkout-Quote sind die stärksten Prädiktoren, aber auch Soft Facts wie das Preisgeld tragen signifikant bei. Mit einem R² von 0,93 erklären wir über 92 Prozent der Varianz.

Das Projekt demonstriert eine vollständige End-to-End Pipeline – vom Web-Scraping über Feature Engineering und Modelltraining bis zum containerisierten Deployment mit Web-Interface und API.

Natürlich gibt es auch Limitationen: Wir arbeiten aktuell nur mit Snapshot-Daten, keine Zeitreihen. Psychologische Faktoren wie Nervenstärke oder mentale Verfassung können wir nicht messen. Und es besteht ein gewisses Risiko von Data Leakage, falls DartsOrakel das FDI-Rating auf ähnlichen Features basiert.

Für die Zukunft wären Rolling Form-Features interessant – also gleitende Durchschnitte der letzten Turniere. Auch eine Unterscheidung zwischen Major-Turnieren und Floor Events könnte helfen. MLflow-Integration für besseres Experiment-Tracking steht auf meiner Liste, ebenso wie Drift-Monitoring für automatische Alerts bei Modell-Degradation.

Vielen Dank für Ihre Aufmerksamkeit. Ich freue mich auf Ihre Fragen.

---

## Mögliche Fragen & Antworten

**F: Warum performt Random Forest schlechter als lineare Modelle?**

A: Das liegt wahrscheinlich an der relativ linearen Beziehung zwischen Features und Target. Die Daten haben keine komplexen nicht-linearen Muster, die ein Random Forest ausnutzen könnte. Außerdem sind 2.500 Beobachtungen für einen Random Forest mit 108 Features relativ wenig Daten.

**F: Wie geht ihr mit der Multikollinearität um?**

A: Wir verwenden Lasso-Regularisierung, die automatisch korrelierte Features penalisiert. Alternativ haben wir ein reduziertes Feature-Set mit 11 Features entwickelt, das VIF-optimiert ist und für Interpretationszwecke geeignet ist.

**F: Wie oft werden die Daten aktualisiert?**

A: Der Scheduler ist auf wöchentliches Update konfiguriert – das entspricht dem Update-Rhythmus von DartsOrakel. Man kann das Intervall über die Umgebungsvariable FDI_REFRESH_MINUTES anpassen.

**F: Wie schnell ist die Inference?**

A: Eine einzelne Vorhersage dauert unter 100 Millisekunden. Das beinhaltet Feature-Transformation und Modell-Prediction. Der Hauptteil der Zeit geht für das One-Hot-Encoding der Länder drauf.

**F: Was würdet ihr anders machen, wenn ihr mehr Zeit hättet?**

A: Drei Dinge: Erstens Zeitreihen-Features einbauen, um Form-Entwicklung zu tracken. Zweitens A/B-Testing verschiedener Feature-Sets mit MLflow. Drittens ein Dashboard für Drift-Monitoring, um zu sehen, wann das Modell neu trainiert werden muss.
