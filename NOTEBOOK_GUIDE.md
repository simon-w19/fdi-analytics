# 📚 Vollständiger Leitfaden durch alle Notebooks

## Überblick: Die Data-Science-Pipeline

Dieses Projekt folgt einer **Standard-Data-Science-Pipeline** mit vier Hauptphasen:

```
┌─────────────────────────────────────────────────────────────┐
│  1. EXPLORATIVE DATENANALYSE (EDA)  → eda.ipynb            │
│     "Verstehen wir die Daten?"                             │
│                          ↓                                  │
│  2. FEATURE ENGINEERING & PREPROCESSING → fdi_rating_modeling.ipynb (Teil 1)
│     "Welche neuen Features erstellen wir?"                │
│                          ↓                                  │
│  3. MODELLTRAINING & VERGLEICH → fdi_rating_modeling.ipynb (Teil 2)
│     "Welches Modell funktioniert am besten?"              │
│                          ↓                                  │
│  4. EVALUATION & DIAGNOSE → fdi_rating_modeling.ipynb (Teil 3)
│     "Warum funktioniert es? Fehler verstehen?"            │
│                          ↓                                  │
│  5. DOKUMENTATION → report.ipynb + slides.md              │
│     "Ergebnisse kommunizieren"                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ NOTEBOOK: `notebooks/eda.ipynb`
### Explorative Datenanalyse

**Ziel:** Die Rohdaten verstehen, Muster entdecken, Anomalien identifizieren.

**Methodologie:** Wir folgen dem **OpenIntro-Prinzip** (Introduction to Modern Statistics):
- Univariate Analyse (eine Variable allein)
- Bivariate Analyse (zwei Variablen zusammen)
- Multivariate Muster (Gruppen und Interaktionen)

---

### 🔍 Abschnitt 1: Datenstruktur & Messniveaus

**WAS:** Klassifikation aller 34 Variablen nach Typ (numerisch/kategorisch) und Messniveau.

**WIE:**
```python
# Jede Spalte wird untersucht:
for col in df.columns:
    dtype = df[col].dtype  # int64, float64, object?
    unique_count = df[col].nunique()  # Wieviele verschiedene Werte?
    missing_pct = df[col].isna().mean() * 100  # % fehlende Werte?
    
    if is_bool_dtype(col):
        measurement = "binary"  # Ja/Nein
    elif is_numeric_dtype(col) and unique_count > 10:
        measurement = "interval"  # Kontinuierliche Zahlen
    else:
        measurement = "nominal"  # Kategorien ohne Ordnung
```

**WARUM:** 
- Messniveaus bestimmen, welche Statistik sinnvoll ist
- Fehlende Werte müssen adressiert werden (Imputation)
- Kategorische Variablen brauchen One-Hot-Encoding

**OUTPUT-INTERPRETATION:**
| Feature | Type | Messniveau | Missing% | Bedeutung |
|---------|------|-----------|----------|-----------|
| profile_fdi_rating | float64 | interval | 0% | Zielvarinale - keine NaNs ✅ |
| age | float64 | interval | ~5% | Braucht Imputation |
| country | object | nominal | <1% | Kategorisch, ~30 Länder |

---

### 📊 Abschnitt 2: Univariate Verteilungen

**WAS:** Histogramme + Dichte-Kurven für 6 Kern-Variablen:
- `profile_fdi_rating` (unser Ziel)
- `last_12_months_averages` (Durchschnitt)
- `last_12_months_first_9_averages` (Early-Game-Durchschnitt)
- `last_12_months_checkout_pct` (Checkout-Erfolg)
- `last_12_months_functional_doubles_pct` (Double-Erfolg)
- `last_12_months_pct_legs_won` (Gewinnrate)

**WARUM:**
- Normalverteilung? → Linear Regression funktioniert besser
- Schiefe Verteilung? → Log-Transformation oder robuste Modelle nötig
- Ausreißer sichtbar? → Data Cleaning nötig

**INTERPRETATION:**

| Variable | Form | Interpretation |
|----------|------|---|
| FDI Rating | ∩ (glockenförmig) | Normalverteilt! Linear Regression sollte funktionieren ✅ |
| Averages | ∩ leicht links-schief | Normal mit Schwanz nach Links (schwache Spieler existieren) |
| Checkout% | Breite Streuung | Spieler haben sehr unterschiedliche Checkout-Skills |
| Legs Won % | Links-schief | Viele Spieler ~60% Win-Rate, wenige bei 20-30% |
| Log-Earnings | Stark rechts-schief! | BRAUCHT Log-Transform um normalverteilt zu sein |

**KEY-INSIGHT:** Most metrics sind annehmbar normalverteilt → Lineare Modelle sind reasonable choice.

---

### 🔗 Abschnitt 3: Korrelationsmatrix

**WAS:** Eine 13×13 Heatmap zeigt Pearson-Korrelationen zwischen allen Hauptvariablen.

**KORRELATION r:**
- r = 1: Perfekt positiv (wenn A steigt, steigt B auch)
- r = 0: Keine Beziehung
- r = -1: Perfekt negativ

**FARBCODIERUNG:**
```
Rot (r > 0.7)    = Starke positive Korrelation ⚠️
Orange (r >0.5)  = Moderate positive Korrelation ⚠️
Blau (r < 0.5)   = Schwache Korrelation ✅
```

**HAUPTFINDINGS:**

1. **Mit FDI-Rating (Oberste Zeile):**
   ```
   r = 0.95 mit last_12_months_averages        → Strongest single predictor!
   r = 0.95 mit last_12_months_first_9_averages → Almost as strong
   r = 0.88 mit last_12_months_checkout_pct    → Strong, but weaker
   r = 0.95 mit api_overall_stat               → SUSPICIOUS! (r=1.0 = perfect!)
   ```
   **Interpretation:** Recent performance (last 12 months) ist der beste Indikator für FDI-Rating.

2. **Multikollinearität entdeckt (Große rote Zellen überall):**
   ```
   r = 0.99 zwischen last_12_months_averages ↔ last_12_months_first_9_averages
   r = 0.99 zwischen checkout_pct ↔ functional_doubles_pct
   r = 0.93 zwischen functional_doubles_pct ↔ checkout_pct
   ```
   **Interpretation:** Features sind untereinander stark abhängig! 
   → Einzeln starke Prädiktoren, aber zusammen problematisch (Multikollinearität).
   → Später mit Regularisierung (Ridge/Lasso) adressieren.

---

### 📍 Abschnitt 4: Geografische Muster (Ridge Plots)

**WAS:** Kernel-Dichte-Plots der FDI-Verteilung für Top-6 Länder überlagert.

**INTERPRETATION:**

```
FDI Rating
2000 │     ╭──────╮
     │    ╭│ (NED) │╮
     │   ╭│ (ENG)  ││╮
1500 │  ╭│        │││╮
     │ ╭│         │││
1000 │ │          │││  ← UNK (Unknown) Spieler sind hier!
     │ │          ││
 500 │ │          │
     └─┴──────────┴┴────
```

**KERN-BEFUNDE:**

| Land | Position | Interpretation |
|------|----------|---|
| ENG (England) | Nach RECHTS verschoben | Höchste durchschn. FDI (~1400) |
| NED (Niederlande) | Ähnlich ENG | Höchste durchschn. FDI (~1350) |
| GER (Deutschland) | Nach LINKS! | Niedrigere FDI (~1200) |
| AUS (Australien) | Mittelmäßig | ~1300 durchschn. |
| UNK (Unbekannt) | WEIT LINKS! | Sehr niedrig (~1050) |

**BUSINESS-INSIGHT:** 
"Elite-Darts-Nationen" (ENG, NED) haben deutlich höhere durchschnittliche Spielerstärke. Das ist **nicht zufällig**, sondern ein echte geografischer Effekt. Mögliche Ursachen:
- Darts-Kultur & Training in UK/Niederlande stärker
- Selektionsbias: Nur die Besten verlassen UK/NED zum Spielen
- Dataset-Sampling: Vielleicht wurden UK/NED-Spieler gezielt gesammelted

**MODELLIERUNGS-KONSEQUENZ:** Land-Dummies müssen ins Modell!

---

### 🔀 Abschnitt 5: Faceting (Simpson's Paradox Check)

**WAS:** Die Beziehung zwischen zwei Features (FDI vs. First-9-Average) wird **getrennt nach Land** geplottet.

**WARUM:** Test auf **Simpson's Paradox** - ein statistisches Phänomen, bei dem ein Trend in gesamten Daten sich in Untergruppen umkehrt.

**BEISPIEL Simpson's Paradox:**
```
Gesamt-Trend: "Bessere Spieler verdienen mehr"
Aber wenn wir nach Jahr aufsplittet:
  - 2020: "Bessere Spieler verdienen WENIGER" (weil 2020 weniger Geld da)
  - 2021: "Bessere Spieler verdienen WENIGER" (weil 2021 andere Turniere)
```

**UNSERER BEFUND:** 
```
Alle 4 Länder (ENG, NED, GER, UNK) zeigen 
die gleiche Beziehung: Höhere First-9-Average → Höheres FDI
Keine Umkehr! ✅
```

**INTERPRETATION:**
The relationship is **robust and universal** - es ist nicht confounded durch Land-Effekte. Diese Variable wird in allen Ländern gleich gut funktionieren.

---

### 🎲 Abschnitt 6: Kontingenztabellen & Chi-Quadrat-Tests

**WAS:** Kreuztabellen zwischen **kategorischen Variablen** (Land × FDI-Kategorie).

**TEST 1: Land × FDI Top-Quartil (Ist ein Land überrepräsentiert in Top 25%?)**

```
                 Top 25%  Rest 75%  | Total | % in Top25%
────────────────────────────────────────────────────────
England (ENG)        90       159  |  249  |  36.1% ← Over-repräsentiert!
Niederlande (NED)    65       120  |  185  |  35.1% ← Over-repräsentiert!
Australien (AUS)     42       140  |  182  |  23.1% ← Unter-repräsentiert
Deutschland (GER)    25       115  |  140  |  17.9% ← Stark unter-repräsentiert!
Unbekannt (UNK)      15       156  |  171  |   8.8% ← Extrem unter-repräsentiert!
```

**Chi-Quadrat-Test Ergebnis:**
```
χ² = 134.17
p-value = 0.0000 (< 0.0001)
DoF = 7
```

**INTERPRETATION:**
- χ² = 134 ist SEHR GROss (bei DoF=7, kritischer Wert ~12)
- p-value < 0.0001 → **Extrem signifikant**
- **Konklusion:** Länder und FDI-Erfolg sind DEFINITIV nicht unabhängig!
- **Effekt-Größe:** Riesig. England/Niederlande sind wirklich besser.

---

**TEST 2: Land × Checkout-Performance (4 Buckets)**

```
                Low     Medium  High   VeryHigh | χ² Test
────────────────────────────────────────────────────────
England         20%     25%     30%    25%
Niederlande     18%     24%     32%    26%
Australien      30%     30%     28%    12%  ← Weniger High/VeryHigh
Deutschland     40%     30%     20%    10%  ← Deutlich schwächer
Unbekannt       45%     35%     15%     5%  ← Am schwächsten

χ² = 407.19 (!!!)
p-value = 0.0000
```

**INTERPRETATION:**
- χ² = 407 ist *gigantisch* (noch größer als Test 1)
- **Checkout-Skill differenziert Länder EXTREM stark**
- England/Niederlande haben deutlich höhere High/VeryHigh Checkout-Quoten
- Deutschland und Unknown Länder sind viel schwächer

**GESAMTFAZIT EDA:**
✅ Daten sind gut strukturiert, ~3000 Spieler mit relevanten Features  
✅ Zielvarinale (FDI) ist normalverteilt  
✅ Starke bivariate Beziehungen erkannt (Durchschnitte, Checkout-Erfolg)  
⚠️ Massive Multikollinearität (Features korrelieren zu 0.9+)  
⚠️ Starke geografische Effekte (Land-Dummies essentiell)  

---

## 2️⃣ NOTEBOOK: `notebooks/fdi_rating_modeling.ipynb`
### Modellentwicklung & Diagnose

**Ziel:** Machine-Learning-Modelle trainieren, die FDI-Rating vorhersagen, und verstehen, was funktioniert.

---

### Phase 1: Datenladen & Feature Engineering

#### Schritt 1: Data Leakage Prevention

**WAS:** Entfernen von `api_rank` und `api_overall_stat`.

**WARUM:** Diese Features sind DIREKT vom FDI-Rating abgeleitet oder sind das Ranking selbst!

**ANALOGE:** 
```
❌ FALSCH: "Vorhersagen Sie eine Prüfungsnote" + "Geben Sie die Lösung vor"
✅ RICHTIG: Vorhersagen Sie die Prüfungsnote NUR basierend auf Studienzeit & Vorwissen
```

**KONSEQUENZ:** 
- Mit Data Leakage würde Modell 99% Genauigkeit zeigen, aber nur im Training
- In der Praxis (neue Spieler) würde es 0% Genauigkeit haben
- Wir entfernen beide Features SOFORT

**CODE:**
```python
LEAKY_FEATURES = ["api_rank", "api_overall_stat"]
df = df.drop(columns=LEAKY_FEATURES, errors="ignore")
```

---

#### Schritt 2: Feature Engineering (Neue Features aus Rohdaten)

**WAS:** 12 neue Features erstellen aus den bestehenden 37.

**WARUM:** Domain-Knowledge in Daten codieren → Modell schneller lernen.

| Feature | Berechnung | Intuition |
|---------|-----------|-----------|
| `log_total_earnings` | log(earnings) | Logarithmieren wegen extrem rechtsschiefer Verteilung (Reiche Spieler sind Ausreißer) |
| `season_win_rate` | win_pct / 100 | Normalisiert von Prozent (0-100) zu Dezimal (0-1) |
| `checkout_combo` | (checkout% × double%) / 100 | Multiplikation: muss BEIDE Fähigkeiten haben |
| `first9_delta` | first9_avg - 12m_avg | Konsistenz-Check: starker Start oder nur später gut? |
| `momentum_gap` | throw_avg - against_throw_avg | Psychologisches "Clutch": besser unter Druck oder nicht? |
| `experience_intensity` | tour_card_years / age | "Intensität": wieviel des Lebens haben Darts gespielt? |
| `earnings_per_year` | total_earnings / tour_card_years | "Kapitaleffizienz": verdient pro Jahr aktiv |
| `first9_ratio` | first9_avg / 12m_avg | Prozentsatz: Start-Phase Überdurchschnitt? |
| `break_efficiency` | legs_won_throwing2 / legs_won_throwing1 | "Breaking": besser gegnerische Aufschläge brechenls halten? |
| `hold_break_spread` | legs_won_throw1 - legs_won_throw2 | Differenz: absoluter Unterschied Halten vs. Brechen |
| `power_scoring_ratio` | (180s + 171-180s) / (140s + 131-140s) | Hochpunkte vs. Standard-Punkte: aggressive oder konservativ? |
| `tv_stage_delta` | tv_avg - general_avg | Nervenstärke? Höher bei Kamera oder nicht? |

**BEISPIEL - Warum ist `first9_delta` wertvoll?**

Zwei Spieler mit gleicher 12-Monats-Average = 75:
```
Spieler A: First-9 = 85, Rest = 72  → Delta = +13 (Starker Start, später weniger)
Spieler B: First-9 = 65, Rest = 78  → Delta = -13 (Schwacher Start, später stark)
```

Ein lineares Modell könnte nicht den Unterschied erkennen (beide haben Average=75).
Mit `first9_delta` explizit: Der Unterschied wird dem Modell "geschenkt".

**ERGEBNIS:** 
- 37 ursprüngliche numerische Features
- 12 neue engineered Features
- Total: **38 Input-Features** für Modelle

---

#### Schritt 3: Featuresets Definieren

**WAS:** Gruppierung der Features nach Type.

```python
numeric_features = [
    # "Hard Facts" (direkt gemessen)
    "age", "last_12_months_averages", "last_12_months_checkout_pcnt",
    # ... weitere 35 numerische Features
]

categorical_features = ["country"]  # 30+ verschiedene Länder

# Total: 38 Features
```

#### Schritt 4: Train/Test-Split

**WAS:** 80% Training (1,981 Spieler), 20% Test (496 Spieler).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**WARUM:** 
- Training: Modell "lernt" Muster
- Test: "Unbekannte" Spieler → echte Vorhersage-Genauigkeit

**SEED (random_state=42):** Sorgt für **Reproduzierbarkeit** (gleiche Zahl → gleicher Split immer).

---

### Phase 2: Baseline & Preprocessing

#### Baseline-Dummy-Modell

**WAS:** Einfachste mögliche Vorhersage: "Sage immer den Durchschnitt des Training-Sets"

```python
baseline_value = y_train.mean()  # z.B. 1340
baseline_pred = [1340, 1340, 1340, ...]  # Für alle Test-Spieler
```

**METRIKEN:**
```
R² = 0.0        (0% erklärt)
RMSE = 188.7    (durchschnittlicher Fehler: 188.7 FDI-Punkte)
MAE = 151.1     (mittlerer absoluter Fehler: 151.1 Punkte)
```

**INTERPRETATION:** 
Alle echten Modelle müssen BESSER als diese sein, sonst sind sie unnötig.

---

#### Preprocessing Pipeline

**WAS:** 4 Schritte automatisiert:

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),    # Schritt 1: Fehlwerte füllen
    ('scaler', StandardScaler()),                      # Schritt 2: Normalisieren
])
```

**SCHRITT 1: Imputation (Fehlwerte)**
- Numerisch: Median verwenden (robust gegen Ausreißer)
- Kategorisch: Mode (häufigster Wert) verwenden
- Warum Median statt Mean? Median unempfindlich gegen Ausreißer

**SCHRITT 2: Skalierung**
- StandardScaler: (X - mean) / std
- Transformiert jedes Feature auf Mean=0, Std=1
- WARUM: Ridge/Lasso/KNN brauchen skalierte Features (sonst große Features dominieren)

**SCHRITT 3: Encoding (Kategorische Variablen)**
- One-Hot-Encoding: `country = "ENG"` → `country_ENG=1, country_NED=0, ...`
- WARUM: ML-Modelle verstehen nur Zahlen, nicht Text

---

### Phase 3: Modelltraining & Vergleich

#### 4 verschiedene Modelle trainiert:

**1. LINEAR REGRESSION**
```
FDI = β₀ + β₁×age + β₂×averages + β₃×checkout% + ... + β₃₈×country
```
- Simplest, schnellste
- Aber: Annahme = lineare Beziehungen (stimmt nicht überall)

**2. RIDGE REGRESSION (L2 Regularisierung)**
```
minimize: MSE(Y - Ŷ) + λ × Σ(β²)
```
- Strafe für große Koeffizienten
- Stabilisiert bei Multikollinearität
- Koeffizienten werden klein, aber nie exakt 0

**3. LASSO REGRESSION (L1 Regularisierung)**
```
minimize: MSE(Y - Ŷ) + λ × Σ|β|
```
- Aggressivere Strafe
- Setzt unwichtige Koeffizienten auf EXAKT 0
- = Automatische Feature-Selektion

**4. RANDOM FOREST (Tree Ensemble)**
```
FDI = Durchschnitt aus 600 Entscheidungsbäumen
```
- Kann nonlineare Muster lernen
- Black-box (schwer interpretierbar)
- Robust gegen Multikollinearität (Bäume "ignorieren" Korrelationen)

---

#### ERGEBNISSE (Test-Metriken):

| Modell | R² | RMSE | MAE | Train-Test Gap | 
|--------|-----|------|-----|---|
| Baseline | 0.00 | 188.7 | 151.1 | - |
| Linear | 0.924 | 47.8 | 35.9 | 0.000 |
| Ridge (α=5) | 0.929 | 45.8 | 34.4 | -0.001 |
| Lasso (α=0.1) | 0.914 | 52.4 | 38.3 | 0.003 |
| **Random Forest** | **0.923** | **48.3** | **35.2** | **0.011** |

**INTERPRETATION:**

1. **Alle ML-Modelle > Baseline** ✅
   - Linear: 92.4% der Varianz erklärt (vs. 0% Baseline) = HUGE Verbesserung!

2. **Ridge ist leicht besser als Linear** 
   - R²: 0.929 vs 0.924 = nur +0.5% Verbesserung
   - Aber: Stabilere Koeffizienten wegen Multikollinearität-Handling

3. **Lasso schlechter als Ridge**
   - Wahrscheinlich weil es zu viele Features auf 0 setzt (zu aggressiv)

4. **Random Forest vergleichbar mit Linear/Ridge**
   - Auch ~92% erklärt
   - Train-Test Gap = 0.011 (akzeptabel, kein großes Overfitting)
   - Nicht besser als Linear, aber auch nicht schlechter

**CHOICE:** Ridge oder Random Forest würden beide funktionieren. 
Wir werden **Random Forest wählen** für GridSearchCV-Tuning.

---

### Phase 4: Hyperparameter-Tuning

#### GridSearchCV für Random Forest

**WAS:** Test ALLER Kombinationen von Hyperparameter-Optionen.

```python
param_grid = {
    'n_estimators': [300, 600],                # Anz. Bäume
    'max_depth': [None, 10, 20],                # Baum-Tiefe
    'min_samples_leaf': [1, 2, 5],              # Min Beobachtungen pro Blatt
    'max_features': [0.5, 'sqrt', 'log2']      # Features pro Split
}
# 2 × 3 × 3 × 3 = 54 Kombinationen
# Mit 5-Fold CV = 270 Trainings-Durchläufe
```

**BESTE PARAMETER GEFUNDEN:**
```
n_estimators = 600      (mittlere Anzahl)
max_depth = None        (keine Beschränkung - Bäume wachsen unbegrenzt)
min_samples_leaf = 2    (aggressiv splitten)
max_features = 0.5      (nutze nur 50% der Features pro Split)
```

**ERGEBNIS NACH TUNING:**
```
Test R² = 0.923 (vorher: 0.923)
Test RMSE = 48.3 (vorher: 48.3)
```

**FAZIT:** Tuning hat praktisch NICHTS verbessert! 
Das bedeutet: Standardparameter waren bereits gut.

---

### Phase 5: Multikollinearität-Diagnose

#### VIF (Variance Inflation Factor) Berechnung

**WAS:** Misst, wie sehr Koeffizienten durch Multikollinearität "aufgeblasen" werden.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

for i, col in enumerate(numeric_features):
    vif = 1 / (1 - R²_i)  # R² = wie gut kann Feature_i von anderen vorhergesagt werden?
```

**VIF-INTERPRETATION:**
```
VIF = 1.0   → Kein Problem, unabhängig
VIF = 5.0   → Moderat problematisch (feature = 80% erklärbar von anderen)
VIF = 10.0  → Kritisch (90% erklärbar)
VIF = 100+  → Quasi perfekt redundant
```

**UNSER RESULTAT:**
```
✓ VORHER (alle 37 numerischen Features):
  36 von 37 haben VIF > 5 (nur 1 Feature ist "clean")
  
✓ NACHHER (nach Versuch der Reduktion):
  27 von 37 haben VIF > 5 (nur 10 Features sind "clean")
  
Top 5 problematische Features:
  - api_sum_field1: VIF = >1000 (perfekt redundant!)
  - last_12_months_averages: VIF = >1000
  - last_12_months_functional_doubles_pct: VIF = 500+
  - ... etc
```

**INTERPRETATION:**
- **Massive Multikollinearität erkannt!**
- Features sind untereinander so stark korreliert, dass einzelne Koeffizienten unzuverlässig sind
- Aber: Für **Vorhersage** ist das weniger problematisch als für **Inferenz**
  - Ridge/Lasso/Random-Forest können damit umgehen
  - Nur lineare Regression's Koeffizienten werden instabil

---

#### Hochkorrelierte Feature-Paare

**WAS:** Alle Paare mit |r| > 0.8 identifiziert.

**BEISPIELE:**
```
r = 0.9999: last_12_months_averages ↔ api_sum_field1
r = 0.9999: last_12_months_functional_doubles_pct ↔ last_12_months_checkout_pcnt
r = 0.9903: last_12_months_with_throw ↔ last_12_months_first_9
```

**BEDEUTUNG:** r = 0.99 = wenn ich A kenne, kann ich B quasi exakt vorhersagen.
Features sind **praktisch identisch** → eines ist redundant.

---

### Phase 6: Feature-Selektion via Lasso

#### Was Lasso tut

**WAS:** Trainiert ein lineares Modell MIT Strafe für große Koeffizienten.

```python
Lasso(alpha=0.01)  # Strafe: 0.01 × Σ|Koeffizienten|
```

**EFFEKT:** Koeffizienten werden kleiner, und bei α groß genug, werden sie EXAKT 0.

#### ERGEBNIS

```
✅ SELECTED FEATURES: 93 von 109 (85%)
   Top 5 nach |Koeffizient|:
   1. country_BOT (Botswana): 140.2  ← Botswana Spieler haben höheres FDI!
   2. country_BRU (Brasilien): 124.8
   3. last_12_months_averages: 105.3 ← Aveeages sind wichtig
   4. country_CPV (Kap Verde): 94.2
   5. country_ISR (Israel): 88.5

❌ ELIMINATED FEATURES: 16 von 109 (15%)
   - hold_break_spread → zu redundant
   - tv_stage_delta → zu redundant
   - 13 country_dummies mit kleinen Populationen (z.B. country_ARM, country_BHR)
```

**INTERPRETATION:**

1. **Lasso konservativ** - eliminiert nur offensichtlich redundante Features
2. **Land-Dummies dominieren** - viele der Top-Koeffizienten sind countries
   - Bestätigt unsere EDA-Findung: Länder sind wichtig!
3. **Eliminierte Features:**
   - `hold_break_spread`: Redundant zu `break_efficiency` (beide messen Halten vs. Brechen)
   - `tv_stage_delta`: Zu wenig Varianz oder redundant zu anderen Durchschnitten
   - Kleine Länder: Zu wenig Daten, starkes Overfitting-Risiko

---

## 3️⃣ NOTEBOOK: `reports/report.ipynb`
### Formaler Bericht

**Ziel:** Ergebnisse kompakt für Stakeholder zusammenfassen.

**Struktur:**

1. **Einführung**: Was ist das Problem? Was ist FDI-Rating?
2. **Daten**: Woher kommen die Daten? Wieviele Spieler? Wieviele Features?
3. **Methodologie**: Welche Modelle? Welche Evaluations-Metriken?
4. **Ergebnisse**: Beste Modelle? Genauigkeit? Feature-Wichtigkeit?
5. **Schlussfolgerungen**: Was haben wir gelernt? Was sind Limitationen?
6. **Literatur**: Welche Bücher/Papers haben wir verwendet?

---

## 4️⃣ ZUSÄTZLICHE DATEIEN: `slides.md` & `README.md`
### Präsentation & Dokumentation

**`slides.md`:** 8-Slide-Präsentation für Google Slides/PowerPoint
**`README.md`:** Technische Dokumentation + Lessons Learned

---

## 📊 ZUSAMMENFASSUNG: Welches Modell ist am besten?

| Modell | R² Test | RMSE | Interpretierbarkeit | Multikollinearität-Robust | Wahl |
|--------|---------|------|------|-----|------|
| Linear | 0.924 | 47.8 | ⭐⭐⭐⭐⭐ (Einfach) | ❌ (VIF-Probleme) | Interpretierbar, aber instabil |
| Ridge | 0.929 | 45.8 | ⭐⭐⭐⭐ (Moderat) | ✅ Robust | **Best für Production** |
| Lasso | 0.914 | 52.4 | ⭐⭐⭐⭐⭐ (Sparse) | ✅ Robust | Gut für Sparse-Daten |
| Random Forest | 0.923 | 48.3 | ⭐⭐⭐ (Black-box) | ✅ Robust | Flexibel, aber hard to explain |

**EMPFEHLUNG:** **Ridge Regression** 
- Höchster R² (92.9%)
- Robust gegen Multikollinearität
- Interpretierbar (Koeffizienten)
- Schnell & einfach zu deployen

---

## 🎓 KEY LEARNINGS

### 1. **Normalverteilung ist wichtig**
   - EDA zeigte: FDI ist (annähernd) normalverteilt
   - Daher funktionieren lineare Modelle gut
   - Log-Transformation der Earnings war essentiell (sonst zu schief)

### 2. **Multikollinearität != Katastrophe**
   - 36 von 37 Features haben VIF > 5 (massive Redundanz!)
   - Aber: Modelle funktionieren trotzdem gut (R² > 0.92)
   - Ridge/Lasso/RF-Modelle können damit umgehen
   - Problem ist nur für Linear-Regression's Koeffizienten (werden instabil)

### 3. **Feature Engineering zahlt sich aus**
   - 12 neue Features aus Domain-Knowledge generiert
   - Features wie `first9_delta`, `break_efficiency` sind interpretierbar
   - Modell brauchte das Wissen nicht zu "erfinden"

### 4. **Geografische Effekte sind real**
   - Chi-Quadrat-Tests zeigen: Länder beeinflussen FDI & Checkout-Erfolg signifikant
   - England/Niederlande haben 35% der Top 25% Spieler (vs. 25% erwartet)
   - Deutschland überraschend schwach (Sampling-Bias oder echte Unterschiede?)

### 5. **Data Leakage is critical**
   - `api_rank` und `api_overall_stat` MÜSSEN entfernt werden
   - Würden unrealistisch hohe Genauigkeit vortäuschen
   - Echte Vorhersagen brauchen nur "echte" Features

### 6. **Random Forest bringt keine Verbesserung**
   - "Intelligentere" Modelle sind nicht automatisch besser
   - RF: R² = 0.923, Linear: R² = 0.924 → praktisch identisch
   - Aber: RF ist robuster gegen Annahmen, Linear ist interpretierbar
   - **Trade-off:** Einfachheit vs. Flexibilität

---

## 🚀 NEXT STEPS (Wenn Zeit/Ressourcen vorhanden)

1. **Feature Importance Analysis (Random Forest):**
   - Welche 5 Features sind AM WICHTIGSTEN?
   - Könnten wir mit 5-10 Features 85% Genauigkeit erreichen?

2. **Residual Analysis:**
   - Wo macht das Modell Fehler?
   - Gibt es systematische Fehler (z.B. überschätzt neue Spieler)?

3. **Ensemble Methods:**
   - Kombinieren Sie Ridge + Random Forest
   - Weighted Average der Vorhersagen könnte besser sein

4. **Cross-Validation Stability:**
   - TimeSeriesSplit statt Random Split?
   - Wie stabil sind Modelle wenn wir Daten zeitlich trennen?

5. **Deployment:**
   - API bauen (FastAPI/Flask)
   - Gradio App (bereits vorhanden) mit beste Modell verbinden
   - Docker-Container für Skalierbarkeit

---

## 📚 Referenzen

- **OpenIntro: Introduction to Modern Statistics** (2e) - Methodologie für EDA
- **ISLP: An Introduction to Statistical Learning with Python** (James et al. 2023) - Regularisierung, Modellselection
- **sklearn Dokumentation** - Implementierungen

---

**Notebook-Autor:** Simon  
**Projekt:** FDI Analytics Pipeline  
**Datum:** 2026
