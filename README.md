# Schwere Ränder: Wann zahlt sich Median-of-Means aus? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-median-of-means-demo.streamlit.app/)** -->

Stück 6 der Reihe **Mittelwerte und Lagemaße** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Kante im Abhängigkeitsgraphen: Stück 4 (Trimmen, Winsorisieren, Huber) → 6 als Fix, denn Median und Trimmen zielen bei Schiefe auf eine andere Größe als den Mittelwert, und das Mittel hat keine Garantie für den seltenen Ernstfall. Die Demo setzt Mittel, Median und Trimmen als bekannt voraus und wiederholt sie nur in einem Satz.

Vehikel: **ein Lager schätzt aus n Sendungen die mittlere Größe** (Ziel Mittelwert, weil Gesamtmenge = Anzahl mal Mittel): die Sendungsmenge je Kunde (lognormal), die Standzeit an der Rampe (Pareto) oder den Lieferverzug (symmetrisch um 0: Normal, t, symmetrisch mit Pareto-Rand). Alle Verteilungen sind **erfundene Modelle, keine Messdaten**. Die Demo zeigt (a) die **Kippkarte**: bei welchem Niveau δ schlägt Median-of-Means das Mittel, in Abhängigkeit von Verteilung und n; (b) was die Absicherung kostet: **typischer Fehler und Verzerrung**; (c) die Blockzahl K als Regler; (d) **Trimmen mit wenigen Werten** als Gegenkandidat; (e) die Zielgröße (Mittelwert oder Median); (f) einen negativen Befund zu Intervallen.

**Buchstaben.** n ist die Stichprobengröße, δ das Niveau (Q(1 − δ) ist der Fehler, den nur ein Anteil δ der Läufe überschreitet), K die Zahl der Blöcke, k die Zahl der abgeschnittenen Werte je Seite beim Trimmen, c der Faktor in k = ⌈c ln(1/δ)⌉. Fehler: bei symmetrischen Verteilungen |Schätzer|, bei schiefen (Schätzer − Mittelwert) / Mittelwert (negativ: zu klein).

Abgrenzung (Stichwortsuche in 455 lokalen und den per `gh api` gelesenen READMEs, Stand 2026-10-24, laut Novitätsprüfung im Einzelplan; für diese Demo erneut gelesen: `trimmen-huber-demo`, `permutation-test-demo`): `trimmen-huber-demo` ist Stück 4 und misst Effizienz, Verzerrung und Verschiebungskurven von Lage-Schätzern bei Ausreißern; das Fehlerquantil Q(1 − δ) als Funktion des Niveaus ist dort ausdrücklich offen. `permutation-test-demo` winsorisiert nur als Vorbehandlung eines Tests, `multiple-testing-demo` zeigt den Gewinnerfluch bei vielen Varianten (anderes Problem), `robust-filter-demo` gewichtet Messvarianzen nach Huber im Kalman-Filter, `pearson-demo` prüft Intervalle für Korrelationen bei schweren Rändern. In keiner dieser Demos ist Median-of-Means oder die Kippkarte der Gegenstand.

## Warum dieses Problem

„Median-of-Means schützt vor schweren Rändern“ stimmt, aber nur **bei seltenen Ereignissen und schiefen oder sehr schweren Rändern**, und die Absicherung hat einen Preis. Der Hook ist deshalb nicht „Median-of-Means gewinnt“, sondern die **Kippkarte und die Verzerrung**: (1) bei Normal und t(5) schlägt Median-of-Means das Mittel in keinem Niveau (δ = 0,5 bis 0,001) und keinem n (50 bis 5000) des Rasters; (2) bei schiefen Rändern schlägt es das Mittel erst bei seltenen Ereignissen (Kippniveau, wo es eines gibt, höchstens 0,05), und das Kippniveau **sinkt mit n**; bei symmetrisch schwerem Pareto-Rand (α = 2,5) ist es dagegen schon bei n = 50 und 200 auf jedem Niveau des Rasters besser; (3) der Preis ist bei Normal, t(5) und schiefen Rändern ein größerer typischer Fehler, bei schiefen Rändern **zusätzlich eine Verzerrung nach unten**; bei symmetrisch schwerem Pareto-Rand und bei t(3) mit n = 50 ist der typische Fehler dagegen kleiner; (4) **Trimmen mit wenigen Werten** (eigene Wahl des Faktors c) gewinnt die Dreier-Wahl mit Mittel und Median-of-Means mit c = 1 in 92 von 160 Zellen (bei symmetrischen Rändern in 48 von 64), mit c = 4 in 49; (5) ein brauchbares Median-of-Means-Intervall für den Mittelwert habe ich nicht gefunden.

Kernfrage: *Wann ist das Mittel bei schweren Rändern gefährlich, was kostet die Absicherung, und wer sichert besser ab?*

## Befunde und Korrekturen gegenüber dem Plan

Die Hypothesen des Einzelplans (`mittelwerte-planung/STUECK-6-median-of-means.md`) wurden im Bau mit dem **eigenen SplitMix64-Strom** neu gemessen; die Vormessung des Plans (numpy-Zufall, Wegwerfskripte) liefert keine README-Zahl. Alle Zahlen: 40 000 Stichproben je Seed und Zelle, vier feste Zufalls-Seeds, gleiche Stichproben für alle Schätzer (gepaarter Vergleich); Median-of-Means mit der Regel K = ⌈8 ln(1/δ)⌉ (gedeckelt bei n/2), wo nichts anderes steht.

| | Hypothese | Ergebnis | Zahl (diese Demo) |
|---|---|---|---|
| H1 | Median-of-Means schlägt das Mittel erst bei seltenen Ereignissen; nie bei Normal und t(5) | **Ja** bei schiefen Rändern, Normal und t(5); nicht bei symmetrisch schwerem Pareto-Rand | Lognormal (σ = 1,5), n = 200, Q(1 − δ) Mittel / Median-of-Means: 0,295 / 0,376 (δ = 0,1), 0,371 / 0,430 (0,05), 0,643 / 0,531 (0,01), 1,333 / 0,625 (0,001). Kippniveau (besser bei diesem und allen kleineren Niveaus, in allen vier Seeds): Lognormal (σ = 1,5) 0,02, t(3) 0,01 bei n = 200; Normal und t(5) in keinem Niveau und keinem n. In 56 von 160 Zellen (10 Verteilungen · 4 n · δ = 0,1 / 0,05 / 0,01 / 0,001) ist Median-of-Means in allen Seeds besser, davon 20 von 64 bei symmetrischen Verteilungen. Ausnahme: bei symmetrisch mit Pareto-Rand (α = 2,5) ist Median-of-Means bei n = 50 und 200 auf jedem Niveau des Rasters besser (Kippniveau 0,5; Verhältnis zum Mittel bei n = 50: 0,87 bei δ = 0,5, 0,24 bei δ = 0,001) |
| H2 | Das Kippniveau sinkt mit n | **Ja** | Lognormal (σ = 1,5) bei n = 50 / 200 / 1000: 0,05 / 0,02 / 0,005; bei n = 5000 ist Median-of-Means bis δ = 0,001 (Ende des Rasters) nicht durchgehend besser (der Plan fand dort 0,0005, mein Raster endet bei 0,001) |
| H3 | Der Preis ist ein größerer typischer Fehler und eine Verzerrung nach unten, wachsend mit K und Schiefe | **Ja** bei schiefen Rändern; bei Normal und t(5) nur der größere typische Fehler; bei symmetrisch schwerem Pareto-Rand und t(3) mit n = 50 ist der typische Fehler kleiner | n = 200, δ = 0,01: typischer Fehler von Median-of-Means das 2,64-fache des Mittels bei Lognormal (σ = 1,5) (2,53 bei σ = 2, 2,43 bei Pareto α = 2,1, 1,25 bei Normal; √(π/2) = 1,2533). Dagegen kleiner als 1: bei symmetrisch mit Pareto-Rand (α = 2,5) 0,56 bei n = 50 und 0,74 bei n = 200, bei t(3) 0,94 bei n = 50. Verzerrung bei Lognormal (σ = 1,5): −31,4 % (K = 37), −12,0 % bei n = 1000; K = 4 / 8 / 16 / 32 / 64: −5,0 / −10,8 / −18,6 / −28,9 / −42,0 %; Median −67,2 %, Trimmen 10 % −48,1 %. Auszahlung bei δ = 0,001 (Verhältnis zum Mittel): Lognormal (σ = 2) 0,16, Pareto (α = 2,1) 0,28, Lognormal (σ = 1,5) 0,47, Pareto (α = 3) 0,73, t(3) 0,84; bei Lognormal (σ = 1) und Normal keine (1,08 und 1,23) |
| H4 | Die Regel K = ⌈8 ln(1/δ)⌉ ist bei Schiefe zu groß | **Ja** (das beste K ist im Nachhinein gewählt) | Lognormal (σ = 1,5), n = 200, δ = 0,01: Regel K = 37 gibt 0,531, das beste gemessene K = 8 gibt 0,433, der Fehlerquantil-Wert der Regel liegt 23 % über dem des besten K. Bei schiefen Rändern, wo der Deckel n/2 nicht greift (84 Zellen), verliert die Regel 10 bis 33 % (Median 22 %); das beste K ist bei Lognormal (σ = 1,5) 4 / 8 / 16 für δ = 0,1 / 0,01 / 0,001, bei t(3) 100 (= n/2) |
| H5 | Trimmen mit k = ⌈c ln(1/δ)⌉ je Seite schlägt Median-of-Means | **Ja für symmetrische Ränder; bei schiefen Rändern gewinnt abgestimmtes Median-of-Means; das Bild hängt an meiner Wahl von c** | Dreier-Wahl Mittel / Median-of-Means / Trimmen (kleinstes Q(1 − δ), 160 Zellen): mit c = 1 gewinnt das Trimmen 92, das Mittel 67, Median-of-Means 1 Zelle; mit c = 2: 57 / 75 / 28; mit c = 4: 49 / 75 / 36. Median-of-Means ist besser als das beste Trimmen aus c ∈ {1, 2, 4} in allen Seeds in 1 von 160 Zellen. Der Faktor c ist **eigene Wahl, nicht aus einer Quelle**. Beide im Nachhinein abgestimmt: bei schiefen Rändern gewinnt Median-of-Means 64 von 96 Zellen, das Mittel 32, das Trimmen 0 (das Trimmen ist dort verzerrt: Trimmen 10 % bei Lognormal (σ = 1,5), n = 200: −48,1 %); bei symmetrischen das Trimmen 38 von 64, das Mittel 14, Median-of-Means 12 |
| H6 | Bei symmetrisch schweren Rändern gewinnt Trimmen | **Ja** (Huber nicht gemessen) | Mit c = 1 gewinnt bei den 64 symmetrischen Zellen das Trimmen 48, das Mittel 15 (alle bei Normal), Median-of-Means 1; t(3), n = 200, δ = 0,05: Q(1 − δ) Mittel 0,237, Median-of-Means 0,247, Trimmen mit c = 1 0,200 |
| H7 | Zielgröße Median: der Median-Schätzer ist im typischen Fehler weit besser als Median-of-Means | **Ja** | Typischer Fehler gegen das Ziel Median bei schiefen Verteilungen, δ = 0,05: Median-of-Means ist 2,0- bis 236,5-mal so groß wie der des Median-Schätzers (24 Zellen); Lognormal (σ = 1,5), n = 200: 1,296 gegen 0,089 |
| H8 | Es gibt keine brauchbare Median-of-Means-Intervallkonstruktion | **Ja, als negativer Befund** (nur die geprüften Konstruktionen) | Überdeckung des Mittelwerts bei Soll 95 %, n = 50 / 200 / 1000 / 5000: Student bei Lognormal (σ = 1,5) 81,9 / 87,5 / 91,8 / 93,7 %, Vorzeichen-Intervall auf den Blockmitteln 46,5 / 73,2 / 88,7 / 94,7 %, Median-of-Means ± s·√(32 ln(1/δ)/n) mindestens 99,7 % (Lognormal σ = 1,5, Soll 95 %), aber 4,9- bis 5,0-mal so breit wie Student bei Normal (Soll 95 %). Bei symmetrischen Verteilungen weicht Student in 48 Zellen (4 Verteilungen · 4 n · 3 Niveaus) um höchstens 1,1 Prozentpunkte vom Soll ab. Das Vorzeichen-Intervall hält bei symmetrischen Verteilungen das Niveau (bei Soll 95 % kleinste Überdeckung 97,6 %, 0,9- bis 1,5-mal so breit), bei schiefen bei n = 50 nicht (alle unter 85 %); eine brauchbare Konstruktion bleibt **offen** |
| H9 | Die Konstante 32 im Intervall ist nötig | **Teilweise** | Mit c = 32 liegt die kleinste Überdeckung über alle Verteilungen und n bei 95,7 % (Soll 90 %), 97,0 % (95 %), 98,6 % (99 %: knapp unter dem Soll, Lognormal σ = 2, n = 50); mit c = 8 sinkt sie auf 76,4 / 79,8 / 87,1 %. Die Konstante 32 stammt aus meiner Erinnerung und ist **nicht an einer Quelle geprüft** |

**Korrekturen gegenüber dem Plan.** (1) „Trimmen und Huber sind überall besser“ gilt nur bei **symmetrischen** Rändern; bei schiefen Rändern mit Ziel Mittelwert ist das Trimmen 10 % um 48,1 % verzerrt (Huber ist nicht Teil dieser Demo, der Plan nannte −56 %). (2) In der Dreier-Wahl gewinnt Median-of-Means in 1 von 160 Zellen (Plan: 0); die Zelle: symmetrisch mit Pareto-Rand, n = 50, δ = 0,05. (3) Statt t(2,5) steht **symmetrisch mit Pareto-Rand (α = 2,5)** im Verteilungsraster, weil der Erzeuger nur ganzzahlige Freiheitsgrade der t-Verteilung zieht; die Zahlen dazu sind neu gemessen und nicht mit dem Plan vergleichbar. (4) Das Raster der Niveaus endet bei δ = 0,001 (Plan: 0,0005): bei 40 000 Läufen je Seed gäbe δ = 0,0005 nur 20 Überschreitungen je Seed. (5) Die Streuung der Messung ist nicht klein: das Fehlerquantil des Mittels bei δ = 0,001 streut über die Seeds (Standardabweichung relativ zum Mittelwert der Seeds) bis zu 12,7 % (Lognormal σ = 2, n = 200), bei δ = 0,01 bis zu 3,6 %; Aussagen über Kippniveaus verlangen deshalb „in allen vier Seeds“. (6) Die Presets „Zielgröße Median“ und „Intervall“ des Plans sind Experimente (E5, E6) statt Presets, weil die Zielgröße kein Regler ist (bei symmetrischen Verteilungen wäre er wirkungslos). (7) Bootstrap-Intervalle, Huber und Catoni sind nicht Teil dieser Demo.

## Modell

Gegeben n unabhängige Werte aus einer Verteilung mit Mittelwert μ und endlicher Varianz; gesucht der Schätzer mit kleinem Fehler im Normalfall **und** im seltenen Ernstfall. **Median-of-Means:** K zusammenhängende Blöcke (Größe ⌊n/K⌋ oder ⌈n/K⌉), Mittel je Block, Median der Blockmittel; K = 1 ist das Mittel, K = n der Median. **Trimmen:** die k kleinsten und k größten Werte weglassen. **Regeln:** K = min(⌊n/2⌋, ⌈8 ln(1/δ)⌉) (Faustregel der Vorlage, Herkunft nicht geprüft), k = min(⌊(n − 1)/2⌋, ⌈c ln(1/δ)⌉). Formeln und Herleitungen im 📐-Expander der App.

## Methodik

Je Zelle (Verteilung, n) und Seed 40 000 Stichproben aus dem SplitMix64-Strom (`mom_scenario.py`); alle Schätzer rechnen auf denselben Stichproben. Gemessen werden die Quantile des Fehlers bei den Niveaus 0,5 (typischer Fehler) bis 0,999, die Verzerrung (mittlerer Fehler) und der RMSE; bei schiefen Verteilungen zusätzlich gegen den Median als Ziel. Verteilungen: Normal, t(5), t(3), symmetrisch mit Pareto-Rand (α = 2,5), Lognormal (σ = 1 / 1,5 / 2), Pareto (α = 2,1 / 2,5 / 3); n = 50 / 200 / 1000 / 5000. Die Intervalle (E6) rechnen 8 000 Stichproben je Seed (n = 5000: 4 000). Die Messreihen liegen vorgerechnet in `data/mom_results.json` mit der Prüfsumme der Quelldateien; die App rechnet beliebige Einstellungen mit denselben Funktionen nach (Nachmessen, Läufe · n · Seeds ≤ 20 Millionen Werte).

## Befunde (gemessen, keine Behauptungen)

Die Zahlen stehen in der Tabelle oben; jede ist in `tests/test_claims.py` aus der Messreihe nachgerechnet. Die drei stärksten Aussagen: (1) Bei Lognormal (σ = 1,5) und n = 200 sinkt das Fehlerquantil des Mittels bei δ = 0,001 auf 1,333 gegen 0,625 bei Median-of-Means, bei δ = 0,1 ist es umgekehrt (0,295 gegen 0,376). (2) Die Absicherung kostet −31,4 % Verzerrung und das 2,64-fache des typischen Fehlers. (3) Mit dem Faktor c = 1 gewinnt das Trimmen die Dreier-Wahl in 92 von 160 Zellen, Median-of-Means in 1.

## Ehrliche Grenzen

* **Erfundene Verteilungen, kein Realdatenbezug.** Lognormal und Pareto sind übliche Annahmen für schiefe Mengen und Zeiten, aber kein Beleg, dass echte Sendungsdaten so aussehen. Unabhängig und gleich verteilt: Serien und Saison sind nicht Teil.
* **Der Faktor c und die Blockzahl wurden im Nachhinein verglichen.** Der Vergleich in E4 wählt beide am selben Datensatz; fair wäre die Wahl auf einer Hälfte der Daten (nicht gemessen). Das Bild der Dreier-Wahl hängt an c (H5).
* **Die Regel K = ⌈8 ln(1/δ)⌉ und die Konstante 32 sind nicht an einer Quelle geprüft.** Die Literatur zu Median-of-Means und zum getrimmten Mittel bei schweren Rändern habe ich **nicht eingesehen**; die Demo misst, sie beweist nichts. Eine datengetriebene Wahl von K gibt es nicht.
* **Ein Intervall für den Mittelwert aus Median-of-Means habe ich nicht gefunden** (H8). Das ist die Auswahl der geprüften Konstruktionen, keine Unmöglichkeitsaussage.
* **Seltene Ereignisse brauchen viele Läufe.** Q(1 − δ) bei δ = 0,001 streut zwischen den Seeds (Korrektur 5 oben); das Nachmessen in der App kennzeichnet Ergebnisse mit weniger als 100 Überschreitungen als wenig belastbar.
* **Eine einzelne Stichprobe (Hauptansicht) beweist nichts.** Sie zeigt, wie sich die Schätzer unterscheiden können, nicht, wie sie im Mittel abschneiden.
* **Browser-Durchlauf:** Standardansicht, ein Preset („Zu viele Blöcke“), Nachmessen und alle Abbildungen wurden im Browser angesehen; die PDF-Erzeugung ist nur im AppTest geprüft, nicht mit einem Klick im Browser.

## Tests

**278 Tests** (lokal gemessen 2026-10-26, rund 30 bis 50 s, im Test-venv mit Python 3.14), in Gruppen:

* **Einheiten an Handrechnungen** (`test_estimators.py`, `test_edge_cases.py`, Mini-Instanzen in `tests/conftest.py`): Blockbildung (Daten 1, 2, 3, 4, 5, 100 mit K = 3 gibt 3,5; n = 10, K = 4 gibt Blockgrößen 3, 3, 2, 2; Daten 2, 4, …, 14, 200 mit K = 4 gibt 9), Trimmen, Regel (K = 24 / 37 / 56 bei δ = 0,05 / 0,01 / 0,001, Deckel n/2), Vorzeichen-Intervall (K = 24, δ = 0,05: Rang 7, Überdeckung 0,97735); n = 1, n = 2, alle Werte gleich, inklusive Grenzen.
* **Orakel** (`test_oracle_median_of_means.py`, 66 Tests, anderer Rechenweg): Median-of-Means gegen die Definition mit `numpy.array_split` auf 200 Zufallsinstanzen und gegen die Aufzählung **aller** Zuteilungen auf Blöcke für n ≤ 8 (105 Zuteilungen bei n = 8, K = 4; 15 bei n = 6, K = 3); Trimmen gegen `scipy.stats.trim_mean` auf 200 Instanzen; Vorzeichen-Intervall gegen `scipy.stats.binom`; Student-Quantile der Tabelle gegen `scipy.stats.t`; Erzeuger gegen scipy (Mittelwert, Varianz, Median, Kolmogorov-Smirnov-Abstand); das Fehlerquantil des Mittels bei Normal gegen z/√n und die Verzerrung des Medians gegen e^(−σ²/2) − 1 bzw. 2^(1/α)(α − 1)/α − 1.
* **Eigenschaften** (`test_eigenschaften_lagemasse.py`): affine Gleichheit, Permutation innerhalb und zwischen gleich großen Blöcken, Monotonie, Schranken, konstante Stichprobe, Trimmen zwischen Mittel und Median, Monotonie der Regeln in δ und c, Faktor √(π/2) bei Normal.
* **Zweige (Nullspalten-Regel)** (`test_zweige.py`): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab; die Siegerkarte hat in beiden Verteilungsgruppen mindestens zwei Sieger; der Deckel n/2 greift bei n = 50.
* **Messreihen** (`test_results.py`, `test_results_current.py`, `test_experiments.py`): vollständig, gültiges JSON, Raster gleich den Konstanten, Prüfsumme der Quelldateien (gegen eine unabhängige SHA-256-Rechnung), die Zelle n = 50 wird live mit allen 40 000 Läufen nachgerechnet und stimmt überein, Stapelgrenzen.
* **Aussagen** (`test_claims.py`: jede Zahl dieser README und der Preset-Hilfetexte; `test_verdicts.py`, `test_evaluation*.py`: Schwellen und Zahlen der Verdikte an handgesetzten Tabellen).
* **Oberfläche**: AppTest (Standard, jeder Button, jeder Regler an Minimum und Maximum, Eckfall-Matrix der sechs Regler, alle K-Optionen, Nachmessen, PDF), Layout (Schnellstart in der Seitenleiste, Hervorhebung des aktiven Presets, jede Reglerwahl hebt sie auf), Footer, Abbildungen, PDF, Laufzeit-Importe gegen `requirements.txt`.
* **Laufzeit-Netze**: `test_termination.py` führt die Schleifen im eigenen Prozess mit Zeitgrenze (30 s) aus und prüft zwei `PYTHONHASHSEED`; `test_worstcase.py` rechnet das Maximum (n = 5000, K = n/2, alle Verteilungen, Nachmessen mit dem vollen Budget).

**Mutationsprüfung** (`tools/mutate_demo.py`, mit `--exclude mom_rng.py`, `MOM_RESULTS_PATH` auf die Messdatei und ohne `test_results_current.py`, `test_layout.py`, `test_pdf.py`; Stand `tests/mutation_state.json`): 192 Stellen in `mom_estimators.py`, `mom_experiments.py`, `mom_results.py`, `mom_grid.py` und `mom_evaluation.py`: **180 gefangen, 10 äquivalent** (festgehalten: Rückgabewert von `check_delta` ungenutzt; Schleifengrenzen von `sign_interval_rank` jenseits des Medians; `K <= half` neben der Menge `{half}`; `<=` gegen `<` bei Intervallgrenzen mit Wahrscheinlichkeit 0 für Gleichheit; `>` gegen `>=` beim Gleitkomma-Schwellenwert 0,1; `pay >= 0` nach dem ersten Zweig) und **2 offen**: zwei Endlosschleifen (`while done < runs` zu `<=`), die das Werkzeug als Zeitüberschreitung meldet und nicht als gefangen zählt; von Hand bestätigt, dass `test_termination.py` beide nach 30 s fängt. **Nicht „sauber“**, weil 2 offen sind. Nicht mutiert: `mom_scenario.py` (Erzeuger, durch die Orakel gegen scipy abgesichert), `mom_rng.py` (wörtliche Kopie, Test gegen das Original), Darstellung, App. Der Lauf für `mom_evaluation.py` brauchte eine Arbeitskopie mit einem zusätzlichen Import im Orakel (das Werkzeug mutiert nur Module, die das Orakel erreicht); der Stand ist in `tests/mutation_state.json` übernommen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets, Zwischenstände der Messung |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Hauptansicht, Nachmessen, E1 bis E6, Grenzen, Mathematik) |
| `mom_rng.py` | SplitMix64, wörtliche Kopie von `reduktionskette-demo/red_scenario.py` |
| `mom_scenario.py` | Verteilungen (Sendungsmengen, Standzeiten, Lieferverzug), blockweiser SplitMix64-Strom, Fehlerdefinition |
| `mom_estimators.py` | Median-of-Means, Trimmen, Blockzahl-Regel, Vorzeichen-Intervall |
| `mom_grid.py` | Raster der Messreihen (Seeds, n, Niveaus, Läufe, Student-Quantile), Teil der Prüfsumme |
| `mom_experiments.py` | Messung einer Zelle und der Intervalle, Teile der Messreihe, Prüfsumme |
| `mom_results.py` | Laden und Abfragen von `data/mom_results.json` |
| `mom_evaluation.py` | Kippniveau, Prämie, Dreier-Wahl, Verdikte als reine Funktionen, Nachmessen |
| `mom_constants.py` | Regler-Grenzen, Farben, Presets mit Hilfetexten |
| `mom_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `mom_layout.py` | Fester Kopf (Stylesheet) |
| `mom_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mom_pdf_export.py` | PDF-Export (fpdf2), erst nach Klick erzeugt |
| `tools/make_results.py`, `tools/run_parts.sh` | Erzeugt die Messreihen (Teile parallel) |
| `data/mom_results.json` | Vorgerechnete Messreihen mit Prüfsumme der Quelldateien |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly, fpdf2); dev: pytest, scipy (nur Orakel) |
| `tests/` | Einheiten, Orakel, Eigenschaften, Randfälle, Zweige, Messreihen, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

* **Huber und Catoni** → Huber gehört zu `trimmen-huber-demo` (Stück 4); Catoni ist dort grob gemessen und nicht Teil dieser Demo.
* **Tests auf Mittelwertdifferenz** → `permutation-test-demo`.
* **Eine datengetriebene Wahl von K** und die Wahl von K und c auf einer Hälfte der Daten.
* **Bootstrap-Intervalle für Median-of-Means** und adaptive Intervalle (E6 zeigt nur die geprüften Konstruktionen).
* **Hohe Dimension** (Mittelwertschätzung für Vektoren) und abhängige Daten.
* **Realdaten**: nur erfundene Verteilungen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Streamlit-Cloud-Einstellung der Python-Version (die CI nutzt Python 3.12 <!-- lint-ok: dezimalpunkt -->) ist beim Deploy zu prüfen und hier zu vermerken.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Messreihen neu erzeugen: `sh tools/run_parts.sh` (20 Teile, höchstens 6 Prozesse gleichzeitig); danach schlägt `tests/test_results.py` nur noch an, wenn sich eine Quelldatei der Messung geändert hat.

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
