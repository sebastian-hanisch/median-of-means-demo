"""Raster der vorgerechneten Messreihen (data/mom_results.json, erzeugt von tools/make_results.py). Diese Datei gehört zur Prüfsumme der Messung: ändert sich ein Wert,
ist die Ergebnisdatei veraltet (tests/test_results.py). Buchstaben wie in mom_constants.py."""

from __future__ import annotations

SEEDS = (11, 12, 13, 14)       # vier feste Zufalls-Seeds je Zelle: nie ein einzelner Seed als Befund
RESULTS_FILE = "mom_results.json"
N_RASTER = (50, 200, 1000, 5000)
RUNS_PER_SEED = 40_000
Q_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)          # Quantile des Fehlers; δ = 1 − Level
DELTA_GRID = tuple(round(1.0 - q, 3) for q in Q_LEVELS)                       # 0,5 bis 0,001
K_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)                                   # feste Blockzahlen der Messung (dazu n/2 und die Regel-K)
MIN_EXCEEDANCES = 100          # ein Quantil Q(1 − δ) gilt als belastbar, wenn δ · (Läufe über alle Seeds) mindestens so groß ist
INTERVAL_NS = (50, 200, 1000, 5000)
INTERVAL_DELTAS = (0.1, 0.05, 0.01)
INTERVAL_RUNS = {50: 8000, 200: 8000, 1000: 8000, 5000: 4000}
INTERVAL_CONSTANTS = (8.0, 32.0)    # c in MoM ± σ̂ · √(c · ln(1/δ) / n): 32 ist eine Konstante aus der Erinnerung (nicht geprüft), 8 die kleinere Vergleichswahl
# Zweiseitige Student-t-Quantile t_{1 − δ/2; n − 1}, mit scipy.stats.t.ppf erzeugt und in tests/test_oracle_median_of_means.py dagegen geprüft (die Laufzeit braucht kein scipy).
T_QUANTILES = {
    (50, 0.1): 1.6765508926168535, (50, 0.05): 2.0095752371292392, (50, 0.01): 2.679951973631552,
    (200, 0.1): 1.6525467461665633, (200, 0.05): 1.9719565442517533, (200, 0.01): 2.600760216058516,
    (1000, 0.1): 1.6463803454275352, (1000, 0.05): 1.9623414611334493, (1000, 0.01): 2.5807596372676365,
    (5000, 0.1): 1.6451584985826642, (5000, 0.05): 1.9604386466615247, (5000, 0.01): 2.576813163398982,
}

