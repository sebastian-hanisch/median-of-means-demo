"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets. Das Raster der vorgerechneten Messreihen steht in mom_grid.py.

Buchstaben: n Stichprobengröße, δ das Niveau (das Fehlerquantil Q(1 − δ) ist der Fehler, der nur in einem Anteil δ der Läufe überschritten wird), K die Zahl der Blöcke von
Median-of-Means, k die Zahl der abgeschnittenen Werte je Seite beim Trimmen, c der Faktor in k = ⌈c ln(1/δ)⌉, μ der wahre Mittelwert.
Fehler: bei symmetrischen Verteilungen |T|, bei schiefen |T − μ| / μ (relativ zum Mittelwert); die Zielgröße Median zeigt E5."""

from __future__ import annotations

from mom_grid import DELTA_GRID, SEEDS  # noqa: F401
from mom_scenario import DISTRIBUTIONS, SYMMETRIC

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
DELTA_OPTIONS = DELTA_GRID                                   # 0,5 bis 0,001
K_OPTIONS = ("Regel", "1", "2", "4", "8", "16", "32", "64", "128", "256", "n/2")   # „Regel“: K = ⌈8 ln(1/δ)⌉; jede Zahl wird bei n/2 gedeckelt
C_CHOICES = (1, 2, 4)

N_MIN, N_MAX, DEFAULT_N = 20, 5000, 200        # Rechenaufwand am Maximum: Sortieren O(n log n), Blockmittel O(n), Median der Blockmittel O(K)
SEED_MAX, DEFAULT_SEED = 999999, 7
DEFAULT_DIST = "Lognormal (σ = 1,5)"
DEFAULT_DELTA = 0.01
DEFAULT_K = "Regel"
DEFAULT_C = 1
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS = 200, 5000, 1000   # Läufe je Seed beim Nachmessen

# --- Rechenbudget ---------------------------------------------------------------------------------------------------------------------------
LIVE_BUDGET = 20_000_000       # Läufe · n · Seeds höchstens (Nachmessen in der App); bei n = 5000 und 4 Seeds also 1000 Läufe je Seed
LIVE_SEEDS = (11, 12, 13, 14)

# --- Farben -------------------------------------------------------------------------------------------------------------------------------
COLOR_POINT = "rgba(80,80,80,0.55)"
COLOR_TARGET = "#2F6B65"
COLOR_LINE = "#4c78a8"
COLOR_LINE2 = "#e07b00"
COLOR_BAD = "#d62728"
COLOR_GOOD = "#2F6B65"
SPEC_COLORS = {"Mittel": "#7f7f7f", "Median": "#59a14f", "MoM": "#d62728", "Trimmen": "#4c78a8"}
BLOCK_COLORS = ("#4c78a8", "#e07b00", "#59a14f", "#b07aa1", "#9c755f", "#edc948", "#76b7b2", "#ff9da7")

# --- Presets (Werte für jeden Regler; Verhalten in tests/test_presets.py geprüft) -----------------------------------------------------------
_BASE = {"dist": DEFAULT_DIST, "n": DEFAULT_N, "delta": DEFAULT_DELTA, "kopt": DEFAULT_K, "c": DEFAULT_C, "seed": DEFAULT_SEED}


def _preset(**changes):
    unknown = set(changes) - set(_BASE)
    assert not unknown, unknown
    return {**_BASE, **changes}


PRESETS = {
    "Schiefe Mengen (Standard)": _preset(),
    "Normal: die Absicherung kostet nur": _preset(dist="Normal"),
    "Selten zahlt sie sich aus": _preset(delta=0.001),
    "Zu viele Blöcke": _preset(kopt="n/2"),
    "Standzeiten mit Pareto-Tail": _preset(dist="Pareto (α = 2,1)", n=1000),
    "Symmetrisch schwer (t(3))": _preset(dist="t(3)"),
}
# Hilfetexte: nach der Messung geschrieben, jede Zahl darin steht in tests/test_claims.py (test_preset_help_numbers).
PRESET_HELP = {
    "Schiefe Mengen (Standard)": "Sendungsmengen, lognormal (σ = 1,5), n = 200, δ = 0,01: Median-of-Means (K = 37) hat im seltenen Ernstfall einen kleineren Fehler als das Mittel (Verhältnis 0,83), "
                                 "im Normalfall aber einen 2,64-mal so großen und liegt im Mittel 31,4 % unter dem Mittelwert (E1 bis E3).",
    "Normal: die Absicherung kostet nur": "Normalverteilte Verspätungen: Das Mittel ist am genauesten. Das Fehlerquantil von Median-of-Means ist auf jedem Niveau und bei jedem n der Messung zwischen 1,13- und 1,26-mal so groß wie das des Mittels (bei δ = 0,01 und n = 200: 1,24); "
                                          "die Absicherung kostet hier nur (E1).",
    "Selten zahlt sie sich aus": "Wie der Standard, aber δ = 0,001 (K = 56): Das Fehlerquantil des Mittels ist 1,333, das von Median-of-Means 0,625 (Verhältnis 0,47). Je seltener der Ernstfall, desto eher zahlt sich die Absicherung aus.",
    "Zu viele Blöcke": "K = n/2 = 100 Blöcke (zwei Werte je Block): Die Verzerrung wächst auf −51,5 % des Mittelwerts, und das Fehlerquantil (0,656 bei δ = 0,01) ist größer als das des Mittels (0,643) (E2).",
    "Standzeiten mit Pareto-Tail": "Standzeiten, Pareto (α = 2,1), n = 1000, δ = 0,01: Median-of-Means schlägt das Mittel im Ernstfall (Verhältnis 0,81), kostet im Normalfall das 2,01-fache und liegt 4,9 % unter dem Mittelwert (E1 bis E3).",
    "Symmetrisch schwer (t(3))": "Verspätungen, t-verteilt mit 3 Freiheitsgraden (symmetrisch, kein Verzerrungsproblem): Bei δ = 0,01 und n = 200 liegt Median-of-Means gleichauf mit dem Mittel (Verhältnis 0,99); "
                                 "das Trimmen mit wenigen Werten ist besser (E4).",
}
assert set(DISTRIBUTIONS) >= set(SYMMETRIC) and len(set(PRESETS)) == len(PRESETS)
