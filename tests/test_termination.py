"""Schleifen mit Abbruchbedingung (Rang des Vorzeichen-Intervalls, Stapelschleife der Messung) enden im eigenen Prozess innerhalb einer Zeitgrenze, auch auf Randfällen.

Ein Mutant, der eine solche Schleife endlos laufen lässt, wird sonst erst vom Zeitlimit des Mutationswerkzeugs gefangen (und nicht in tests/mutation_state.json vermerkt).
Zweiter Teil: dasselbe Szenario mit zwei verschiedenen PYTHONHASHSEED rechnet dieselben Zahlen (kein Zufall aus der Hash-Reihenfolge)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import numpy as np
import mom_estimators as E
import mom_experiments as X
import mom_scenario as S

for K in (1, 2, 3, 24, 100, 2500):                                                                    # 2500 = größtes K des Reglers (n/2 bei n = 5000)
    for delta in (0.5, 0.05, 0.001):
        r = E.sign_interval_rank(K, delta)
        assert r is None or (1 <= r[0] <= K // 2 + 1 and 1 - delta - 1e-9 <= r[1] <= 1.0)
cell = X.measure_cell("Pareto (α = 2,1)", 37, 7, (1,), ["mean", "median", "mom:18", "trim:4"])       # Läufe kleiner als eine Stapelgröße, ungerades n
assert len(cell["rows"]) == 4
cell = X.measure_cell("t(3)", 5000, 450, (1,), ["mean", "mom:2500"])                                  # mehrere Stapel (400 Zeilen je Stapel) und ein Rest
assert np.isfinite(np.asarray(cell["rows"][1]["q"])).all()
iv = X.measure_intervals("Normal", 50, 9, (1,))                                                     # Stapelschleife der Intervall-Messung
assert len(iv["rows"]) == 12
x = S.batch("Lognormal (σ = 2)", 3, 5000, S.Stream(1))
assert np.isfinite(x).all()
print("fertig")
"""

REPRO = r"""
import mom_experiments as X
cell = X.measure_cell("t(3)", 51, 60, (1, 2), ["mean", "median", "mom:7", "trim:3"])
print(repr(cell["rows"]))
"""


def test_all_loops_terminate_within_the_time_limit():
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("fertig")


def test_results_do_not_depend_on_the_hash_seed():
    outputs = []
    for hash_seed in ("1", "2"):
        done = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.'); " + REPRO.strip().replace("\n", "; ")],
                              cwd=ROOT, capture_output=True, text=True, timeout=30, encoding="utf-8", env={**os.environ, "PYTHONHASHSEED": hash_seed})
        assert done.returncode == 0, done.stderr[-2000:]
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1] and outputs[0].startswith("[{")
