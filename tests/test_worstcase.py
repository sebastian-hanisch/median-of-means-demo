"""Worst-Case-Test am Slider-Maximum (DEMO-PLAYBOOK §1c): keine Zeit-Assertion, nur ein Timeout im eigenen Prozess als Netz gegen Hänger und Aufblähen.

Komplexität am Maximum n = 5000: Stichprobe, Sortieren O(n log n), Blockmittel O(n), Median der K Blockmittel O(K) (K ≤ 2500). Nachmessen: Läufe · n · Seeds ≤ 20 Mio. Werte (4 Seeds · 1000 Läufe · 5000)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import numpy as np
import mom_constants as C
import mom_evaluation as V
import mom_scenario as S

for dist in S.DISTRIBUTIONS:
    for kopt in ("Regel", "n/2", "1"):
        a = V.analyse(V.Settings(dist=dist, n=C.N_MAX, delta=0.001, kopt=kopt, c=4))
        assert a.x.shape == (C.N_MAX,) and np.isfinite(list(a.values.values())).all(), (dist, kopt)
live = V.live_measure(V.Settings(dist="t(3)", n=C.N_MAX, delta=0.001, kopt="n/2", c=4), C.RUNS_MAX)
assert live["runs"] == 4000 and all(np.isfinite(r["q"]) for r in live["rows"].values())
print("fertig")
"""


def test_largest_instance_finishes():
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=300, encoding="utf-8")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("fertig")
