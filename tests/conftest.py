"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Die Erwartungswerte stehen im Test als Rechnung.

Handrechnungen:
* x = (1, 2, 3, 4, 5, 100), K = 3: Blöcke (1, 2 | 3, 4 | 5, 100), Blockmittel 1,5 | 3,5 | 52,5, Median der Blockmittel 3,5. Mittel 115/6 = 19,1667, Median (3 + 4)/2 = 3,5.
* n = 10, K = 4: 10 // 4 = 2 Rest 2, also Blockgrößen 3, 3, 2, 2 (die ersten beiden Blöcke bekommen den Rest).
* x = (2, 4, 6, 8, 10, 12, 14, 200), K = 4: Blöcke (2, 4 | 6, 8 | 10, 12 | 14, 200), Blockmittel 3 | 7 | 11 | 107, K gerade: Mittel der beiden mittleren Blockmittel (7 + 11)/2 = 9.
  Es gibt 8!/(2!⁴ · 4!) = 105 Zuteilungen in vier Paare; kein Paar mit 200 hat ein Mittel unter 101, die beiden mittleren Paarmittel liegen immer zwischen 7 und 11 (siehe Aufzählung im Orakel).
* x = (0, 1, 2, 5, 12): Mittel 20/5 = 4, Median 2; Trimmen mit k = 1 je Seite: (1 + 2 + 5)/3 = 8/3.
* Vorzeichen-Intervall, K = 24, δ = 0,05: P(Bin(24; ½) ≤ 6) = (1 + 24 + 276 + 2024 + 10626 + 42504 + 134596)/2²⁴ = 190051/16777216 = 0,011327 ≤ 0,025, P(Bin ≤ 7) = 536155/16777216 = 0,031958 > 0,025:
  also l = 7 und Überdeckung 1 − 2 · 0,011327 = 0,97735.
* Regel K = ⌈8 ln(1/δ)⌉: δ = 0,05 → ⌈23,97⌉ = 24; δ = 0,01 → ⌈36,84⌉ = 37; δ = 0,001 → ⌈55,26⌉ = 56; δ = 0,5 → ⌈5,55⌉ = 6; bei n = 50 gedeckelt bei n/2 = 25.
* Trimmen k = ⌈c ln(1/δ)⌉: c = 1, δ = 0,01 → ⌈4,605⌉ = 5; c = 2 → ⌈9,21⌉ = 10; c = 4 → ⌈18,42⌉ = 19.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand_x():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])


@pytest.fixture
def pairs_x():
    return np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 200.0])


@pytest.fixture
def trim_x():
    return np.array([0.0, 1.0, 2.0, 5.0, 12.0])
