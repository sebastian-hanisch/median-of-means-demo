"""Vorgerechnete Messreihen (data/mom_results.json, erzeugt von tools/make_results.py aus mom_experiments.py): Laden und Abfragen.

Die Datei trägt die Prüfsumme der Quelldateien; `is_current()` vergleicht sie mit dem heutigen Stand (tests/test_results.py).
Jede Zelle (Verteilung, n) hat je Schätzer und Seed die Quantile des Fehlers bei den Niveaus Q_LEVELS, die Verzerrung (mittlerer vorzeichenbehafteter Fehler) und den RMSE.
Abfragen geben, wenn nicht anders gesagt, den Mittelwert über die Seeds zurück; `*_seeds` die Liste je Seed."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np

import mom_grid as G
import mom_scenario as S

PATH = Path(os.environ.get("MOM_RESULTS_PATH") or Path(__file__).resolve().parent / "data" / G.RESULTS_FILE)


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return load()["meta"]


def is_current() -> bool:
    import mom_experiments as X
    return meta()["erzeuger_hash"] == X.source_hash()


def level_index(delta: float) -> int:
    """Index des Niveaus 1 − δ in Q_LEVELS; δ muss im Raster DELTA_GRID liegen."""
    for i, d in enumerate(G.DELTA_GRID):
        if abs(d - delta) < 1e-9:
            return i
    raise ValueError(f"δ = {delta} liegt nicht im Raster {G.DELTA_GRID}")


def nearest_grid_n(n: int) -> int:
    """Das gemessene n aus N_RASTER, das n am nächsten liegt (im Verhältnis, nicht im Abstand: 700 liegt näher an 1000 als an 200)."""
    return min(G.N_RASTER, key=lambda m: abs(np.log(m / max(n, 1))))


@functools.lru_cache(maxsize=None)
def _cells(dist: str) -> dict:
    i = S.DISTRIBUTIONS.index(dist)
    return {c["n"]: c for c in load()[f"kipp_{i}"]["cells"]}


def cell(dist: str, n: int) -> dict:
    return _cells(dist)[n]


def runs_total(dist: str, n: int) -> int:
    c = cell(dist, n)
    return c["runs_per_seed"] * len(c["seeds"])


def specs(dist: str, n: int) -> list[str]:
    return [r["spec"] for r in cell(dist, n)["rows"]]


def row(dist: str, n: int, spec: str) -> dict:
    (r,) = [r for r in cell(dist, n)["rows"] if r["spec"] == spec]
    return r


def _key(base: str, target: str) -> str:
    if target == "Mittelwert":
        return base
    if target == "Median":
        return base + "_med"
    raise ValueError(f"unbekannte Zielgröße: {target}")


def _target(dist: str, target: str) -> str:
    """Bei symmetrischen Verteilungen gibt es nur ein Ziel (0): dort liest jede Abfrage die Spalten für den Mittelwert."""
    return "Mittelwert" if S.DIST_INFO[dist].symmetric else target


def q_seeds(dist: str, n: int, spec: str, delta: float, target: str = "Mittelwert") -> list[float]:
    """Q(1 − δ) des Fehlers je Seed."""
    j = level_index(delta)
    return [q[j] for q in row(dist, n, spec)[_key("q", _target(dist, target))]]


def q(dist: str, n: int, spec: str, delta: float, target: str = "Mittelwert") -> float:
    return float(np.mean(q_seeds(dist, n, spec, delta, target)))


def q_std(dist: str, n: int, spec: str, delta: float, target: str = "Mittelwert") -> float:
    """Standardabweichung von Q(1 − δ) über die Seeds (Maß für die Streuung der Messung)."""
    return float(np.std(q_seeds(dist, n, spec, delta, target), ddof=1))


def typical(dist: str, n: int, spec: str, target: str = "Mittelwert") -> float:
    """Typischer Fehler: der Median des Fehlers (Niveau 0,5, δ = 0,5)."""
    return q(dist, n, spec, 0.5, target)


def bias(dist: str, n: int, spec: str, target: str = "Mittelwert") -> float:
    """Mittlerer vorzeichenbehafteter Fehler, bei schiefen Verteilungen relativ zum Ziel (−0,3 heißt: 30 % darunter)."""
    return float(np.mean(row(dist, n, spec)[_key("bias", _target(dist, target))]))


def rmse(dist: str, n: int, spec: str, target: str = "Mittelwert") -> float:
    return float(np.mean(row(dist, n, spec)[_key("rmse", _target(dist, target))]))


# --- Intervalle -----------------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _interval_cells(dist: str) -> dict:
    i = S.DISTRIBUTIONS.index(dist)
    return {c["n"]: c for c in load()[f"int_{i}"]["cells"]}


def interval_cell(dist: str, n: int) -> dict:
    return _interval_cells(dist)[n]


def interval_row(dist: str, n: int, method: str, delta: float) -> dict | None:
    rows = [r for r in interval_cell(dist, n)["rows"] if r["method"] == method and abs(r["delta"] - delta) < 1e-9]
    return rows[0] if rows else None


def coverage(dist: str, n: int, method: str, delta: float) -> float | None:
    """Überdeckung des wahren Mittelwerts, Mittel über die Seeds; None, wenn das Verfahren für diese Größe nicht definiert ist."""
    r = interval_row(dist, n, method, delta)
    return None if r is None else float(np.mean(r["cov"]))


def width(dist: str, n: int, method: str, delta: float) -> float | None:
    """Mittlere (über die Seeds) mediane Breite, relativ zum Mittelwert (schief) bzw. absolut (symmetrisch)."""
    r = interval_row(dist, n, method, delta)
    return None if r is None else float(np.mean(r["wmed"]))
