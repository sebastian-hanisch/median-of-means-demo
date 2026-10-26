"""Messreihen: Fehlerquantile aller Schätzer über Verteilung × n (Kippkarte) und Überdeckung von Intervallen. Dieselben Funktionen rechnen in der App nach (Nachmessen).

`measure_cell` erzeugt für eine Verteilung und ein n je Seed `runs` Stichproben aus dem SplitMix64-Strom (mom_scenario) und wertet auf DENSELBEN Stichproben alle Schätzer aus
(gepaarter Vergleich). Je Schätzer und Seed: Quantile des Fehlers bei den Niveaus Q_LEVELS, mittlerer vorzeichenbehafteter Fehler (Verzerrung) und Wurzel des mittleren
quadratischen Fehlers. Bei schiefen Verteilungen zusätzlich dasselbe gegen den Median als Ziel. Die Teile (`PARTS`) laufen für sich und liefern JSON-fähige Dicts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

import mom_grid as G
import mom_scenario as S
from mom_estimators import (block_means, k_of_delta, median_of_means, sign_interval_rank, trim_count, trimmed_mean_sorted)

ROOT = Path(__file__).resolve().parent
SOURCES = ("mom_rng.py", "mom_scenario.py", "mom_estimators.py", "mom_grid.py", "mom_experiments.py", "tools/make_results.py")
CHUNK_VALUES = 2_000_000       # Werte je Stapel: begrenzt den Speicher; die Stapelgröße hängt nur von n ab, das Ergebnis also nicht vom Rechner


def source_hash() -> str:
    """Prüfsumme der Quelldateien der Messreihen (Zeilenenden normalisiert): stimmt sie nicht mit der in der JSON-Datei überein, ist die Datei veraltet."""
    h = hashlib.sha256()
    for name in SOURCES:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def trim10_k(n: int) -> int:
    """Werte je Seite beim Trimmen mit 10 % (ganzzahliger Anteil, wie scipy.stats.trim_mean): ⌊0,1 n⌋, mit kleinem Zuschlag gegen Gleitkomma-Kanten."""
    return int(0.1 * n + 1e-9)


def mom_spec(K: int) -> str:
    return f"mom:{K}"


def trim_spec(k: int) -> str:
    return f"trim:{k}"


def k_values(n: int) -> list[int]:
    """Alle Blockzahlen, die für n gemessen werden: die Regel-K aller Niveaus, das feste Raster (soweit ≤ n/2) und n/2."""
    half = max(n // 2, 1)
    return sorted({k_of_delta(d, n) for d in G.DELTA_GRID} | {K for K in G.K_GRID if K <= half} | {half})


def trim_values(n: int) -> list[int]:
    """Alle Trimm-Zahlen k je Seite, die für n gemessen werden: ⌈c ln(1/δ)⌉ für c ∈ {1, 2, 4} und alle Niveaus, dazu 10 %."""
    from mom_estimators import C_OPTIONS
    return sorted({trim_count(c, d, n) for c in C_OPTIONS for d in G.DELTA_GRID} | {trim10_k(n)})


def specs_for(n: int) -> list[str]:
    return ["mean", "median"] + [trim_spec(k) for k in trim_values(n)] + [mom_spec(K) for K in k_values(n)]


def estimate_specs(x: np.ndarray, specs: list[str]) -> dict[str, np.ndarray]:
    """Wert jedes Schätzers (Name → Feld über die Zeilen von x). Namen: mean, median, trim:k, mom:K."""
    xs = np.sort(x, axis=1)
    out = {}
    for s in specs:
        if s == "mean":
            out[s] = x.mean(axis=1)
        elif s == "median":
            out[s] = np.median(xs, axis=1)
        elif s.startswith("trim:"):
            out[s] = trimmed_mean_sorted(xs, int(s[5:]))
        elif s.startswith("mom:"):
            out[s] = median_of_means(x, int(s[4:]))
        else:
            raise ValueError(f"unbekannter Schätzer: {s}")
    return out


def _sig(v: float) -> float:
    return float(f"{v:.9g}")


def _summary(est: np.ndarray, target: float, denom: float) -> tuple[list[float], float, float]:
    e = (est - target) / denom
    return [_sig(v) for v in np.quantile(np.abs(e), G.Q_LEVELS)], _sig(float(e.mean())), _sig(float(np.sqrt((e * e).mean())))


def measure_cell(dist: str, n: int, runs: int, seeds=G.SEEDS, specs: list[str] | None = None) -> dict:
    """Eine Zelle (Verteilung, n): je Seed `runs` Stichproben, alle Schätzer auf denselben Stichproben. Rückgabe {dist, n, runs_per_seed, seeds, levels, rows}; jede Zeile
    {spec, q [Seed][Niveau], bias [Seed], rmse [Seed]} und bei schiefen Verteilungen zusätzlich q_med, bias_med, rmse_med (Ziel Median)."""
    specs = list(specs) if specs is not None else specs_for(n)
    targets = [("", S.error_scale(dist, "Mittelwert"))]
    if not S.DIST_INFO[dist].symmetric:
        targets.append(("_med", S.error_scale(dist, "Median")))
    rows = {s: {"spec": s} for s in specs}
    step = max(1, CHUNK_VALUES // n)
    for seed in seeds:
        stream = S.Stream(S.cell_seed(seed, "kipp", dist, n))
        parts = {s: [] for s in specs}
        done = 0
        while done < runs:
            b = min(step, runs - done)
            est = estimate_specs(S.batch(dist, b, n, stream), specs)
            for s in specs:
                parts[s].append(est[s])
            done += b
        for s in specs:
            est = np.concatenate(parts[s])
            for suffix, (target, denom) in targets:
                q, bias, rmse = _summary(est, target, denom)
                rows[s].setdefault("q" + suffix, []).append(q)
                rows[s].setdefault("bias" + suffix, []).append(bias)
                rows[s].setdefault("rmse" + suffix, []).append(rmse)
    return {"dist": dist, "n": n, "runs_per_seed": runs, "seeds": list(seeds), "levels": list(G.Q_LEVELS), "rows": [rows[s] for s in specs]}


# --- Intervalle (E6) ------------------------------------------------------------------------------------------------------------------------

def interval_methods() -> list[str]:
    return ["student", "vorzeichen"] + [f"mom_sigma:{c:g}" for c in G.INTERVAL_CONSTANTS]


def interval_bounds(x: np.ndarray, delta: float, method: str):
    """(unten, oben) je Zeile für das Niveau 1 − δ, oder None, wenn das Verfahren für diese Größe nicht definiert ist (Vorzeichen-Intervall mit zu wenigen Blöcken).
    student: Mittel ± t · s/√n. vorzeichen: Vorzeichen-Intervall auf den Blockmitteln (Blockzahl nach der Regel; gilt für den MEDIAN der Blockmittel).
    mom_sigma:c: Median-of-Means ± s · √(c ln(1/δ) / n) mit der Standardabweichung s der Stichprobe."""
    n = x.shape[1]
    s = x.std(axis=1, ddof=1)
    if method == "student":
        half = G.T_QUANTILES[(n, delta)] * s / math.sqrt(n)
        m = x.mean(axis=1)
        return m - half, m + half
    K = k_of_delta(delta, n)
    if method == "vorzeichen":
        rank = sign_interval_rank(K, delta)
        if rank is None:
            return None
        b = np.sort(block_means(x, K), axis=1)
        return b[:, rank[0] - 1], b[:, K - rank[0]]
    if method.startswith("mom_sigma:"):
        c = float(method[10:])
        half = s * math.sqrt(c * math.log(1.0 / delta) / n)
        m = median_of_means(x, K)
        return m - half, m + half
    raise ValueError(f"unbekanntes Intervall: {method}")


def measure_intervals(dist: str, n: int, runs: int, seeds=G.SEEDS) -> dict:
    """Überdeckung des wahren Mittelwerts und mittlere Breite (relativ: Breite / Mittelwert bei schiefen, absolut bei symmetrischen Verteilungen) je Verfahren, Niveau und Seed."""
    d = S.DIST_INFO[dist]
    rel = 1.0 if d.symmetric else d.mean
    methods = interval_methods()
    acc = {(m, dl): {"cov": [], "wmed": []} for m in methods for dl in G.INTERVAL_DELTAS}
    step = max(1, CHUNK_VALUES // n)
    for seed in seeds:
        stream = S.Stream(S.cell_seed(seed, "int", dist, n))
        hits = {k: [] for k in acc}
        widths = {k: [] for k in acc}
        done = 0
        while done < runs:
            b = min(step, runs - done)
            x = S.batch(dist, b, n, stream)
            for (m, dl) in acc:
                lo_hi = interval_bounds(x, dl, m)
                if lo_hi is None:
                    continue
                lo, hi = lo_hi
                hits[(m, dl)].append((lo <= d.mean) & (d.mean <= hi))
                widths[(m, dl)].append((hi - lo) / rel)
            done += b
        for k in acc:
            if hits[k]:
                acc[k]["cov"].append(_sig(float(np.concatenate(hits[k]).mean())))
                acc[k]["wmed"].append(_sig(float(np.median(np.concatenate(widths[k])))))
    rows = [{"method": m, "delta": dl, **v} for (m, dl), v in acc.items() if v["cov"]]
    return {"dist": dist, "n": n, "runs_per_seed": runs, "seeds": list(seeds), "rows": rows}


# --- Teile ----------------------------------------------------------------------------------------------------------------------------------

def part_kipp(i: int):
    dist = S.DISTRIBUTIONS[i]
    return lambda: {"dist": dist, "cells": [measure_cell(dist, n, G.RUNS_PER_SEED) for n in G.N_RASTER]}


def part_int(i: int):
    dist = S.DISTRIBUTIONS[i]
    return lambda: {"dist": dist, "cells": [measure_intervals(dist, n, G.INTERVAL_RUNS[n]) for n in G.INTERVAL_NS]}


PARTS = {**{f"kipp_{i}": part_kipp(i) for i in range(len(S.DISTRIBUTIONS))}, **{f"int_{i}": part_int(i) for i in range(len(S.DISTRIBUTIONS))}}
