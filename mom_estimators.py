"""Schätzer: Median-of-Means, getrimmtes Mittel, Blockzahl-Regel und das Vorzeichen-Intervall auf den Blockmitteln (nur numpy und Standardbibliothek).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b). Stichproben-Stapel haben die Form (Läufe, n); eine einzelne Stichprobe (n,) wird wie
ein Stapel mit einer Zeile behandelt und liefert eine Zahl.

Buchstaben: n Stichprobengröße, K Zahl der Blöcke, δ das Niveau (Fehlerquantil 1 − δ), k Zahl der abgeschnittenen Werte je Seite, c der Faktor in k = ⌈c ln(1/δ)⌉."""

from __future__ import annotations

import math

import numpy as np

RULE_FACTOR = 8.0        # K = ⌈8 ln(1/δ)⌉: die Faustregel der Vorlage; ihre Herkunft ist nicht an einer Quelle geprüft (siehe README)
C_OPTIONS = (1, 2, 4)    # Faktoren c für Trimmen mit k = ⌈c ln(1/δ)⌉ je Seite: eigene Wahl


def _rows(x) -> np.ndarray:
    return np.atleast_2d(np.asarray(x, dtype=float))


def block_sizes(n: int, K: int) -> np.ndarray:
    """Größen der K zusammenhängenden Blöcke: n // K, die ersten n % K Blöcke einen Wert mehr (nichts wird weggeworfen)."""
    if not 1 <= K <= n:
        raise ValueError(f"K = {K} liegt außerhalb von 1 bis n = {n}")
    sizes = np.full(K, n // K, dtype=np.int64)
    sizes[: n % K] += 1
    return sizes


def block_edges(n: int, K: int) -> np.ndarray:
    """Anfangsindex jedes Blocks (für np.add.reduceat)."""
    return np.concatenate(([0], np.cumsum(block_sizes(n, K))[:-1])).astype(np.int64)


def block_labels(n: int, K: int) -> np.ndarray:
    """Blocknummer (0 bis K − 1) jedes der n Werte, für die Einfärbung."""
    return np.repeat(np.arange(K), block_sizes(n, K))


def block_means(x, K: int) -> np.ndarray:
    """Mittel jedes der K Blöcke, Form (Läufe, K)."""
    x = _rows(x)
    sizes = block_sizes(x.shape[1], K)
    return np.add.reduceat(x, block_edges(x.shape[1], K), axis=1) / sizes


def median_of_means(x, K: int):
    """Median-of-Means: K Blöcke, Mittel je Block, Median der Blockmittel (bei geradem K das Mittel der beiden mittleren). K = 1 ist das Mittel, K = n der Median."""
    arr = _rows(x)
    out = np.median(block_means(arr, K), axis=1)
    return float(out[0]) if np.ndim(x) == 1 else out


def trimmed_mean_sorted(xs, k: int):
    """Mittel der sortierten Zeilen ohne die k kleinsten und k größten Werte. Verlangt 0 ≤ k und 2k < n."""
    xs = _rows(xs)
    n = xs.shape[1]
    if k < 0 or 2 * k >= n:
        raise ValueError(f"k = {k} je Seite passt nicht zu n = {n}")
    return xs[:, k: n - k].mean(axis=1)


def trimmed_mean(x, k: int):
    """Getrimmtes Mittel mit k abgeschnittenen Werten je Seite (nach dem Sortieren)."""
    arr = _rows(x)
    out = trimmed_mean_sorted(np.sort(arr, axis=1), k)
    return float(out[0]) if np.ndim(x) == 1 else out


def _ceil(x: float) -> int:
    """Aufrunden mit Toleranz gegen Gleitkomma-Kanten: ⌈24,000000000000004⌉ soll 24 sein, nicht 25."""
    return math.ceil(x - 1e-9)


def check_delta(delta: float) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError(f"Niveau δ = {delta} liegt nicht in (0, 1)")
    return float(delta)


def k_of_delta(delta: float, n: int) -> int:
    """Blockzahl nach der Regel K = ⌈8 ln(1/δ)⌉, mindestens 1 und höchstens n/2 (mindestens zwei Werte je Block, außer n = 1)."""
    check_delta(delta)
    return int(max(1, min(max(n // 2, 1), _ceil(RULE_FACTOR * math.log(1.0 / delta)))))


def trim_count(c: float, delta: float, n: int) -> int:
    """Zahl der abgeschnittenen Werte je Seite, k = ⌈c ln(1/δ)⌉, höchstens (n − 1) // 2 (es bleibt mindestens ein Wert)."""
    check_delta(delta)
    return int(max(0, min((n - 1) // 2, _ceil(c * math.log(1.0 / delta)))))


def sign_interval_rank(K: int, delta: float):
    """Vorzeichen-Intervall auf den K Blockmitteln: das größte l ≥ 1 mit P(Bin(K; ½) ≤ l − 1) ≤ δ/2. Das Intervall [b_(l), b_(K+1−l)] der sortierten Blockmittel überdeckt dann den
    MEDIAN der Verteilung der Blockmittel mit Wahrscheinlichkeit 1 − 2 P(Bin ≤ l − 1) ≥ 1 − δ (verteilungsfrei). Rückgabe (l, genaue Überdeckung) oder None, wenn K zu klein ist.
    Achtung: das Intervall gilt für den Median der Blockmittel, nicht für den Mittelwert der Verteilung."""
    check_delta(delta)
    cdf, best = 0.0, None
    for j in range(0, K // 2 + 1):
        cdf += math.comb(K, j) / (1 << K)       # ganzzahlige Division: auch für große K ohne Überlauf
        if cdf <= delta / 2.0:
            best = (j + 1, 1.0 - 2.0 * cdf)
        else:
            break
    return best


def sign_interval(x, K: int, delta: float):
    """Zeilenweise Grenzen (unten, oben) des Vorzeichen-Intervalls auf den Blockmitteln, oder None, wenn K für das Niveau zu klein ist."""
    rank = sign_interval_rank(K, delta)
    if rank is None:
        return None
    l = rank[0]
    b = np.sort(block_means(x, K), axis=1)
    return b[:, l - 1], b[:, K - l]


def estimate_all(x, delta: float, c: float = 1.0, K: int | None = None) -> dict:
    """Alle Schätzer einer einzelnen Stichprobe (n,): Mittel, Median, Trimmen mit k = ⌈c ln(1/δ)⌉ und Median-of-Means mit K (Standard: die Regel). Dazu K und k."""
    arr = np.asarray(x, dtype=float)
    n = arr.shape[0]
    K = k_of_delta(delta, n) if K is None else int(K)
    k = trim_count(c, delta, n)
    return {"Mittel": float(arr.mean()), "Median": float(np.median(arr)), "MoM": median_of_means(arr, K), "Trimmen": trimmed_mean(arr, k), "K": K, "k": k}
