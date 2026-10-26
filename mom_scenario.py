"""Szenario: Sendungsmengen, Standzeiten und Lieferverzug mit schweren Rändern, aus dem eigenen SplitMix64-Strom.

Alle Zufallszahlen stammen aus dem SplitMix64-Strom von `mom_rng.py` (wörtliche Kopie); nichts aus `numpy.random`, damit jede Reihe auf jeder Plattform und in jeder Python-Version
gleich ist. SplitMix64 ist ein Zähler-Generator: der k-te Wert des Stroms hängt nur vom Seed und von k ab, ein ganzer Block lässt sich also auf einmal mit numpy berechnen
(`mix_words`); tests/test_scenario.py prüft ihn Wert für Wert gegen die Python-Klasse. Normalwerte nach Box-Muller; log, cos, sin und Potenzen kommen aus numpy, die letzte
Binärstelle kann je Plattform abweichen.

Vehikel (Einheiten sind frei gewählt, die Verteilungen sind erfundene Modelle, keine Messdaten):
* Sendungsmenge je Kunde: lognormal, Einheit = Median (Median 1).
* Standzeit an der Rampe: Pareto mit Mindestwert 1 (Einheit = Mindeststandzeit), Tail-Exponent α.
* Lieferverzug gegenüber Plan: symmetrisch um 0 (t-Verteilung bzw. beidseitig mit Pareto-Rand), Ziel 0."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from mom_rng import MASK, SplitMix64  # noqa: F401  (SplitMix64: Gegenprobe in den Tests)

GAMMA = 0x9E3779B97F4A7C15
_C1 = 0xBF58476D1CE4E5B9
_C2 = 0x94D049BB133111EB
TWO_M53 = 2.0 ** -53


@dataclass(frozen=True)
class Dist:
    kind: str
    param: float
    mean: float
    median: float
    var: float | None          # None: unendliche Varianz
    symmetric: bool
    vehicle: str               # Vehikel in einem Wort für Texte


def _lognormal(sigma: float) -> Dist:
    return Dist("lognormal", sigma, math.exp(sigma * sigma / 2.0), 1.0, (math.exp(sigma * sigma) - 1.0) * math.exp(sigma * sigma), False, "Sendungsmenge je Kunde")


def _pareto(alpha: float) -> Dist:
    var = alpha / ((alpha - 1.0) ** 2 * (alpha - 2.0)) if alpha > 2 else None
    return Dist("pareto", alpha, alpha / (alpha - 1.0), 2.0 ** (1.0 / alpha), var, False, "Standzeit an der Rampe")


DIST_INFO = {
    "Normal": Dist("normal", 0.0, 0.0, 0.0, 1.0, True, "Lieferverzug"),
    "t(5)": Dist("t", 5.0, 0.0, 0.0, 5.0 / 3.0, True, "Lieferverzug"),
    "t(3)": Dist("t", 3.0, 0.0, 0.0, 3.0, True, "Lieferverzug"),
    "Symmetrisch mit Pareto-Rand (α = 2,5)": Dist("pareto_sym", 2.5, 0.0, 0.0, 2.0 / (1.5 * 0.5), True, "Lieferverzug"),
    "Lognormal (σ = 1)": _lognormal(1.0),
    "Lognormal (σ = 1,5)": _lognormal(1.5),
    "Lognormal (σ = 2)": _lognormal(2.0),
    "Pareto (α = 2,1)": _pareto(2.1),
    "Pareto (α = 2,5)": _pareto(2.5),
    "Pareto (α = 3)": _pareto(3.0),
}
DISTRIBUTIONS = tuple(DIST_INFO)
SYMMETRIC = tuple(name for name, d in DIST_INFO.items() if d.symmetric)
SKEWED = tuple(name for name, d in DIST_INFO.items() if not d.symmetric)


def mix_words(seed: int, start: int, count: int) -> np.ndarray:
    """Die Werte Nr. start … start + count − 1 (ab 0 gezählt) des SplitMix64-Stroms zum Seed, als uint64-Feld."""
    k = np.arange(start + 1, start + 1 + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = k * np.uint64(GAMMA) + np.uint64(seed & MASK)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(_C1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(_C2)
        return z ^ (z >> np.uint64(31))


def cell_seed(base: int, *tags) -> int:
    """Stabiler Seed einer Zelle aus dem Basis-Seed und beliebig vielen Marken (Zahlen oder Texte)."""
    value = int(base)
    for tag in tags:
        part = zlib.crc32(tag.encode("utf-8")) if isinstance(tag, str) else int(round(float(tag) * 1000))
        value = (value * 1_000_003 + part) % (1 << 61)
    return value


class Stream:
    """Zufallsstrom mit Position: jeder Aufruf zieht die nächsten Werte des SplitMix64-Stroms zum Seed (als Block)."""

    def __init__(self, seed: int):
        self.seed = int(seed) & MASK
        self.pos = 0

    def words(self, count: int) -> np.ndarray:
        w = mix_words(self.seed, self.pos, count)
        self.pos += count
        return w

    def uniform(self, count: int) -> np.ndarray:
        """Gleichverteilt in [0, 1), 53 Bit."""
        return (self.words(count) >> np.uint64(11)).astype(np.float64) * TWO_M53

    def uniform_open(self, count: int) -> np.ndarray:
        """Gleichverteilt in (0, 1), nie 0 oder 1 (für Logarithmen und Umkehrfunktionen)."""
        return ((self.words(count) >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53

    def normal(self, count: int) -> np.ndarray:
        """Standardnormal nach Box-Muller: aus zwei Strom-Werten (u1, u2) die zwei Werte r·cos(2π u2), r·sin(2π u2), r = √(−2 ln u1). Bei ungerader Zahl entfällt der letzte."""
        pairs = (count + 1) // 2
        w = self.words(2 * pairs)
        u1 = ((w[0::2] >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53
        u2 = ((w[1::2] >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53
        r = np.sqrt(-2.0 * np.log(u1))
        out = np.empty(2 * pairs)
        out[0::2] = r * np.cos(2.0 * math.pi * u2)
        out[1::2] = r * np.sin(2.0 * math.pi * u2)
        return out[:count]


def draw(kind: str, param: float, stream: Stream, count: int) -> np.ndarray:
    """`count` Werte in Standardform: normal(0; 1), t mit `param` Freiheitsgraden (ganzzahlig), lognormal(0; param) mit Median 1, pareto(1; param) mit Mindestwert 1,
    pareto_sym(param) = (Pareto(1; param) − 1) mit zufälligem Vorzeichen: symmetrisch um 0 mit positiver Dichte in 0 (ohne den Abzug 1 hätte die Verteilung eine Lücke um 0,
    und der Median der Stichprobe wäre bei ±1 statt bei 0), Varianz 2/((α − 1)(α − 2)), Tail-Exponent α."""
    if kind == "normal":
        return stream.normal(count)
    if kind == "t":
        nu = int(param)
        z = stream.normal(count * (nu + 1)).reshape(count, nu + 1)
        return z[:, 0] / np.sqrt((z[:, 1:] ** 2).sum(axis=1) / nu)
    if kind == "lognormal":
        return np.exp(param * stream.normal(count))
    if kind == "pareto":
        return stream.uniform_open(count) ** (-1.0 / param)
    if kind == "pareto_sym":
        magnitude = stream.uniform_open(count) ** (-1.0 / param) - 1.0
        return np.where(stream.uniform(count) < 0.5, -magnitude, magnitude)
    raise ValueError(f"unbekannte Verteilung: {kind}")


def batch(dist: str, runs: int, n: int, stream: Stream) -> np.ndarray:
    """Stapel (runs, n) unabhängiger Stichproben der Verteilung `dist` (Name aus DIST_INFO)."""
    d = DIST_INFO[dist]
    return draw(d.kind, d.param, stream, runs * n).reshape(runs, n)


def one_sample(dist: str, n: int, seed: int) -> np.ndarray:
    """Eine Stichprobe der Länge n, deterministisch aus (dist, n, seed). Die Reihenfolge ist die des Stroms (zufällig, unabhängig)."""
    return batch(dist, 1, n, Stream(cell_seed(seed, "stichprobe", dist)))[0]


def population(dist: str) -> tuple[float, float]:
    """(Mittelwert, Median) der Verteilung."""
    d = DIST_INFO[dist]
    return d.mean, d.median


def error_scale(dist: str, target: str = "Mittelwert") -> tuple[float, float]:
    """(Ziel, Nenner) der Fehlerdefinition: Fehler = |T − Ziel| / Nenner. Symmetrische Verteilungen: Ziel 0 (Mittelwert = Median), Nenner 1 (absolut);
    schiefe: Ziel Mittelwert bzw. Median der Verteilung, Nenner derselbe Wert (relativer Fehler)."""
    d = DIST_INFO[dist]
    if d.symmetric:
        return 0.0, 1.0
    if target not in ("Mittelwert", "Median"):
        raise ValueError(f"unbekannte Zielgröße: {target}")
    value = d.mean if target == "Mittelwert" else d.median
    return value, value
