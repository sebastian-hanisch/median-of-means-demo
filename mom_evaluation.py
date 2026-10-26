"""Kennzahlen, Zusammenfassungen der Messreihen und Verdikt-Logik. Alles reine Funktionen: die App ruft sie auf, tests/ prüft sie an handgesetzten Werten.

Buchstaben: δ das Niveau (Q(1 − δ) ist das Fehlerquantil), K die Zahl der Blöcke, k die Zahl der abgeschnittenen Werte je Seite, c der Faktor in k = ⌈c ln(1/δ)⌉.
Fehler: bei symmetrischen Verteilungen |T|, bei schiefen |T − Ziel| / Ziel (Ziel Mittelwert oder Median der Verteilung)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import mom_constants as C
import mom_estimators as E
import mom_experiments as X
import mom_grid as G
import mom_results as R
import mom_scenario as S

COUNT_DELTAS = (0.1, 0.05, 0.01, 0.001)       # Niveaus der Zählungen über die „160 Zellen“ (10 Verteilungen · 4 n · 4 Niveaus)
PARTS_OF_MOM = ("Mittel", "Median", "Trimmen", "MoM")


def de(x: float, nd: int = 3) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    """Anteil als Prozentzahl mit Dezimalkomma (0,314 → „31,4 %“)."""
    return f"{100.0 * x:.{nd}f} %".replace(".", ",")


def de_delta(delta: float) -> str:
    """Niveau δ mit so vielen Stellen, wie es braucht (0,5; 0,05; 0,001)."""
    return f"{delta:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def k_from_option(option: str, delta: float, n: int) -> int:
    """Blockzahl aus der Reglerwahl: „Regel“ (⌈8 ln(1/δ)⌉, gedeckelt bei n/2), „n/2“ oder eine Zahl; höchstens n/2, mindestens 1."""
    half = max(n // 2, 1)
    if option == "Regel":
        return E.k_of_delta(delta, n)
    if option == "n/2":
        return half
    return max(1, min(half, int(option)))


# --- Eine Stichprobe (Hauptansicht) ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    dist: str = C.DEFAULT_DIST
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    kopt: str = "Regel"
    c: int = C.DEFAULT_C
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: np.ndarray
    K: int
    k: int
    values: dict            # Mittel, Median, Trimmen, MoM
    labels: np.ndarray      # Blocknummer je Wert
    block_means: np.ndarray
    mean_true: float
    median_true: float
    errors: dict            # Fehler je Schätzer gegen den Mittelwert (relativ bei schiefen, absolut bei symmetrischen Verteilungen), vorzeichenbehaftet


def signed_error(dist: str, value: float, target: str = "Mittelwert") -> float:
    t, denom = S.error_scale(dist, target)
    return (value - t) / denom


def analyse(settings: Settings) -> Analysis:
    x = S.one_sample(settings.dist, settings.n, settings.seed)
    K = k_from_option(settings.kopt, settings.delta, settings.n)
    est = E.estimate_all(x, settings.delta, c=settings.c, K=K)
    values = {k: est[k] for k in PARTS_OF_MOM}
    mean_true, median_true = S.population(settings.dist)
    errors = {k: signed_error(settings.dist, v) for k, v in values.items()}
    return Analysis(settings, x, K, est["k"], values, E.block_labels(settings.n, K), E.block_means(x, K)[0], mean_true, median_true, errors)


# --- Messreihen: Kippkarte ------------------------------------------------------------------------------------------------------------------

def mom_spec(delta: float, n: int) -> str:
    return X.mom_spec(E.k_of_delta(delta, n))


def trim_spec(c: float, delta: float, n: int) -> str:
    return X.trim_spec(E.trim_count(c, delta, n))


def ratio_seeds(dist: str, n: int, spec: str, ref: str, delta: float, target: str = "Mittelwert") -> list[float]:
    """Q(1 − δ) von `spec` geteilt durch Q(1 − δ) von `ref`, je Seed (gleiche Stichproben: gepaarter Vergleich)."""
    return [a / b for a, b in zip(R.q_seeds(dist, n, spec, delta, target), R.q_seeds(dist, n, ref, delta, target))]


def kipp_ratio(dist: str, n: int, delta: float) -> float:
    """Q(1 − δ) von Median-of-Means mit der Regel-Blockzahl zu Q(1 − δ) des Mittels (Verhältnis der Seed-Mittel). Unter 1: Median-of-Means ist bei diesem Niveau besser."""
    return R.q(dist, n, mom_spec(delta, n), delta) / R.q(dist, n, "mean", delta)


def better_in_all_seeds(dist: str, n: int, delta: float) -> bool:
    """Ist Median-of-Means (Regel) bei diesem Niveau in jedem Seed besser als das Mittel?"""
    return all(r < 1.0 for r in ratio_seeds(dist, n, mom_spec(delta, n), "mean", delta))


def tipping_level(better: dict) -> float | None:
    """Kippniveau: das größte δ, ab dem Median-of-Means bei diesem und allen kleineren Niveaus besser ist. `better`: δ → bool (in allen Seeds besser). None, wenn schon beim kleinsten δ nicht."""
    best = None
    for delta in sorted(better):              # von klein nach groß: solange besser, wächst das Kippniveau
        if not better[delta]:
            break
        best = delta
    return best


def tipping_level_of(dist: str, n: int, deltas=G.DELTA_GRID) -> float | None:
    return tipping_level({d: better_in_all_seeds(dist, n, d) for d in deltas})


def count_better_cells(deltas=COUNT_DELTAS) -> tuple[int, int]:
    """(Zellen, in denen Median-of-Means das Mittel in allen Seeds schlägt, Zellen insgesamt) über alle Verteilungen, n und die Niveaus `deltas`."""
    cells = [(d, n, dl) for d in S.DISTRIBUTIONS for n in G.N_RASTER for dl in deltas]
    return sum(better_in_all_seeds(*c) for c in cells), len(cells)


def reliable(delta: float, runs_total: int, min_exceedances: int = G.MIN_EXCEEDANCES) -> bool:
    """Ein Fehlerquantil Q(1 − δ) ist nur belastbar, wenn ihn genug Läufe überschreiten: δ · Läufe ≥ min_exceedances."""
    return delta * runs_total >= min_exceedances - 1e-9


# --- Prämie und Auszahlung, Verzerrung, Blockzahl --------------------------------------------------------------------------------------------

def premium(dist: str, n: int, delta: float) -> float:
    """Prämie: typischer Fehler (Median des Fehlers) von Median-of-Means (Regel zu δ) geteilt durch den des Mittels. Über 1: im Normalfall schlechter."""
    return R.typical(dist, n, mom_spec(delta, n)) / R.typical(dist, n, "mean")


def best_k(dist: str, n: int, delta: float, target: str = "Mittelwert") -> tuple[int, float]:
    """(K, Q(1 − δ)) des im Nachhinein besten Median-of-Means aus den gemessenen Blockzahlen (K = 1 ist das Mittel)."""
    options = [(R.q(dist, n, X.mom_spec(K), delta, target), K) for K in X.k_values(n)]
    qv, K = min(options)
    return K, qv


def rule_loss(dist: str, n: int, delta: float) -> float:
    """Wie viel schlechter ist die Regel-Blockzahl als die im Nachhinein beste? (Q Regel / Q beste − 1)"""
    return R.q(dist, n, mom_spec(delta, n), delta) / best_k(dist, n, delta)[1] - 1.0


def k_curve(dist: str, n: int, delta: float, target: str = "Mittelwert") -> list[dict]:
    """Je gemessener Blockzahl: K, Q(1 − δ), typischer Fehler und Verzerrung."""
    return [{"K": K, "q": R.q(dist, n, X.mom_spec(K), delta, target), "typ": R.typical(dist, n, X.mom_spec(K), target), "bias": R.bias(dist, n, X.mom_spec(K), target)}
            for K in X.k_values(n)]


# --- Dreier-Wahl: Mittel, Median-of-Means, Trimmen ------------------------------------------------------------------------------------------

def three_way(dist: str, n: int, delta: float, c: int = 1) -> dict:
    """Q(1 − δ) von Mittel, Median-of-Means (Regel) und Trimmen mit k = ⌈c ln(1/δ)⌉ je Seite, dazu der Sieger (kleinstes Q; bei Gleichstand in dieser Reihenfolge)."""
    qs = {"Mittel": R.q(dist, n, "mean", delta), "MoM": R.q(dist, n, mom_spec(delta, n), delta), "Trimmen": R.q(dist, n, trim_spec(c, delta, n), delta)}
    winner = min(qs, key=lambda k: (qs[k], list(qs).index(k)))
    return {**qs, "Sieger": winner}


def winner_counts(c: int = 1, deltas=COUNT_DELTAS, dists=S.DISTRIBUTIONS) -> dict:
    """Zählt die Sieger der Dreier-Wahl über alle Zellen (Verteilung × n × Niveau)."""
    counts = {"Mittel": 0, "MoM": 0, "Trimmen": 0}
    for d in dists:
        for n in G.N_RASTER:
            for dl in deltas:
                counts[three_way(d, n, dl, c)["Sieger"]] += 1
    return counts


def mom_beats_trimming_in_all_seeds(dist: str, n: int, delta: float) -> bool:
    """Ist Median-of-Means (Regel) in jedem Seed besser als das beste Trimmen aus c ∈ {1, 2, 4}?"""
    mom = R.q_seeds(dist, n, mom_spec(delta, n), delta)
    best_trim = [min(vals) for vals in zip(*[R.q_seeds(dist, n, trim_spec(c, delta, n), delta) for c in E.C_OPTIONS])]
    return all(a < b for a, b in zip(mom, best_trim))


def tuned_winner(dist: str, n: int, delta: float) -> str:
    """Beide im Nachhinein abgestimmt: das beste K (nur K ≥ 2) gegen das beste c ∈ {1, 2, 4} beim Trimmen und gegen das Mittel."""
    q_mom = min(R.q(dist, n, X.mom_spec(K), delta) for K in X.k_values(n) if K >= 2)
    q_trim = min(R.q(dist, n, trim_spec(c, delta, n), delta) for c in E.C_OPTIONS)
    q_mean = R.q(dist, n, "mean", delta)
    qs = {"Mittel": q_mean, "MoM": q_mom, "Trimmen": q_trim}
    return min(qs, key=lambda k: (qs[k], list(qs).index(k)))


# --- Intervalle -----------------------------------------------------------------------------------------------------------------------------

METHOD_LABELS = {"student": "Student-Intervall des Mittels", "vorzeichen": "Vorzeichen-Intervall auf den Blockmitteln",
                 "mom_sigma:8": "Median-of-Means ± s·√(8 ln(1/δ)/n)", "mom_sigma:32": "Median-of-Means ± s·√(32 ln(1/δ)/n)"}


def interval_table(dist: str, delta: float) -> list[dict]:
    """Je Verfahren und n: Überdeckung des wahren Mittelwerts und Breite relativ zum Student-Intervall."""
    out = []
    for method in X.interval_methods():
        for n in G.INTERVAL_NS:
            cov, w = R.coverage(dist, n, method, delta), R.width(dist, n, method, delta)
            w_student = R.width(dist, n, "student", delta)
            out.append({"method": method, "n": n, "coverage": cov, "width": w, "width_vs_student": None if w is None else w / w_student})
    return out


def min_coverage(dist_list, method: str, delta: float) -> tuple[float, str, int]:
    """Kleinste Überdeckung eines Verfahrens über Verteilungen und n: (Wert, Verteilung, n)."""
    best = None
    for d in dist_list:
        for n in G.INTERVAL_NS:
            c = R.coverage(d, n, method, delta)
            if c is not None and (best is None or c < best[0]):
                best = (c, d, n)
    return best


# --- Nachmessen in der App -----------------------------------------------------------------------------------------------------------------

def live_runs_allowed(n: int, seeds=C.LIVE_SEEDS) -> int:
    """Wie viele Läufe je Seed das Budget bei diesem n erlaubt (Läufe · n · Seeds ≤ LIVE_BUDGET), höchstens RUNS_MAX."""
    return int(max(1, min(C.RUNS_MAX, C.LIVE_BUDGET // (n * len(seeds)))))


def live_measure(settings: Settings, runs: int) -> dict:
    """Misst die vier Schätzer der Hauptansicht (Mittel, Median, Trimmen, Median-of-Means) mit den Einstellungen nach: `runs` Stichproben je Seed in C.LIVE_SEEDS, gleiche Stichproben für alle.
    Rückgabe {spec → {typ, q, bias}} (Mittel über die Seeds), dazu die Läufe insgesamt."""
    runs = min(int(runs), live_runs_allowed(settings.n))
    K = k_from_option(settings.kopt, settings.delta, settings.n)
    k = E.trim_count(settings.c, settings.delta, settings.n)
    specs = ["mean", "median", X.trim_spec(k), X.mom_spec(K)]
    cell = X.measure_cell(settings.dist, settings.n, runs, C.LIVE_SEEDS, specs)
    j_q = min(range(len(G.Q_LEVELS)), key=lambda i: abs(G.Q_LEVELS[i] - (1.0 - settings.delta)))
    rows = {r["spec"]: r for r in cell["rows"]}
    names = dict(zip(specs, PARTS_OF_MOM))
    out = {names[s]: {"spec": s, "typ": float(np.mean([q[0] for q in rows[s]["q"]])), "q": float(np.mean([q[j_q] for q in rows[s]["q"]])),
                      "bias": float(np.mean(rows[s]["bias"]))} for s in specs}
    return {"rows": out, "runs": runs * len(C.LIVE_SEEDS), "K": K, "k": k, "level": G.Q_LEVELS[j_q]}


# --- Verdikte (reine Funktionen: (Art, Text); die App ruft st.<Art>(Text)) ------------------------------------------------------------------

def _tip_text(level: float | None) -> str:
    return f"ab δ ≤ {de_delta(level)}" if level is not None else f"bis δ = {de_delta(min(G.DELTA_GRID))} in keinem Niveau durchgehend"


def verdict_main(a: Analysis) -> tuple[str, str]:
    """Einzelne Stichprobe: Fehler der vier Schätzer. Eine Stichprobe zeigt nichts über das Verfahren, das sagt der Text dazu."""
    s = a.settings
    rel = "" if S.DIST_INFO[s.dist].symmetric else " (relativ zum Mittelwert)"
    e = a.errors
    best = min(e, key=lambda k: abs(e[k]))
    return "info", (f"In dieser einen Stichprobe (n = {s.n}, Zufalls-Seed {s.seed}) liegt der Fehler{rel} beim Mittel bei {de(e['Mittel'])}, beim Median bei {de(e['Median'])}, "
                    f"beim Trimmen ({a.k} Werte je Seite) bei {de(e['Trimmen'])} und bei Median-of-Means ({a.K} Blöcke) bei {de(e['MoM'])}; am nächsten am Mittelwert liegt hier **{best}**. "
                    "Eine einzelne Stichprobe ist ein Zufallsfall: wie die Schätzer im Mittel und im seltenen Ernstfall abschneiden, zeigen die Experimente unten.")


def verdict_kipp(dist: str, delta: float) -> tuple[str, str]:
    """E1: für eine Verteilung das Verhältnis Median-of-Means / Mittel beim gewählten Niveau über die vier n und die Kippniveaus."""
    ratios = {n: kipp_ratio(dist, n, delta) for n in G.N_RASTER}
    tips = {n: tipping_level_of(dist, n) for n in G.N_RASTER}
    ratio_text = ", ".join(f"n = {n}: {de(r, 2)}" for n, r in ratios.items())
    tip_text = "; ".join(f"n = {n}: {_tip_text(t)}" for n, t in tips.items())
    head = f"{dist}, Niveau δ = {de_delta(delta)}: Q(1 − δ) von Median-of-Means geteilt durch das des Mittels ist {ratio_text} (unter 1: Median-of-Means besser). "
    tail = f"Kippniveau (Median-of-Means ist bei diesem und allen kleineren Niveaus in allen vier Seeds besser): {tip_text}."
    if all(r < 1 for r in ratios.values()):
        return "success", head + tail
    if all(r >= 1 for r in ratios.values()):
        return "warning", head + "Bei diesem Niveau ist das Mittel in jedem n mindestens so genau. " + tail
    return "info", head + tail


def verdict_k(dist: str, n: int, delta: float) -> tuple[str, str]:
    """E2: Regel-Blockzahl gegen im Nachhinein beste Blockzahl, mit der Verzerrung beider."""
    K = E.k_of_delta(delta, n)
    best, q_best = best_k(dist, n, delta)
    q_rule = R.q(dist, n, X.mom_spec(K), delta)
    if best == K:
        return "info", f"{dist}, n = {n}, δ = {de_delta(delta)}: Die Regel-Blockzahl K = {K} ist hier auch die beste der gemessenen Blockzahlen (Q(1 − δ) = {de(q_rule)})."
    loss = q_rule / q_best - 1.0
    if S.DIST_INFO[dist].symmetric:
        bias_text = "Bei symmetrischen Verteilungen ist die Verzerrung 0."
    else:
        bias_text = f"Verzerrung der Regel: {pct(R.bias(dist, n, X.mom_spec(K)))}, des besten K: {pct(R.bias(dist, n, X.mom_spec(best)))}."
    best_text = "K = 1, das ist das Mittel" if best == 1 else f"K = {best}"
    return ("warning" if loss > 0.1 else "info"), (
        f"{dist}, n = {n}, δ = {de_delta(delta)}: Die Regel liefert K = {K} mit Q(1 − δ) = {de(q_rule)}; das im Nachhinein beste gemessene K ist {best_text} mit {de(q_best)}. "
        f"Die Regel ist damit **{pct(loss, 0)} schlechter** als das beste K des Rasters. {bias_text} "
        "Das beste K ist hier im Nachhinein gewählt; eine Regel aus den Daten gibt es in dieser Demo nicht.")


def verdict_premium(dist: str, n: int, delta: float) -> tuple[str, str]:
    """E3: typischer Fehler (Prämie) gegen Fehler im seltenen Ernstfall (Auszahlung) von Median-of-Means gegenüber dem Mittel."""
    prem = premium(dist, n, delta)
    pay = kipp_ratio(dist, n, delta)
    typ_mean = R.typical(dist, n, "mean")
    typ_mom = R.typical(dist, n, mom_spec(delta, n))
    head = (f"{dist}, n = {n}, δ = {de_delta(delta)}: Im Normalfall (typischer Fehler) ist der Fehler von Median-of-Means **{de(prem, 2)}-mal** so groß wie beim Mittel ({de(typ_mom)} gegen {de(typ_mean)}). "
            f"Im seltenen Ernstfall (Q(1 − δ)) ist das Verhältnis {de(pay, 2)}")
    if pay < 1 <= prem:
        return "info", head + ": Die Absicherung kostet im Alltag und zahlt im Ernstfall."
    if pay >= 1 and prem >= 1:
        return "warning", head + ": Hier kostet die Absicherung, ohne im Ernstfall zu zahlen."
    return "info", head + "."


def verdict_three_way(dist: str, n: int, delta: float, c: int) -> tuple[str, str]:
    """E4: Dreier-Wahl Mittel, Median-of-Means, Trimmen mit k = ⌈c ln(1/δ)⌉, dazu die Zählung über alle Zellen."""
    t = three_way(dist, n, delta, c)
    counts = winner_counts(c)
    total = sum(counts.values())
    k = E.trim_count(c, delta, n)
    return "info", (f"{dist}, n = {n}, δ = {de_delta(delta)}: Q(1 − δ) beim Mittel {de(t['Mittel'])}, bei Median-of-Means (K = {E.k_of_delta(delta, n)}) {de(t['MoM'])}, "
                    f"beim Trimmen ({k} Werte je Seite, c = {c}) {de(t['Trimmen'])}; kleinster Wert: **{t['Sieger']}**. "
                    f"Über alle {total} Zellen (10 Verteilungen · 4 n · 4 Niveaus) gewinnt bei diesem c das Mittel {counts['Mittel']}, Median-of-Means {counts['MoM']} und das Trimmen {counts['Trimmen']} Zellen. "
                    "Der Faktor c ist eine eigene Wahl dieser Demo, nicht aus einer Quelle.")


def verdict_target(dist: str, n: int, delta: float) -> tuple[str, str]:
    """E5: typischer Fehler gegen den Mittelwert und gegen den Median als Ziel (nur bei schiefen Verteilungen verschieden)."""
    if S.DIST_INFO[dist].symmetric:
        return "info", f"{dist} ist symmetrisch um 0: Mittelwert und Median sind dasselbe Ziel, die Wahl der Zielgröße ändert hier nichts. Für den Kontrast eine schiefe Verteilung wählen."
    t = {k: (R.typical(dist, n, s, "Mittelwert"), R.typical(dist, n, s, "Median")) for k, s in (("Mittel", "mean"), ("Median", "median"), ("MoM", mom_spec(delta, n)))}
    ratio = t["MoM"][1] / t["Median"][1]
    return "info", (f"{dist}, n = {n}: Typischer Fehler gegen das Ziel Mittelwert: Mittel {de(t['Mittel'][0])}, Median {de(t['Median'][0])}, Median-of-Means {de(t['MoM'][0])}. "
                    f"Gegen das Ziel Median: Mittel {de(t['Mittel'][1])}, Median {de(t['Median'][1])}, Median-of-Means {de(t['MoM'][1])}, also **{de(ratio, 1)}-mal** so groß wie beim Median-Schätzer. "
                    "Welcher Schätzer besser ist, hängt davon ab, welche Größe man schätzen will; wer die Gesamtmenge braucht, will den Mittelwert.")


def width_range_text(lo: float, hi: float) -> str:
    """„1,5-mal Student“ bzw. „2,4- bis 2,5-mal Student“."""
    return f"{de(lo, 1)}-mal Student" if de(lo, 1) == de(hi, 1) else f"{de(lo, 1)}- bis {de(hi, 1)}-mal Student"


def interval_width_ratio(dist: str, n: int, method: str, delta: float) -> float:
    return R.width(dist, n, method, delta) / R.width(dist, n, "student", delta)


def verdict_intervals(dist: str, delta: float) -> tuple[str, str]:
    """E6: Überdeckung und Breite der Intervalle für den Mittelwert; brauchbar heißt: Soll-Überdeckung (Toleranz 1 Prozentpunkt) in jedem n und höchstens 1,5-mal so breit wie das Student-Intervall."""
    target = 1.0 - delta
    lines, usable = [], []
    for method in X.interval_methods():
        covs = [R.coverage(dist, n, method, delta) for n in G.INTERVAL_NS]
        if any(c is None for c in covs):
            lines.append(f"{METHOD_LABELS[method]}: für kleine n nicht definiert")
            continue
        widths = [interval_width_ratio(dist, n, method, delta) for n in G.INTERVAL_NS]
        if min(covs) >= target - 0.01 and max(widths) <= 1.5 and method != "student":
            usable.append(method)
        lines.append(f"{METHOD_LABELS[method]}: kleinste Überdeckung {pct(min(covs))}, Breite {width_range_text(min(widths), max(widths))}")
    head = f"{dist}, Soll {pct(target, 0)}. " + "; ".join(lines) + ". "
    if usable:
        names = " und ".join(METHOD_LABELS[m] for m in usable)
        return "success", head + f"Das Niveau halten und höchstens 1,5-mal so breit wie Student sind: {names}."
    return "warning", head + "Keine der gemessenen Median-of-Means-Konstruktionen hält das Niveau in jedem n und ist zugleich höchstens 1,5-mal so breit wie das Student-Intervall (offen, nicht widerlegt: andere Konstruktionen)."


def verdict_live(settings: Settings, live: dict) -> tuple[str, str]:
    """Nachmessen mit den Einstellungen der Seitenleiste: Q(1 − δ) und Verzerrung von Mittel und Median-of-Means."""
    rows = live["rows"]
    q_mean, q_mom = rows["Mittel"]["q"], rows["MoM"]["q"]
    ratio = q_mom / q_mean
    rel = "" if S.DIST_INFO[settings.dist].symmetric else " (relativ zum Mittelwert)"
    head = (f"{live['runs']} Läufe (4 Seeds), {settings.dist}, n = {settings.n}, δ = {de_delta(settings.delta)}: Q(1 − δ) beim Mittel {de(q_mean)}, bei Median-of-Means (K = {live['K']}) {de(q_mom)}"
            f"{rel}, Verhältnis {de(ratio, 2)}. Verzerrung von Median-of-Means: {pct(rows['MoM']['bias'])}.")
    if not reliable(settings.delta, live["runs"]):
        return "info", head + f" **Wenig belastbar:** bei δ = {de_delta(settings.delta)} überschreiten nur etwa {settings.delta * live['runs']:.0f} Läufe das Quantil (nötig sind {G.MIN_EXCEEDANCES}); mehr Läufe oder die vorgerechnete Messreihe nutzen."
    return ("success" if ratio < 1 else "info"), head


def tuned_counts(dists, deltas=COUNT_DELTAS) -> dict:
    """Zählt die Sieger der nachträglich abgestimmten Wahl (bestes K ≥ 2, bestes c, Mittel) über die Zellen der Verteilungen `dists`."""
    counts = {"Mittel": 0, "MoM": 0, "Trimmen": 0}
    for d in dists:
        for n in G.N_RASTER:
            for dl in deltas:
                counts[tuned_winner(d, n, dl)] += 1
    return counts


def verdict_count() -> tuple[str, str]:
    """E1 über alle Zellen: in wie vielen schlägt Median-of-Means (Regel) das Mittel in jedem Seed?"""
    better, total = count_better_cells()
    sym_better = sum(better_in_all_seeds(d, n, dl) for d in S.SYMMETRIC for n in G.N_RASTER for dl in COUNT_DELTAS)
    sym_total = len(S.SYMMETRIC) * len(G.N_RASTER) * len(COUNT_DELTAS)
    return "info", (f"In **{better} von {total}** Zellen (10 Verteilungen · 4 n · Niveaus {', '.join(de_delta(d) for d in COUNT_DELTAS)}) ist Median-of-Means mit der Regel-Blockzahl in allen vier Seeds besser als das Mittel; "
                    f"davon {sym_better} von {sym_total} bei symmetrischen und {better - sym_better} von {total - sym_total} bei schiefen Verteilungen. Bei Normal und t(5) in keiner.")


def verdict_tuned() -> tuple[str, str]:
    """E4: nachträglich abgestimmte Wahl (bestes K, bestes c) gegen das Mittel, getrennt nach symmetrischen und schiefen Verteilungen."""
    sym, skew = tuned_counts(S.SYMMETRIC), tuned_counts(S.SKEWED)
    return "warning", (f"Beide Parameter im Nachhinein am selben Datensatz gewählt (bestes K ≥ 2, bestes c ∈ {{1, 2, 4}}; keine Wahl auf einer Hälfte der Daten, der Vergleich ist also zugunsten beider Verfahren verzerrt): "
                       f"bei den {sum(sym.values())} symmetrischen Zellen gewinnt das Trimmen {sym['Trimmen']}, das Mittel {sym['Mittel']} und Median-of-Means {sym['MoM']}; "
                       f"bei den {sum(skew.values())} schiefen Zellen Median-of-Means {skew['MoM']}, das Mittel {skew['Mittel']} und das Trimmen {skew['Trimmen']}. "
                       "Bei schiefen Rändern ist das Trimmen aber zu einer anderen Größe verzerrt (Tabelle der Verzerrung unter E2).")


def bias_table(dist: str, n: int, delta: float, c: int) -> list[dict]:
    """Verzerrung (mittlerer Fehler gegen den Mittelwert; relativ bei schiefen Verteilungen) und typischer Fehler der Schätzer, gemessen für das n der Messreihe."""
    entries = [("Mittel", "mean"), ("Median", "median"), (f"Trimmen 10 % ({X.trim10_k(n)} je Seite)", X.trim_spec(X.trim10_k(n))),
               (f"Trimmen c = {c} ({E.trim_count(c, delta, n)} je Seite)", trim_spec(c, delta, n)),
               (f"Median-of-Means, Regel (K = {E.k_of_delta(delta, n)})", mom_spec(delta, n)), (f"Median-of-Means, K = n/2 ({max(n // 2, 1)})", X.mom_spec(max(n // 2, 1)))]
    return [{"Schätzer": name, "Verzerrung": R.bias(dist, n, spec), "typischer Fehler": R.typical(dist, n, spec)} for name, spec in entries]
