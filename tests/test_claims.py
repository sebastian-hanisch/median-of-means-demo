"""Jede Zahl der README wird hier aus der vorgerechneten Messreihe (data/mom_results.json) gebildet und in der README gesucht (DEMO-PLAYBOOK §4); dasselbe für die Hilfetexte der Presets.

Die Datei selbst ist durch tests/test_results.py gegen die Quelldateien abgesichert; dort werden Zellen auch live nachgerechnet."""

from pathlib import Path

import numpy as np
import pytest

import mom_constants as C
import mom_estimators as E
import mom_evaluation as V
import mom_experiments as X
import mom_grid as G
import mom_results as R
import mom_scenario as S

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
L = "Lognormal (σ = 1,5)"
de, pct = V.de, V.pct


def mn(text):
    """Minuszeichen der README (U+2212) statt des Bindestrichs von Python."""
    return text.replace("-", "−")


def has(text):
    assert text in README, text


def q(dist, n, spec, delta):
    return R.q(dist, n, spec, delta)


def mom(delta, n=200):
    return V.mom_spec(delta, n)


def test_h1_the_tipping_map():
    parts = []
    for d in (0.1, 0.05, 0.01, 0.001):
        parts.append(f"{de(q(L, 200, 'mean', d))} / {de(q(L, 200, mom(d), d))}")
    has(f"{parts[0]} (δ = 0,1), {parts[1]} (0,05), {parts[2]} (0,01), {parts[3]} (0,001)")
    assert q(L, 200, "mean", 0.1) < q(L, 200, mom(0.1), 0.1) and q(L, 200, "mean", 0.001) > 2 * q(L, 200, mom(0.001), 0.001)
    assert V.tipping_level_of(L, 200) == 0.02 and V.tipping_level_of("t(3)", 200) == 0.01
    has(f"Lognormal (σ = 1,5) {V.de_delta(V.tipping_level_of(L, 200))}, t(3) {V.de_delta(V.tipping_level_of('t(3)', 200))} bei n = 200")
    assert all(V.tipping_level_of(d, n) is None and not any(V.better_in_all_seeds(d, n, dl) for dl in G.DELTA_GRID) for d in ("Normal", "t(5)") for n in G.N_RASTER)
    better, total = V.count_better_cells()
    sym = sum(V.better_in_all_seeds(d, n, dl) for d in S.SYMMETRIC for n in G.N_RASTER for dl in V.COUNT_DELTAS)
    has(f"In {better} von {total} Zellen (10 Verteilungen · 4 n · δ = 0,1 / 0,05 / 0,01 / 0,001) ist Median-of-Means in allen Seeds besser, davon {sym} von 64 bei symmetrischen Verteilungen")

    sp = "Symmetrisch mit Pareto-Rand (α = 2,5)"
    assert V.tipping_level_of(sp, 50) == 0.5 and V.tipping_level_of(sp, 200) == 0.5
    has(f"Kippniveau 0,5; Verhältnis zum Mittel bei n = 50: {de(V.kipp_ratio(sp, 50, 0.5), 2)} bei δ = 0,5, {de(V.kipp_ratio(sp, 50, 0.001), 2)} bei δ = 0,001")
    assert all(V.kipp_ratio(sp, n, dl) < 1 for n in (50, 200) for dl in G.DELTA_GRID)
    assert max(V.tipping_level_of(d, n) or 0 for d in S.SKEWED for n in G.N_RASTER) == 0.05
    has("(Kippniveau, wo es eines gibt, höchstens 0,05)")


def test_h2_tipping_level_falls_with_n():
    levels = [V.tipping_level_of(L, n) for n in G.N_RASTER]
    assert levels[0] > levels[1] > levels[2] and levels[3] is None
    has(f"bei n = 50 / 200 / 1000: {' / '.join(V.de_delta(x) for x in levels[:3])}; bei n = 5000 ist Median-of-Means bis δ = 0,001")
    assert not V.better_in_all_seeds(L, 5000, 0.001)


def test_h3_price_is_typical_error_and_bias():
    has(f"typischer Fehler von Median-of-Means das {de(V.premium(L, 200, 0.01), 2)}-fache des Mittels bei Lognormal (σ = 1,5) "
        f"({de(V.premium('Lognormal (σ = 2)', 200, 0.01), 2)} bei σ = 2, {de(V.premium('Pareto (α = 2,1)', 200, 0.01), 2)} bei Pareto α = 2,1, {de(V.premium('Normal', 200, 0.01), 2)} bei Normal; √(π/2) = {de(np.sqrt(np.pi / 2), 4)})")
    assert abs(V.premium("Normal", 200, 0.01) - np.sqrt(np.pi / 2)) < 0.03
    sp = "Symmetrisch mit Pareto-Rand (α = 2,5)"
    has(f"bei symmetrisch mit Pareto-Rand (α = 2,5) {de(V.premium(sp, 50, 0.01), 2)} bei n = 50 und {de(V.premium(sp, 200, 0.01), 2)} bei n = 200, bei t(3) {de(V.premium('t(3)', 50, 0.01), 2)} bei n = 50")
    assert V.premium(sp, 50, 0.01) < 1 and V.premium(sp, 200, 0.01) < 1 and V.premium("t(3)", 50, 0.01) < 1
    b200, b1000 = R.bias(L, 200, mom(0.01)), R.bias(L, 1000, mom(0.01, 1000))
    has(f"Verzerrung bei Lognormal (σ = 1,5): {mn(pct(b200))} (K = 37), {mn(pct(b1000))} bei n = 1000")
    ks = (4, 8, 16, 32, 64)
    has(f"K = 4 / 8 / 16 / 32 / 64: {mn(' / '.join(pct(R.bias(L, 200, X.mom_spec(K)))[:-2] for K in ks))} %")
    biases = [R.bias(L, 200, X.mom_spec(K)) for K in (1,) + ks]
    assert all(a > b for a, b in zip(biases, biases[1:])) and abs(biases[0]) < 0.02
    has(f"Median {mn(pct(R.bias(L, 200, 'median')))}, Trimmen 10 % {mn(pct(R.bias(L, 200, X.trim_spec(X.trim10_k(200)))))}")
    pay = {d: V.kipp_ratio(d, 200, 0.001) for d in S.DISTRIBUTIONS}
    has(f"Lognormal (σ = 2) {de(pay['Lognormal (σ = 2)'], 2)}, Pareto (α = 2,1) {de(pay['Pareto (α = 2,1)'], 2)}, Lognormal (σ = 1,5) {de(pay[L], 2)}, Pareto (α = 3) {de(pay['Pareto (α = 3)'], 2)}, t(3) {de(pay['t(3)'], 2)}; "
        f"bei Lognormal (σ = 1) und Normal keine ({de(pay['Lognormal (σ = 1)'], 2)} und {de(pay['Normal'], 2)})")
    assert pay["Lognormal (σ = 2)"] < pay["Pareto (α = 2,1)"] < pay[L] < pay["Pareto (α = 3)"] < pay["t(3)"] < 1 < pay["Lognormal (σ = 1)"] < pay["Normal"]


def test_h4_the_rule_is_too_large_for_skewed_tails():
    best, qb = V.best_k(L, 200, 0.01)
    has(f"Regel K = {E.k_of_delta(0.01, 200)} gibt {de(q(L, 200, mom(0.01), 0.01))}, das beste gemessene K = {best} gibt {de(qb)}, der Fehlerquantil-Wert der Regel liegt {pct(V.rule_loss(L, 200, 0.01), 0)} über dem des besten K")
    losses = [V.rule_loss(d, n, dl) for d in S.SKEWED for n in G.N_RASTER for dl in V.COUNT_DELTAS if E.k_of_delta(dl, n) < n // 2]
    has(f"wo der Deckel n/2 nicht greift ({len(losses)} Zellen), verliert die Regel {pct(min(losses), 0).replace(' %', '')} bis {pct(max(losses), 0)} (Median {pct(float(np.median(losses)), 0)})")
    has(f"das beste K ist bei Lognormal (σ = 1,5) {' / '.join(str(V.best_k(L, 200, d)[0]) for d in (0.1, 0.01, 0.001))} für δ = 0,1 / 0,01 / 0,001, bei t(3) {V.best_k('t(3)', 200, 0.01)[0]} (= n/2)")
    assert all(V.E.k_of_delta(d, 200) > V.best_k(L, 200, d)[0] for d in (0.1, 0.01, 0.001))


def test_h5_trimming_with_few_values():
    counts = {c: V.winner_counts(c) for c in (1, 2, 4)}
    has(f"mit c = 1 gewinnt das Trimmen {counts[1]['Trimmen']}, das Mittel {counts[1]['Mittel']}, Median-of-Means {counts[1]['MoM']} Zelle; "
        f"mit c = 2: {counts[2]['Trimmen']} / {counts[2]['Mittel']} / {counts[2]['MoM']}; mit c = 4: {counts[4]['Trimmen']} / {counts[4]['Mittel']} / {counts[4]['MoM']}")
    assert counts[1]["Trimmen"] > counts[1]["Mittel"] > counts[1]["MoM"] and counts[4]["MoM"] > counts[2]["MoM"] > counts[1]["MoM"]
    beats = sum(V.mom_beats_trimming_in_all_seeds(d, n, dl) for d in S.DISTRIBUTIONS for n in G.N_RASTER for dl in V.COUNT_DELTAS)
    has(f"in allen Seeds in {beats} von 160 Zellen")
    sym, skew = V.tuned_counts(S.SYMMETRIC), V.tuned_counts(S.SKEWED)
    has(f"bei schiefen Rändern gewinnt Median-of-Means {skew['MoM']} von 96 Zellen, das Mittel {skew['Mittel']}, das Trimmen {skew['Trimmen']}")
    has(f"bei symmetrischen das Trimmen {sym['Trimmen']} von 64, das Mittel {sym['Mittel']}, Median-of-Means {sym['MoM']}")
    has(f"Trimmen 10 % bei Lognormal (σ = 1,5), n = 200: {mn(pct(R.bias(L, 200, X.trim_spec(X.trim10_k(200)))))}")
    assert skew["Trimmen"] == 0 and skew["MoM"] > skew["Mittel"]


def test_h6_symmetric_tails():
    sym = V.winner_counts(1, dists=S.SYMMETRIC)
    has(f"bei den 64 symmetrischen Zellen das Trimmen {sym['Trimmen']}, das Mittel {sym['Mittel']} (alle bei Normal), Median-of-Means {sym['MoM']}")
    assert V.winner_counts(1, dists=["Normal"])["Mittel"] == sym["Mittel"]
    has(f"Mittel {de(q('t(3)', 200, 'mean', 0.05))}, Median-of-Means {de(q('t(3)', 200, mom(0.05), 0.05))}, Trimmen mit c = 1 {de(q('t(3)', 200, V.trim_spec(1, 0.05, 200), 0.05))}")
    normal_winners = {V.three_way("Normal", n, d, 1)["Sieger"] for n in G.N_RASTER for d in V.COUNT_DELTAS}
    assert "MoM" not in normal_winners and "Mittel" in normal_winners


def test_h7_target_median():
    ratios = [R.typical(d, n, mom(0.05, n), "Median") / R.typical(d, n, "median", "Median") for d in S.SKEWED for n in G.N_RASTER]
    has(f"{de(min(ratios), 1)}- bis {de(max(ratios), 1)}-mal so groß wie der des Median-Schätzers ({len(ratios)} Zellen)")
    has(f"{de(R.typical(L, 200, mom(0.05), 'Median'), 3)} gegen {de(R.typical(L, 200, 'median', 'Median'), 3)}")


def test_h8_intervals_negative():
    has(f"Student bei Lognormal (σ = 1,5) {' / '.join(pct(R.coverage(L, n, 'student', 0.05))[:-2] for n in G.INTERVAL_NS)} %")
    has(f"Vorzeichen-Intervall auf den Blockmitteln {' / '.join(pct(R.coverage(L, n, 'vorzeichen', 0.05))[:-2] for n in G.INTERVAL_NS)} %")
    c32 = min(R.coverage(L, n, "mom_sigma:32", 0.05) for n in G.INTERVAL_NS)
    has(f"mindestens {pct(c32)}")
    w32 = [V.interval_width_ratio("Normal", n, "mom_sigma:32", 0.05) for n in G.INTERVAL_NS]
    has(f"{de(min(w32), 1)}- bis {de(max(w32), 1)}-mal so breit wie Student bei Normal")
    sym_cov = min(R.coverage(d, n, "vorzeichen", 0.05) for d in S.SYMMETRIC for n in G.INTERVAL_NS)
    sym_w = [V.interval_width_ratio(d, n, "vorzeichen", 0.05) for d in S.SYMMETRIC for n in G.INTERVAL_NS]
    has(f"kleinste Überdeckung {pct(sym_cov)}, {de(min(sym_w), 1)}- bis {de(max(sym_w), 1)}-mal so breit")
    dev = [abs(R.coverage(d, n, "student", dl) - (1 - dl)) for d in S.SYMMETRIC for n in G.INTERVAL_NS for dl in G.INTERVAL_DELTAS]
    has(f"Bei symmetrischen Verteilungen weicht Student in {len(dev)} Zellen (4 Verteilungen · 4 n · 3 Niveaus) um höchstens {de(100 * max(dev), 1)} Prozentpunkte vom Soll ab")
    assert all(R.coverage(d, 50, "vorzeichen", 0.05) < 0.95 - 0.1 for d in S.SKEWED)


def test_h9_constants():
    c32 = [V.min_coverage(S.DISTRIBUTIONS, "mom_sigma:32", dl)[0] for dl in G.INTERVAL_DELTAS]
    c8 = [V.min_coverage(S.DISTRIBUTIONS, "mom_sigma:8", dl)[0] for dl in G.INTERVAL_DELTAS]
    has(f"Mit c = 32 liegt die kleinste Überdeckung über alle Verteilungen und n bei {pct(c32[0])} (Soll 90 %), {pct(c32[1])} (95 %), {pct(c32[2])} (99 %: knapp unter dem Soll, Lognormal σ = 2, n = 50)")
    has(f"mit c = 8 sinkt sie auf {pct(c8[0])[:-2]} / {pct(c8[1])[:-2]} / {pct(c8[2])[:-2]} %")
    assert c32[0] > 0.9 and c32[1] > 0.95 and 0.98 < c32[2] < 0.99 and V.min_coverage(S.DISTRIBUTIONS, 'mom_sigma:32', 0.01)[1:] == ('Lognormal (σ = 2)', 50)
    assert all(a < 1 - dl for a, dl in zip(c8, G.INTERVAL_DELTAS))


def test_corrections():
    winners = [(d, n, dl) for d in S.DISTRIBUTIONS for n in G.N_RASTER for dl in V.COUNT_DELTAS if V.three_way(d, n, dl, 1)["Sieger"] == "MoM"]
    assert winners == [("Symmetrisch mit Pareto-Rand (α = 2,5)", 50, 0.05)]
    has("symmetrisch mit Pareto-Rand, n = 50, δ = 0,05")
    worst = max((R.q_std(d, n, "mean", 0.001) / R.q(d, n, "mean", 0.001), d, n) for d in S.DISTRIBUTIONS for n in G.N_RASTER)
    has(f"bis zu {pct(worst[0])} ({worst[1].replace('Lognormal (σ = 2)', 'Lognormal σ = 2')}, n = {worst[2]})")
    has(f"bei δ = 0,01 bis zu {pct(max(R.q_std(d, n, 'mean', 0.01) / R.q(d, n, 'mean', 0.01) for d in S.DISTRIBUTIONS for n in G.N_RASTER))}")
    assert all(V.reliable(d, R.runs_total("Normal", 50)) for d in G.DELTA_GRID) and not V.reliable(0.0005, R.runs_total("Normal", 50))
    has(f"Trimmen 10 % um {pct(abs(R.bias(L, 200, X.trim_spec(X.trim10_k(200)))))} verzerrt")


def test_main_claims_in_the_befunde_paragraph():
    has(f"sinkt das Fehlerquantil des Mittels bei δ = 0,001 auf {de(q(L, 200, 'mean', 0.001))} gegen {de(q(L, 200, mom(0.001), 0.001))} bei Median-of-Means, bei δ = 0,1 ist es umgekehrt ({de(q(L, 200, 'mean', 0.1))} gegen {de(q(L, 200, mom(0.1), 0.1))})")
    has(f"kostet {mn(pct(R.bias(L, 200, mom(0.01))))} Verzerrung und das {de(V.premium(L, 200, 0.01), 2)}-fache")
    c = V.winner_counts(1)
    has(f"in {c['Trimmen']} von 160 Zellen, Median-of-Means in {c['MoM']}")


def test_normal_ratio_range_in_the_preset_help_and_the_k_numbers():
    ratios = [V.kipp_ratio("Normal", n, d) for n in G.N_RASTER for d in G.DELTA_GRID]
    text = C.PRESET_HELP["Normal: die Absicherung kostet nur"]
    assert f"zwischen {de(min(ratios), 2)}- und {de(max(ratios), 2)}-mal" in text and f"(bei δ = 0,01 und n = 200: {de(V.kipp_ratio('Normal', 200, 0.01), 2)})" in text


def test_preset_help_numbers():
    h = C.PRESET_HELP
    t = h["Schiefe Mengen (Standard)"]
    assert f"(Verhältnis {de(V.kipp_ratio(L, 200, 0.01), 2)})" in t and f"einen {de(V.premium(L, 200, 0.01), 2)}-mal so großen" in t and f"{pct(abs(R.bias(L, 200, mom(0.01))))} unter" in t and "K = 37" in t
    t = h["Selten zahlt sie sich aus"]
    assert f"des Mittels ist {de(q(L, 200, 'mean', 0.001), 3)}, das von Median-of-Means {de(q(L, 200, mom(0.001), 0.001), 3)} (Verhältnis {de(V.kipp_ratio(L, 200, 0.001), 2)})" in t and f"K = {E.k_of_delta(0.001, 200)}" in t
    t = h["Zu viele Blöcke"]
    assert f"auf {pct(R.bias(L, 200, 'mom:100'))} des Mittelwerts".replace("-", "−") in t and f"({de(q(L, 200, 'mom:100', 0.01), 3)} bei δ = 0,01)" in t and f"des Mittels ({de(q(L, 200, 'mean', 0.01), 3)})" in t
    assert q(L, 200, "mom:100", 0.01) > q(L, 200, "mean", 0.01)
    p = "Pareto (α = 2,1)"
    t = h["Standzeiten mit Pareto-Tail"]
    assert f"(Verhältnis {de(V.kipp_ratio(p, 1000, 0.01), 2)})" in t and f"das {de(V.premium(p, 1000, 0.01), 2)}-fache" in t and f"liegt {pct(abs(R.bias(p, 1000, mom(0.01, 1000))))} unter" in t
    t = h["Symmetrisch schwer (t(3))"]
    assert f"(Verhältnis {de(V.kipp_ratio('t(3)', 200, 0.01), 2)})" in t and V.three_way("t(3)", 200, 0.01, 1)["Sieger"] == "Trimmen"
    assert abs(R.bias("t(3)", 200, mom(0.01))) < 0.01


def test_method_numbers_in_the_readme():
    has("Je Zelle (Verteilung, n) und Seed 40 000 Stichproben")
    assert G.RUNS_PER_SEED == 40_000 and len(G.SEEDS) == 4 and G.N_RASTER == (50, 200, 1000, 5000) and G.DELTA_GRID[0] == 0.5 and G.DELTA_GRID[-1] == 0.001
    has("8 000 Stichproben je Seed (n = 5000: 4 000)")
    assert G.INTERVAL_RUNS == {50: 8000, 200: 8000, 1000: 8000, 5000: 4000}
    has("≤ 20 Millionen Werte")
    assert C.LIVE_BUDGET == 20_000_000
    has("weniger als 100 Überschreitungen")
    assert G.MIN_EXCEEDANCES == 100
    has("(20 Teile, höchstens 6 Prozesse gleichzeitig)")
    assert len(X.PARTS) == 20
