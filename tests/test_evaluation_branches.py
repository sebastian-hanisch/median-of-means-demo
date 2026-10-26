"""Grenzfälle der Auswertung an handgesetzten Tabellen (monkeypatch auf mom_results): Schwellen mit Gleichstand, ausgeschlossene Blockzahlen, Niveau-Wahl des Nachmessens.

Die Messreihe trifft solche Grenzen nie genau; ohne diese Datei überleben die Mutanten der Vergleiche (`<` gegen `<=`) und der Ausschlüsse (`K >= 2`) im Mutationstest."""

import pytest

import mom_estimators as E
import mom_evaluation as V
import mom_experiments as X
import mom_grid as G
import mom_results as R

DIST = "Lognormal (σ = 1,5)"


def fake_q(monkeypatch, fn, typ=None, bias=None):
    monkeypatch.setattr(R, "q", lambda dist, n, spec, delta, target="Mittelwert": fn(spec))
    monkeypatch.setattr(R, "q_seeds", lambda dist, n, spec, delta, target="Mittelwert": [fn(spec)] * 4)
    monkeypatch.setattr(R, "typical", lambda dist, n, spec, target="Mittelwert": (typ or (lambda s: 0.5))(spec))
    monkeypatch.setattr(R, "bias", lambda dist, n, spec, target="Mittelwert": (bias or (lambda s: 0.0))(spec))


def test_tuned_winner_excludes_exactly_k_equal_one_and_includes_k_equal_two(monkeypatch):
    trim = V.trim_spec(1, 0.01, 200)
    only_k1_small = {"mean": 1.0, "mom:1": 0.1, trim: 1.5}
    fake_q(monkeypatch, lambda s: only_k1_small.get(s, 2.0))
    assert V.tuned_winner(DIST, 200, 0.01) == "Mittel"                           # K = 1 (das Mittel selbst) zählt nicht als Median-of-Means
    k2_small = {"mean": 1.0, "mom:2": 0.5, trim: 1.5}
    fake_q(monkeypatch, lambda s: k2_small.get(s, 2.0))
    assert V.tuned_winner(DIST, 200, 0.01) == "MoM"                              # K = 2 zählt
    k4_small = {"mean": 1.0, "mom:4": 0.5, trim: 1.5}
    fake_q(monkeypatch, lambda s: k4_small.get(s, 2.0))
    assert V.tuned_winner(DIST, 200, 0.01) == "MoM"
    trim_small = {"mean": 1.0, "mom:4": 0.9, V.trim_spec(2, 0.01, 200): 0.3}
    fake_q(monkeypatch, lambda s: trim_small.get(s, 2.0))
    assert V.tuned_winner(DIST, 200, 0.01) == "Trimmen"


def test_mom_beats_trimming_needs_a_strict_win_in_every_seed(monkeypatch):
    mom = V.mom_spec(0.01, 200)
    trims = {V.trim_spec(c, 0.01, 200) for c in E.C_OPTIONS}
    fake_q(monkeypatch, lambda s: 0.5 if s == mom else (0.6 if s in trims else 1.0))
    assert V.mom_beats_trimming_in_all_seeds(DIST, 200, 0.01)
    fake_q(monkeypatch, lambda s: 0.6 if s in trims else 0.6)                      # Gleichstand: kein Sieg
    assert not V.mom_beats_trimming_in_all_seeds(DIST, 200, 0.01)
    best_trim = V.trim_spec(4, 0.01, 200)
    fake_q(monkeypatch, lambda s: 0.5 if s == mom else (0.4 if s == best_trim else 0.6))     # das BESTE Trimmen zählt, nicht das schlechteste
    assert not V.mom_beats_trimming_in_all_seeds(DIST, 200, 0.01)


def test_min_coverage_returns_the_smallest_and_keeps_the_first_of_a_tie(monkeypatch):
    table = {("Normal", 50): 0.9, ("Normal", 200): 0.8, ("t(3)", 50): 0.8, ("t(3)", 200): 0.95}
    monkeypatch.setattr(R, "coverage", lambda dist, n, method, delta: table.get((dist, n)))
    assert V.min_coverage(["Normal", "t(3)"], "student", 0.05) == (0.8, "Normal", 200)                 # Gleichstand: der erste bleibt
    monkeypatch.setattr(R, "coverage", lambda dist, n, method, delta: None if n == 50 else table.get((dist, n)))
    assert V.min_coverage(["Normal", "t(3)"], "vorzeichen", 0.05)[2] == 200                            # nicht definierte Werte werden übersprungen
    monkeypatch.setattr(R, "coverage", lambda dist, n, method, delta: 0.99 if (dist, n) != ("t(3)", 1000) else 0.5)
    assert V.min_coverage(["Normal", "t(3)"], "student", 0.05) == (0.5, "t(3)", 1000)


def fake_intervals(monkeypatch, coverage, width_ratio):
    monkeypatch.setattr(R, "coverage", lambda dist, n, method, delta: coverage(method, n, delta))
    monkeypatch.setattr(R, "width", lambda dist, n, method, delta: 1.0 if method == "student" else width_ratio(n))


def test_interval_verdict_thresholds_are_inclusive_and_use_the_worst_n(monkeypatch):
    delta = 0.05
    target = 1.0 - delta
    # genau an den Schwellen: Überdeckung = Soll − 0,01, Breite = 1,5 → brauchbar (success)
    fake_intervals(monkeypatch, lambda m, n, d: target - 0.01, lambda n: 1.5)
    assert V.verdict_intervals("Normal", delta)[0] == "success"
    # knapp darunter: Überdeckung unter Soll − 0,01 → nicht brauchbar
    fake_intervals(monkeypatch, lambda m, n, d: target - 0.0101, lambda n: 1.5)
    assert V.verdict_intervals("Normal", delta)[0] == "warning"
    # Soll gerade erreicht, Breite knapp über 1,5 → nicht brauchbar
    fake_intervals(monkeypatch, lambda m, n, d: target, lambda n: 1.5001)
    assert V.verdict_intervals("Normal", delta)[0] == "warning"
    # nur ein n zu breit / zu schmal überdeckt: das schlechteste n entscheidet
    fake_intervals(monkeypatch, lambda m, n, d: target if n != 200 else 0.5, lambda n: 1.0)
    assert V.verdict_intervals("Normal", delta)[0] == "warning"
    fake_intervals(monkeypatch, lambda m, n, d: target, lambda n: 1.0 if n != 5000 else 3.0)
    assert V.verdict_intervals("Normal", delta)[0] == "warning"
    fake_intervals(monkeypatch, lambda m, n, d: target + 0.05, lambda n: 1.0)
    assert V.verdict_intervals("Normal", delta)[0] == "success"
    # nur Student hält das Niveau: Student selbst zählt nicht als Median-of-Means-Konstruktion
    monkeypatch.setattr(R, "coverage", lambda dist, n, method, delta: target if method == "student" else 0.5)
    assert V.verdict_intervals("Normal", delta)[0] == "warning"


def test_premium_boundaries_and_combinations(monkeypatch):
    def run(pay, prem):
        fake_q(monkeypatch, lambda s: {"mean": 1.0, "mom:37": pay}.get(s, 1.0), typ=lambda s: {"mean": 0.1, "mom:37": 0.1 * prem}.get(s, 0.5))
        return V.verdict_premium(DIST, 200, 0.01)
    assert run(0.9, 1.0)[0] == "info" and "zahlt im Ernstfall" in run(0.9, 1.0)[1]            # Prämie genau 1 zählt als Kosten
    assert run(0.9, 1.5)[0] == "info" and "zahlt im Ernstfall" in run(0.9, 1.5)[1]
    kind, text = run(1.5, 3.0)
    assert kind == "warning" and "ohne im Ernstfall zu zahlen" in text                         # teurer und im Ernstfall nicht besser
    kind, text = run(1.2, 0.8)
    assert kind == "info" and text.endswith("Verhältnis 1,20.")                                # billiger im Alltag, im Ernstfall nicht besser: nur Zahlen
    kind, text = run(0.9, 0.8)
    assert kind == "info" and text.endswith("Verhältnis 0,90.")
    assert run(1.0, 2.0)[0] == "warning"                                                       # Verhältnis genau 1 gilt als „zahlt nicht“


def test_k_verdict_threshold_between_info_and_warning(monkeypatch):
    rule, best = "mom:37", "mom:8"
    base = {f"mom:{K}": 9.0 for K in X.k_values(200)}
    fake_q(monkeypatch, lambda s: {**base, rule: 1.0625, best: 1.0}.get(s, 1.0))
    assert V.verdict_k(DIST, 200, 0.01)[0] == "info"                                           # 6,25 % unter der Schwelle
    fake_q(monkeypatch, lambda s: {**base, rule: 1.25, best: 1.0}.get(s, 1.0))
    assert V.verdict_k(DIST, 200, 0.01)[0] == "warning"                                        # 25 % darüber


def test_live_measure_reads_the_quantile_at_the_level_of_delta():
    for delta in G.DELTA_GRID:                                                  # alle neun Niveaus, auch das erste und das letzte
        s = V.Settings(dist="Pareto (α = 2,5)", n=50, delta=delta)
        live = V.live_measure(s, 300)
        j = G.Q_LEVELS.index(round(1.0 - delta, 3)) if round(1.0 - delta, 3) in G.Q_LEVELS else None
        assert live["level"] == pytest.approx(1.0 - delta) and j is not None
        cell = X.measure_cell(s.dist, 50, 300, V.C.LIVE_SEEDS, ["mean", f"mom:{live['K']}"])
        expected = sum(q[j] for q in cell["rows"][0]["q"]) / 4
        assert live["rows"]["Mittel"]["q"] == pytest.approx(expected) and live["rows"]["MoM"]["q"] == pytest.approx(sum(q[j] for q in cell["rows"][1]["q"]) / 4)
        assert live["rows"]["Mittel"]["typ"] == pytest.approx(sum(q[0] for q in cell["rows"][0]["q"]) / 4)


def test_live_measure_caps_the_runs_at_the_budget_and_returns_what_was_asked_below_it():
    assert V.live_measure(V.Settings(n=5000), 5000)["runs"] == 4 * 1000
    assert V.live_measure(V.Settings(n=50), 300)["runs"] == 4 * 300
    assert V.live_runs_allowed(1) == 5000 and V.live_runs_allowed(10 ** 6) == 5


def test_bias_table_last_rows_use_the_rule_and_half_the_sample(monkeypatch):
    rows = V.bias_table(DIST, 200, 0.01, 2)
    assert rows[-2]["Schätzer"] == "Median-of-Means, Regel (K = 37)" and rows[-1]["Schätzer"] == "Median-of-Means, K = n/2 (100)"
    assert rows[-1]["Verzerrung"] == R.bias(DIST, 200, "mom:100") and rows[-2]["Verzerrung"] == R.bias(DIST, 200, "mom:37")
    assert rows[3]["Schätzer"] == "Trimmen c = 2 (10 je Seite)" and rows[2]["Schätzer"] == "Trimmen 10 % (20 je Seite)"


def test_best_k_picks_the_smallest_value_and_the_smallest_k_on_ties(monkeypatch):
    fake_q(monkeypatch, lambda s: {"mom:8": 0.5, "mom:16": 0.5}.get(s, 2.0))
    assert V.best_k(DIST, 200, 0.01) == (8, 0.5)
    fake_q(monkeypatch, lambda s: 0.1 if s == "mom:100" else 2.0)
    assert V.best_k(DIST, 200, 0.01) == (100, 0.1)


def test_better_in_all_seeds_needs_a_strict_win(monkeypatch):
    mom = V.mom_spec(0.01, 200)
    fake_q(monkeypatch, lambda s: 0.9 if s == mom else 1.0)
    assert V.better_in_all_seeds(DIST, 200, 0.01)
    fake_q(monkeypatch, lambda s: 1.0)
    assert not V.better_in_all_seeds(DIST, 200, 0.01)


def test_values_returned_by_the_ratio_helpers_are_numbers(monkeypatch):
    fake_q(monkeypatch, lambda s: 2.0 if s == "mean" else 1.0, typ=lambda s: 0.4 if s == "mean" else 0.2)
    assert V.kipp_ratio(DIST, 200, 0.01) == 0.5 and V.premium(DIST, 200, 0.01) == 0.5 and V.interval_width_ratio is not None
    monkeypatch.setattr(R, "width", lambda dist, n, method, delta: 3.0 if method != "student" else 1.5)
    assert V.interval_width_ratio("Normal", 50, "vorzeichen", 0.05) == 2.0
