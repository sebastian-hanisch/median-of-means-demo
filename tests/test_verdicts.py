"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text). Die Messreihen werden hier durch handgesetzte Tabellen ersetzt (monkeypatch auf mom_results), damit Schwellen und Zahlen im Text
direkt geprüft werden; ohne diese Datei überleben fast alle Mutanten der Verdikt-Arithmetik, weil die Messreihe sie nicht trifft."""

import pytest

import mom_evaluation as V
import mom_grid as G
import mom_results as R

DIST = "Lognormal (σ = 1,5)"


def fake_results(monkeypatch, table, typ=None, bias=None):
    """table: spec → Q(1 − δ) (für alle Seeds gleich); spec nicht in table → 1,0. typ: spec → typischer Fehler."""
    typ = typ or {}
    bias = bias or {}
    monkeypatch.setattr(R, "q_seeds", lambda dist, n, spec, delta, target="Mittelwert": [table.get(spec, 1.0)] * 4)
    monkeypatch.setattr(R, "q", lambda dist, n, spec, delta, target="Mittelwert": table.get(spec, 1.0))
    monkeypatch.setattr(R, "typical", lambda dist, n, spec, target="Mittelwert": typ.get(spec, 0.5))
    monkeypatch.setattr(R, "bias", lambda dist, n, spec, target="Mittelwert": bias.get(spec, 0.0))


def test_premium_and_payout_numbers_and_kind(monkeypatch):
    fake_results(monkeypatch, {"mean": 1.0, "mom:37": 0.5}, typ={"mean": 0.1, "mom:37": 0.25})
    kind, text = V.verdict_premium(DIST, 200, 0.01)
    assert kind == "info" and "**2,50-mal**" in text and "(0,250 gegen 0,100)" in text and "Verhältnis 0,50" in text and "kostet im Alltag und zahlt im Ernstfall" in text


def test_premium_warning_when_it_costs_and_does_not_pay_with_the_inclusive_boundary(monkeypatch):
    fake_results(monkeypatch, {"mean": 1.0, "mom:37": 1.0}, typ={"mean": 0.1, "mom:37": 0.1})              # Verhältnis genau 1, Prämie genau 1
    kind, text = V.verdict_premium(DIST, 200, 0.01)
    assert kind == "warning" and "ohne im Ernstfall zu zahlen" in text and "**1,00-mal**" in text
    fake_results(monkeypatch, {"mean": 1.0, "mom:37": 0.9}, typ={"mean": 0.1, "mom:37": 0.1})
    assert V.verdict_premium(DIST, 200, 0.01)[0] == "info"                                                 # zahlt, Prämie genau 1: gilt als Kosten
    fake_results(monkeypatch, {"mean": 1.0, "mom:37": 0.9}, typ={"mean": 0.1, "mom:37": 0.09})
    kind, text = V.verdict_premium(DIST, 200, 0.01)
    assert kind == "info" and text.endswith("Verhältnis 0,90.") and "kostet im Alltag" not in text


def test_kipp_verdict_kinds_and_numbers(monkeypatch):
    def mom_spec(n):
        return V.mom_spec(0.01, n)
    monkeypatch.setattr(R, "q_seeds", lambda dist, n, spec, delta, target="Mittelwert": [0.8 if spec.startswith("mom:") else 1.0] * 4)
    monkeypatch.setattr(R, "q", lambda dist, n, spec, delta, target="Mittelwert": 0.8 if spec.startswith("mom:") else 1.0)
    kind, text = V.verdict_kipp(DIST, 0.01)
    assert kind == "success" and "n = 50: 0,80, n = 200: 0,80, n = 1000: 0,80, n = 5000: 0,80" in text and "ab δ ≤ 0,5" in text
    fake_results(monkeypatch, {"mean": 1.0, **{mom_spec(n): 1.0 for n in G.N_RASTER}})
    kind, text = V.verdict_kipp(DIST, 0.01)
    assert kind == "warning" and "mindestens so genau" in text and "bis δ = 0,001 in keinem Niveau durchgehend" in text            # Verhältnis genau 1: nicht besser
    fake_results(monkeypatch, {"mean": 1.0, mom_spec(50): 0.5, mom_spec(200): 1.5, mom_spec(1000): 1.5, mom_spec(5000): 1.5})
    assert V.verdict_kipp(DIST, 0.01)[0] == "info"


def test_k_verdict_branches_and_thresholds(monkeypatch):
    fake_results(monkeypatch, {"mom:37": 1.0})                                                              # alle Blockzahlen gleich gut: bei Gleichstand gewinnt das kleinste K (= 1, das Mittel)
    kind, text = V.verdict_k(DIST, 200, 0.01)
    assert kind == "info" and "K = 1, das ist das Mittel" in text and "**0 % schlechter**" in text
    table = {"mom:37": 1.0625, "mom:8": 1.0}
    fake_results(monkeypatch, {**{f"mom:{K}": 2.0 for K in (1, 2, 4, 6, 8, 13, 16, 19, 24, 32, 37, 43, 50, 56, 64, 100)}, **table}, bias={"mom:37": -0.3, "mom:8": -0.1})
    kind, text = V.verdict_k(DIST, 200, 0.01)
    assert kind == "info" and "K = 37" in text and "K = 8 mit 1,000" in text and "**6 % schlechter**" in text and "Verzerrung der Regel: -30,0 %, des besten K: -10,0 %" in text
    fake_results(monkeypatch, {**{f"mom:{K}": 2.0 for K in (1, 2, 4, 6, 8, 13, 16, 19, 24, 32, 37, 43, 50, 56, 64, 100)}, "mom:37": 1.125, "mom:8": 1.0})
    assert V.verdict_k(DIST, 200, 0.01)[0] == "warning"                                                     # 12,5 % > 10 %
    kind, text = V.verdict_k("Normal", 200, 0.01)
    assert "Bei symmetrischen Verteilungen ist die Verzerrung 0" in text


def test_k_verdict_when_the_rule_is_the_best(monkeypatch):
    values = {f"mom:{K}": 2.0 for K in (1, 2, 4, 6, 8, 13, 16, 19, 24, 32, 37, 43, 50, 56, 64, 100)}
    fake_results(monkeypatch, {**values, "mom:37": 0.7})
    kind, text = V.verdict_k(DIST, 200, 0.01)
    assert kind == "info" and "K = 37 ist hier auch die beste" in text and "Q(1 − δ) = 0,700" in text


def test_three_way_numbers_and_winner(monkeypatch):
    fake_results(monkeypatch, {"mean": 0.6, "mom:37": 0.5, "trim:5": 0.4})
    kind, text = V.verdict_three_way(DIST, 200, 0.01, 1)
    assert kind == "info" and "beim Mittel 0,600" in text and "Median-of-Means (K = 37) 0,500" in text and "(5 Werte je Seite, c = 1) 0,400" in text and "**Trimmen**" in text
    fake_results(monkeypatch, {"mean": 0.4, "mom:37": 0.5, "trim:5": 0.4})
    assert "**Mittel**" in V.verdict_three_way(DIST, 200, 0.01, 1)[1]                                      # Gleichstand: Reihenfolge Mittel, MoM, Trimmen


def test_target_verdict_symmetric_and_skewed(monkeypatch):
    kind, text = V.verdict_target("Normal", 200, 0.05)
    assert kind == "info" and "dasselbe Ziel" in text
    monkeypatch.setattr(R, "typical", lambda dist, n, spec, target="Mittelwert": {("mean", "Mittelwert"): 0.1, ("median", "Mittelwert"): 0.6, ("mom:24", "Mittelwert"): 0.3,
                                                                                    ("mean", "Median"): 2.0, ("median", "Median"): 0.1, ("mom:24", "Median"): 1.2}[(spec, target)])
    kind, text = V.verdict_target(DIST, 200, 0.05)
    assert "Mittel 0,100, Median 0,600, Median-of-Means 0,300" in text and "Mittel 2,000, Median 0,100, Median-of-Means 1,200" in text and "**12,0-mal**" in text


def test_interval_verdict_on_the_measured_cells():
    kind, text = V.verdict_intervals("Normal", 0.05)
    assert kind == "success" and "Soll 95 %" in text and "höchstens 1,5-mal so breit wie Student sind: Vorzeichen-Intervall auf den Blockmitteln." in text and "1,5-mal Student" in text
    kind, text = V.verdict_intervals(DIST, 0.05)
    assert kind == "warning" and "Keine der gemessenen Median-of-Means-Konstruktionen" in text and "kleinste Überdeckung 46,5 %" in text
    assert V.width_range_text(1.0, 1.04) == "1,0-mal Student" and V.width_range_text(2.44, 2.5) == "2,4- bis 2,5-mal Student"


def test_live_verdict_reliability_and_kind():
    s = V.Settings(delta=0.01)
    live = {"rows": {"Mittel": {"q": 1.0, "typ": 0.1, "bias": 0.0}, "MoM": {"q": 0.5, "typ": 0.2, "bias": -0.2}}, "runs": 40000, "K": 37}
    kind, text = V.verdict_live(s, live)
    assert kind == "success" and "40000 Läufe (4 Seeds)" in text and "Q(1 − δ) beim Mittel 1,000, bei Median-of-Means (K = 37) 0,500 (relativ zum Mittelwert), Verhältnis 0,50" in text
    assert "Verzerrung von Median-of-Means: -20,0 %" in text and "Wenig belastbar" not in text
    kind, text = V.verdict_live(s, {**live, "runs": 9999})
    assert kind == "info" and "**Wenig belastbar:** bei δ = 0,01 überschreiten nur etwa 100 Läufe" in text
    kind, text = V.verdict_live(s, {**live, "rows": {**live["rows"], "MoM": {"q": 1.0, "typ": 0.2, "bias": -0.2}}})
    assert kind == "info" and "Verhältnis 1,00" in text
    assert "(relativ zum Mittelwert)" not in V.verdict_live(V.Settings(dist="Normal"), live)[1]


def test_main_verdict_names_every_error_and_the_closest_estimator():
    a = V.analyse(V.Settings())
    kind, text = V.verdict_main(a)
    e = a.errors
    assert kind == "info" and f"beim Mittel bei {V.de(e['Mittel'])}" in text and f"bei Median-of-Means ({a.K} Blöcke) bei {V.de(e['MoM'])}" in text and f"({a.k} Werte je Seite)" in text
    assert f"liegt hier **{min(e, key=lambda k: abs(e[k]))}**" in text and "(relativ zum Mittelwert)" in text and "Zufalls-Seed 7" in text
    assert "(relativ zum Mittelwert)" not in V.verdict_main(V.analyse(V.Settings(dist="Normal")))[1]


def test_count_and_tuned_verdicts_contain_the_counts():
    kind, text = V.verdict_count()
    better, total = V.count_better_cells()
    assert kind == "info" and f"**{better} von {total}**" in text and "Niveaus 0,1, 0,05, 0,01, 0,001" in text and "Bei Normal und t(5) in keiner" in text
    kind, text = V.verdict_tuned()
    sym, skew = V.tuned_counts(V.S.SYMMETRIC), V.tuned_counts(V.S.SKEWED)
    assert kind == "warning" and f"das Trimmen {sym['Trimmen']}, das Mittel {sym['Mittel']} und Median-of-Means {sym['MoM']}" in text
    assert f"Median-of-Means {skew['MoM']}, das Mittel {skew['Mittel']} und das Trimmen {skew['Trimmen']}" in text and f"bei den {sum(sym.values())} symmetrischen" in text


def test_tip_text():
    assert V._tip_text(0.02) == "ab δ ≤ 0,02" and V._tip_text(None).startswith("bis δ = 0,001")
