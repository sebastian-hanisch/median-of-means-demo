"""Auswertung: Formatierung, Kippniveau, Zuverlässigkeit, Blockzahl-Wahl und Zusammenfassungen der Messreihen an handgesetzten Werten und an der Messreihe."""

import pytest

import mom_evaluation as V
import mom_grid as G
import mom_results as R
import mom_scenario as S


def test_german_number_formats():
    assert V.de(0.8333, 2) == "0,83" and V.de(12.0, 1) == "12,0"
    assert V.pct(0.314) == "31,4 %" and V.pct(-0.5151, 1) == "-51,5 %" and V.pct(0.226, 0) == "23 %"
    assert [V.de_delta(d) for d in (0.5, 0.05, 0.001, 0.2, 0.1)] == ["0,5", "0,05", "0,001", "0,2", "0,1"]


def test_tipping_level_by_hand():
    """Das größte δ, ab dem es bei diesem und allen kleineren besser ist: Lücke bei einem kleineren δ beendet die Reihe."""
    assert V.tipping_level({0.001: True, 0.01: True, 0.05: True, 0.1: False}) == 0.05
    assert V.tipping_level({0.001: True, 0.01: False, 0.05: True, 0.1: True}) == 0.001                  # nicht durchgehend: nur bis zur Lücke
    assert V.tipping_level({0.001: False, 0.01: True}) is None
    assert V.tipping_level({}) is None
    assert V.tipping_level({0.1: True, 0.001: True, 0.01: True}) == 0.1                                  # Reihenfolge der Eingabe ist gleichgültig


def test_reliable_needs_enough_exceedances():
    assert V.reliable(0.001, 160_000) and not V.reliable(0.001, 40_000) and V.reliable(0.01, 10_000) and not V.reliable(0.01, 9_999)
    assert V.reliable(0.5, 200) and not V.reliable(0.5, 199)


def test_live_budget_bounds_the_runs():
    assert V.live_runs_allowed(5000) == 1000 and V.live_runs_allowed(20) == 5000 and V.live_runs_allowed(200) == 5000
    assert all(1 <= V.live_runs_allowed(n) <= 5000 for n in (20, 100, 5000))


def test_signed_error_is_absolute_for_symmetric_and_relative_for_skewed():
    assert V.signed_error("Normal", -0.3) == pytest.approx(-0.3)
    mu = S.DIST_INFO["Lognormal (σ = 1,5)"].mean
    assert V.signed_error("Lognormal (σ = 1,5)", mu * 0.7) == pytest.approx(-0.3)
    assert V.signed_error("Lognormal (σ = 1,5)", 1.0, "Median") == pytest.approx(0.0)


def test_analyse_is_consistent():
    a = V.analyse(V.Settings())
    assert a.K == 37 and a.k == 5 and len(a.x) == 200 and len(a.labels) == 200 and len(a.block_means) == 37
    assert a.values["MoM"] == pytest.approx(float(sorted(a.block_means)[18]))                       # K = 37 ungerade: der mittlere Wert
    assert a.errors["Mittel"] == pytest.approx(a.values["Mittel"] / a.mean_true - 1.0)


def test_analyse_reproducible_and_seed_sensitive():
    a, b, c = V.analyse(V.Settings(seed=3)), V.analyse(V.Settings(seed=3)), V.analyse(V.Settings(seed=4))
    assert (a.x == b.x).all() and a.values == b.values and not (a.x == c.x).all()


def test_spec_names_follow_the_rules():
    assert V.mom_spec(0.01, 200) == "mom:37" and V.mom_spec(0.01, 50) == "mom:25" and V.trim_spec(2, 0.01, 200) == "trim:10"
    for n in G.N_RASTER:
        for d in G.DELTA_GRID:
            assert V.mom_spec(d, n) in R.specs("Normal", n) and all(V.trim_spec(c, d, n) in R.specs("Normal", n) for c in (1, 2, 4))


def test_ratio_seeds_pairs_the_same_seeds():
    r = V.ratio_seeds("Lognormal (σ = 1,5)", 200, "mom:37", "mean", 0.01)
    assert len(r) == 4 and r == pytest.approx([a / b for a, b in zip(R.q_seeds("Lognormal (σ = 1,5)", 200, "mom:37", 0.01), R.q_seeds("Lognormal (σ = 1,5)", 200, "mean", 0.01))])
    assert V.kipp_ratio("Lognormal (σ = 1,5)", 200, 0.01) == pytest.approx(R.q("Lognormal (σ = 1,5)", 200, "mom:37", 0.01) / R.q("Lognormal (σ = 1,5)", 200, "mean", 0.01))


def test_best_k_is_the_minimum_of_the_measured_curve_and_the_rule_loss_is_non_negative():
    for d in ("Normal", "Lognormal (σ = 1,5)", "t(3)"):
        best, qv = V.best_k(d, 200, 0.01)
        curve = V.k_curve(d, 200, 0.01)
        assert qv == pytest.approx(min(c["q"] for c in curve)) and best in [c["K"] for c in curve]
        assert V.rule_loss(d, 200, 0.01) >= -1e-12
    assert V.best_k("Normal", 200, 0.01)[0] == 1                                                         # bei Normal ist das Mittel (K = 1) das beste


def test_three_way_winner_and_ties_by_order():
    t = V.three_way("Lognormal (σ = 1,5)", 200, 0.01, 1)
    assert t["Sieger"] == min(("Mittel", "MoM", "Trimmen"), key=lambda k: t[k])
    counts = V.winner_counts(1)
    normal = V.winner_counts(1, dists=("Normal",))
    assert sum(counts.values()) == 160 and sum(normal.values()) == 16
    assert normal["MoM"] == 0 and normal["Mittel"] >= 14                                                   # Normal: das Mittel, in 1 Zelle gewinnt das Trimmen mit wenigen Werten durch Rauschen


def test_count_better_cells_matches_the_cell_by_cell_count():
    better, total = V.count_better_cells()
    assert total == 160 and better == sum(V.better_in_all_seeds(d, n, dl) for d in S.DISTRIBUTIONS for n in G.N_RASTER for dl in V.COUNT_DELTAS)
    assert not any(V.better_in_all_seeds("Normal", n, d) for n in G.N_RASTER for d in G.DELTA_GRID)


def test_bias_table_rows():
    rows = V.bias_table("Lognormal (σ = 1,5)", 200, 0.01, 1)
    assert [r["Schätzer"].split(" (")[0].split(",")[0] for r in rows] == ["Mittel", "Median", "Trimmen 10 %", "Trimmen c = 1", "Median-of-Means", "Median-of-Means"]
    bias = {r["Schätzer"]: r["Verzerrung"] for r in rows}
    assert abs(bias["Mittel"]) < 0.02 and bias["Median"] < bias[rows[4]["Schätzer"]] < 0


def test_interval_table_has_every_method_and_n():
    rows = V.interval_table("Normal", 0.05)
    assert len(rows) == 4 * len(G.INTERVAL_NS) and all(r["coverage"] is not None for r in rows)
    assert all(r["width_vs_student"] == pytest.approx(1.0) for r in rows if r["method"] == "student")
    assert V.min_coverage(["Normal"], "student", 0.05)[0] == pytest.approx(0.95, abs=0.01)


def test_live_measure_returns_the_four_estimators():
    live = V.live_measure(V.Settings(n=50), 500)
    assert list(live["rows"]) == ["Mittel", "Median", "Trimmen", "MoM"] and live["runs"] == 2000 and live["K"] == 25 and live["k"] == 5
    assert all(r["q"] > 0 and r["typ"] > 0 for r in live["rows"].values())
    capped = V.live_measure(V.Settings(n=5000), 5000)
    assert capped["runs"] == 4000                                                                          # Budget: 1000 Läufe je Seed bei n = 5000
