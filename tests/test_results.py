"""Die vorgerechneten Messreihen (data/mom_results.json): vollständig, aktuell gegenüber den Quelldateien und in sich stimmig; Zellen werden live nachgerechnet."""

import json

import numpy as np
import pytest

import mom_estimators as E
import mom_experiments as X
import mom_grid as G
import mom_results as R
import mom_scenario as S


def test_results_file_is_valid_utf8_json_with_all_parts():
    text = R.PATH.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"meta", *X.PARTS}
    assert "Infinity" not in text and "NaN" not in text and "Lognormal (σ = 1,5)" in text           # Umlaute und Sonderzeichen im Klartext, gültiges JSON


def test_grids_in_the_file_equal_the_constants():
    for d in S.DISTRIBUTIONS:
        assert sorted(R._cells(d)) == list(G.N_RASTER)
        for n in G.N_RASTER:
            c = R.cell(d, n)
            assert c["runs_per_seed"] == G.RUNS_PER_SEED and c["seeds"] == list(G.SEEDS) and c["levels"] == list(G.Q_LEVELS)
            assert R.specs(d, n) == X.specs_for(n)
            assert len(c["rows"][0]["q"]) == len(G.SEEDS) and len(c["rows"][0]["q"][0]) == len(G.Q_LEVELS)
            assert ("q_med" in c["rows"][0]) == (not S.DIST_INFO[d].symmetric)
        ic = R.interval_cell(d, 50)
        assert ic["runs_per_seed"] == G.INTERVAL_RUNS[50]


def test_every_quantile_row_is_sorted_finite_and_positive():
    for d in S.DISTRIBUTIONS:
        for n in G.N_RASTER:
            for r in R.cell(d, n)["rows"]:
                q = np.asarray(r["q"])
                assert np.isfinite(q).all() and (q > 0).all() and (np.diff(q, axis=1) >= 0).all(), (d, n, r["spec"])
                assert np.isfinite(r["bias"]).all() and (np.asarray(r["rmse"]) > 0).all()


def test_no_all_zero_column():
    """Nullspalten-Regel: kein Schätzer hat überall den Fehler 0 (sonst läuft die Variante nie)."""
    for d in S.DISTRIBUTIONS:
        for r in R.cell(d, 200)["rows"]:
            assert max(max(q) for q in r["q"]) > 0


def test_stored_cell_is_reproduced_by_the_live_function():
    """Das Nachmessen in der App liefert dieselben Zahlen wie die vorgerechnete Zelle (gleiche Seeds, gleiche Läufe), bis auf Gleitkomma-Rundung der letzten Stelle."""
    stored = R.cell("Lognormal (σ = 1,5)", 50)
    again = X.measure_cell("Lognormal (σ = 1,5)", 50, G.RUNS_PER_SEED, G.SEEDS, ["mean", "median", "mom:25", "trim:5"])
    for r in again["rows"]:
        old = R.row("Lognormal (σ = 1,5)", 50, r["spec"])
        for key in ("q", "bias", "rmse", "q_med", "bias_med", "rmse_med"):
            assert np.asarray(r[key]) == pytest.approx(np.asarray(old[key]), rel=1e-6), (r["spec"], key)
    assert stored["n"] == 50


def test_stored_interval_cell_is_reproduced_by_the_live_function():
    again = X.measure_intervals("Normal", 50, G.INTERVAL_RUNS[50])
    old = R.interval_cell("Normal", 50)
    for r in again["rows"]:
        (o,) = [x for x in old["rows"] if x["method"] == r["method"] and x["delta"] == r["delta"]]
        assert np.asarray(r["cov"]) == pytest.approx(np.asarray(o["cov"]), abs=1e-9) and np.asarray(r["wmed"]) == pytest.approx(np.asarray(o["wmed"]), rel=1e-6)
    assert len(again["rows"]) == len(old["rows"])


def test_mean_at_k_one_equals_median_of_means_with_one_block():
    """mom:1 ist das Mittel (andere Summationsreihenfolge, daher nur bis auf Rundung)."""
    for d in ("Normal", "Lognormal (σ = 2)"):
        assert R.q(d, 200, "mom:1", 0.01) == pytest.approx(R.q(d, 200, "mean", 0.01), rel=1e-6)


def test_bias_grows_from_mean_over_half_the_blocks_to_median_for_skewed_distributions():
    """K = n/2 ist nicht der Median (das wäre K = n); der Median liegt in der Messung als eigener Schätzer, mit dem größten Betrag der Verzerrung bei schiefen Verteilungen."""
    for d in S.SKEWED:
        assert abs(R.bias(d, 1000, "median")) > abs(R.bias(d, 1000, "mom:500")) > abs(R.bias(d, 1000, "mom:1"))


def test_interval_methods_and_missing_values():
    assert R.coverage("Normal", 50, "vorzeichen", 0.1) is not None and R.coverage("Normal", 50, "student", 0.01) == pytest.approx(0.99, abs=0.005)
    with pytest.raises(ValueError):
        R.level_index(0.07)
    assert R.nearest_grid_n(700) == 1000 and R.nearest_grid_n(20) == 50 and R.nearest_grid_n(5000) == 5000 and R.nearest_grid_n(120) == 200
