"""Zweig-Tests (Nullspalten-Regel, DEMO-PLAYBOOK §0): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab, sonst läuft sie nie."""

import numpy as np
import pytest

import mom_estimators as E
import mom_evaluation as V
import mom_grid as G
import mom_results as R
import mom_scenario as S


def test_mom_with_the_rule_differs_from_fixed_k_and_from_the_mean_on_a_sample():
    x = S.one_sample("Lognormal (σ = 1,5)", 200, 7)
    assert E.median_of_means(x, 37) != E.median_of_means(x, 8) and E.median_of_means(x, 37) != x.mean() and E.median_of_means(x, 37) != np.median(x)


def test_every_trimming_variant_differs_from_its_neighbour():
    x = S.one_sample("Pareto (α = 2,1)", 200, 3)
    values = [E.trimmed_mean(x, E.trim_count(c, 0.01, 200)) for c in (1, 2, 4)] + [E.trimmed_mean(x, 20)]
    assert len(set(values)) == 4


def test_measured_variants_differ_in_the_results():
    d, n = "Lognormal (σ = 1,5)", 200
    qs = {s: R.q(d, n, s, 0.01) for s in ("mean", "median", V.mom_spec(0.01, n), "mom:8", V.trim_spec(1, 0.01, n), V.trim_spec(4, 0.01, n))}
    assert len(set(qs.values())) == len(qs)


def test_target_median_differs_from_target_mean_for_skewed_distributions_only():
    for d in S.SKEWED:
        assert R.typical(d, 200, "median", "Median") != R.typical(d, 200, "median", "Mittelwert")
    for d in S.SYMMETRIC:
        assert R.typical(d, 200, "median", "Median") == R.typical(d, 200, "median", "Mittelwert")                 # symmetrisch: dasselbe Ziel, dieselbe Spalte


def test_the_winner_map_has_at_least_two_different_winners_in_each_distribution_group():
    for group in (S.SYMMETRIC, S.SKEWED):
        winners = {V.three_way(d, n, dl, 1)["Sieger"] for d in group for n in G.N_RASTER for dl in V.COUNT_DELTAS}
        assert len(winners) >= 2
    assert {V.three_way(d, n, 0.01, 1)["Sieger"] for d in S.DISTRIBUTIONS for n in G.N_RASTER} == {"Mittel", "Trimmen"}


def test_cap_at_n_half_binds_at_n_50_and_the_two_levels_give_the_same_k():
    assert E.k_of_delta(0.01, 50) == E.k_of_delta(0.001, 50) == 25 and E.k_of_delta(0.01, 200) != E.k_of_delta(0.001, 200)
    # kein Fehler, aber gleiche Zelle: Median-of-Means ist bei n = 50 für beide Niveaus derselbe Schätzer
    assert V.mom_spec(0.01, 50) == V.mom_spec(0.001, 50)


def test_every_preset_changes_a_measured_or_displayed_quantity():
    import mom_constants as C
    base = V.analyse(V.Settings())
    outcomes = []
    for name, p in C.PRESETS.items():
        a = V.analyse(V.Settings(**p))
        outcomes.append((a.settings.dist, a.settings.n, a.K, a.k))
    assert len(set(outcomes)) == len(outcomes) and (base.settings.dist, base.settings.n, base.K, base.k) in outcomes


def test_c_changes_the_trimming_only():
    a1, a4 = V.analyse(V.Settings(c=1)), V.analyse(V.Settings(c=4))
    assert a1.k != a4.k and a1.values["Trimmen"] != a4.values["Trimmen"] and a1.values["MoM"] == a4.values["MoM"] and a1.values["Mittel"] == a4.values["Mittel"]


def test_k_option_changes_median_of_means_only():
    a, b = V.analyse(V.Settings(kopt="Regel")), V.analyse(V.Settings(kopt="4"))
    assert a.values["MoM"] != b.values["MoM"] and a.values["Trimmen"] == b.values["Trimmen"] and a.K == 37 and b.K == 4


def test_delta_changes_both_k_and_trimming():
    a, b = V.analyse(V.Settings(delta=0.5)), V.analyse(V.Settings(delta=0.001))
    assert a.K < b.K and a.k < b.k
