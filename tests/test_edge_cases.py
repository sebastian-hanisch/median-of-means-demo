"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, inklusive Grenzen, alle Werte gleich, Ausreißer an den Rändern."""

import numpy as np
import pytest

import mom_estimators as E
import mom_evaluation as V


def test_exactly_two_elements():
    """n = 2: höchstens K = 2 Blöcke (der Median, hier das Mittel der zwei Werte); Trimmen lässt keinen Wert weg (2k < 2 verlangt k = 0)."""
    x = np.array([1.0, 5.0])
    assert E.median_of_means(x, 1) == pytest.approx(3.0) and E.median_of_means(x, 2) == pytest.approx(3.0)
    assert E.trimmed_mean(x, 0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        E.trimmed_mean(x, 1)
    assert E.k_of_delta(0.01, 2) == 1 and E.trim_count(5, 0.01, 2) == 0


def test_single_element():
    """n = 1: jeder Schätzer ist der Wert selbst; die Regel liefert K = 1, die Trimm-Zahl 0."""
    x = np.array([7.0])
    assert E.median_of_means(x, 1) == 7.0 and E.trimmed_mean(x, 0) == 7.0
    assert E.k_of_delta(0.001, 1) == 1 and E.trim_count(4, 0.001, 1) == 0
    r = E.estimate_all(x, 0.01)
    assert r["Mittel"] == r["Median"] == r["MoM"] == r["Trimmen"] == 7.0


def test_all_values_equal():
    """Alle Werte gleich: jeder Schätzer liefert diesen Wert, auch mit Rest (n nicht durch K teilbar)."""
    x = np.full(37, 2.5)
    for K in (1, 2, 5, 18, 37):
        assert E.median_of_means(x, K) == 2.5
    assert E.trimmed_mean(x, 10) == 2.5


def test_k_equal_to_n_and_k_one_are_allowed_and_k_above_n_is_not():
    x = np.arange(1.0, 8.0)
    assert E.median_of_means(x, 7) == pytest.approx(4.0) and E.median_of_means(x, 1) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        E.median_of_means(x, 8)
    with pytest.raises(ValueError):
        E.median_of_means(x, 0)


def test_trim_limit_is_inclusive_at_the_largest_valid_k():
    """Odd n: k = (n − 1)/2 lässt genau den mittleren Wert stehen (der Median), k = (n + 1)/2 ist ungültig. Gerades n: k = n/2 − 1 lässt die zwei mittleren stehen."""
    assert E.trimmed_mean(np.array([9.0, 1.0, 5.0, 3.0, 7.0]), 2) == 5.0
    assert E.trimmed_mean(np.array([4.0, 1.0, 3.0, 2.0]), 1) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        E.trimmed_mean(np.array([1.0, 2.0, 3.0, 4.0]), 2)


def test_rule_cap_is_inclusive_at_n_half():
    """Die Regel wird bei n/2 gedeckelt: bei n = 50 liefern δ = 0,01 und δ = 0,001 beide 25; bei n = 74 ist 37 noch erlaubt, bei n = 73 nicht (73 // 2 = 36)."""
    assert E.k_of_delta(0.01, 74) == 37 and E.k_of_delta(0.01, 73) == 36 and E.k_of_delta(0.01, 75) == 37


def test_an_extreme_outlier_moves_median_of_means_by_at_most_one_block_position():
    """Ein Ausreißer verdirbt höchstens ein Blockmittel: der Median der 10 Blockmittel (5,5; 15,5; …; 95,5) rückt um höchstens einen Rangplatz, hier in [45,5; 65,5]; das Mittel geht mit."""
    base = np.arange(1.0, 101.0)
    for pos in (0, 99, 50):
        x = base.copy()
        x[pos] = 1e12
        assert 45.5 - 1e-9 <= E.median_of_means(x, 10) <= 65.5 + 1e-9
        assert x.mean() > 1e9


def test_settings_at_the_smallest_and_largest_sizes_run():
    for n in (20, 5000):
        for kopt in ("Regel", "1", "n/2", "256"):
            a = V.analyse(V.Settings(n=n, kopt=kopt))
            assert 1 <= a.K <= n // 2 and np.isfinite(list(a.values.values())).all() and a.x.shape == (n,)
    assert V.k_from_option("256", 0.01, 20) == 10 and V.k_from_option("n/2", 0.01, 21) == 10 and V.k_from_option("1", 0.5, 20) == 1


def test_k_option_rule_depends_on_delta_and_cap():
    assert V.k_from_option("Regel", 0.05, 200) == 24 and V.k_from_option("Regel", 0.001, 50) == 25 and V.k_from_option("4", 0.001, 200) == 4
