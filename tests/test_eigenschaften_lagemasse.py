"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen, die für Median-of-Means und das getrimmte Mittel gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Seed. Alle Aussagen sind Eigenschaften der Definitionen (Blockmittel und Median bzw. Sortieren und Abschneiden)."""

import numpy as np
import pytest

import mom_estimators as E
import mom_scenario as S


def instances(count=30, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(3, 60))
        K = int(rng.integers(1, n + 1))
        yield rng.standard_t(2.5, n), K


def test_affine_equivariance_for_positive_and_negative_scale():
    """MoM(a·x + b) = a·MoM(x) + b, auch für a < 0 (Mittel und Median vertauschen sich mit dem Vorzeichen)."""
    for x, K in instances():
        for a, b in [(2.5, 3.0), (0.01, -7.0), (-1.7, 4.0)]:
            assert E.median_of_means(a * x + b, K) == pytest.approx(a * E.median_of_means(x, K) + b, rel=1e-9, abs=1e-9)


def test_permutation_inside_blocks_and_of_blocks_does_not_change_the_value():
    rng = np.random.default_rng(2)
    for x, K in instances():
        n = len(x)
        edges = list(E.block_edges(n, K)) + [n]
        blocks = [x[edges[i]: edges[i + 1]] for i in range(K)]
        shuffled_inside = np.concatenate([rng.permutation(b) for b in blocks])
        reordered = np.concatenate([blocks[i] for i in rng.permutation(K)]) if len(set(len(b) for b in blocks)) == 1 else None
        assert E.median_of_means(shuffled_inside, K) == pytest.approx(E.median_of_means(x, K), rel=1e-9, abs=1e-12)
        if reordered is not None:
            assert E.median_of_means(reordered, K) == pytest.approx(E.median_of_means(x, K), rel=1e-9, abs=1e-12)


def test_monotone_in_every_value():
    """Erhöht man einen Wert, sinkt Median-of-Means nie und das getrimmte Mittel auch nicht."""
    rng = np.random.default_rng(3)
    for x, K in instances():
        i = int(rng.integers(0, len(x)))
        y = x.copy()
        y[i] += abs(rng.standard_cauchy()) + 0.1
        assert E.median_of_means(y, K) >= E.median_of_means(x, K) - 1e-12
        k = (len(x) - 1) // 3
        assert E.trimmed_mean(y, k) >= E.trimmed_mean(x, k) - 1e-12


def test_bounded_by_the_range_and_by_the_block_means():
    for x, K in instances():
        m = E.median_of_means(x, K)
        bm = E.block_means(x, K)[0]
        assert bm.min() - 1e-12 <= m <= bm.max() + 1e-12 and x.min() - 1e-12 <= m <= x.max() + 1e-12


def test_constant_sample_is_reproduced_by_every_estimator():
    for x, K in instances():
        const = np.full(len(x), 3.25)
        assert E.median_of_means(const, K) == pytest.approx(3.25) and E.trimmed_mean(const, (len(x) - 1) // 2) == pytest.approx(3.25)


def test_trimming_interpolates_between_mean_and_median():
    for x, _ in instances():
        n = len(x)
        assert E.trimmed_mean(x, 0) == pytest.approx(x.mean())
        assert E.trimmed_mean(x, (n - 1) // 2) == pytest.approx(np.median(x))


def test_rule_is_monotone_in_delta_and_trim_count_in_c():
    deltas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for n in (20, 50, 200, 5000):
        ks = [E.k_of_delta(d, n) for d in deltas]
        assert ks == sorted(ks) and all(1 <= k <= max(n // 2, 1) for k in ks)
        for d in deltas:
            assert E.trim_count(1, d, n) <= E.trim_count(2, d, n) <= E.trim_count(4, d, n) <= (n - 1) // 2


def test_same_seed_same_sample_and_estimates():
    a = S.one_sample("Pareto (α = 2,5)", 200, 5)
    b = S.one_sample("Pareto (α = 2,5)", 200, 5)
    assert np.array_equal(a, b) and E.estimate_all(a, 0.01) == E.estimate_all(b, 0.01)


def test_sign_interval_is_wider_for_a_higher_level_and_shrinks_with_data_in_the_normal_case():
    x = S.batch("Normal", 300, 480, S.Stream(5))
    narrow = E.sign_interval(x, 24, 0.2)
    wide = E.sign_interval(x, 24, 0.01)
    assert (wide[1] - wide[0]).mean() > (narrow[1] - narrow[0]).mean()
    small = E.sign_interval(S.batch("Normal", 300, 240, S.Stream(5)), 24, 0.05)
    big = E.sign_interval(S.batch("Normal", 300, 2400, S.Stream(5)), 24, 0.05)
    assert (big[1] - big[0]).mean() < (small[1] - small[0]).mean()


def test_normal_efficiency_of_median_of_means_is_below_the_mean_by_the_sqrt_pi_over_two_limit():
    """Bei Normal ist der Fehler des Mittels der kleinste mögliche; Median-of-Means mit vielen Blöcken verliert den Faktor √(π/2) = 1,2533 an Streuung (Median der Blockmittel)."""
    x = S.batch("Normal", 20000, 480, S.Stream(6))
    ratio = np.std(E.median_of_means(x, 96)) / np.std(x.mean(axis=1))
    assert ratio == pytest.approx(np.sqrt(np.pi / 2), rel=0.06) and ratio > 1.15
