"""Orakel-Test median-of-means-demo: Schätzer, Intervalle und Erzeuger gegen einen anderen Rechenweg (ORACLE-PLAYBOOK, DEMO-PLAYBOOK §12).

Orakel (jeweils ein anderer Weg als der Code der Demo):
* Median-of-Means gegen die Definition mit einer Python-Schleife über `numpy.array_split` (gleiche Konvention: die ersten n mod K Blöcke eine Zahl mehr) und gegen die Aufzählung aller
  Zuteilungen auf Blöcke für n ≤ 8: die zusammenhängende Zuteilung liegt in der Menge, jede andere Zuteilung bleibt zwischen den Grenzen der Aufzählung.
* Trimmen gegen `scipy.stats.trim_mean` (Anteil (k + ½)/n, damit int(Anteil·n) = k auch bei Gleitkomma-Kanten), Mittel gegen `numpy.mean`, Median gegen `statistics.median`.
* Vorzeichen-Intervall: Rang und Überdeckung gegen `scipy.stats.binom`; Überdeckung im Versuch gegen die genaue Formel.
* Student-Quantile der Konstantentabelle gegen `scipy.stats.t.ppf`.
* Erzeuger: Mittelwert und Varianz aller zehn Verteilungen gegen scipy.stats bzw. die geschlossene Formel; Kolmogorov-Smirnov-Abstand gegen die scipy-Verteilungsfunktion.
* Messreihe: Fehlerquantil des Mittels bei Normal gegen z/√n; Verzerrung des Medians gegen die geschlossene Formel; Verzerrung wächst mit K (Mini-Rechnung).
scipy liegt nur im Test-venv (requirements-dev.txt), nicht in der Laufzeit."""

import itertools
import statistics

import numpy as np
import pytest

import mom_estimators as E
import mom_grid as G
import mom_scenario as S

scipy_stats = pytest.importorskip("scipy.stats")


def reference_mom(x, K):
    """Definition: np.array_split in K Teile, Mittel je Teil, Median der Blockmittel (reine Python-Schleife)."""
    means = [float(np.mean(part)) for part in np.array_split(np.asarray(x, dtype=float), K)]
    return statistics.median(means)


def test_median_of_means_equals_the_definition_on_random_instances():
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(2, 60))
        K = int(rng.integers(1, n + 1))
        x = rng.standard_t(2.5, n)
        assert E.median_of_means(x, K) == pytest.approx(reference_mom(x, K), rel=1e-9, abs=1e-12)


def test_stack_of_rows_equals_the_row_by_row_definition():
    rng = np.random.default_rng(6)
    x = rng.lognormal(0.0, 1.5, (40, 37))
    for K in (1, 2, 5, 18, 37):
        assert E.median_of_means(x, K) == pytest.approx([reference_mom(row, K) for row in x], rel=1e-9)


def all_assignments(n, sizes):
    """Alle Zuteilungen der Positionen 0 … n − 1 auf Blöcke der vorgegebenen Größen (Blöcke gleicher Größe vertauschbar, daher als Menge von Blockmengen)."""
    def rec(rest, sizes):
        if not sizes:
            yield []
            return
        size = sizes[0]
        for chosen in itertools.combinations(rest, size):
            remaining = [i for i in rest if i not in chosen]
            for tail in rec(remaining, sizes[1:]):
                yield [chosen] + tail
    seen = set()
    for assignment in rec(list(range(n)), list(sizes)):
        key = frozenset(assignment)
        if key not in seen:
            seen.add(key)
            yield assignment


def mom_of_assignment(x, assignment):
    return statistics.median([float(np.mean([x[i] for i in block])) for block in assignment])


def test_contiguous_blocks_are_one_of_all_assignments_and_bounded_by_them(pairs_x):
    """x = (2, 4, …, 14, 200), K = 4: 105 Zuteilungen in vier Paare; die zusammenhängende liegt darin (Wert 9), und alle Werte liegen im Bereich der Aufzählung."""
    assignments = list(all_assignments(8, [2, 2, 2, 2]))
    assert len(assignments) == 105
    values = [mom_of_assignment(pairs_x, a) for a in assignments]
    contiguous = E.median_of_means(pairs_x, 4)
    assert contiguous == pytest.approx(9.0) and any(v == pytest.approx(contiguous) for v in values)
    assert min(values) >= 3.0 - 1e-12 and max(values) <= 107.0
    assert 7.0 - 1e-12 <= np.median(values) <= 11.0 + 1e-12


def test_every_contiguous_assignment_for_small_n_is_in_the_enumeration():
    rng = np.random.default_rng(8)
    for n, K in [(6, 3), (7, 3), (8, 4), (5, 2), (7, 7), (6, 1)]:
        x = rng.normal(size=n)
        sizes = list(E.block_sizes(n, K))
        enum = {round(mom_of_assignment(x, a), 12) for a in all_assignments(n, sizes)}
        assert round(E.median_of_means(x, K), 12) in enum


def test_the_hand_instance_has_exactly_fifteen_assignments_and_the_documented_range(hand_x):
    """n = 6, K = 3: 6!/(2!³ · 3!) = 15 Zuteilungen; Blockmittel zweier Werte, der Median der drei liegt zwischen 2,5 und 4,5 (Handrechnung der Grenzen unten)."""
    assignments = list(all_assignments(6, [2, 2, 2]))
    assert len(assignments) == 15
    values = [mom_of_assignment(hand_x, a) for a in assignments]
    assert E.median_of_means(hand_x, 3) == pytest.approx(3.5) and any(v == pytest.approx(3.5) for v in values)
    # Handrechnung der Grenzen: das Paar mit der 100 hat das größte Blockmittel (≥ 50,5), der Median ist also das größere Mittel der beiden anderen Paare.
    # Minimum: 100 mit 5, Rest (1, 4 | 2, 3): beide Mittel 2,5. Maximum: 100 mit 1, Rest (2, 3 | 4, 5): Mittel 2,5 und 4,5, also 4,5.
    assert min(values) == pytest.approx(2.5) and max(values) == pytest.approx(4.5)


def test_trimmed_mean_equals_scipy_trim_mean():
    rng = np.random.default_rng(9)
    for _ in range(200):
        n = int(rng.integers(3, 80))
        k = int(rng.integers(0, (n - 1) // 2 + 1))
        x = rng.standard_cauchy(n)
        assert E.trimmed_mean(x, k) == pytest.approx(scipy_stats.trim_mean(x, (k + 0.5) / n), rel=1e-9, abs=1e-12)


def test_mean_and_median_equal_numpy_and_statistics():
    rng = np.random.default_rng(10)
    x = rng.lognormal(0.0, 1.0, (30, 51))
    assert E.median_of_means(x, 1) == pytest.approx(x.mean(axis=1), rel=1e-12)
    assert E.median_of_means(x, 51) == pytest.approx([statistics.median(r) for r in x], rel=1e-12)
    assert E.median_of_means(x[:, :50], 50) == pytest.approx([statistics.median(r) for r in x[:, :50]], rel=1e-12)


@pytest.mark.parametrize("K", [4, 8, 12, 24, 37, 56, 100])
@pytest.mark.parametrize("delta", [0.2, 0.1, 0.05, 0.01, 0.001])
def test_sign_interval_rank_and_cover_equal_the_binomial(K, delta):
    rank = E.sign_interval_rank(K, delta)
    binom = scipy_stats.binom(K, 0.5)
    ls = [l for l in range(1, K // 2 + 1) if binom.cdf(l - 1) <= delta / 2]
    if not ls:
        assert rank is None
        return
    l, cover = rank
    assert l == max(ls) and cover == pytest.approx(1 - 2 * binom.cdf(l - 1), abs=1e-12) and cover >= 1 - delta - 1e-12
    assert binom.cdf(l) > delta / 2                                                  # l + 1 wäre nicht mehr erlaubt


def test_sign_interval_covers_the_center_of_symmetric_block_means_with_the_exact_probability():
    """Normalverteilung: der Median der Blockmittel ist der Mittelwert 0. Überdeckung im Versuch gegen 1 − 2·P(Bin(24; ½) ≤ 6)."""
    x = S.batch("Normal", 20000, 240, S.Stream(21))
    lo, hi = E.sign_interval(x, 24, 0.05)
    covered = float(((lo <= 0) & (0 <= hi)).mean())
    exact = 1 - 2 * scipy_stats.binom(24, 0.5).cdf(6)
    assert exact == pytest.approx(0.97735, abs=1e-5) and covered == pytest.approx(exact, abs=0.004)


def test_student_quantile_table_equals_scipy():
    for (n, delta), value in G.T_QUANTILES.items():
        assert value == pytest.approx(scipy_stats.t.ppf(1 - delta / 2, n - 1), rel=1e-12)
    assert len(G.T_QUANTILES) == len(G.INTERVAL_NS) * len(G.INTERVAL_DELTAS)


SCIPY_DISTS = {
    "Normal": scipy_stats.norm(), "t(5)": scipy_stats.t(5), "t(3)": scipy_stats.t(3),
    "Lognormal (σ = 1)": scipy_stats.lognorm(s=1.0), "Lognormal (σ = 1,5)": scipy_stats.lognorm(s=1.5), "Lognormal (σ = 2)": scipy_stats.lognorm(s=2.0),
    "Pareto (α = 2,1)": scipy_stats.pareto(b=2.1), "Pareto (α = 2,5)": scipy_stats.pareto(b=2.5), "Pareto (α = 3)": scipy_stats.pareto(b=3.0),
}


@pytest.mark.parametrize("dist", list(SCIPY_DISTS))
def test_documented_moments_equal_scipy(dist):
    mean, var = SCIPY_DISTS[dist].stats("mv")
    info = S.DIST_INFO[dist]
    assert info.mean == pytest.approx(float(mean), rel=1e-12, abs=1e-12) and info.var == pytest.approx(float(var), rel=1e-12)
    assert info.median == pytest.approx(float(SCIPY_DISTS[dist].median()), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("dist", list(SCIPY_DISTS))
def test_generator_matches_the_scipy_distribution_function(dist):
    x = S.batch(dist, 1, 100000, S.Stream(S.cell_seed(3, dist)))[0]
    assert scipy_stats.kstest(x, SCIPY_DISTS[dist].cdf).statistic < 0.0075                # kritischer Wert (α = 1 %) bei n = 100 000: 0,0051


def test_symmetric_pareto_generator_matches_its_distribution_function_and_variance():
    pareto = scipy_stats.pareto(b=2.5)
    x = S.batch("Symmetrisch mit Pareto-Rand (α = 2,5)", 1, 200000, S.Stream(33))[0]
    cdf = lambda v: 0.5 + np.sign(v) * 0.5 * pareto.cdf(np.abs(v) + 1.0)                 # |X| = Pareto − 1, Vorzeichen zufällig
    assert scipy_stats.kstest(x, cdf).statistic < 0.0075
    second_moment_of_pareto = float(pareto.var()) + float(pareto.mean()) ** 2
    assert S.DIST_INFO["Symmetrisch mit Pareto-Rand (α = 2,5)"].var == pytest.approx(second_moment_of_pareto - 2 * float(pareto.mean()) + 1.0, rel=1e-9)         # E(V − 1)² = E V² − 2 E V + 1


def test_measured_normal_mean_quantile_equals_z_over_sqrt_n():
    """Kontrolle der Auswertung: bei Normal ist |Mittel| halbnormal, Q(1 − δ) = z_{1−δ/2}/√n (Streuung der Messung über 160 000 Läufe unter 1 %)."""
    import mom_results as R
    for n in (200, 1000):
        for delta in (0.5, 0.1, 0.05, 0.01):
            assert R.q("Normal", n, "mean", delta) == pytest.approx(scipy_stats.norm.ppf(1 - delta / 2) / np.sqrt(n), rel=0.03)


def test_measured_median_bias_equals_the_closed_form_at_large_n():
    """Der Stichprobenmedian zielt auf den Median der Verteilung: relative Verzerrung gegen den Mittelwert = Median/Mittelwert − 1."""
    import mom_results as R
    for dist in ("Lognormal (σ = 1)", "Lognormal (σ = 1,5)", "Pareto (α = 3)", "Pareto (α = 2,5)"):
        mean, median = S.population(dist)
        assert R.bias(dist, 5000, "median") == pytest.approx(median / mean - 1.0, abs=0.01)
    assert S.population("Lognormal (σ = 1,5)")[1] / S.population("Lognormal (σ = 1,5)")[0] - 1 == pytest.approx(np.exp(-1.125) - 1)            # e^(−σ²/2) − 1
    assert S.population("Pareto (α = 3)")[1] / S.population("Pareto (α = 3)")[0] - 1 == pytest.approx(2 ** (1 / 3) * 2 / 3 - 1)               # 2^(1/α)·(α − 1)/α − 1


def test_expected_mom_bias_has_the_sign_and_order_of_the_blocks_for_lognormal():
    """Mini-Rechnung ohne Messreihe: der Median von K Blockmitteln einer rechtsschiefen Verteilung liegt unter dem Mittelwert, und das umso mehr, je mehr Blöcke (je kleiner die Blöcke)."""
    x = S.batch("Lognormal (σ = 1,5)", 4000, 200, S.Stream(41))
    biases = [float(E.median_of_means(x, K).mean()) for K in (1, 4, 16, 64)]
    assert biases[0] > biases[1] > biases[2] > biases[3] and biases[3] < S.population("Lognormal (σ = 1,5)")[0]
