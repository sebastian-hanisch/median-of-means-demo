"""Szenario: der blockweise berechnete Zufallsstrom stimmt mit der Python-Klasse überein, Verteilungen haben die dokumentierten Lagen, Fehlerdefinition und Kopie von SplitMix64."""

from pathlib import Path

import numpy as np
import pytest

import mom_scenario as S
from mom_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"mom_rng.py": ("reduktionskette-demo", "red_scenario.py")}


@pytest.mark.parametrize("copy", list(SOURCES))
def test_copy_is_verbatim_apart_from_the_header(copy):
    """Der kopierte Baustein (mom_rng.py, Entscheidung der Reihe: wörtlich kopieren, Quelle im Kopf der Datei): Vergleich mit dem Original, solange es im Portfolio-Ordner liegt."""
    folder, original = SOURCES[copy]
    source = ROOT.parent / folder / original
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    mine = (ROOT / copy).read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """Die Kopie läuft für sich: SplitMix64 mit Seed 0 liefert die bekannten ersten Werte der Konvention (Vigna)."""
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]
    assert SplitMix64(7).below(10) == SplitMix64(7).below(10) < 10


def test_block_stream_equals_the_python_class_value_by_value():
    for seed in (0, 7, 123456789, 2 ** 63 + 5):
        rng = SplitMix64(seed)
        expected = [rng.next() for _ in range(50)]
        assert [int(v) for v in S.mix_words(seed, 0, 50)] == expected
        assert [int(v) for v in S.mix_words(seed, 20, 10)] == expected[20:30]


def test_stream_continues_where_it_stopped():
    a, b = S.Stream(9), S.Stream(9)
    first = np.concatenate([a.uniform(7), a.uniform(5)])
    assert first == pytest.approx(b.uniform(12), abs=0)
    assert a.pos == 12


def test_uniform_ranges_and_normal_moments():
    u = S.Stream(3).uniform_open(100000)
    assert 0 < u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.01
    z = S.Stream(4).normal(200000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01 and len(S.Stream(4).normal(7)) == 7


def test_cell_seed_is_stable_and_distinguishes_tags():
    assert S.cell_seed(5, "a", 1.5) == S.cell_seed(5, "a", 1.5)
    assert len({S.cell_seed(5, "a", 1), S.cell_seed(5, "b", 1), S.cell_seed(5, "a", 2), S.cell_seed(6, "a", 1)}) == 4


def test_ten_distributions_four_symmetric_and_documented_moments():
    assert len(S.DISTRIBUTIONS) == 10 and S.SYMMETRIC == ("Normal", "t(5)", "t(3)", "Symmetrisch mit Pareto-Rand (α = 2,5)") and len(S.SKEWED) == 6
    assert S.DIST_INFO["Lognormal (σ = 1,5)"].mean == pytest.approx(np.exp(1.125)) and S.DIST_INFO["Lognormal (σ = 1,5)"].median == 1.0
    assert S.DIST_INFO["Pareto (α = 3)"].mean == pytest.approx(1.5) and S.DIST_INFO["Pareto (α = 3)"].median == pytest.approx(2 ** (1 / 3))
    assert S.DIST_INFO["Pareto (α = 3)"].var == pytest.approx(0.75)                       # 3 / (2² · 1)
    assert S.DIST_INFO["t(3)"].var == 3.0 and S.DIST_INFO["Symmetrisch mit Pareto-Rand (α = 2,5)"].var == pytest.approx(8 / 3)
    assert all(d.var is not None for d in S.DIST_INFO.values()) and all(d.vehicle for d in S.DIST_INFO.values())


@pytest.mark.parametrize("dist", S.DISTRIBUTIONS)
def test_every_distribution_has_the_documented_median(dist):
    x = S.batch(dist, 1, 400000, S.Stream(S.cell_seed(1, dist)))[0]
    assert np.median(x) == pytest.approx(S.population(dist)[1], abs=0.02)


def test_pareto_has_minimum_one_and_the_tail_exponent():
    x = S.draw("pareto", 3.0, S.Stream(8), 200000)
    assert x.min() >= 1.0 and np.mean(x > 2.0) == pytest.approx(2.0 ** -3, abs=0.005)


def test_symmetric_pareto_tail_has_both_signs_density_at_zero_and_the_tail_exponent():
    x = S.draw("pareto_sym", 2.5, S.Stream(8), 200000)
    assert abs(np.mean(x > 0) - 0.5) < 0.01 and np.mean(np.abs(x) < 0.1) > 0.1          # kein Loch um 0
    assert np.mean(np.abs(x) > 1.0) == pytest.approx(2.0 ** -2.5, abs=0.005)          # P(|X| > t) = (1 + t)^(−α)


def test_unknown_distribution_is_rejected():
    with pytest.raises(ValueError):
        S.draw("cauchy", 1.0, S.Stream(1), 3)


def test_batches_and_samples_are_reproducible_and_independent_of_other_cells():
    a = S.batch("t(3)", 5, 20, S.Stream(4))
    assert np.array_equal(a, S.batch("t(3)", 5, 20, S.Stream(4))) and a.shape == (5, 20)
    assert not np.array_equal(a, S.batch("t(3)", 5, 20, S.Stream(5)))
    x = S.one_sample("Lognormal (σ = 1)", 50, 7)
    assert np.array_equal(x, S.one_sample("Lognormal (σ = 1)", 50, 7)) and not np.array_equal(x, S.one_sample("Lognormal (σ = 1)", 50, 8))
    assert not np.array_equal(x, S.one_sample("Lognormal (σ = 1,5)", 50, 7))


def test_error_scale_is_absolute_for_symmetric_and_relative_for_skewed():
    assert S.error_scale("Normal") == (0.0, 1.0) and S.error_scale("t(3)", "Median") == (0.0, 1.0)
    mu, med = S.DIST_INFO["Lognormal (σ = 1,5)"].mean, 1.0
    assert S.error_scale("Lognormal (σ = 1,5)") == (mu, mu) and S.error_scale("Lognormal (σ = 1,5)", "Median") == (med, med)
    with pytest.raises(ValueError):
        S.error_scale("Lognormal (σ = 1,5)", "Modus")
