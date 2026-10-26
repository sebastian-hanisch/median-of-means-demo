"""Einheiten der Schätzer an den Mini-Instanzen von Hand (tests/conftest.py): Blockbildung, Median-of-Means, Trimmen, Blockzahl-Regel, Vorzeichen-Intervall."""

import math

import numpy as np
import pytest

import mom_estimators as E


def test_block_sizes_distribute_the_remainder_to_the_first_blocks():
    assert list(E.block_sizes(10, 4)) == [3, 3, 2, 2]
    assert list(E.block_sizes(6, 3)) == [2, 2, 2] and list(E.block_sizes(7, 7)) == [1] * 7 and list(E.block_sizes(5, 1)) == [5]
    assert list(E.block_sizes(11, 3)) == [4, 4, 3]


@pytest.mark.parametrize("n, K", [(5, 0), (5, 6), (5, -1)])
def test_block_sizes_reject_invalid_k(n, K):
    with pytest.raises(ValueError):
        E.block_sizes(n, K)


def test_block_edges_and_labels():
    assert list(E.block_edges(10, 4)) == [0, 3, 6, 8]
    assert list(E.block_labels(10, 4)) == [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]


def test_block_means_by_hand(hand_x):
    assert E.block_means(hand_x, 3)[0] == pytest.approx([1.5, 3.5, 52.5])


def test_median_of_means_by_hand(hand_x, pairs_x):
    assert E.median_of_means(hand_x, 3) == pytest.approx(3.5)                 # Blockmittel 1,5 | 3,5 | 52,5
    assert E.median_of_means(pairs_x, 4) == pytest.approx(9.0)                # Blockmittel 3 | 7 | 11 | 107: (7 + 11)/2
    assert hand_x.mean() == pytest.approx(115 / 6)                            # das Mittel läuft dem Ausreißer hinterher, MoM nicht


def test_k_equal_one_is_the_mean_and_k_equal_n_is_the_median(hand_x):
    assert E.median_of_means(hand_x, 1) == pytest.approx(hand_x.mean())
    assert E.median_of_means(hand_x, 6) == pytest.approx(np.median(hand_x))
    odd = np.array([5.0, 1.0, 9.0, 2.0, 7.0])
    assert E.median_of_means(odd, 5) == pytest.approx(5.0)


def test_stack_of_rows_gives_one_value_per_row(hand_x):
    stack = np.vstack([hand_x, hand_x + 1.0, hand_x * 2.0])
    assert E.median_of_means(stack, 3) == pytest.approx([3.5, 4.5, 7.0])
    assert isinstance(E.median_of_means(hand_x, 3), float)


def test_trimmed_mean_by_hand(trim_x):
    assert E.trimmed_mean(trim_x, 1) == pytest.approx(8 / 3)
    assert E.trimmed_mean(trim_x, 0) == pytest.approx(4.0)                    # k = 0 ist das Mittel
    assert E.trimmed_mean(trim_x, 2) == pytest.approx(2.0)                    # k = (n − 1)/2 ist der Median
    assert E.trimmed_mean(np.array([12.0, 0.0, 5.0, 2.0, 1.0]), 1) == pytest.approx(8 / 3)   # sortiert wird in der Funktion


@pytest.mark.parametrize("k", [-1, 3, 5])
def test_trimmed_mean_rejects_k_that_leaves_nothing(trim_x, k):
    with pytest.raises(ValueError):
        E.trimmed_mean(trim_x, k)


def test_rule_k_of_delta_by_hand():
    assert [E.k_of_delta(d, 200) for d in (0.5, 0.05, 0.01, 0.001)] == [6, 24, 37, 56]
    assert E.k_of_delta(0.01, 50) == 25 and E.k_of_delta(0.001, 50) == 25     # gedeckelt bei n/2: beide Niveaus liefern dasselbe K, kein Fehler
    assert E.k_of_delta(0.999, 200) == 1                                      # mindestens ein Block
    assert E.k_of_delta(0.01, 1) == 1 and E.k_of_delta(0.01, 3) == 1          # n/2 < 1: ein Block


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.5])
def test_rule_rejects_delta_outside_the_open_interval(delta):
    with pytest.raises(ValueError):
        E.k_of_delta(delta, 100)
    with pytest.raises(ValueError):
        E.trim_count(1, delta, 100)


def test_trim_count_by_hand():
    assert [E.trim_count(c, 0.01, 200) for c in (1, 2, 4)] == [5, 10, 19]
    assert E.trim_count(4, 0.001, 50) == 24                                   # höchstens (n − 1) // 2: es bleibt ein Wert
    assert E.trim_count(1, 0.999, 200) == 1                                   # ⌈0,001⌉ = 1
    assert E.trim_count(4, 0.01, 3) == 1 and E.trim_count(1, 0.01, 1) == 0


def test_sign_interval_rank_by_hand():
    l, cover = E.sign_interval_rank(24, 0.05)
    assert l == 7 and cover == pytest.approx(1 - 2 * 190051 / 2 ** 24)
    assert E.sign_interval_rank(4, 0.05) is None                              # zu wenige Blöcke: P(Bin(4; ½) = 0) = 1/16 > 0,025
    l5, c5 = E.sign_interval_rank(8, 0.05)
    assert (l5, c5) == (1, pytest.approx(1 - 2 / 256))                        # P(Bin(8; ½) = 0) = 1/256 ≤ 0,025, P(≤ 1) = 9/256 > 0,025


def test_sign_interval_uses_the_l_th_smallest_and_largest_block_mean():
    rng_x = np.arange(1.0, 241.0)[None, :]                                     # K = 24 Blöcke zu je 10: Blockmittel 5,5; 15,5; …
    lo, hi = E.sign_interval(rng_x, 24, 0.05)
    assert lo[0] == pytest.approx(5.5 + 10 * 6) and hi[0] == pytest.approx(5.5 + 10 * 17)       # 7. kleinstes und 7. größtes: Rang 7 bzw. 18
    assert E.sign_interval(rng_x, 4, 0.05) is None


def test_estimate_all_on_the_hand_sample(hand_x):
    r = E.estimate_all(hand_x, 0.05, c=1.0, K=3)
    assert r["K"] == 3 and r["MoM"] == pytest.approx(3.5) and r["Mittel"] == pytest.approx(115 / 6) and r["Median"] == pytest.approx(3.5)
    assert r["k"] == 2 and r["Trimmen"] == pytest.approx((3.0 + 4.0) / 2)      # n = 6: k = ⌈ln 20⌉ = 3, gedeckelt bei (6 − 1) // 2 = 2
    assert E.estimate_all(hand_x, 0.05)["K"] == 3                              # Regel 24, gedeckelt bei n/2 = 3


def test_math_helper_ceil_is_not_off_by_float():
    assert E.k_of_delta(math.exp(-3), 1000) == 24                              # 8 · 3 = 24 genau; math.ceil(24.000000000000004) wäre 25


def test_sign_interval_rank_near_the_median_and_at_exact_ties():
    """Randfälle der Schleife: δ nahe 1 reicht bis dicht an den Median (K = 10, δ = 0,9: P(Bin ≤ 4) = 386/1024 = 0,377 ≤ 0,45, P(Bin ≤ 5) = 638/1024 = 0,623 > 0,45, also l = 5, Überdeckung 1 − 2 · 0,377 = 0,246);
    ein Gleichstand cdf = δ/2 gilt als erlaubt (K = 2, δ = 0,5: P(Bin(2; ½) = 0) = 0,25 = δ/2, also l = 1 und Überdeckung 0,5)."""
    l, cover = E.sign_interval_rank(10, 0.9)
    assert l == 5 and cover == pytest.approx(1 - 2 * 386 / 1024)
    assert E.sign_interval_rank(2, 0.5) == (1, 0.5)
    assert E.sign_interval_rank(1, 0.5) is None                                 # P(Bin(1; ½) = 0) = 0,5 > 0,25
