"""Abbildungen: jede Figur lässt sich für jede Verteilung bauen, alle Achsen sind gesperrt, Beschriftungen stehen in der Legende statt als Text-Spur."""

import numpy as np
import pytest

import mom_evaluation as V
import mom_grid as G
import mom_scenario as S
import mom_visualization as W


def _locked(fig):
    return all(ax.fixedrange for ax in (fig.layout[k] for k in fig.layout if k.startswith(("xaxis", "yaxis"))))


@pytest.mark.parametrize("dist", S.DISTRIBUTIONS)
def test_every_figure_builds_for_every_distribution(dist):
    a = V.analyse(V.Settings(dist=dist))
    figs = [W.build_blocks(a), W.build_location(a), W.build_kipp_curves(dist), W.build_k_curve(dist, 200, 0.01), W.build_typ_vs_worst(dist, 200, 0.01, 1),
            W.build_target(dist, 200, 0.01, 1), W.build_interval_coverage(dist, 0.05)]
    for fig in figs:
        assert len(fig.data) >= 1 and _locked(fig)


def test_heatmaps_cover_all_distributions_and_grid_sizes():
    for fig in (W.build_kipp_heatmap(0.01), W.build_winner_map(0.01, 1)):
        (heat,) = fig.data
        assert list(heat.y) == list(S.DISTRIBUTIONS) and len(heat.x) == len(G.N_RASTER) and _locked(fig)
    z = np.asarray(W.build_kipp_heatmap(0.01).data[0].z)
    assert z.shape == (10, 4) and z[S.DISTRIBUTIONS.index("Normal")].min() > 1.1
    winners = np.asarray(W.build_winner_map(0.01, 1).data[0].z)
    assert set(np.unique(winners)) <= {0, 1, 2}


def test_block_figure_uses_log_axis_only_for_positive_values_and_draws_one_segment_per_block():
    pos = V.analyse(V.Settings(dist="Lognormal (σ = 1,5)", kopt="8"))
    fig = W.build_blocks(pos)
    assert fig.layout.yaxis.type == "log"
    segments = [t for t in fig.data if t.name and t.name.startswith("Blockmittel")][0]
    assert sum(1 for v in segments.x if v is None) == 8
    assert W.build_blocks(V.analyse(V.Settings(dist="t(3)"))).layout.yaxis.type != "log"


def test_k_curve_marks_rule_and_best_k_and_axes_are_logarithmic_in_k():
    fig = W.build_k_curve("Lognormal (σ = 1,5)", 200, 0.01)
    names = [t.name for t in fig.data]
    assert "Regel: K = 37" in names and "bestes K: K = 8" in names and fig.layout.xaxis.type == "log"
    assert all(t.mode != "text" for t in fig.data if hasattr(t, "mode"))


def test_interval_figure_skips_methods_without_values():
    fig = W.build_interval_coverage("Normal", 0.1)
    assert len(fig.data) == 4 and all(len(t.x) == 4 for t in fig.data)
