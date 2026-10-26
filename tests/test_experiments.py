"""Messfunktionen: die Läufe werden genau gezählt (Stapelgrenzen), das Ergebnis hängt bei geradem n nicht von der Stapelgröße ab, Prüfsumme und Aktualität gegen eine unabhängige Rechnung."""

import hashlib

import numpy as np
import pytest

import mom_experiments as X
import mom_grid as G
import mom_results as R


def _count_rows(monkeypatch, fn, *args):
    seen = []
    original = X.S.batch

    def counting(dist, runs, n, stream):
        seen.append(runs)
        return original(dist, runs, n, stream)
    monkeypatch.setattr(X.S, "batch", counting)
    fn(*args)
    return seen


@pytest.mark.parametrize("n, runs", [(50, 7), (5000, 450), (5000, 800), (5000, 400), (200, 1)])
def test_measure_cell_draws_exactly_the_requested_runs_in_chunks_of_the_documented_size(monkeypatch, n, runs):
    seen = _count_rows(monkeypatch, X.measure_cell, "Normal", n, runs, (1, 2), ["mean"])
    step = X.CHUNK_VALUES // n
    per_seed = [step] * (runs // step) + ([runs % step] if runs % step else [])               # volle Stapel, dann der Rest
    assert seen == per_seed * 2


@pytest.mark.parametrize("n, runs", [(50, 7), (5000, 450)])
def test_measure_intervals_draws_exactly_the_requested_runs(monkeypatch, n, runs):
    seen = _count_rows(monkeypatch, X.measure_intervals, "Normal", n, runs, (1,))
    step = X.CHUNK_VALUES // n
    assert seen == [step] * (runs // step) + ([runs % step] if runs % step else [])


def test_result_does_not_depend_on_the_chunk_size_for_even_n(monkeypatch):
    big = X.measure_cell("t(3)", 50, 300, (1,), ["mean", "mom:5", "trim:2"])
    monkeypatch.setattr(X, "CHUNK_VALUES", 50 * 64)
    small = X.measure_cell("t(3)", 50, 300, (1,), ["mean", "mom:5", "trim:2"])
    assert big["rows"] == small["rows"]
    iv_big = X.measure_intervals("Normal", 50, 300, (1,))
    monkeypatch.setattr(X, "CHUNK_VALUES", 50 * 64)
    assert iv_big["rows"] == X.measure_intervals("Normal", 50, 300, (1,))["rows"]


def test_source_hash_is_the_documented_sha256_over_names_and_normalised_sources():
    h = hashlib.sha256()
    for name in X.SOURCES:
        text = (X.ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    assert X.source_hash() == h.hexdigest()[:16] and len(X.source_hash()) == 16
    assert R.is_current() == (R.meta()["erzeuger_hash"] == h.hexdigest()[:16])


def test_rmse_is_the_stored_mean_and_at_least_the_absolute_bias():
    for spec in ("mean", "median", "mom:37"):
        stored = R.row("Lognormal (σ = 1,5)", 200, spec)["rmse"]
        assert R.rmse("Lognormal (σ = 1,5)", 200, spec) == pytest.approx(float(np.mean(stored)))
        assert R.rmse("Lognormal (σ = 1,5)", 200, spec) >= abs(R.bias("Lognormal (σ = 1,5)", 200, spec)) - 1e-9
    assert R.rmse("Lognormal (σ = 1,5)", 200, "median", "Median") == pytest.approx(float(np.mean(R.row("Lognormal (σ = 1,5)", 200, "median")["rmse_med"])))


def test_interval_bounds_reject_unknown_methods_and_undefined_sign_intervals():
    x = np.arange(1.0, 51.0)[None, :] + np.arange(3)[:, None]
    assert X.interval_bounds(x, 0.1, "vorzeichen") is not None and X.interval_bounds(x[:, :6], 0.1, "vorzeichen") is None
    with pytest.raises(ValueError):
        X.interval_bounds(x, 0.1, "bootstrap")
    with pytest.raises(ValueError):
        X.estimate_specs(x, ["huber"])
