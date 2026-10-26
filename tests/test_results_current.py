"""Die Messdatei ist aktuell gegenüber den Quelldateien der Messung. Eigene Datei, damit das Mutationswerkzeug sie auslassen kann (`--suite-ignore test_results_current.py`):
jede Mutation an einer hash-gedeckten Quelldatei ließe sie sonst fehlschlagen und würde als gefangen gezählt, ohne dass ein Test die Wirkung der Mutation prüft."""

import mom_experiments as X
import mom_grid as G
import mom_results as R


def test_results_file_is_current_with_the_generator_sources():
    """Schlägt fehl, sobald sich eine der Quelldateien ändert (mom_rng, mom_scenario, mom_estimators, mom_grid, mom_experiments, tools/make_results.py):
    dann `sh tools/run_parts.sh` (alle Teile und Zusammenfügen) neu laufen lassen."""
    assert R.meta()["erzeuger_hash"] == X.source_hash()
    assert R.meta()["quellen"] == list(X.SOURCES) and R.meta()["seeds"] == list(G.SEEDS)
