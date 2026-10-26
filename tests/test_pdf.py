"""PDF-Export: gültige Datei, alle Zeichen aus dem Latin-1-Satz, Kernschriften stürzen nicht ab."""

import mom_constants as C
import mom_evaluation as V
from mom_pdf_export import _latin, build_pdf, estimate_rows


def test_pdf_is_built_for_every_preset():
    for name, preset in C.PRESETS.items():
        a = V.analyse(V.Settings(**preset))
        data = build_pdf(a, [V.verdict_main(a)])
        assert data[:5] == b"%PDF-" and len(data) > 1500, name


def test_latin_replaces_special_characters_and_markdown():
    assert _latin("p = ∞ → „x“ ≥ 2 · 3 – 4 **fett** `c` ε α λ δ μ ⌈x⌉") == 'p = unendlich -> "x" >= 2 * 3 - 4 fett c eps alpha lambda delta mu [x]'
    assert _latin("Größe ä ö ü ß") == "Größe ä ö ü ß"


def test_estimate_rows_name_every_estimator_with_its_error():
    a = V.analyse(V.Settings())
    rows = estimate_rows(a)
    assert [r[0] for r in rows] == ["Mittel", "Median", "Trimmen", "MoM"] and all(r[2] == a.errors[r[0]] for r in rows)
