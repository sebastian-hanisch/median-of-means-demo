"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def _set(at, key, value):
    for art in ("selectbox", "select_slider", "slider", "number_input"):
        try:
            getattr(at.sidebar, art)(key=key).set_value(value)
            return
        except KeyError:
            continue
    raise AssertionError(key)


def test_corner_matrix_of_the_controls():
    """Jeder Regler auf Minimum und Maximum (einzeln), dazu alle Minima und alle Maxima zusammen (DEMO-PLAYBOOK §1d)."""
    import mom_constants as C
    from mom_scenario import DISTRIBUTIONS
    lo = {"dist_select": DISTRIBUTIONS[0], "n_slider": C.N_MIN, "delta_slider": C.DELTA_OPTIONS[0], "kopt_slider": C.K_OPTIONS[0], "c_slider": C.C_CHOICES[0], "seed_input": 0}
    hi = {"dist_select": DISTRIBUTIONS[-1], "n_slider": C.N_MAX, "delta_slider": C.DELTA_OPTIONS[-1], "kopt_slider": C.K_OPTIONS[-1], "c_slider": C.C_CHOICES[-1], "seed_input": C.SEED_MAX}
    runs = [(k, v) for k in lo for v in (lo[k], hi[k])]
    for key, value in runs:
        at = _fresh_app()
        _set(at, key, value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (key, value, [str(e) for e in at.exception])
    for combo in (lo, hi):
        at = _fresh_app()
        for key, value in combo.items():
            _set(at, key, value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, ([str(e) for e in at.exception])


def test_every_block_option_runs_and_live_measurement_and_pdf_work():
    import mom_constants as C
    at = _fresh_app()
    for option in C.K_OPTIONS:
        _set(at, "kopt_slider", option)
        at.run(timeout=TIMEOUT)
        assert not at.exception, option
    at.button(key="live_go").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Läufe (4 Seeds)" in m.value for m in list(at.info) + list(at.success))
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception and at.get("download_button")
