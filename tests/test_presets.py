"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen/Einrasten) und die Wirkung jedes Presets."""

import pytest

import mom_constants as C
import mom_evaluation as V
import mom_presets as P
import mom_scenario as S


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_for_their_controls():
    for name, preset in C.PRESETS.items():
        assert preset["dist"] in S.DISTRIBUTIONS and C.N_MIN <= preset["n"] <= C.N_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["delta"] in C.DELTA_OPTIONS and preset["kopt"] in C.K_OPTIONS and preset["c"] in C.C_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert V.Settings(**C.PRESETS["Schiefe Mengen (Standard)"]) == V.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_delta_options_are_the_measured_grid():
    import mom_grid as G
    assert tuple(C.DELTA_OPTIONS) == tuple(G.DELTA_GRID)


@pytest.mark.parametrize("state_key, raw, expected", [
    ("n_slider", "99999", C.N_MAX), ("n_slider", "1", C.N_MIN), ("n_slider", "abc", None), ("delta_slider", "0.01", 0.01), ("delta_slider", "0.011", None),
    ("kopt_slider", "n/2", "n/2"), ("kopt_slider", "7", None), ("dist_select", "Normal", "Normal"), ("dist_select", "Cauchy", None), ("c_slider", "2", 2), ("c_slider", "3", None),
    ("seed_input", "-5", 0),
])
def test_permalink_values_are_clamped_or_snapped_or_ignored(state_key, raw, expected):
    assert P.parse_value(P.SETTING_SPECS[state_key], raw) == expected


def test_each_preset_changes_the_main_result_against_the_standard():
    std = V.analyse(V.Settings())
    for name, preset in C.PRESETS.items():
        if name == "Schiefe Mengen (Standard)":
            continue
        a = V.analyse(V.Settings(**preset))
        assert (a.settings != std.settings) and (a.K != std.K or a.k != std.k or a.settings.dist != std.settings.dist or a.settings.n != std.settings.n), name


def test_the_too_many_blocks_preset_uses_n_half():
    a = V.analyse(V.Settings(**C.PRESETS["Zu viele Blöcke"]))
    assert a.K == C.DEFAULT_N // 2 and V.analyse(V.Settings()).K == 37
