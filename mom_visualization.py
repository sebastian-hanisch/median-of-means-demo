"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Beschriftungen von Linien stehen in der Legende (Spuren), Text auf Heatmaps als Annotation der Heatmap selbst (texttemplate), nie als Text-Spur."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mom_constants as C
import mom_evaluation as V
import mom_experiments as X
import mom_grid as G
import mom_results as R
import mom_scenario as S
from mom_estimators import k_of_delta


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.3, yanchor="top") if legend_below else dict())
    return lock_axes(fig)


def _positive(x) -> bool:
    return bool(np.all(np.asarray(x) > 0))


# --- Hauptansicht: eine Stichprobe ----------------------------------------------------------------------------------------------------------

def build_blocks(a: V.Analysis):
    """Die Stichprobe in Reihenfolge des Stroms, farbig nach Block; waagerechte Striche sind die Blockmittel, die Linie der Median der Blockmittel (= Median-of-Means).
    Bei positiven Werten logarithmische y-Achse (sonst bestimmen wenige Ausreißer die ganze Achse)."""
    x, labels, K = a.x, a.labels, a.K
    n = len(x)
    colors = [C.BLOCK_COLORS[int(b) % len(C.BLOCK_COLORS)] for b in labels]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, n + 1), y=x, mode="markers", marker=dict(color=colors, size=6 if n <= 400 else 4), name="Werte (Farbe: Block)",
                             hovertemplate="Nr. %{x}: %{y:.3f}<extra></extra>"))
    edges = np.concatenate(([0], np.cumsum(np.bincount(labels, minlength=K))))
    xs, ys = [], []
    for b in range(K):
        xs += [edges[b] + 0.5, edges[b + 1] + 0.5, None]
        ys += [a.block_means[b], a.block_means[b], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="black", width=3), name=f"Blockmittel ({K} Blöcke)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0.5, n + 0.5], y=[a.values["MoM"]] * 2, mode="lines", line=dict(color=C.SPEC_COLORS["MoM"], width=3, dash="dash"),
                             name="Median der Blockmittel (MoM)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0.5, n + 0.5], y=[a.values["Mittel"]] * 2, mode="lines", line=dict(color=C.SPEC_COLORS["Mittel"], width=2, dash="dot"), name="Mittel", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0.5, n + 0.5], y=[a.mean_true] * 2, mode="lines", line=dict(color=C.COLOR_TARGET, width=2), name="wahrer Mittelwert", hoverinfo="skip"))
    if _positive(x) and _positive(a.block_means):
        fig.update_yaxes(type="log")
    fig.update_xaxes(title_text="Nr. der Sendung im Strom")
    fig.update_yaxes(title_text="Wert" + (" (log)" if _positive(x) else ""))
    return _base(fig, 420)


def build_location(a: V.Analysis):
    """Verteilung der Stichprobe (Histogramm) mit den vier Schätzern und dem wahren Mittelwert und Median als senkrechte Linien."""
    x = a.x
    fig = go.Figure()
    if _positive(x):
        edges = np.logspace(np.log10(x.min()), np.log10(x.max()), 41)
        counts, _ = np.histogram(x, bins=edges)
        fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges), marker_color="rgba(120,120,120,0.45)", name="Verteilung der Stichprobe", hoverinfo="skip"))
        fig.update_xaxes(type="log", title_text="Wert (log)")
    else:
        lo, hi = np.percentile(x, [1, 99])
        pad = 0.15 * (hi - lo)
        counts, edges = np.histogram(x, bins=41, range=(lo - pad, hi + pad))
        fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges), marker_color="rgba(120,120,120,0.45)", name="Verteilung der Stichprobe", hoverinfo="skip"))
        fig.update_xaxes(range=[lo - pad, hi + pad], title_text="Wert (Ausschnitt 1 % bis 99 % der Stichprobe)")
    top = max(1, int(counts.max()))
    lines = [("Mittel", a.values["Mittel"], C.SPEC_COLORS["Mittel"], "dot"), ("Median", a.values["Median"], C.SPEC_COLORS["Median"], "dot"),
             ("Trimmen", a.values["Trimmen"], C.SPEC_COLORS["Trimmen"], "dash"), ("MoM", a.values["MoM"], C.SPEC_COLORS["MoM"], "solid"),
             ("wahrer Mittelwert", a.mean_true, C.COLOR_TARGET, "solid")]
    for name, value, color, dash in lines:
        fig.add_trace(go.Scatter(x=[value, value], y=[0, top], mode="lines", line=dict(color=color, width=3, dash=dash), name=f"{name}", hovertemplate=f"{name}: %{{x:.3f}}<extra></extra>"))
    fig.update_yaxes(title_text="Anzahl")
    return _base(fig, 340)


# --- E1: Kippkarte --------------------------------------------------------------------------------------------------------------------------

def build_kipp_heatmap(delta: float):
    """Verhältnis Q(1 − δ) von Median-of-Means (Regel) zum Mittel: Verteilungen (Zeilen) × n (Spalten). Unter 1 (grün) ist Median-of-Means besser."""
    z = [[V.kipp_ratio(d, n, delta) for n in G.N_RASTER] for d in S.DISTRIBUTIONS]
    fig = go.Figure(go.Heatmap(z=z, x=[f"n = {n}" for n in G.N_RASTER], y=list(S.DISTRIBUTIONS), zmin=0.4, zmax=1.6, zmid=1.0,
                               colorscale=[[0, C.COLOR_GOOD], [0.5, "#f2f2f2"], [1, C.COLOR_BAD]], text=[[f"{v:.2f}".replace(".", ",") for v in row] for row in z],
                               texttemplate="%{text}", hovertemplate="%{y}, %{x}: %{text}<extra></extra>", colorbar=dict(title="MoM / Mittel", thickness=12)))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 420, legend_below=False)


def build_kipp_curves(dist: str):
    """Für eine Verteilung: das Verhältnis Q(1 − δ) Median-of-Means / Mittel über dem Niveau δ, eine Linie je n."""
    fig = go.Figure()
    fig.add_hline(y=1.0, line=dict(color="black", width=1, dash="dash"))
    palette = ("#9ecae1", "#6baed6", "#3182bd", "#08519c")
    for n, color in zip(G.N_RASTER, palette):
        fig.add_trace(go.Scatter(x=list(G.DELTA_GRID), y=[V.kipp_ratio(dist, n, d) for d in G.DELTA_GRID], mode="lines+markers", line=dict(color=color, width=3), name=f"n = {n}",
                                 hovertemplate="δ = %{x}: %{y:.2f}<extra>n = " + str(n) + "</extra>"))
    fig.update_xaxes(type="log", autorange="reversed", title_text="Niveau δ (nach rechts: selteneres Ereignis)", tickvals=list(G.DELTA_GRID), ticktext=[V.de_delta(d) for d in G.DELTA_GRID])
    fig.update_yaxes(title_text="Q(1 − δ): MoM / Mittel (unter 1: MoM besser)")
    return _base(fig, 360)


# --- E2: Blockzahl K ------------------------------------------------------------------------------------------------------------------------

def build_k_curve(dist: str, n: int, delta: float):
    """Links: Q(1 − δ) über der Blockzahl K (K = 1 ist das Mittel; Markierung: Regel und das im Nachhinein beste K). Rechts: Verzerrung (mittlerer Fehler, relativ) über K."""
    curve = V.k_curve(dist, n, delta)
    ks = [c["K"] for c in curve]
    rule, best = k_of_delta(delta, n), V.best_k(dist, n, delta)[0]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Q(1 − δ) und typischer Fehler", "Verzerrung (mittlerer Fehler)"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=ks, y=[c["q"] for c in curve], mode="lines+markers", line=dict(color=C.COLOR_LINE, width=3), name="Q(1 − δ)", hovertemplate="K = %{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=[c["typ"] for c in curve], mode="lines+markers", line=dict(color=C.COLOR_LINE2, width=2, dash="dot"), name="typischer Fehler", hovertemplate="K = %{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=[c["bias"] for c in curve], mode="lines+markers", line=dict(color=C.COLOR_BAD, width=3), name="Verzerrung", hovertemplate="K = %{x}: %{y:.3f}<extra></extra>"), row=1, col=2)
    for K, label, color in ((rule, "Regel", C.SPEC_COLORS["MoM"]), (best, "bestes K", C.COLOR_GOOD)):
        row = next((c for c in curve if c["K"] == K), None)
        if row is not None:
            fig.add_trace(go.Scatter(x=[K], y=[row["q"]], mode="markers", marker=dict(color=color, size=14, symbol="diamond", line=dict(color="black", width=1)), name=f"{label}: K = {K}",
                                     hovertemplate=f"{label}: K = {K}<extra></extra>"), row=1, col=1)
    fig.update_xaxes(type="log", title_text="Blockzahl K")
    fig.update_yaxes(title_text="Fehler", row=1, col=1)
    fig.update_yaxes(title_text="mittlerer Fehler" + (" (relativ)" if not S.DIST_INFO[dist].symmetric else ""), zeroline=True, row=1, col=2)
    return _base(fig, 380)


# --- E3: typisch gegen Worst-Case ------------------------------------------------------------------------------------------------------------

def build_typ_vs_worst(dist: str, n: int, delta: float, c: int):
    """Gruppiert: typischer Fehler und Q(1 − δ) für Mittel, Median, Trimmen und Median-of-Means (Regel)."""
    specs = {"Mittel": "mean", "Median": "median", "Trimmen": V.trim_spec(c, delta, n), "MoM": V.mom_spec(delta, n)}
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(specs), y=[R.typical(dist, n, s) for s in specs.values()], name="typischer Fehler (Median des Fehlers)", marker_color="rgba(76,120,168,0.75)",
                         hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=list(specs), y=[R.q(dist, n, s, delta) for s in specs.values()], name=f"Q(1 − δ), δ = {V.de_delta(delta)}", marker_color="rgba(214,39,40,0.75)",
                        hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Fehler" + (" (relativ zum Mittelwert)" if not S.DIST_INFO[dist].symmetric else ""))
    return _base(fig, 340)


# --- E4: Dreier-Wahl --------------------------------------------------------------------------------------------------------------------------

def build_winner_map(delta: float, c: int):
    """Sieger der Dreier-Wahl (Mittel, MoM, Trimmen) je Verteilung und n: kleinstes Q(1 − δ)."""
    names = ("Mittel", "MoM", "Trimmen")
    z = [[names.index(V.three_way(d, n, delta, c)["Sieger"]) for n in G.N_RASTER] for d in S.DISTRIBUTIONS]
    text = [[names[v] for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=[f"n = {n}" for n in G.N_RASTER], y=list(S.DISTRIBUTIONS), zmin=-0.5, zmax=2.5, text=text, texttemplate="%{text}", showscale=False,
                               colorscale=[[0, C.SPEC_COLORS["Mittel"]], [1 / 3, C.SPEC_COLORS["Mittel"]], [1 / 3, C.SPEC_COLORS["MoM"]], [2 / 3, C.SPEC_COLORS["MoM"]],
                                           [2 / 3, C.SPEC_COLORS["Trimmen"]], [1, C.SPEC_COLORS["Trimmen"]]], hovertemplate="%{y}, %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 420, legend_below=False)


# --- E5: Zielgröße ----------------------------------------------------------------------------------------------------------------------------

def build_target(dist: str, n: int, delta: float, c: int):
    """Typischer Fehler gegen das Ziel Mittelwert und gegen das Ziel Median (nur schiefe Verteilungen)."""
    specs = {"Mittel": "mean", "Median": "median", "Trimmen": V.trim_spec(c, delta, n), "MoM": V.mom_spec(delta, n)}
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(specs), y=[R.typical(dist, n, s, "Mittelwert") for s in specs.values()], name="Ziel: Mittelwert", marker_color="rgba(76,120,168,0.75)", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=list(specs), y=[R.typical(dist, n, s, "Median") for s in specs.values()], name="Ziel: Median", marker_color="rgba(89,161,79,0.75)", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="typischer Fehler (relativ zum Ziel)")
    return _base(fig, 340)


# --- E6: Intervalle ---------------------------------------------------------------------------------------------------------------------------

def build_interval_coverage(dist: str, delta: float):
    """Überdeckung des wahren Mittelwerts je Verfahren über n; die gestrichelte Linie ist das Soll 1 − δ."""
    fig = go.Figure()
    fig.add_hline(y=1 - delta, line=dict(color="black", width=1, dash="dash"))
    palette = {"student": "#7f7f7f", "vorzeichen": "#59a14f", "mom_sigma:8": "#e07b00", "mom_sigma:32": "#d62728"}
    for m in X.interval_methods():
        ns = [n for n in G.INTERVAL_NS if R.coverage(dist, n, m, delta) is not None]
        fig.add_trace(go.Scatter(x=[f"n = {n}" for n in ns], y=[100 * R.coverage(dist, n, m, delta) for n in ns], mode="lines+markers", line=dict(color=palette[m], width=3),
                                 name=V.METHOD_LABELS[m], hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text="Überdeckung des wahren Mittelwerts (%)", range=[max(0, 100 * (1 - delta) - 40), 100.5])
    return _base(fig, 380)
