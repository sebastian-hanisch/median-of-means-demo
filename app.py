"""Schwere Ränder: Wann zahlt sich Median-of-Means aus? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import mom_constants as C
import mom_evaluation as V
import mom_grid as G
import mom_results as R
import mom_scenario as S
from mom_layout import dunkles_design, kopf_css
from mom_pdf_export import build_pdf
from mom_presets import (
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mom_visualization import (
    build_blocks,
    build_interval_coverage,
    build_k_curve,
    build_kipp_curves,
    build_kipp_heatmap,
    build_location,
    build_target,
    build_typ_vs_worst,
    build_winner_map,
)

st.set_page_config(page_title="Schwere Ränder – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return V.analyse(settings)


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mom_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Schwere Ränder: Wann zahlt sich Median-of-Means aus?")
st.markdown(
    """
Stück 6 der Reihe **Mittelwerte und Lagemaße**. Ein Lager schätzt aus n Sendungen die **mittlere** Menge, Standzeit oder Verspätung. Bei **schweren Rändern** (seltene, sehr große Werte)
kann das Mittel in einzelnen Stichproben weit danebenliegen. **Median-of-Means** teilt die Stichprobe in K Blöcke, mittelt jeden Block und nimmt den Median der Blockmittel: eine
Absicherung gegen seltene große Fehler. Die Demo misst, **wann** sich das auszahlt (Kippkarte über Verteilung, n und Niveau δ), **was** es kostet (typischer Fehler und Verzerrung)
und **ob** ein Trimmen mit wenigen Werten die bessere Absicherung ist. Alle Verteilungen sind erfundene Modelle, keine Messdaten.

Weiter unten: **Nachmessen**, die Experimente **🔬 E1 bis E6**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Median-of-Means", expanded=True):
    st.markdown(
        """
1. **Blöcke bilden:** Die n Werte werden der Reihe nach in K zusammenhängende Blöcke geteilt (die ersten n mod K Blöcke bekommen einen Wert mehr; nichts wird weggeworfen).
2. **Blockmittel:** Jeder Block liefert sein Mittel. Ein einzelner Ausreißer verdirbt höchstens ein Blockmittel.
3. **Median der Blockmittel:** Der Median der K Blockmittel ist der Schätzer. **K = 1 ist das Mittel, K = n der Median.**
4. **Der Preis:** Bei den gemessenen rechtsschiefen Modellen (Lognormal, Pareto) liegt der Median der Blockmittel unter ihrem Mittelwert (Verzerrung), und bei Normal und den schiefen Modellen ist der typische Fehler größer als beim Mittel.
   Die Regel **K = ⌈8 ln(1/δ)⌉** ist eine Faustregel der Vorlage; ihre Herkunft habe ich nicht an einer Quelle geprüft, die Demo behandelt K als Regler und misst.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Verteilung und Stichprobe**")
    dist = st.selectbox("Verteilung", S.DISTRIBUTIONS, key="dist_select",
                        help="Sendungsmenge je Kunde (Lognormal, Einheit = Median), Standzeit an der Rampe (Pareto, Einheit = Mindeststandzeit), Lieferverzug (Normal, t, symmetrisch mit Pareto-Rand; Ziel 0). "
                             "Erfundene Modelle, keine Messdaten.")
    n = st.slider("Stichprobengröße n", C.N_MIN, C.N_MAX, key="n_slider", help="Wie viele Sendungen die Stichprobe der Hauptansicht hat und wie groß die Stichproben beim Nachmessen sind.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", 0, C.SEED_MAX, key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Absicherung**")
    delta = st.select_slider("Niveau δ", options=list(C.DELTA_OPTIONS), key="delta_slider", format_func=V.de_delta,
                             help="Das Fehlerquantil Q(1 − δ) ist der Fehler, den nur ein Anteil δ der Läufe überschreitet. Kleines δ heißt: seltener Ernstfall (1 von 1 000 bei 0,001).")
    kopt = st.select_slider("Blockzahl K", options=list(C.K_OPTIONS), key="kopt_slider",
                            help="„Regel“ ist K = ⌈8 ln(1/δ)⌉ (Faustregel der Vorlage, Herkunft nicht geprüft). Die festen Werte werden bei n/2 gedeckelt (mindestens zwei Werte je Block). K = 1 ist das Mittel.")
    c = st.select_slider("Trimm-Faktor c", options=list(C.C_CHOICES), key="c_slider",
                         help="Trimmen schneidet k = ⌈c ln(1/δ)⌉ Werte je Seite ab. Der Faktor c ist eine eigene Wahl dieser Demo, nicht aus einer Quelle.")

sync_query_params({"dist_select": dist, "n_slider": int(n), "delta_slider": delta, "kopt_slider": kopt, "c_slider": int(c), "seed_input": int(seed)})

settings = V.Settings(dist, int(n), float(delta), kopt, int(c), int(seed))
a = _analysis(settings)
symmetric = S.DIST_INFO[dist].symmetric
grid_n = R.nearest_grid_n(settings.n)
grid_note = "" if grid_n == settings.n else f" (vorgerechnet ist n = {grid_n}, am nächsten zu Ihrem n = {settings.n})"

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Eine Stichprobe")
d_info = S.DIST_INFO[dist]
st.caption(f"Vehikel: {d_info.vehicle}. Wahrer Mittelwert der Verteilung {V.de(a.mean_true, 3)}, wahrer Median {V.de(a.median_true, 3)}. "
           + ("Symmetrisch um 0: Fehler = |Schätzer|." if symmetric else "Schief: Fehler = (Schätzer − Mittelwert) / Mittelwert, negativ heißt zu klein."))
m1, m2, m3, m4 = st.columns(4)
for col, label, key in ((m1, "Mittel", "Mittel"), (m2, "Median", "Median"), (m3, f"Trimmen ({a.k} je Seite)", "Trimmen"), (m4, f"Median-of-Means (K = {a.K})", "MoM")):
    col.metric(label, V.de(a.values[key], 3))
    col.caption(f"Fehler: {V.de(a.errors[key], 3)}")
st.plotly_chart(build_blocks(a), width="stretch", key="blocks_chart")
st.caption("Punkte: die Stichprobe in der Reihenfolge des Stroms, Farbe = Block (bei mehr als 8 Blöcken wiederholen sich die Farben). Schwarze Striche: Blockmittel. "
           "Gestrichelt rot: der Median der Blockmittel = Median-of-Means. Grün: der wahre Mittelwert.")
st.plotly_chart(build_location(a), width="stretch", key="location_chart")
show(V.verdict_main(a))

pdf_key = repr(settings)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf"] = (pdf_key, build_pdf(a, [V.verdict_main(a)]))
if st.session_state.get("pdf", (None, None))[0] == pdf_key:
    st.download_button("⬇️ PDF herunterladen", data=st.session_state["pdf"][1], file_name="median-of-means.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann zahlt sich Median-of-Means aus?")
show(V.verdict_kipp(dist, settings.delta))
show(V.verdict_count())

st.markdown("---")

# --- Nachmessen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔁 Nachmessen mit Ihren Einstellungen")
allowed = V.live_runs_allowed(settings.n)
st.caption(f"Misst Mittel, Median, Trimmen und Median-of-Means auf denselben Stichproben (gepaarter Vergleich) in vier festen Seeds. Bei n = {settings.n} erlaubt das Budget bis zu {allowed} Läufe je Seed.")
runs_options = sorted({r for r in (200, 500, 1000, 2000, 5000) if r <= allowed} | {allowed if allowed < 200 else 200})
runs = st.select_slider("Läufe je Seed", options=runs_options, value=runs_options[-1], key="live_runs")
live_key = (repr(settings), runs)
if st.button("▶ Nachmessen", key="live_go"):
    with st.spinner(f"Messe {runs * len(C.LIVE_SEEDS)} Stichproben mit n = {settings.n} (Sekunden)…"):
        t0 = time.perf_counter()
        st.session_state["live"] = (live_key, V.live_measure(settings, runs), time.perf_counter() - t0)
stored = st.session_state.get("live")
if stored and stored[0] == live_key:
    live = stored[1]
    show(V.verdict_live(settings, live))
    st.caption(f"Rechenzeit {stored[2]:.1f} s.")
    st.dataframe(
        [{"Schätzer": k, "typischer Fehler": V.de(v["typ"], 3), f"Q(1 − δ), δ = {V.de_delta(settings.delta)}": V.de(v["q"], 3), "Verzerrung": V.pct(v["bias"])} for k, v in live["rows"].items()],
        hide_index=True, width="stretch",
    )

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 E1: Kippkarte – bei welchem Niveau schlägt Median-of-Means das Mittel?")
runs_text = f"{G.RUNS_PER_SEED:_}".replace("_", ".")
st.caption(f"Vorgerechnet: {runs_text} Stichproben je Seed und Zelle, vier feste Zufalls-Seeds, gleiche Stichproben für alle Schätzer; Median-of-Means mit der Regel-Blockzahl. "
           "Unter 1 (grün): besser als das Mittel. Das Kippniveau steht in der Tabelle darunter.")
st.plotly_chart(build_kipp_heatmap(settings.delta), width="stretch", key="kipp_heat")
st.plotly_chart(build_kipp_curves(dist), width="stretch", key="kipp_curves")
st.dataframe(
    [{"Verteilung": d, **{f"n = {m}": (V.de_delta(t) if (t := V.tipping_level_of(d, m)) is not None else "–") for m in G.N_RASTER}} for d in S.DISTRIBUTIONS],
    hide_index=True, width="stretch",
)
st.caption("Kippniveau: das größte δ, ab dem Median-of-Means bei diesem und allen kleineren δ des Rasters (bis 0,001) in allen vier Seeds besser ist; „–“: auch bei δ = 0,001 nicht durchgehend.")

st.subheader("🔬 E2: Blockzahl K als Regler – und was die Verzerrung kostet")
st.caption(f"Vorgerechnet für {dist}, n = {grid_n}{grid_note}, δ = {V.de_delta(settings.delta)}. K = 1 ist das Mittel.")
st.plotly_chart(build_k_curve(dist, grid_n, settings.delta), width="stretch", key="k_curve")
show(V.verdict_k(dist, grid_n, settings.delta))
st.dataframe(
    [{"Schätzer": r["Schätzer"], "Verzerrung": V.pct(r["Verzerrung"]), "typischer Fehler": V.de(r["typischer Fehler"], 3)} for r in V.bias_table(dist, grid_n, settings.delta, settings.c)],
    hide_index=True, width="stretch",
)
st.caption("Verzerrung = mittlerer Fehler über alle Läufe" + (" (symmetrische Verteilung: 0 bis auf Rauschen)." if symmetric else ", relativ zum Mittelwert; minus heißt: im Mittel zu klein."))

st.subheader("🔬 E3: Typisch gegen Ernstfall – Prämie und Auszahlung")
st.caption(f"Vorgerechnet für {dist}, n = {grid_n}{grid_note}. Typischer Fehler = Median des Fehlers; Q(1 − δ) = Fehler im seltenen Ernstfall.")
st.plotly_chart(build_typ_vs_worst(dist, grid_n, settings.delta, settings.c), width="stretch", key="typ_worst")
show(V.verdict_premium(dist, grid_n, settings.delta))

st.subheader("🔬 E4: Trimmen als Gegenkandidat")
st.caption(f"Sieger der Dreier-Wahl Mittel, Median-of-Means (Regel), Trimmen mit k = ⌈{settings.c} ln(1/δ)⌉ je Seite bei δ = {V.de_delta(settings.delta)}: kleinstes Q(1 − δ).")
st.plotly_chart(build_winner_map(settings.delta, settings.c), width="stretch", key="winner_map")
show(V.verdict_three_way(dist, grid_n, settings.delta, settings.c))
show(V.verdict_tuned())

st.subheader("🔬 E5: Die Zielgröße entscheidet vor dem Verfahren")
st.caption(f"Vorgerechnet für {dist}, n = {grid_n}{grid_note}. Fehler relativ zum jeweiligen Ziel.")
if not symmetric:
    st.plotly_chart(build_target(dist, grid_n, settings.delta, settings.c), width="stretch", key="target_chart")
show(V.verdict_target(dist, grid_n, settings.delta))

st.subheader("🔬 E6: Gibt es ein brauchbares Intervall für den Mittelwert? (negativer Befund)")
st.caption("Überdeckung des wahren Mittelwerts bei Niveau 1 − δ; 8 000 Stichproben je Seed (n = 5 000: 4 000), vier Seeds. Student: Mittel ± t·s/√n. Vorzeichen-Intervall: verteilungsfrei, aber für den "
           "Median der Blockmittel. Median-of-Means ± s·√(c·ln(1/δ)/n): c = 32 ist eine Konstante aus der Erinnerung, nicht an einer Quelle geprüft; c = 8 die kleinere Vergleichswahl.")
delta_i = min(G.INTERVAL_DELTAS, key=lambda d: abs(d - settings.delta))
if abs(delta_i - settings.delta) > 1e-9:
    st.caption(f"Für Intervalle vorgerechnet sind δ = 0,1, 0,05 und 0,01; gezeigt wird das nächste zu Ihrem δ: {V.de_delta(delta_i)}.")
st.plotly_chart(build_interval_coverage(dist, delta_i), width="stretch", key="interval_chart")
show(V.verdict_intervals(dist, delta_i))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Werte sind unabhängig und gleich verteilt.** | Die Messreihen ziehen unabhängige Stichproben aus einem Modell; bei abhängigen Werten (Serien, Saison) ist die Blockbildung in zeitlicher Reihenfolge eine andere Aufgabe. Nicht gemessen. | – |
| **Die Verteilungen sind erfundene Modelle.** | Kein Realdatenbezug: Lognormal und Pareto sind übliche Annahmen für schiefe Mengen und Zeiten, aber kein Beleg, dass echte Sendungsdaten so aussehen. | – |
| **Das Ziel ist der Mittelwert.** | Wer die Gesamtmenge braucht, braucht den Mittelwert. Zielt man auf den Median, ist der Median-Schätzer typischerweise um ein Vielfaches besser (E5). | Mittelwert und Median (Wurzel der Reihe) |
| **Median-of-Means ist erwartungstreu genug.** | Bei schiefen Rändern liegt Median-of-Means im Mittel unter dem Mittelwert, umso mehr, je mehr Blöcke (E2). | – |
| **K folgt einer Regel.** | Die Regel K = ⌈8 ln(1/δ)⌉ ist eine Faustregel, deren Herkunft nicht geprüft ist; das im Nachhinein beste K liegt bei schiefen Rändern in allen Zellen darunter, in denen der Deckel n/2 nicht greift (E2). Eine datengetriebene Wahl von K ist nicht gebaut. | – |
| **Trimm-Faktor c und K wurden im Nachhinein verglichen.** | Der Vergleich in E4 wählt beide am selben Datensatz; fair wäre die Wahl auf einer Hälfte der Daten (nicht gemessen). | `trimmen-huber-demo` (Trimmen, Winsorisieren, Huber) |
| **Eine Aussage über das seltene Ereignis braucht viele Läufe.** | Q(1 − δ) bei δ = 0,001 streut zwischen den Seeds (E1, README nennt die Streuung); bei weniger als 100 Überschreitungen kennzeichnet das Nachmessen das Ergebnis als wenig belastbar. | – |
| **Ein Intervall für den Mittelwert lässt sich aus Median-of-Means bauen.** | Nicht brauchbar gefunden (E6): entweder zu schmal bei schiefen Rändern oder mehr als doppelt so breit wie Student. Das ist die Auswahl der geprüften Konstruktionen, keine Unmöglichkeitsaussage. | `pearson-demo` (Intervalle für Korrelationen bei schweren Rändern) |
"""
)

st.markdown("---")

# --- Verfahren ------------------------------------------------------------------------------------------------------------------------------

with st.expander("🧮 Die Schätzer im Überblick"):
    st.markdown(
        """
| Schätzer | Idee | Wen schützt er? | Preis |
|---|---|---|---|
| **Mittel** | Summe geteilt durch n | niemanden, ist aber bei Normal am genauesten | ein Ausreißer verschiebt es beliebig weit |
| **Median** | mittlerer Wert der sortierten Stichprobe | gegen fast jeden Ausreißer | zielt bei Schiefe auf den Median, nicht den Mittelwert (E5) |
| **Trimmen** | k kleinste und k größte Werte weglassen, den Rest mitteln | gegen bis zu k Ausreißer je Seite | bei schiefen Verteilungen zu klein (Verzerrung), k = ⌈c ln(1/δ)⌉ ist eigene Wahl |
| **Median-of-Means** | K Blockmittel, davon der Median | gegen bis zu etwa K/2 verdorbene Blöcke | bei Normal ungenauer (Faktor √(π/2) ≈ 1,25), bei den gemessenen schiefen Verteilungen zu klein |
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind $n$ unabhängige Werte $X_1, \dots, X_n$ mit gemeinsamer Verteilung (Mittelwert $\mu$, endliche Varianz). Gesucht ist ein Schätzer $\hat\mu$, dessen Fehler
$|\hat\mu - \mu|$ auch im seltenen Ernstfall klein bleibt.

| Symbol | Bedeutung |
|---|---|
| $n$ | Stichprobengröße |
| $K$ | Zahl der Blöcke |
| $B_j$ | der $j$-te Block (zusammenhängende Positionen) |
| $\delta$ | Niveau: der Fehler wird nur in einem Anteil $\delta$ der Läufe überschritten |
| $k$ | Zahl der abgeschnittenen Werte je Seite beim Trimmen |
| $c$ | Faktor in $k = \lceil c \ln(1/\delta) \rceil$ |
| $X_{(i)}$ | der $i$-kleinste Wert |

**Median-of-Means.**

$$\hat\mu_{\mathrm{MoM}} = \operatorname{median}\big(\bar X_1, \dots, \bar X_K\big), \qquad \bar X_j = \frac{1}{|B_j|} \sum_{i \in B_j} X_i$$

Die Blöcke haben die Größen $\lfloor n/K \rfloor$ oder $\lceil n/K \rceil$:

$$|B_j| \in \{\lfloor n/K \rfloor, \lceil n/K \rceil\}, \qquad \sum_j |B_j| = n$$

$K = 1$ ergibt das Mittel, $K = n$ den Median.

**Getrimmtes Mittel.**

$$\hat\mu_{\mathrm{TM}} = \frac{1}{n - 2k} \sum_{i = k+1}^{n-k} X_{(i)}$$

**Fehlerquantil.** Gemessen wird das Quantil des Fehlers:

$$Q(1-\delta) = \inf\big\{ t : P(|\hat\mu - \mu| > t) \le \delta \big\}$$

**Blockzahl und Trimm-Zahl** (die Regel ist die Faustregel der Vorlage, $c$ ist eine eigene Wahl):

$$K = \min\big(\lfloor n/2 \rfloor, \lceil 8 \ln(1/\delta) \rceil\big), \qquad k = \min\big(\lfloor (n-1)/2 \rfloor, \lceil c \ln(1/\delta) \rceil\big)$$

**Preis bei Normal.** Für viele Blöcke ist der Median der Blockmittel asymptotisch normal mit der Streuung des Mittels mal $\sqrt{\pi/2}$:

$$\frac{\operatorname{sd}(\hat\mu_{\mathrm{MoM}})}{\operatorname{sd}(\bar X)} \to \sqrt{\pi/2} \approx 1{,}2533$$

**Verzerrung bei Schiefe.** Der Stichprobenmedian zielt auf den Median $m$ statt auf $\mu$; sein relativer Fehler ist $m/\mu - 1$. Für die Lognormalverteilung mit Parameter $\sigma$ und für die Pareto-Verteilung mit Exponent $\alpha$:

$$\frac{m}{\mu} - 1 = e^{-\sigma^2/2} - 1, \qquad \frac{m}{\mu} - 1 = 2^{1/\alpha}\,\frac{\alpha - 1}{\alpha} - 1$$

**Vorzeichen-Intervall auf den Blockmitteln.** Mit den sortierten Blockmitteln $b_{(1)} \le \dots \le b_{(K)}$ und dem größten $l$ mit $P(\mathrm{Bin}(K, \tfrac12) \le l-1) \le \delta/2$ gilt für den **Median der Blockmittel** (nicht für $\mu$):

$$P\big(b_{(l)} \le \operatorname{med} \le b_{(K+1-l)}\big) = 1 - 2\,P\big(\mathrm{Bin}(K, \tfrac12) \le l-1\big) \ge 1-\delta$$

**Literatur.** Die Garantien aus der Literatur zu Median-of-Means und zum getrimmten Mittel bei schweren Rändern habe ich nicht eingesehen. Die Demo misst an Modellen, sie beweist nichts.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
