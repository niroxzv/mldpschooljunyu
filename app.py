"""
HDB Resale Price Estimator - Streamlit web app
================================================
Loads the tuned Gradient Boosting model saved by the notebook (section 6.4,
round 1: 600 trees / max_depth 5 / learning_rate 0.1) and estimates the resale
price of a flat in Punggol, Sengkang or Hougang.

Run locally with:  streamlit run app.py
(from the `mldp` environment, which has joblib and scikit-learn installed)

The app never reads the CSV. It only needs the two artefacts the notebook saves
(resale_price_model.pkl, model_columns.pkl), which keeps the deployment small.
Any figure quoted in the interface is therefore hard-coded below and labelled
with where it came from.

Every input is pre-filled with the most common value for that kind of flat, so
the form can be submitted immediately, and inputs are validated against what
the training data actually contains before a prediction is shown. The result
is explained with numbers ("three floors higher: +$5,019") because those are
directly actionable for a buyer or seller.
"""
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- page config
st.set_page_config(
    page_title="HDB Resale Price Estimator",
    page_icon=":material/apartment:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ===================================================================
# REFERENCE DATA
# ===================================================================
# Measured from the same 47,864 transactions the model was trained on (Punggol /
# Sengkang / Hougang, Jan 2017 - Jul 2026). These let the app offer only
# combinations the model has actually seen, warn when an input falls outside
# that experience, and pre-fill typical values so someone who does not know
# their flat's details still gets a sensible answer.

# A street belongs to exactly one town, so the street list is filtered by town.
TOWN_STREETS = {
    "HOUGANG": [
        "BUANGKOK CRES", "BUANGKOK GREEN", "BUANGKOK LINK", "HOUGANG AVE 1",
        "HOUGANG AVE 10", "HOUGANG AVE 2", "HOUGANG AVE 3", "HOUGANG AVE 4",
        "HOUGANG AVE 5", "HOUGANG AVE 6", "HOUGANG AVE 7", "HOUGANG AVE 8",
        "HOUGANG AVE 9", "HOUGANG CTRL", "HOUGANG ST 11", "HOUGANG ST 21",
        "HOUGANG ST 22", "HOUGANG ST 31", "HOUGANG ST 32", "HOUGANG ST 51",
        "HOUGANG ST 52", "HOUGANG ST 61", "HOUGANG ST 91", "HOUGANG ST 92",
        "LOR AH SOO", "UPP SERANGOON CRES", "UPP SERANGOON RD",
        "UPP SERANGOON VIEW",
    ],
    "PUNGGOL": [
        "EDGEDALE PLAINS", "EDGEFIELD PLAINS", "NORTHSHORE DR", "PUNGGOL CTRL",
        "PUNGGOL DR", "PUNGGOL EAST", "PUNGGOL FIELD", "PUNGGOL FIELD WALK",
        "PUNGGOL PL", "PUNGGOL RD", "PUNGGOL WALK", "PUNGGOL WAY",
        "SUMANG LANE", "SUMANG LINK", "SUMANG WALK",
    ],
    "SENGKANG": [
        "ANCHORVALE CRES", "ANCHORVALE DR", "ANCHORVALE LANE", "ANCHORVALE LINK",
        "ANCHORVALE RD", "ANCHORVALE ST", "COMPASSVALE BOW", "COMPASSVALE CRES",
        "COMPASSVALE DR", "COMPASSVALE LANE", "COMPASSVALE LINK",
        "COMPASSVALE RD", "COMPASSVALE ST", "COMPASSVALE WALK", "FERNVALE LANE",
        "FERNVALE LINK", "FERNVALE RD", "FERNVALE ST", "JLN KAYU",
        "RIVERVALE CRES", "RIVERVALE DR", "RIVERVALE ST", "RIVERVALE WALK",
        "SENGKANG CTRL", "SENGKANG EAST AVE", "SENGKANG EAST RD",
        "SENGKANG EAST WAY", "SENGKANG WEST AVE", "SENGKANG WEST RD",
        "SENGKANG WEST WAY",
    ],
}

# flat_type -> (smallest, largest, most common) floor area seen in the data.
FLAT_TYPE_AREA = {
    "2 ROOM":    (37.0, 50.0, 47.0),
    "3 ROOM":    (59.0, 92.0, 67.0),
    "4 ROOM":    (82.0, 123.0, 93.0),
    "5 ROOM":    (108.0, 152.0, 112.0),
    "EXECUTIVE": (125.0, 177.0, 137.0),
}

# Plain-English hint per flat type, so the user does not need to know HDB's
# naming to pick the right one.
FLAT_TYPE_HINT = {
    "2 ROOM": "1 bedroom",
    "3 ROOM": "2 bedrooms",
    "4 ROOM": "3 bedrooms",
    "5 ROOM": "3 bedrooms, larger",
    "EXECUTIVE": "largest, often double-storey",
}

# flat_type -> the flat models that actually exist for it. A 2 ROOM flat is
# never a Maisonette, so offering that would invite a nonsense prediction.
FLAT_TYPE_MODELS = {
    "2 ROOM":    ["2-Room", "Model A"],
    "3 ROOM":    ["DBSS", "Improved", "Model A", "New Generation",
                  "Premium Apartment", "Simplified"],
    "4 ROOM":    ["DBSS", "Improved", "Model A", "Model A2", "New Generation",
                  "Premium Apartment", "Simplified"],
    "5 ROOM":    ["DBSS", "Improved", "Model A", "Other", "Premium Apartment"],
    "EXECUTIVE": ["Apartment", "Maisonette", "Other", "Premium Apartment"],
}

# Remaining lease observed per town, and the typical value used as the default.
# Punggol and Sengkang are young estates, so a 50-year lease there is not real.
TOWN_LEASE = {
    "HOUGANG":  (48.2, 95.9, 69.0),
    "PUNGGOL":  (75.4, 96.0, 92.0),
    "SENGKANG": (71.5, 96.2, 87.0),
}

# HDB records storeys in 3-floor bands. The model was trained on the band's
# midpoint, so the app collects the band and converts, exactly like the notebook.
STOREY_BANDS = {
    "01 to 03": 2.0, "04 to 06": 5.0, "07 to 09": 8.0, "10 to 12": 11.0,
    "13 to 15": 14.0, "16 to 18": 17.0, "19 to 21": 20.0, "22 to 24": 23.0,
    "25 to 27": 26.0,
}

# Median price over the last 12 months of transactions, so the estimate can be
# shown against what comparable flats have really been selling for.
RECENT_MEDIAN = {
    ("HOUGANG", "2 ROOM"): 380_000,  ("HOUGANG", "3 ROOM"): 455_000,
    ("HOUGANG", "4 ROOM"): 622_000,  ("HOUGANG", "5 ROOM"): 770_000,
    ("HOUGANG", "EXECUTIVE"): 980_000,
    ("PUNGGOL", "2 ROOM"): 398_000,  ("PUNGGOL", "3 ROOM"): 548_000,
    ("PUNGGOL", "4 ROOM"): 683_888,  ("PUNGGOL", "5 ROOM"): 758_000,
    ("PUNGGOL", "EXECUTIVE"): 835_000,
    ("SENGKANG", "2 ROOM"): 390_500, ("SENGKANG", "3 ROOM"): 545_000,
    ("SENGKANG", "4 ROOM"): 645_000, ("SENGKANG", "5 ROOM"): 706_500,
    ("SENGKANG", "EXECUTIVE"): 850_000,
}

# Rooms per flat_type (EXECUTIVE assumed = 6). Must match the notebook, because
# the model was trained on area_per_room = floor_area_sqm / rooms.
ROOMS = {"2 ROOM": 2, "3 ROOM": 3, "4 ROOM": 4, "5 ROOM": 5, "EXECUTIVE": 6}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MAX_MONTH = 114          # Jul 2026 = the last month in the training data

# Held-out test-set performance of the deployed model (notebook section 6.5).
TEST_MAE = 17_539
TEST_R2 = 0.9721
N_TRANSACTIONS = 47_864
N_TEST = 9_573

def month_label(m):
    """Turn months-since-Jan-2017 into a label a person can read."""
    return f"{MONTH_NAMES[m % 12]} {2017 + m // 12}"

MONTH_OPTIONS = [month_label(m) for m in range(MAX_MONTH + 1)]
MONTH_LOOKUP = {label: m for m, label in enumerate(MONTH_OPTIONS)}

# ===================================================================
# STYLING
# ===================================================================
# One typeface, one accent colour, thin rules instead of shadows.
#
# The app is pinned to a light appearance for everyone, rather than following
# the viewer's dark preference or Streamlit's theme menu. Two reasons: a
# valuation tool should read like a printed document, and a fixed appearance
# keeps the screenshots in the report consistent with what a marker sees.
# Pinning it also removes the whole class of bug where Streamlit's own theme
# and the browser's prefers-color-scheme disagree with each other.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

      :root {
          --navy:      #1E3A5F;
          --blue:      #2563EB;
          --ink:       #0F172A;
          --ink-muted: #64748B;
          --rule:      #E2E8F0;
          --well:      #F8FAFC;
      }

      /* ---- Pin the light appearance --------------------------------------
         color-scheme tells the browser to render scrollbars and native
         controls light. The rest repaints Streamlit's own surfaces, which are
         the only things that would otherwise go dark. */
      :root, .stApp { color-scheme: light !important; }

      .stApp, [data-testid="stHeader"], [data-testid="stMain"],
      [data-testid="stBottomBlockContainer"] {
          background: #FFFFFF !important;
          color: var(--ink) !important;
      }
      [data-testid="stSidebar"],
      [data-testid="stSidebarContent"] {
          background: #F4F6F9 !important;
          color: var(--ink) !important;
      }
      /* Text everywhere: Streamlit sets near-white in dark mode, which would
         be invisible on the white page above. */
      .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2,
      .stApp h3, .stApp h4, .stApp td, .stApp th,
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] li,
      [data-testid="stSidebar"] span, [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
          color: var(--ink) !important;
      }
      /* Input surfaces and the dropdown list, which is a separate portal. */
      [data-testid="stSelectbox"] div:not([data-testid="stWidgetLabel"] *),
      [data-testid="stSelectbox"] input,
      [data-testid="stNumberInput"] div:not([data-testid="stWidgetLabel"] *),
      [data-testid="stNumberInput"] input,
      [data-testid="stNumberInput"] button,
      [data-testid="stSelectboxVirtualDropdown"],
      [data-testid="stSelectboxVirtualDropdown"] li,
      [data-testid="stExpander"], [data-testid="stExpander"] details,
      [data-testid="stExpander"] summary,
      [data-testid="stExpander"] summary *,
      [data-testid="stExpanderDetails"] {
          background-color: var(--well) !important;
          color: var(--ink) !important;
      }
      [data-testid="stExpander"] details { border-color: #CBD5E1 !important; }
      [data-testid="stSelectboxVirtualDropdown"] li:hover {
          background-color: #E8ECF2 !important;
      }
      /* Metric cards. */
      [data-testid="stMetric"] {
          background-color: #FFFFFF !important;
          color: var(--ink) !important;
          border-color: #CBD5E1 !important;
      }

      /* Tables (see show_table). Plain HTML, so they can actually be styled to
         match the page - which st.dataframe cannot, being canvas-drawn. */
      .wtable {
          width: 100%; border-collapse: collapse; margin: .2rem 0 .1rem 0;
          font-size: .93rem; background: #FFFFFF;
          border: 1px solid #CBD5E1; border-radius: 4px;
      }
      .wtable thead th {
          text-align: left; font-weight: 600; color: #334155;
          background: #F1F5F9; padding: .6rem .85rem;
          border-bottom: 1px solid #CBD5E1; white-space: nowrap;
      }
      .wtable tbody td {
          padding: .55rem .85rem; color: var(--ink);
          border-bottom: 1px solid #E8ECF2;
      }
      .wtable tbody tr:last-child td { border-bottom: none; }
      .wtable tbody tr:nth-child(even) td { background: #FBFCFE; }
      /* Money columns right-aligned on tabular figures so they line up. */
      .wtable thead th:not(:first-child),
      .wtable tbody td:not(:first-child) {
          text-align: right;
          font-variant-numeric: tabular-nums;
          font-feature-settings: "tnum";
      }

      /* The secondary button (Reset) keeps Streamlit's dark-theme fill, which
         is black on the light page. Give it a normal outlined-button look. */
      [data-testid^="stBaseButton-secondary"] {
          background: #FFFFFF !important;
          color: var(--ink) !important;
          border: 1px solid #CBD5E1 !important;
      }
      [data-testid^="stBaseButton-secondary"]:hover {
          background: var(--well) !important;
          border-color: var(--blue) !important;
          color: var(--blue) !important;
      }
      [data-testid^="stBaseButton-secondary"] svg {
          fill: currentColor !important; color: inherit !important;
      }
      /* Icons: the dropdown chevrons and the "?" tooltip targets are SVGs that
         inherit Streamlit's near-white dark-theme colour, so they disappear
         against the light surfaces above. */
      [data-testid="stSelectbox"] svg, [data-testid="stNumberInput"] svg,
      [data-testid="stTooltipHoverTarget"] svg, [data-testid="stExpander"] svg,
      [data-testid="stSidebar"] svg, [data-testid="stMetric"] svg {
          fill: var(--ink-muted) !important;
          color: var(--ink-muted) !important;
      }
      /* The "?" icons inside a slider sit within the hue-rotate above, and
         --ink-muted is a blue-grey, so it would rotate to olive. A true neutral
         grey has no hue to rotate. */
      [data-testid="stSlider"] svg {
          fill: #737373 !important;
          color: #737373 !important;
      }

      html, body, .stApp, [data-testid="stSidebar"] {
          font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
      }

      /* Money must use tabular figures, or digits shift width as values change. */
      .result .amount, [data-testid="stMetricValue"],
      [data-testid="stMetricDelta"], .stDataFrame {
          font-variant-numeric: tabular-nums;
          font-feature-settings: "tnum";
      }

      /* Numbered step marker + rule above each section. The number carries the
         sequence so the heading does not have to. */
      .step {
          display: flex; align-items: baseline; gap: .7rem;
          margin: 2.1rem 0 1rem 0; padding-top: 1.1rem;
          border-top: 1px solid var(--rule);
      }
      .step:first-of-type { margin-top: 1rem; }
      .step .n {
          font-size: .72rem; font-weight: 700; letter-spacing: .12em;
          color: var(--blue); flex: none;
      }
      .step .t {
          font-size: 1.16rem; font-weight: 600; color: var(--ink);
          letter-spacing: -.01em;
      }

      /* The headline result. A navy rule on the left rather than a coloured
         fill, so the number itself is the loudest thing on the page. */
      .result {
          border: 1px solid var(--rule);
          border-left: 3px solid var(--navy);
          border-radius: 4px;
          padding: 1.7rem 1.6rem;
          background: #fff;
      }
      .result .caption {
          font-size: .7rem; font-weight: 600; letter-spacing: .14em;
          text-transform: uppercase; color: var(--ink-muted);
      }
      .result .amount {
          font-size: 3.1rem; font-weight: 700; color: var(--ink);
          line-height: 1.05; margin: .45rem 0 .4rem 0; letter-spacing: -.025em;
      }
      .result .band {
          font-size: .93rem; color: var(--ink-muted);
          padding-top: .6rem; border-top: 1px solid var(--rule);
      }
      .result .band b { color: var(--ink); font-weight: 600; }

      /* Empty state, before anything has been estimated. */
      .awaiting {
          border: 1px dashed var(--rule); border-radius: 4px;
          padding: 2.4rem 1.5rem; text-align: center;
          color: var(--ink-muted); font-size: .95rem; background: var(--well);
      }

      [data-testid="stMetricValue"] { font-size: 1.28rem; font-weight: 600; }

      /* Streamlit's accent defaults to red (#FF4B4B). Buttons can be set
         directly; sliders, dropdowns and number fields draw their accent
         inline with no selector to hook, so their red is hue-rotated to blue
         instead. Greys have no saturation, so the rotation leaves them alone. */
      [data-testid^="stBaseButton-primary"] {
          background: var(--blue) !important;
          border-color: var(--blue) !important;
          color: #fff !important;
      }
      [data-testid^="stBaseButton-primary"]:hover {
          background: var(--navy) !important;
          border-color: var(--navy) !important;
      }
      [data-testid="stSlider"] {
          filter: hue-rotate(218deg) saturate(1.05);
      }

      /* Dropdowns and the number field are NOT filtered - the filter would
         drag their tinted surface towards olive. Instead their red focus
         border is overridden directly: :focus-within hits whichever inner div
         carries the border without depending on Streamlit's DOM structure. */
      [data-testid="stSelectbox"] div:focus-within,
      [data-testid="stNumberInput"] div:focus-within {
          border-color: var(--blue) !important;
      }

      /* The dropdowns are type-to-search boxes, so clicking one places a text
         cursor. Typing still filters the list; the blinking caret is just
         hidden because it reads as a glitch on a dropdown. */
      [data-testid="stSelectbox"],
      [data-testid="stSelectbox"] * {
          caret-color: transparent !important;
      }

      /* Keyboard focus must stay visible - never remove the ring. */
      *:focus-visible {
          outline: 2px solid var(--blue) !important;
          outline-offset: 2px !important;
      }

      /* ...except the search input inside a dropdown or number field. Those
         inputs are sized to their text, so any ring draws a box around the
         words rather than around the control. The field already shows focus
         with its own border (above), so nothing is lost. All three ways a
         browser can draw that box are cleared, in every focus state. */
      [data-testid="stSelectbox"] input,
      [data-testid="stSelectbox"] input:focus,
      [data-testid="stSelectbox"] input:focus-visible,
      [data-testid="stSelectbox"] input:focus-within,
      [data-testid="stNumberInput"] input,
      [data-testid="stNumberInput"] input:focus,
      [data-testid="stNumberInput"] input:focus-visible,
      [data-testid="stNumberInput"] input:focus-within {
          outline: none !important;
          border: none !important;
          box-shadow: none !important;
      }

      /* With no config file pinning the theme, the app follows the viewer's
         system setting, so the custom surfaces need a dark variant. Streamlit's
         own dark background is near-black; replace it with a dark slate grey,
         with the sidebar and cards one step lighter so the layers read. */

      @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
              animation-duration: .001ms !important;
              transition-duration: .001ms !important;
          }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

def show_table(frame):
    """Render a DataFrame as a plain HTML table.

    st.dataframe draws to a canvas and takes its colours from Streamlit's own
    theme in JavaScript, so CSS cannot reach it - it stays dark whatever the
    page around it looks like. These tables are small and static, so nothing is
    lost by dropping the interactive grid for something that can be styled.
    """
    st.markdown(
        frame.to_html(index=False, escape=False, border=0, classes="wtable"),
        unsafe_allow_html=True,
    )


def step(number, title):
    """Numbered section marker, in place of a plain heading."""
    st.markdown(
        f'<div class="step"><span class="n">STEP {number}</span>'
        f'<span class="t">{title}</span></div>',
        unsafe_allow_html=True,
    )

# ===================================================================
# LOAD MODEL
# ===================================================================
@st.cache_resource
def load_model():
    """Load the saved model and its column list once, and cache them.

    Returns (model, columns, None) on success, or (None, None, message) so the
    caller can show a readable error instead of a traceback.
    """
    try:
        model = joblib.load("resale_price_model.pkl")
        columns = joblib.load("model_columns.pkl")
        return model, columns, None
    except FileNotFoundError as exc:
        return None, None, (
            f"Could not find `{exc.filename}`. Both `resale_price_model.pkl` "
            "and `model_columns.pkl` must sit next to `app.py`. Re-run the "
            "final cell of the notebook to regenerate them."
        )
    except Exception as exc:                                # noqa: BLE001
        return None, None, (
            f"The model file could not be opened ({type(exc).__name__}: {exc}). "
            "This usually means the scikit-learn version here differs from the "
            "one that saved it - check `requirements.txt` pins scikit-learn==1.8.0."
        )

model, model_columns, load_error = load_model()

if load_error:
    st.title("HDB Resale Price Estimator", anchor=False)
    st.error(load_error, icon=":material/error:")
    st.stop()

def known_categories(prefix):
    """Recover valid categories from the model's one-hot column names."""
    return {c[len(prefix):] for c in model_columns if c.startswith(prefix)}

# Only ever offer a category the model was trained on. If the notebook is re-run
# with different data, the app narrows its options instead of sending the model
# a column it has never seen.
KNOWN_TOWNS = known_categories("town_")
KNOWN_STREETS = known_categories("street_name_")
KNOWN_TYPES = known_categories("flat_type_")
KNOWN_MODELS = known_categories("flat_model_")

# ===================================================================
# SIDEBAR
# ===================================================================
with st.sidebar:
    st.subheader("About this estimator", anchor=False)
    st.write(
        "Prices are predicted by a **Gradient Boosting** model trained on every "
        "HDB resale transaction in Punggol, Sengkang and Hougang since 2017 "
        f"({N_TRANSACTIONS:,} sales)."
    )

    st.markdown("**Accuracy**")
    c1, c2 = st.columns(2)
    c1.metric("Typical error", f"${TEST_MAE:,}", border=True, help=(
        f"Mean absolute error on {N_TEST:,} held-out sales the model never saw "
        "while training."
    ))
    c2.metric("R squared", f"{TEST_R2:.3f}", border=True, help=(
        "The share of the price variation between flats that the model explains."
    ))
    st.caption(
        f"{TEST_MAE:,} dollars is about 3.4% of a typical 520,000 dollar flat — "
        "close enough to anchor an asking price or judge whether a listing is fair."
    )

    st.markdown("**Scope**")
    st.markdown(
        "- Covers **Punggol, Sengkang and Hougang** only.\n"
        f"- Trained on sales up to **{month_label(MAX_MONTH)}**.\n"
        "- An estimate, **not a formal valuation** — an HDB-appointed valuer is "
        "still required for an actual transaction."
    )
    st.caption("Data source: data.gov.sg — HDB Resale Flat Prices.")

# ===================================================================
# HEADER
# ===================================================================
st.title("HDB Resale Price Estimator", anchor=False)
st.markdown(
    "Valuation guide for **Punggol, Sengkang and Hougang**. Enter a flat's "
    "details to see what it is likely to be worth, and what is driving that "
    "figure."
)
st.info(
    "Every field is already set to the most common answer for that kind of flat, "
    "so you can press **Estimate price** straight away and adjust afterwards. "
    "Hover the **?** beside any field for help.",
    icon=":material/lightbulb:",
)

# ===================================================================
# INPUTS - all eight on screen. A field the user cannot see is a field
# they cannot answer, so nothing is hidden behind an optional section.
# ===================================================================
step(1, "Where is the flat?")
loc_left, loc_right = st.columns(2)

with loc_left:
    town = st.selectbox(
        "Town", sorted(t for t in TOWN_STREETS if t in KNOWN_TOWNS),
        help="The three North East Line towns this model covers.",
    )

with loc_right:
    # Filtered by town: this is what stops an impossible address such as
    # Hougang + PUNGGOL DR from ever reaching the model.
    streets = [s for s in TOWN_STREETS[town] if s in KNOWN_STREETS]
    street_name = st.selectbox(
        "Street", streets,
        help=f"Only streets that exist in {town.title()} are listed.",
    )

step(2, "What kind of flat is it?")
type_left, type_right = st.columns(2)

with type_left:
    flat_type = st.selectbox(
        "Flat type",
        [t for t in FLAT_TYPE_AREA if t in KNOWN_TYPES],
        index=2,                       # 4 ROOM: about half of all sales here
        format_func=lambda t: f"{t} — {FLAT_TYPE_HINT[t]}",
        help="How HDB classifies the flat's size.",
    )

with type_right:
    # Filtered by flat type, so e.g. a 2 ROOM flat cannot be a Maisonette.
    models_for_type = [m for m in FLAT_TYPE_MODELS[flat_type] if m in KNOWN_MODELS]
    # Default to the commonest model for this flat type rather than whatever
    # sorts first, so the form opens on a realistic flat.
    default_model = next(
        (m for m in ("Model A", "Premium Apartment", "Improved") if m in models_for_type),
        models_for_type[0],
    )
    flat_model = st.selectbox(
        "Flat model", models_for_type,
        index=models_for_type.index(default_model),
        help="HDB's design name for the layout. It is printed on the resale listing.",
    )

area_min, area_max, area_typical = FLAT_TYPE_AREA[flat_type]
floor_area_sqm = st.slider(
    "Floor area (sqm)",
    min_value=float(area_min), max_value=float(area_max),
    value=float(area_typical), step=1.0,
    help=(
        f"{flat_type} flats in these towns range from {area_min:.0f} to "
        f"{area_max:.0f} sqm."
    ),
)
st.caption(
    f"Most **{flat_type}** flats here are **{area_typical:.0f} sqm** — leave it "
    "as is if you are not sure. The exact figure is on the flat's resale listing."
)

step(3, "Which unit, and when?")
unit_left, unit_right = st.columns(2)

with unit_left:
    # Ask for the actual floor, which is what a person knows, then convert to
    # the band midpoint the model was trained on. HDB only publishes a 3-floor
    # band for privacy, so every flat in a band shares one value - the caption
    # below says so outright rather than leaving the user to notice it.
    floor = st.number_input(
        "Which floor?", min_value=1, max_value=27, value=11, step=1,
        help="The actual floor the flat is on.",
    )
    storey_band = list(STOREY_BANDS)[(int(floor) - 1) // 3]
    storey_mid = STOREY_BANDS[storey_band]
    st.caption(
        f"HDB groups floors into bands of three, so floor **{int(floor)}** is "
        f"published as **{storey_band}**. The model was trained on the middle "
        f"floor of each band, so it prices this as floor **{storey_mid:.0f}** — "
        f"which is why every flat on floors {storey_band} gets the same estimate."
    )

    lease_lo, lease_hi, lease_typical = TOWN_LEASE[town]
    remaining_lease_years = st.slider(
        "Remaining lease (years)",
        min_value=40.0, max_value=99.0,
        value=float(lease_typical), step=0.5,
        help=(
            f"Flats sold in {town.title()} had between {lease_lo:.0f} and "
            f"{lease_hi:.0f} years left. HDB flats start with a 99-year lease."
        ),
    )

with unit_right:
    sale_label = st.select_slider(
        "Sale month", options=MONTH_OPTIONS, value=MONTH_OPTIONS[-1],
        help=(
            "When the sale happens. Prices in these towns rose sharply over this "
            "period, so timing changes the answer a lot."
        ),
    )
    months_since_2017 = MONTH_LOOKUP[sale_label]

    rooms = ROOMS[flat_type]
    st.caption(
        f"**Space per room:** {floor_area_sqm / rooms:.1f} sqm "
        f"({floor_area_sqm:.0f} sqm over {rooms} rooms). The model uses this to "
        "judge how generously the flat is laid out for its size."
    )

with st.expander("Not sure what these terms mean?", icon=":material/help:"):
    st.markdown(
        """
| Term | What it means | Where to find it |
|---|---|---|
| **Flat type** | Size class — a *4 ROOM* has 3 bedrooms plus a living room. | The resale listing |
| **Flat model** | HDB's design name (*Model A*, *Improved*, *Maisonette*…). Two flats the same size can be different models. | The resale listing |
| **Floor area** | Internal floor space in square metres. | The resale listing, or HDB's My Flat Dashboard |
| **Storey** | HDB publishes a 3-floor band instead of the exact unit, for privacy. | The resale listing |
| **Remaining lease** | Years left on the 99-year lease. A shorter lease usually means a lower price. | HDB's My Flat Dashboard |
| **Sale month** | When the sale is agreed. Prices move over time, so this matters. | Your own timeline |

**Don't know one of these?** Leave it at the default — each is pre-set to the
most common value for that kind of flat.
        """
    )

# ===================================================================
# INPUT VALIDATION
# ===================================================================
def validate():
    """Check the inputs against what the training data actually contains.

    Returns (errors, warnings). Errors describe combinations the model has never
    seen and cannot price honestly, so no prediction is shown. Warnings describe
    unusual but possible flats, where the estimate is shown with a caveat.
    """
    errors, warnings = [], []

    # Hard checks: the model has no column for these, so it cannot price them.
    if street_name not in KNOWN_STREETS:
        errors.append(
            f"**{street_name}** is not one of the {len(KNOWN_STREETS)} streets "
            "the model was trained on, so it cannot be priced."
        )
    if street_name not in TOWN_STREETS.get(town, []):
        errors.append(
            f"**{street_name}** is not in **{town.title()}**. Pick a street "
            "that matches the town."
        )
    if flat_model not in FLAT_TYPE_MODELS.get(flat_type, []):
        errors.append(
            f"No **{flat_type}** flat in these towns is a **{flat_model}**. "
            "Choose a different flat model."
        )

    # Soft checks: possible, but outside the range the model learned from, so
    # the estimate is an extrapolation and should be read with more caution.
    lease_lo, lease_hi, _ = TOWN_LEASE[town]
    if not lease_lo - 0.5 <= remaining_lease_years <= lease_hi + 0.5:
        warnings.append(
            f"A **{remaining_lease_years:.0f}-year** lease is outside the "
            f"{lease_lo:.0f}–{lease_hi:.0f} years seen in {town.title()}, so "
            "this estimate is less reliable than usual."
        )

    area_lo, area_hi, _ = FLAT_TYPE_AREA[flat_type]
    if not area_lo <= floor_area_sqm <= area_hi:
        warnings.append(
            f"**{floor_area_sqm:.0f} sqm** is outside the {area_lo:.0f}–"
            f"{area_hi:.0f} sqm recorded for {flat_type} flats here."
        )

    return errors, warnings

errors, warnings = validate()

# ===================================================================
# PREDICTION
# ===================================================================
def base_row(**overrides):
    """The flat as entered, with any field swapped out for a what-if value."""
    row = {
        "town": town,
        "flat_type": flat_type,
        "street_name": street_name,
        "floor_area_sqm": float(floor_area_sqm),
        "flat_model": flat_model,
        "remaining_lease_years": float(remaining_lease_years),
        "months_since_2017": int(months_since_2017),
        "storey_mid": float(storey_mid),
    }
    row.update(overrides)
    return row

def predict(rows):
    """Turn raw form values into the 97 columns the model expects, and predict.

    Accepts a list of row dicts so the comparison table can price several
    what-if flats in a single call.
    """
    frame = pd.DataFrame(rows)
    # Rebuild the engineered feature from notebook section 6.3.
    frame["area_per_room"] = frame["floor_area_sqm"] / frame["flat_type"].map(ROOMS)
    # One-hot, then line the columns up with the model's training columns.
    # reindex adds every column these flats do not have as 0, in the right order.
    frame = pd.get_dummies(frame).reindex(columns=model_columns, fill_value=0)
    return model.predict(frame)

st.write("")
button_col, reset_col = st.columns([3, 1])
if button_col.button("Estimate price", type="primary", width="stretch",
                     icon=":material/calculate:"):
    st.session_state.show_estimate = True
if reset_col.button("Reset", width="stretch", icon=":material/restart_alt:"):
    st.session_state.show_estimate = False

for message in errors:
    st.error(message, icon=":material/error:")
for message in warnings:
    st.warning(message, icon=":material/warning:")

if errors:
    st.info("Fix the problem above to see an estimate.", icon=":material/info:")

elif st.session_state.get("show_estimate"):
    try:
        price = float(predict([base_row()])[0])
    except Exception as exc:                                # noqa: BLE001
        st.error(
            f"The estimate could not be calculated ({type(exc).__name__}: {exc}). "
            "Please adjust the inputs and try again.",
            icon=":material/error:",
        )
    else:
        low, high = price - TEST_MAE, price + TEST_MAE
        st.markdown(
            f"""
            <div class="result">
              <div class="caption">Estimated resale price</div>
              <div class="amount">${price:,.0f}</div>
              <div class="band">Likely range
                <b>${low:,.0f}</b> to <b>${high:,.0f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Streamlit reads $...$ as LaTeX maths, so two dollar amounts in one
        # string would swallow the text between them. Escaped with a backslash.
        st.caption(
            f"The single best guess is \\${price:,.0f}. The range is the model's "
            f"typical error of plus or minus \\${TEST_MAE:,} on flats it had never "
            "seen before — treat it as the sensible negotiating window rather "
            "than a guarantee."
        )

        median = RECENT_MEDIAN.get((town, flat_type))
        if median:
            diff, pct = price - median, (price - median) / median * 100
            st.metric(
                label=f"Compared with recent {flat_type.lower()} sales in {town.title()}",
                value=f"${median:,} median",
                delta=f"{diff:+,.0f} ({pct:+.1f}%)",
                help="Median of the last 12 months of transactions in the dataset.",
                border=True,
            )
            if abs(pct) < 5:
                verdict = "in line with what comparable flats have been selling for."
            elif pct > 0:
                verdict = ("above the typical sale — the storey, size or timing you "
                           "chose is working in this flat's favour.")
            else:
                verdict = ("below the typical sale — usually a smaller unit, a lower "
                           "floor, or an earlier sale date.")
            st.write(f"This flat is **{verdict}**")

        # ---------------------------------------------------------- what-ifs
        # Answered with numbers rather than charts. A row like "three floors
        # higher: +$6,300" needs no axis-reading and is directly actionable.
        step(4, "What is driving this price?")
        st.write(
            "Each row re-asks the model the same question with **one** detail "
            "changed and everything else held exactly as you entered it."
        )

        bands = list(STOREY_BANDS)
        band_index = bands.index(storey_band)
        scenarios = []

        # Storey: one band up, one band down, where those bands exist.
        if band_index + 1 < len(bands):
            scenarios.append((f"Three floors higher (floor {STOREY_BANDS[bands[band_index + 1]]:.0f})",
                              {"storey_mid": STOREY_BANDS[bands[band_index + 1]]}))
        if band_index > 0:
            scenarios.append((f"Three floors lower (floor {STOREY_BANDS[bands[band_index - 1]]:.0f})",
                              {"storey_mid": STOREY_BANDS[bands[band_index - 1]]}))

        # Floor area: 10 sqm either way, but only within this flat type's range.
        if floor_area_sqm + 10 <= area_max:
            scenarios.append((f"10 sqm larger ({floor_area_sqm + 10:.0f} sqm)",
                              {"floor_area_sqm": floor_area_sqm + 10}))
        if floor_area_sqm - 10 >= area_min:
            scenarios.append((f"10 sqm smaller ({floor_area_sqm - 10:.0f} sqm)",
                              {"floor_area_sqm": floor_area_sqm - 10}))

        # Timing: the same flat sold earlier, where that date is still inside
        # the period the model learned from.
        for years, back in ((1, 12), (5, 60)):
            if months_since_2017 - back >= 0:
                scenarios.append(
                    (f"Sold {years} year{'s' if years > 1 else ''} earlier "
                     f"({month_label(months_since_2017 - back)})",
                     {"months_since_2017": months_since_2017 - back}))

        # One batched call rather than one per scenario.
        alt_prices = predict([base_row(**changes) for _, changes in scenarios])
        show_table(pd.DataFrame({
            "If the flat were…": [label for label, _ in scenarios],
            "Estimated price": [f"${p:,.0f}" for p in alt_prices],
            "Difference": [f"{p - price:+,.0f}" for p in alt_prices],
        }))
        st.caption(
            "Keep the flat exactly as entered, change only the one thing in a "
            "row, and the price moves by the amount shown. Floor area and sale "
            "timing move it most — together they account for around 86% of the "
            "model's decision."
        )

        # ---------------------------------------------------------- details
        with st.expander("The details you entered", icon=":material/list_alt:"):
            show_table(pd.DataFrame({
                "Detail": ["Town", "Street", "Flat type", "Flat model",
                           "Floor area", "Space per room", "Storey",
                           "Remaining lease", "Sale month"],
                "Value": [town.title(), street_name.title(), flat_type,
                          flat_model, f"{floor_area_sqm:.0f} sqm",
                          f"{floor_area_sqm / ROOMS[flat_type]:.1f} sqm",
                          f"Floor {int(floor)} (band {storey_band})",
                          f"{remaining_lease_years:.1f} years", sale_label],
            }))

        with st.expander("What this estimate assumes, and when to distrust it",
                         icon=":material/warning:"):
            st.markdown(
                f"""
**How the model decides.** Floor area and sale month together account for around
86% of its decision-making. Street, storey, remaining lease, flat type and flat
model make up the rest.

**What it assumes**
- Your flat is **typical for its street** — the model has never seen this
  specific unit, only the {N_TRANSACTIONS:,} sales around it. A renovated flat,
  a corner unit or one facing a rubbish chute will differ.
- The band midpoint stands in for your exact floor, since HDB only publishes
  `{storey_band}` rather than the unit number.
- Market conditions resemble those in the data. The model learned from
  Jan 2017 to {month_label(MAX_MONTH)} and **cannot forecast past that**; asking
  for a later date returns the {month_label(MAX_MONTH)} answer.

**When to trust it less**
- Rare combinations — a 2 ROOM flat, or an EXECUTIVE, where there were far fewer
  sales to learn from.
- Anything flagged with a warning above.
- Flats with unusual features the dataset does not record at all: renovation
  standard, exact facing, or proximity to an MRT entrance.

**What it is for.** Setting a realistic asking price, or sanity-checking a
listing before you view it. It is **not** a valuation — HDB requires a licensed
valuer for an actual transaction.
                """
            )

else:
    st.markdown(
        '<div class="awaiting">Set the flat\'s details above, then press '
        '<b>Estimate price</b>.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Estimates are generated by a machine learning model from historical resale "
    "transactions and are indicative only. They are not a valuation and should "
    "not be the sole basis for a buying or selling decision."
)
