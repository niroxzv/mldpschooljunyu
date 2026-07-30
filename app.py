"""
HDB Resale Price Estimator - Streamlit web app
================================================
Loads the tuned Gradient Boosting model saved by the notebook (section 6.4,
round 1: 600 trees / max_depth 5 / learning_rate 0.1) and estimates the resale
price of a flat in Punggol, Sengkang or Hougang.

Run locally with:  streamlit run app.py

The app never reads the CSV. It only needs the two artefacts the notebook saves
(resale_price_model.pkl, model_columns.pkl), which keeps the deployment small.
Any figure quoted in the interface is therefore hard-coded below and labelled
with where it came from.

The charts are not stored data - they are produced by asking the model itself
what it would predict as one input is varied and everything else is held fixed.

Visual design
-------------
Swiss / minimalist: a strict grid, one accent colour, generous whitespace and no
decoration that does not carry information. Icons are Material Symbols (via
Streamlit's ":material/name:" syntax) rather than emoji, so they inherit the
theme instead of rendering as a different picture on every operating system.
Money is set in tabular figures so digits line up and the layout does not jitter
as values change. Theme colours live in .streamlit/config.toml.
"""

import altair as alt          # ships with Streamlit; no extra requirement
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
# Every constant below is measured from the same 47,864 transactions the model
# was trained on (Punggol / Sengkang / Hougang, Jan 2017 - Jul 2026). They let
# the app offer only combinations the model has actually seen, warn the user
# when an input falls outside that experience, and pre-fill typical values so
# someone who does not know their flat's details still gets a sensible answer.

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
# The third value pre-fills the slider so the form opens on a realistic flat.
FLAT_TYPE_AREA = {
    "2 ROOM":    (37.0, 50.0, 47.0),
    "3 ROOM":    (59.0, 92.0, 67.0),
    "4 ROOM":    (82.0, 123.0, 93.0),
    "5 ROOM":    (108.0, 152.0, 112.0),
    "EXECUTIVE": (125.0, 177.0, 137.0),
}

# flat_type -> the flat models that actually exist for it. A 2 ROOM flat is never
# a Maisonette, so offering that combination would invite a nonsense prediction.
FLAT_TYPE_MODELS = {
    "2 ROOM":    ["2-Room", "Model A"],
    "3 ROOM":    ["DBSS", "Improved", "Model A", "New Generation",
                  "Premium Apartment", "Simplified"],
    "4 ROOM":    ["DBSS", "Improved", "Model A", "Model A2", "New Generation",
                  "Premium Apartment", "Simplified"],
    "5 ROOM":    ["DBSS", "Improved", "Model A", "Other", "Premium Apartment"],
    "EXECUTIVE": ["Apartment", "Maisonette", "Other", "Premium Apartment"],
}

# Remaining lease actually observed per town, and the typical value used as the
# slider default. Punggol and Sengkang are young estates, so a 50-year lease
# there is not a real flat.
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

# Chart colours, drawn from the same navy/bronze palette as the theme. This pair
# was checked with a palette validator and passes all six checks - lightness
# band, chroma floor, colour-blind separation, normal-vision separation and
# contrast against the surface - in BOTH light and dark mode.
SERIES = "#2563EB"       # the model's response curve
HIGHLIGHT = "#B45309"    # the flat the user actually entered
INK = "#0F172A"
INK_MUTED = "#64748B"
GRID = "#E2E8F0"
CHART_FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"


def month_label(m):
    """Turn months-since-Jan-2017 into a label a person can read."""
    return f"{MONTH_NAMES[m % 12]} {2017 + m // 12}"


MONTH_OPTIONS = [month_label(m) for m in range(MAX_MONTH + 1)]
MONTH_LOOKUP = {label: m for m, label in enumerate(MONTH_OPTIONS)}


# ===================================================================
# STYLING
# ===================================================================
# Swiss / minimalist treatment: one typeface, one accent, thin rules instead of
# heavy shadows, and tabular figures so money columns stay aligned. Values are
# defined once as custom properties so nothing is a stray hex further down.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

      :root {
          --navy:      #0F172A;
          --navy-soft: #1E3A5F;
          --accent:    #B45309;
          --ink:       #0F172A;
          --ink-muted: #64748B;
          --rule:      #E2E8F0;
          --well:      #F8FAFC;
      }

      html, body, .stApp, [data-testid="stSidebar"] {
          font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
      }

      /* Money must use tabular figures, or digits shift width as values change. */
      .tnum, .result .amount, [data-testid="stMetricValue"],
      [data-testid="stMetricDelta"] {
          font-variant-numeric: tabular-nums;
          font-feature-settings: "tnum";
      }

      /* Small uppercase eyebrow above each section, in place of a numbered
         heading. Carries the step number without shouting it. */
      .eyebrow {
          font-size: .7rem; font-weight: 600; letter-spacing: .14em;
          text-transform: uppercase; color: var(--ink-muted);
          margin: 1.9rem 0 .3rem 0;
      }
      .eyebrow:first-of-type { margin-top: .9rem; }
      .section-title {
          font-size: 1.14rem; font-weight: 600; color: var(--ink);
          margin: 0 0 .85rem 0; letter-spacing: -.01em;
      }

      /* The headline result. A navy rule on the left instead of a coloured
         fill, so the number itself is the loudest thing on the page. */
      .result {
          border: 1px solid var(--rule);
          border-left: 3px solid var(--navy-soft);
          border-radius: 4px;
          padding: 1.6rem 1.5rem;
          background: #fff;
      }
      .result .caption {
          font-size: .7rem; font-weight: 600; letter-spacing: .14em;
          text-transform: uppercase; color: var(--ink-muted);
      }
      .result .amount {
          font-size: 2.85rem; font-weight: 700; color: var(--ink);
          line-height: 1.1; margin: .4rem 0 .35rem 0; letter-spacing: -.02em;
      }
      .result .band {
          font-size: .92rem; color: var(--ink-muted);
          padding-top: .55rem; border-top: 1px solid var(--rule);
      }
      .result .band b { color: var(--ink); font-weight: 600; }

      /* Empty state, before anything has been estimated. */
      .awaiting {
          border: 1px dashed var(--rule); border-radius: 4px;
          padding: 2.3rem 1.5rem; text-align: center;
          color: var(--ink-muted); font-size: .95rem; background: var(--well);
      }

      [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }

      /* Streamlit's own primary button is red by default. Recolour it to the
         navy accent here rather than in a .streamlit/config.toml, so the whole
         app stays in this single file. */
      [data-testid^="stBaseButton-primary"] {
          background: var(--navy-soft) !important;
          border-color: var(--navy-soft) !important;
          color: #fff !important;
      }
      [data-testid^="stBaseButton-primary"]:hover {
          background: var(--navy) !important;
          border-color: var(--navy) !important;
      }

      /* Keyboard focus must stay visible - never remove the ring. */
      *:focus-visible {
          outline: 2px solid var(--navy-soft) !important;
          outline-offset: 2px !important;
      }

      /* Without a pinned theme the app follows the viewer's system setting, so
         the custom surfaces need a dark variant too. Steps are chosen for the
         dark background rather than mechanically inverted, and the chart colours
         were validated against both surfaces. */
      @media (prefers-color-scheme: dark) {
          :root {
              --ink:       #E8EDF5;
              --ink-muted: #94A3B8;
              --rule:      #2A3648;
              --well:      #161C27;
              --navy-soft: #5B8DD6;
          }
          .result { background: #131A25; }
          [data-testid^="stBaseButton-primary"] {
              background: #2D5A94 !important; border-color: #2D5A94 !important;
          }
          [data-testid^="stBaseButton-primary"]:hover {
              background: #3A6FB0 !important; border-color: #3A6FB0 !important;
          }
      }

      /* Honour a reduced-motion preference. */
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


def eyebrow(step, title):
    """Small uppercase step label followed by a section heading."""
    st.markdown(
        f'<div class="eyebrow">Step {step}</div>'
        f'<div class="section-title">{title}</div>',
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


# Only ever offer a category the model was actually trained on. If the notebook
# is re-run with different data, the app narrows its options instead of sending
# the model a column it has never seen.
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

    st.markdown('<div class="eyebrow">Accuracy</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Typical error", f"${TEST_MAE:,}", help=(
        f"Mean absolute error on {N_TEST:,} held-out sales the model never saw "
        "while training. Roughly half of all estimates land closer than this."
    ), border=True)
    c2.metric("R squared", f"{TEST_R2:.3f}", help=(
        "The share of the price variation between flats that the model explains."
    ), border=True)
    st.caption(
        f"${TEST_MAE:,} is about 3.4% of a typical $520,000 flat — close enough "
        "to anchor an asking price or judge whether a listing is fair."
    )

    st.markdown('<div class="eyebrow">Scope</div>', unsafe_allow_html=True)
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
    "**First time here?** Every field is pre-filled with the most common answer "
    "for that kind of flat, so you can press **Estimate price** straight away "
    "and adjust afterwards. Hover the **?** beside any field for help.",
    icon=":material/lightbulb:",
)

# ===================================================================
# INPUTS
# ===================================================================
eyebrow(1, "Where is the flat?")
loc_left, loc_right = st.columns(2)

with loc_left:
    town = st.selectbox(
        "Town",
        sorted(t for t in TOWN_STREETS if t in KNOWN_TOWNS),
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

eyebrow(2, "What kind of flat is it?")
type_left, type_right = st.columns(2)

with type_left:
    flat_type = st.selectbox(
        "Flat type",
        [t for t in FLAT_TYPE_AREA if t in KNOWN_TYPES],
        index=2,                       # 4 ROOM: about half of all sales here
        help="How HDB classifies the flat's size. A 4 ROOM has 3 bedrooms.",
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

eyebrow(3, "Which unit, and when?")
unit_left, unit_right = st.columns(2)

with unit_left:
    storey_band = st.selectbox(
        "Storey", list(STOREY_BANDS), index=3,
        help="HDB publishes the floor as a 3-storey band, not the exact unit.",
    )
    storey_mid = STOREY_BANDS[storey_band]

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
        f"({floor_area_sqm:.0f} sqm ÷ {rooms} rooms). The model uses this to "
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
| **Remaining lease** | Years left on the 99-year lease. Shorter lease usually means a lower price. | HDB's My Flat Dashboard |
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

    Accepts a list of row dicts so the sensitivity charts can price dozens of
    what-if flats in a single call.
    """
    frame = pd.DataFrame(rows)
    # Rebuild the engineered feature from notebook section 6.3.
    frame["area_per_room"] = frame["floor_area_sqm"] / frame["flat_type"].map(ROOMS)
    # One-hot, then line the columns up with the model's training columns.
    # reindex adds every column these flats do not have as 0, in the right order.
    frame = pd.get_dummies(frame).reindex(columns=model_columns, fill_value=0)
    return model.predict(frame)


def sensitivity_chart(data, x_field, x_type, x_title, marker, note):
    """One what-if curve: how the estimate moves as a single input changes.

    A single blue series, with the user's own flat marked in bronze and named
    directly on the chart, so identity never depends on colour alone. Grid lines
    stay low-contrast so they never compete with the data.
    """
    y = alt.Y("price:Q", title="Estimated price",
              axis=alt.Axis(format="$,.0f", titlePadding=10),
              scale=alt.Scale(zero=False))
    x = alt.X(f"{x_field}:{x_type}", title=x_title,
              axis=alt.Axis(titlePadding=10),
              scale=alt.Scale(zero=False) if x_type == "Q" else alt.Undefined)

    line = alt.Chart(data).mark_line(
        color=SERIES, strokeWidth=2,
        point=alt.OverlayMarkDef(color=SERIES, size=26),
    ).encode(x=x, y=y, tooltip=[
        alt.Tooltip(f"{x_field}:{x_type}", title=x_title),
        alt.Tooltip("price:Q", title="Estimate", format="$,.0f"),
    ])

    dot = alt.Chart(marker).mark_point(
        color=HIGHLIGHT, fill=HIGHLIGHT, size=150, stroke="white", strokeWidth=2,
    ).encode(x=x, y=y)

    label = alt.Chart(marker).mark_text(
        text="Your flat", color=HIGHLIGHT, dy=-14, fontSize=11,
        fontWeight="bold", font=CHART_FONT,
    ).encode(x=x, y=y)

    chart = (
        (line + dot + label)
        .properties(height=270)
        .configure_view(strokeWidth=0)
        .configure_axis(
            grid=True, gridColor=GRID, gridOpacity=0.9, gridWidth=1,
            domainColor=GRID, tickColor=GRID,
            labelColor=INK_MUTED, titleColor=INK_MUTED,
            labelFont=CHART_FONT, titleFont=CHART_FONT,
            labelFontSize=11, titleFontSize=11, titleFontWeight=600,
        )
    )
    st.altair_chart(chart, width="stretch")
    st.caption(note)


st.divider()

# The button gives a clean starting state (nothing predicted yet). Once it has
# been pressed the estimate stays on screen and refreshes as inputs change, so
# the user can compare flats without clicking again.
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
        st.caption(
            f"**How to read this:** ${price:,.0f} is the single best guess. The "
            f"range is the model's typical error of ±${TEST_MAE:,} on flats it had "
            "never seen before — treat it as the sensible negotiating window "
            "rather than a guarantee."
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
        eyebrow(4, "What is driving this price?")
        st.write(
            "Each chart re-asks the model the same question with **one** detail "
            "changed and everything else held exactly as you entered it. The "
            "bronze dot is your flat."
        )
        tab_time, tab_area, tab_storey = st.tabs([
            ":material/trending_up: Over time",
            ":material/straighten: Floor area",
            ":material/stairs: Storey",
        ])

        with tab_time:
            months = sorted({*range(0, MAX_MONTH + 1, 3), months_since_2017, MAX_MONTH})
            data = pd.DataFrame({
                "date": [pd.Timestamp(2017, 1, 1) + pd.DateOffset(months=m) for m in months],
                "price": predict([base_row(months_since_2017=m) for m in months]),
            })
            sensitivity_chart(
                data, "date", "T", "Sale month",
                data[data["date"] == pd.Timestamp(2017, 1, 1)
                     + pd.DateOffset(months=months_since_2017)],
                "What this same flat would have sold for at any point since 2017. "
                "The line stops at Jul 2026 because that is the last month in the "
                "data — the model cannot forecast beyond it.",
            )

        with tab_area:
            areas = sorted({*range(int(area_min), int(area_max) + 1,
                                   max(1, int((area_max - area_min) / 20))),
                            int(floor_area_sqm), int(area_max)})
            data = pd.DataFrame({
                "area": areas,
                "price": predict([base_row(floor_area_sqm=float(a)) for a in areas]),
            })
            sensitivity_chart(
                data, "area", "Q", "Floor area (sqm)",
                data[data["area"] == int(floor_area_sqm)],
                f"How much each extra square metre is worth for a {flat_type} on "
                f"{street_name.title()}. Floor area is the single strongest driver "
                "in this model.",
            )

        with tab_storey:
            data = pd.DataFrame({
                "storey": list(STOREY_BANDS),
                "price": predict([base_row(storey_mid=v) for v in STOREY_BANDS.values()]),
            })
            sensitivity_chart(
                data, "storey", "O", "Storey band",
                data[data["storey"] == storey_band],
                "The height premium for this flat. Higher floors usually sell for "
                "more — better view, less road noise.",
            )

        # ---------------------------------------------------------- details
        with st.expander("The details you entered", icon=":material/list_alt:"):
            st.dataframe(
                pd.DataFrame({
                    "Detail": ["Town", "Street", "Flat type", "Flat model",
                               "Floor area", "Space per room", "Storey",
                               "Remaining lease", "Sale month"],
                    "Value": [town.title(), street_name.title(), flat_type,
                              flat_model, f"{floor_area_sqm:.0f} sqm",
                              f"{floor_area_sqm / ROOMS[flat_type]:.1f} sqm",
                              f"Floors {storey_band}",
                              f"{remaining_lease_years:.1f} years", sale_label],
                }),
                hide_index=True, width="stretch",
            )

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
  Jan 2017 – {month_label(MAX_MONTH)} and **cannot forecast past that**; asking
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
