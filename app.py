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

Design notes
------------
Built for someone who has never used a valuation tool and may not know their
flat's details. Three principles:

1. Five questions on screen, not eight. Flat model, lease and sale month are
   the ones an ordinary buyer is least likely to know, so they sit behind an
   optional section with sensible defaults already filled in.
2. Every answer is pre-filled with the most common value for that kind of flat,
   so pressing the button immediately gives a believable number.
3. One chart, not three. Price over time is the one people intuitively expect;
   the others needed a paragraph of explanation each to be useful.

Icons are Material Symbols (Streamlit's ":material/name:" syntax) rather than
emoji, so they inherit the theme instead of rendering differently per OS. Money
is set in tabular figures so digits stay aligned as values change.
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
    initial_sidebar_state="collapsed",   # start clean; detail is opt-in
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

# Plain-English hint for each flat type, so someone can pick without knowing
# HDB's naming.
FLAT_TYPE_HINT = {
    "2 ROOM": "1 bedroom", "3 ROOM": "2 bedrooms", "4 ROOM": "3 bedrooms",
    "5 ROOM": "3 bedrooms + extra living space", "EXECUTIVE": "Largest, often double-storey",
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

# Chart colours. This pair was checked with a palette validator and passes all
# six checks - lightness band, chroma floor, colour-blind separation,
# normal-vision separation and contrast against the surface - in light AND dark.
SERIES = "#2563EB"       # the model's response curve
HIGHLIGHT = "#B45309"    # the flat the user actually entered
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
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

      :root {
          --navy:      #0F172A;
          --navy-soft: #1E3A5F;
          --ink:       #0F172A;
          --ink-muted: #64748B;
          --rule:      #E2E8F0;
          --well:      #F8FAFC;
      }

      html, body, .stApp, [data-testid="stSidebar"] {
          font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
      }

      /* Money must use tabular figures, or digits shift width as values change. */
      .result .amount, [data-testid="stMetricValue"],
      [data-testid="stMetricDelta"] {
          font-variant-numeric: tabular-nums;
          font-feature-settings: "tnum";
      }

      /* The headline result. A navy rule on the left rather than a coloured
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
          font-size: 3rem; font-weight: 700; color: var(--ink);
          line-height: 1.1; margin: .4rem 0 .35rem 0; letter-spacing: -.02em;
      }
      .result .band {
          font-size: .95rem; color: var(--ink-muted);
          padding-top: .6rem; border-top: 1px solid var(--rule);
      }
      .result .band b { color: var(--ink); font-weight: 600; }

      /* Empty state, before anything has been estimated. */
      .awaiting {
          border: 1px dashed var(--rule); border-radius: 4px;
          padding: 2.4rem 1.5rem; text-align: center;
          color: var(--ink-muted); font-size: .95rem; background: var(--well);
      }

      [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }

      /* Streamlit's primary button is red by default; recolour to the navy
         accent here so the whole app stays in this single file. */
      [data-testid^="stBaseButton-primary"] {
          background: var(--navy-soft) !important;
          border-color: var(--navy-soft) !important;
          color: #fff !important;
      }
      [data-testid^="stBaseButton-primary"]:hover {
          background: var(--navy) !important; border-color: var(--navy) !important;
      }

      /* Keyboard focus must stay visible - never remove the ring. */
      *:focus-visible {
          outline: 2px solid var(--navy-soft) !important;
          outline-offset: 2px !important;
      }

      /* The app follows the viewer's system setting, so the custom surfaces
         need a dark variant. Steps are chosen for the dark background rather
         than mechanically inverted. */
      @media (prefers-color-scheme: dark) {
          :root {
              --ink: #E8EDF5; --ink-muted: #94A3B8;
              --rule: #2A3648; --well: #161C27; --navy-soft: #5B8DD6;
          }
          .result { background: #131A25; }
          [data-testid^="stBaseButton-primary"] {
              background: #2D5A94 !important; border-color: #2D5A94 !important;
          }
          [data-testid^="stBaseButton-primary"]:hover {
              background: #3A6FB0 !important; border-color: #3A6FB0 !important;
          }
      }

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
# HEADER
# ===================================================================
st.title("What is my HDB flat worth?", anchor=False)
st.markdown(
    "Get an estimated resale price for a flat in **Punggol, Sengkang or "
    "Hougang**, based on 47,864 real sales since 2017."
)
st.caption(
    "Answer the questions below and press Estimate. Every field already has the "
    "most common answer filled in, so you can change only what you know."
)

# ===================================================================
# INPUTS - the five questions almost anyone can answer
# ===================================================================
left, right = st.columns(2)

with left:
    town = st.selectbox("Which town?", sorted(t for t in TOWN_STREETS if t in KNOWN_TOWNS))

    # Filtered by town: this stops an impossible address such as
    # Hougang + PUNGGOL DR from ever reaching the model.
    streets = [s for s in TOWN_STREETS[town] if s in KNOWN_STREETS]
    street_name = st.selectbox("Which street?", streets)

    storey_band = st.selectbox(
        "Which floor?", list(STOREY_BANDS), index=3,
        help="HDB gives the floor as a range of 3, not the exact unit.",
    )
    storey_mid = STOREY_BANDS[storey_band]

with right:
    flat_type = st.selectbox(
        "What size of flat?",
        [t for t in FLAT_TYPE_AREA if t in KNOWN_TYPES],
        index=2,                       # 4 ROOM: about half of all sales here
        format_func=lambda t: f"{t}  ({FLAT_TYPE_HINT[t]})",
    )

    area_min, area_max, area_typical = FLAT_TYPE_AREA[flat_type]
    floor_area_sqm = st.slider(
        "How big is it? (sqm)",
        min_value=float(area_min), max_value=float(area_max),
        value=float(area_typical), step=1.0,
    )
    st.caption(
        f"Most **{flat_type}** flats are **{area_typical:.0f} sqm**. "
        "Leave it if you are not sure."
    )

# The three fields an ordinary buyer is least likely to know, tucked away with
# sensible defaults already applied. Progressive disclosure: the form asks five
# questions, not eight, but nothing is lost.
with st.expander("Know more details? (optional)", icon=":material/tune:"):
    d_left, d_right = st.columns(2)

    with d_left:
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
            help="HDB's design name for the layout. It is on the resale listing.",
        )

        lease_lo, lease_hi, lease_typical = TOWN_LEASE[town]
        remaining_lease_years = st.slider(
            "Years left on the lease",
            min_value=40.0, max_value=99.0,
            value=float(lease_typical), step=0.5,
            help="HDB flats start with 99 years. Check HDB's My Flat Dashboard.",
        )

    with d_right:
        sale_label = st.select_slider(
            "When is the sale?", options=MONTH_OPTIONS, value=MONTH_OPTIONS[-1],
            help="Prices rose a lot over this period, so timing matters.",
        )
        months_since_2017 = MONTH_LOOKUP[sale_label]
        st.caption(
            "Leave these alone if you are unsure — they are already set to the "
            f"most common values for a {flat_type} flat in {town.title()}."
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
            "this tool covers, so it cannot be priced."
        )
    if street_name not in TOWN_STREETS.get(town, []):
        errors.append(
            f"**{street_name}** is not in **{town.title()}**. Please pick a "
            "street that matches the town."
        )
    if flat_model not in FLAT_TYPE_MODELS.get(flat_type, []):
        errors.append(
            f"No **{flat_type}** flat in these towns is a **{flat_model}**. "
            "Please choose a different flat model."
        )

    # Soft checks: possible, but outside the range the model learned from, so
    # the estimate is an extrapolation and should be read with more caution.
    lease_lo, lease_hi, _ = TOWN_LEASE[town]
    if not lease_lo - 0.5 <= remaining_lease_years <= lease_hi + 0.5:
        warnings.append(
            f"Flats in {town.title()} normally have {lease_lo:.0f}–{lease_hi:.0f} "
            f"years left, not {remaining_lease_years:.0f}. The estimate below is "
            "less reliable than usual."
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

    Accepts a list of row dicts so the price-history chart can price dozens of
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
if st.button("Estimate price", type="primary", width="stretch",
             icon=":material/calculate:"):
    st.session_state.show_estimate = True

for message in errors:
    st.error(message, icon=":material/error:")
for message in warnings:
    st.warning(message, icon=":material/warning:")

if errors:
    st.info("Please fix the problem above to see an estimate.", icon=":material/info:")

elif st.session_state.get("show_estimate"):
    try:
        price = float(predict([base_row()])[0])
    except Exception as exc:                                # noqa: BLE001
        st.error(
            f"The estimate could not be calculated ({type(exc).__name__}: {exc}). "
            "Please adjust your answers and try again.",
            icon=":material/error:",
        )
    else:
        low, high = price - TEST_MAE, price + TEST_MAE
        st.markdown(
            f"""
            <div class="result">
              <div class="caption">Estimated price</div>
              <div class="amount">${price:,.0f}</div>
              <div class="band">Most likely between
                <b>${low:,.0f}</b> and <b>${high:,.0f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # One plain-English sentence comparing to what really sold, instead of a
        # metric widget the user has to interpret.
        median = RECENT_MEDIAN.get((town, flat_type))
        if median:
            pct = (price - median) / median * 100
            if abs(pct) < 5:
                verdict = (f"That is about the same as other {flat_type.lower()} flats "
                           f"in {town.title()}, which sold for around **${median:,}** "
                           "over the past year.")
            elif pct > 0:
                verdict = (f"That is **{pct:.0f}% higher** than the typical "
                           f"{flat_type.lower()} flat in {town.title()} "
                           f"(**${median:,}**) — usually a higher floor, a bigger "
                           "unit, or a more recent sale date.")
            else:
                verdict = (f"That is **{abs(pct):.0f}% lower** than the typical "
                           f"{flat_type.lower()} flat in {town.title()} "
                           f"(**${median:,}**) — usually a lower floor, a smaller "
                           "unit, or an earlier sale date.")
            st.write(verdict)

        # ------------------------------------------------- price history chart
        # Kept because "what has this been worth over time" is the one question
        # people ask without prompting, and the answer needs no explanation.
        st.subheader("What this flat would have cost over the years", anchor=False)

        months = sorted({*range(0, MAX_MONTH + 1, 3), months_since_2017, MAX_MONTH})
        history = pd.DataFrame({
            "date": [pd.Timestamp(2017, 1, 1) + pd.DateOffset(months=m) for m in months],
            "price": predict([base_row(months_since_2017=m) for m in months]),
        })
        marker = history[history["date"] == pd.Timestamp(2017, 1, 1)
                         + pd.DateOffset(months=months_since_2017)]

        y = alt.Y("price:Q", title="Price",
                  axis=alt.Axis(format="$,.0f", titlePadding=10),
                  scale=alt.Scale(zero=False))
        x = alt.X("date:T", title="Year", axis=alt.Axis(titlePadding=10, format="%Y"))

        line = alt.Chart(history).mark_line(color=SERIES, strokeWidth=2.5).encode(
            x=x, y=y,
            tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"),
                     alt.Tooltip("price:Q", title="Price", format="$,.0f")],
        )
        # The user's own flat, marked and named directly on the chart so it is
        # never identified by colour alone.
        dot = alt.Chart(marker).mark_point(
            color=HIGHLIGHT, fill=HIGHLIGHT, size=170, stroke="white", strokeWidth=2,
        ).encode(x=x, y=y)
        label = alt.Chart(marker).mark_text(
            text="Your flat", color=HIGHLIGHT, dy=-16, fontSize=12,
            fontWeight="bold", font=CHART_FONT,
        ).encode(x=x, y=y)

        st.altair_chart(
            (line + dot + label).properties(height=260)
            .configure_view(strokeWidth=0)
            .configure_axis(
                grid=True, gridColor=GRID, gridOpacity=0.9,
                domainColor=GRID, tickColor=GRID,
                labelColor=INK_MUTED, titleColor=INK_MUTED,
                labelFont=CHART_FONT, titleFont=CHART_FONT,
                labelFontSize=11, titleFontSize=11, titleFontWeight=600,
            ),
            width="stretch",
        )
        st.caption(
            "The same flat, priced at every point since 2017. Hover any point to "
            "see the figure. The line stops at Jul 2026 — the most recent sales "
            "available, so the tool cannot predict beyond it."
        )

        # ------------------------------------------------------------ details
        with st.expander("How accurate is this?", icon=":material/help:"):
            st.markdown(
                f"""
This estimate is typically within **${TEST_MAE:,}** of the real selling price.
That was measured on **{N_TEST:,} real sales** the model had never seen while
learning — about **3.4%** of a typical \\$520,000 flat.

**It works best for** ordinary flats: a 4- or 5-room unit on a normal floor,
sold recently. Most sales look like this.

**Treat it with more caution for** 2-room or Executive flats (far fewer sales to
learn from), or anything shown with a warning above.

**It cannot know** how renovated your flat is, which way it faces, or how close
it is to an MRT entrance — none of that is in the public data.

**It is not a valuation.** HDB requires a licensed valuer for an actual sale.
                """
            )

        with st.expander("What you told us", icon=":material/list_alt:"):
            st.dataframe(
                pd.DataFrame({
                    "Detail": ["Town", "Street", "Flat size", "Flat model",
                               "Floor area", "Floor", "Lease remaining", "Sale month"],
                    "Your answer": [town.title(), street_name.title(), flat_type,
                                    flat_model, f"{floor_area_sqm:.0f} sqm",
                                    f"Floors {storey_band}",
                                    f"{remaining_lease_years:.0f} years", sale_label],
                }),
                hide_index=True, width="stretch",
            )

else:
    st.markdown(
        '<div class="awaiting">Answer the questions above, then press '
        '<b>Estimate price</b>.</div>',
        unsafe_allow_html=True,
    )


# ===================================================================
# SIDEBAR - background detail, collapsed by default
# ===================================================================
with st.sidebar:
    st.subheader("About", anchor=False)
    st.write(
        "Prices are predicted by a **Gradient Boosting** machine learning model "
        f"trained on **{N_TRANSACTIONS:,} real HDB resale transactions** in "
        "Punggol, Sengkang and Hougang since 2017."
    )
    c1, c2 = st.columns(2)
    c1.metric("Typical error", f"${TEST_MAE:,}", border=True)
    c2.metric("R squared", f"{TEST_R2:.3f}", border=True)
    st.caption(
        f"Measured on {N_TEST:,} held-out sales. R squared is the share of the "
        "price difference between flats that the model explains."
    )
    st.markdown(
        "**Covers:** Punggol, Sengkang, Hougang  \n"
        f"**Sales up to:** {month_label(MAX_MONTH)}  \n"
        "**Source:** data.gov.sg — HDB Resale Flat Prices"
    )

st.divider()
st.caption(
    "Estimates come from a machine learning model trained on past sales and are "
    "indicative only. They are not a valuation and should not be the sole basis "
    "for a buying or selling decision."
)
