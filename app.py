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
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------- page config
st.set_page_config(
    page_title="HDB Resale Price Estimator",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ===================================================================
# REFERENCE DATA
# ===================================================================
# Every constant below is measured from the same 47,864 transactions the model
# was trained on (Punggol / Sengkang / Hougang, Jan 2017 - Jul 2026). They let
# the app offer only combinations the model has actually seen, and warn the user
# when an input falls outside that experience instead of quietly guessing.

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

# Remaining lease actually observed per town. Punggol and Sengkang are young
# estates, so a 50-year lease there is not a real flat.
TOWN_LEASE = {
    "HOUGANG":  (48.2, 95.9),
    "PUNGGOL":  (75.4, 96.0),
    "SENGKANG": (71.5, 96.2),
}

# HDB records storeys in 3-floor bands. The model was trained on the band's
# midpoint, so the app collects the band and converts, exactly like the notebook.
STOREY_BANDS = {
    "01 to 03": 2.0, "04 to 06": 5.0, "07 to 09": 8.0, "10 to 12": 11.0,
    "13 to 15": 14.0, "16 to 18": 17.0, "19 to 21": 20.0, "22 to 24": 23.0,
    "25 to 27": 26.0,
}

# Median price of the last 12 months of transactions, used to give the estimate
# some context rather than showing a bare number.
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


def month_label(m):
    """Turn months-since-Jan-2017 into a label a person can read."""
    return f"{MONTH_NAMES[m % 12]} {2017 + m // 12}"


MONTH_OPTIONS = [month_label(m) for m in range(MAX_MONTH + 1)]
MONTH_LOOKUP = {label: m for m, label in enumerate(MONTH_OPTIONS)}


# ===================================================================
# STYLING
# ===================================================================
# Colours are given as rgba over the theme's own background, so the app stays
# readable whether the viewer is in Streamlit's light or dark theme.
st.markdown(
    """
    <style>
      .price-card {
          border: 1px solid rgba(128,128,128,.28);
          border-radius: 14px;
          padding: 1.5rem 1.25rem;
          text-align: center;
          background: rgba(128,128,128,.06);
      }
      .price-card .label {
          font-size: .8rem; letter-spacing: .09em; text-transform: uppercase;
          opacity: .65;
      }
      .price-card .value {
          font-size: 2.9rem; font-weight: 700; line-height: 1.15;
          margin: .35rem 0 .1rem 0;
      }
      .price-card .range { font-size: .95rem; opacity: .75; }
      .placeholder {
          border: 1px dashed rgba(128,128,128,.4);
          border-radius: 14px; padding: 2.1rem 1.25rem;
          text-align: center; opacity: .7;
      }
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
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
    st.title("🏠 HDB Resale Price Estimator")
    st.error(load_error, icon="🚫")
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
    st.header("About this estimator")
    st.write(
        "Prices are predicted by a **Gradient Boosting** model trained on every "
        f"HDB resale transaction in Punggol, Sengkang and Hougang since 2017 "
        f"({N_TRANSACTIONS:,} sales)."
    )

    st.subheader("How accurate is it?")
    c1, c2 = st.columns(2)
    c1.metric("Typical error", f"${TEST_MAE:,}", help=(
        "Mean absolute error on 9,573 held-out sales the model never saw during "
        "training. Half of all estimates land closer than this, half further."
    ))
    c2.metric("R²", f"{TEST_R2:.3f}", help=(
        "Share of the price variation between flats that the model explains."
    ))
    st.caption(
        f"${TEST_MAE:,} is about 3.4% of a typical $520,000 flat — close enough "
        "to anchor an asking price or judge whether a listing is fair."
    )

    st.subheader("Good to know")
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
st.title("🏠 HDB Resale Price Estimator")
st.write(
    "Thinking of buying or selling in **Punggol, Sengkang or Hougang**? "
    "Fill in the flat's details below to see what it is likely to be worth."
)

# ===================================================================
# INPUTS
# ===================================================================
st.subheader("1 · Where is the flat?")
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

st.subheader("2 · What kind of flat is it?")
type_left, type_right = st.columns(2)

with type_left:
    flat_type = st.selectbox(
        "Flat type",
        [t for t in FLAT_TYPE_AREA if t in KNOWN_TYPES],
        index=2,                       # 4 ROOM: half of all sales in this area
        help="Number of rooms, as HDB classifies it.",
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
        help="HDB's design/layout name for the flat, shown on the resale listing.",
    )

area_min, area_max, area_typical = FLAT_TYPE_AREA[flat_type]
floor_area_sqm = st.slider(
    "Floor area (sqm)",
    min_value=float(area_min), max_value=float(area_max),
    value=float(area_typical), step=1.0,
    help=(
        f"{flat_type} flats in these towns range from {area_min:.0f} to "
        f"{area_max:.0f} sqm; {area_typical:.0f} sqm is the most common."
    ),
)

st.subheader("3 · Which unit, and when?")
unit_left, unit_right = st.columns(2)

with unit_left:
    storey_band = st.selectbox(
        "Storey", list(STOREY_BANDS), index=3,
        help="HDB publishes the floor as a 3-storey band rather than the exact unit.",
    )
    storey_mid = STOREY_BANDS[storey_band]

    lease_lo, lease_hi = TOWN_LEASE[town]
    remaining_lease_years = st.slider(
        "Remaining lease (years)",
        min_value=40.0, max_value=99.0,
        value=float(round((lease_lo + lease_hi) / 2)), step=0.5,
        help=(
            f"Flats sold in {town.title()} had between {lease_lo:.0f} and "
            f"{lease_hi:.0f} years left. It is printed on the resale listing."
        ),
    )

with unit_right:
    sale_label = st.select_slider(
        "Sale month", options=MONTH_OPTIONS, value=MONTH_OPTIONS[-1],
        help=(
            "When the sale happens. Prices in these towns rose roughly 50% "
            "across the period, so timing matters."
        ),
    )
    months_since_2017 = MONTH_LOOKUP[sale_label]

    rooms = ROOMS[flat_type]
    st.caption(
        f"**Space per room:** {floor_area_sqm / rooms:.1f} sqm "
        f"({floor_area_sqm:.0f} sqm ÷ {rooms} rooms) — the model uses this as "
        "a measure of how generous the layout is."
    )


# ===================================================================
# INPUT VALIDATION
# ===================================================================
def validate(town, flat_type, street_name, flat_model, floor_area_sqm,
             remaining_lease_years):
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
    lease_lo, lease_hi = TOWN_LEASE[town]
    if not lease_lo - 0.5 <= remaining_lease_years <= lease_hi + 0.5:
        warnings.append(
            f"A **{remaining_lease_years:.0f}-year** lease is outside the "
            f"{lease_lo:.0f}–{lease_hi:.0f} years seen in {town.title()}. "
            f"{town.title()} flats are mostly {'newer' if lease_lo > 60 else 'older'} "
            "than this, so the estimate is less reliable."
        )

    area_lo, area_hi, _ = FLAT_TYPE_AREA[flat_type]
    if not area_lo <= floor_area_sqm <= area_hi:
        warnings.append(
            f"**{floor_area_sqm:.0f} sqm** is outside the {area_lo:.0f}–"
            f"{area_hi:.0f} sqm recorded for {flat_type} flats here."
        )

    return errors, warnings


errors, warnings = validate(town, flat_type, street_name, flat_model,
                            floor_area_sqm, remaining_lease_years)


# ===================================================================
# PREDICTION
# ===================================================================
def build_row():
    """Turn the form inputs into the exact 97-column row the model expects."""
    row = pd.DataFrame([{
        "town": town,
        "flat_type": flat_type,
        "street_name": street_name,
        "floor_area_sqm": float(floor_area_sqm),
        "flat_model": flat_model,
        "remaining_lease_years": float(remaining_lease_years),
        "months_since_2017": int(months_since_2017),
        "storey_mid": float(storey_mid),
    }])

    # Rebuild the engineered feature from section 6.3 the same way the notebook did.
    row["area_per_room"] = row["floor_area_sqm"] / ROOMS[flat_type]

    # One-hot, then line the columns up with the model's training columns.
    # reindex adds every column this flat does not have as 0, in the right order.
    row = pd.get_dummies(row)
    return row.reindex(columns=model_columns, fill_value=0)


st.divider()

# The button gives a clean starting state (nothing predicted yet). Once it has
# been pressed the estimate stays on screen and refreshes as inputs change, so
# the user can compare flats without clicking again.
button_col, reset_col = st.columns([3, 1])
if button_col.button("Estimate price", type="primary", width="stretch"):
    st.session_state.show_estimate = True
if reset_col.button("Reset", width="stretch"):
    st.session_state.show_estimate = False

for message in errors:
    st.error(message, icon="🚫")
for message in warnings:
    st.warning(message, icon="⚠️")

if errors:
    st.info("Fix the problem above to see an estimate.", icon="ℹ️")

elif st.session_state.get("show_estimate"):
    try:
        price = float(model.predict(build_row())[0])
    except Exception as exc:                                # noqa: BLE001
        st.error(
            f"The estimate could not be calculated ({type(exc).__name__}: {exc}). "
            "Please adjust the inputs and try again.",
            icon="🚫",
        )
    else:
        low, high = price - TEST_MAE, price + TEST_MAE
        st.markdown(
            f"""
            <div class="price-card">
              <div class="label">Estimated resale price</div>
              <div class="value">${price:,.0f}</div>
              <div class="range">Typically within ${low:,.0f} – ${high:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"The range is the model's typical error of ±${TEST_MAE:,} on flats "
            "it had never seen before."
        )

        median = RECENT_MEDIAN.get((town, flat_type))
        if median:
            diff = price - median
            pct = diff / median * 100
            st.metric(
                label=f"Compared with recent {flat_type.lower()} sales in {town.title()}",
                value=f"${median:,} median",
                delta=f"{diff:+,.0f} ({pct:+.1f}%)",
                help="Median of the last 12 months of transactions in the dataset.",
            )
            if abs(pct) < 5:
                verdict = "in line with what comparable flats have been selling for."
            elif pct > 0:
                verdict = (
                    "above the typical sale — the storey, size or timing you chose "
                    "is working in this flat's favour."
                )
            else:
                verdict = (
                    "below the typical sale — usually a smaller unit, a lower "
                    "floor, or an earlier sale date."
                )
            st.write(f"This flat is **{verdict}**")

        with st.expander("What went into this estimate?"):
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
            st.caption(
                "Floor area and sale month are the two strongest drivers in this "
                "model, together accounting for around 86% of its decision-making."
            )

else:
    st.markdown(
        '<div class="placeholder">Set the flat\'s details above, then press '
        '<b>Estimate price</b>.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Estimates are generated by a machine learning model from historical resale "
    "transactions and are indicative only. They are not a valuation and should "
    "not be the sole basis for a buying or selling decision."
)
