"""
HDB Resale Price Predictor - Streamlit web app
Loads the trained Gradient Boosting model and predicts resale price.
Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------ config
st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏠", layout="centered")

# rooms per flat_type (EXECUTIVE assumed = 6) - must match the notebook
ROOMS = {"2 ROOM": 2, "3 ROOM": 3, "4 ROOM": 4, "5 ROOM": 5, "EXECUTIVE": 6}


# ------------------------------------------------------------------ load model
@st.cache_resource
def load_model():
    """Load the saved model + column list once and cache it."""
    model = joblib.load("resale_price_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns


model, model_columns = load_model()


def options_for(prefix):
    """Recover the valid categories from the model's one-hot column names,
    so dropdowns can only offer values the model was actually trained on."""
    return sorted(c[len(prefix):] for c in model_columns if c.startswith(prefix))


towns = options_for("town_")
flat_types = options_for("flat_type_")
flat_models = options_for("flat_model_")
streets = options_for("street_name_")

# ------------------------------------------------------------------ UI
st.title("🏠 HDB Resale Price Predictor")
st.write(
    "Estimate the resale price of an HDB flat in **Punggol, Sengkang or Hougang** "
    "using a Gradient Boosting model trained on resale transactions since 2017."
)

col1, col2 = st.columns(2)
with col1:
    town = st.selectbox("Town", towns)
    flat_type = st.selectbox("Flat type", flat_types)
    flat_model = st.selectbox("Flat model", flat_models)
    street_name = st.selectbox("Street name", streets)
with col2:
    floor_area_sqm = st.slider("Floor area (sqm)", 30, 200, 93)
    storey_mid = st.slider("Storey (mid of range)", 1, 50, 10)
    remaining_lease_years = st.slider("Remaining lease (years)", 40, 99, 90)
    months_since_2017 = st.slider("Months since Jan 2017", 0, 120, 108)

# ------------------------------------------------------------------ predict
if st.button("Predict resale price", type="primary"):
    # 1. build the raw row exactly like a dataset row
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

    # 2. rebuild the engineered feature the model was trained on
    rooms = ROOMS.get(flat_type, 4)
    row["area_per_room"] = row["floor_area_sqm"] / rooms

    # 3. one-hot, then line columns up with the model's training columns
    row = pd.get_dummies(row)
    row = row.reindex(columns=model_columns, fill_value=0)

    # 4. predict
    price = model.predict(row)[0]
    st.success(f"### Estimated resale price: ${price:,.0f}")
    st.caption("Estimate only, based on historical transactions. Actual price may vary.")
