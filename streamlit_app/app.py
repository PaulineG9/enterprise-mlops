"""
app.py
------
Streamlit demo for the Asset Failure-Risk model.

Loads the trained model DIRECTLY from a local folder bundled into this
repo (see model/README.md for how that folder gets populated) -- there is
no live Azure endpoint involved, so this app costs nothing to keep
running beyond free-tier Streamlit Community Cloud hosting. The model
itself was trained in Azure ML / Databricks; this app is just a
read-only, self-contained way to demo its predictions.

Run locally with:
    streamlit run app.py
"""

import os

import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

ASSET_TYPES = ["Haul-Truck", "Conveyor-Belt", "Crusher", "Drill-Rig", "Pump-Station"]
REGIONS = ["Pilbara-WA", "Hunter-Valley-NSW", "Bowen-Basin-QLD", "Goldfields-WA", "Kalgoorlie-WA"]

st.set_page_config(page_title="Asset Failure-Risk Demo", page_icon="\U0001F6A8", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "MLmodel")):
        st.error(
            "No model found in the `model/` folder. See `model/README.md` for "
            "how to download it from Azure ML and place it here."
        )
        st.stop()
    return mlflow.sklearn.load_model(MODEL_DIR)


def main():
    st.title("\U0001F6A8 Asset Failure-Risk Predictor")
    st.caption(
        "A live demo of a model trained end-to-end on Databricks + Azure ML, "
        "as part of the [enterprise-mlops](https://github.com/) portfolio project. "
        "This app loads the trained model directly -- no live cloud endpoint, "
        "no ongoing compute cost."
    )

    model = load_model()

    st.subheader("Asset details")
    col1, col2 = st.columns(2)
    with col1:
        asset_type = st.selectbox("Asset type", ASSET_TYPES)
        region = st.selectbox("Region", REGIONS)
        age_days = st.number_input("Asset age (days)", min_value=0, value=900, step=1)
        cycle_count = st.number_input("Cycle count", min_value=0, value=20000, step=100)
    with col2:
        runtime_hours = st.number_input("Runtime in last reading (hours)", min_value=0.0, value=20.0, step=0.5)
        vibration = st.slider("Vibration (mm/s)", 0.0, 15.0, 2.5, 0.1)
        heat = st.slider("Heat (\u00b0C)", 40.0, 130.0, 65.0, 0.5)
        pressure = st.slider("Pressure (kPa)", 60.0, 150.0, 120.0, 0.5)

    st.subheader("Recent trend (rolling averages)")
    st.caption(
        "In production these come from a short rolling window over recent "
        "telemetry readings (see `src/etl_pipeline.py`). For this demo, they "
        "default to match the current readings above -- adjust them to "
        "simulate a rising trend."
    )
    col3, col4 = st.columns(2)
    with col3:
        rolling_vibration_avg = st.slider("Rolling avg vibration (mm/s)", 0.0, 15.0, vibration, 0.1)
    with col4:
        rolling_heat_avg = st.slider("Rolling avg heat (\u00b0C)", 40.0, 130.0, heat, 0.5)

    risk_flag = st.checkbox(
        "Risk flag already triggered (rolling vibration AND heat both above the fleet-wide threshold)",
        value=False,
    )

    input_df = pd.DataFrame([{
        "vibration_mm_s": vibration,
        "heat_celsius": heat,
        "pressure_kpa": pressure,
        "runtime_hours": runtime_hours,
        "cycle_count": cycle_count,
        "age_days": age_days,
        "rolling_vibration_avg": rolling_vibration_avg,
        "rolling_heat_avg": rolling_heat_avg,
        "risk_flag": int(risk_flag),
        "asset_type": asset_type,
        "region": region,
    }])

    st.divider()

    if st.button("Predict failure risk", type="primary", use_container_width=True):
        proba = model.predict_proba(input_df)[0, 1]
        prediction = int(proba >= 0.5)

        st.metric("Predicted probability of failure within 7 days", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

        if prediction == 1:
            st.error("\u26a0\ufe0f Model predicts HIGH risk -- recommend scheduling maintenance.")
        else:
            st.success("\u2705 Model predicts LOW risk -- no action needed based on current readings.")

        with st.expander("Raw input sent to the model"):
            st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()
