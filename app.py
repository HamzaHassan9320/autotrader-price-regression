# app.py
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
from src.utils import list_makes, list_models

MODEL_PATH = Path("models/gbr.pkl") 

@st.cache_resource(show_spinner="Loading model…")
def load_model(path=MODEL_PATH):
    return joblib.load(path)

pipe = load_model()

st.title("🚗 AutoTrader Price Estimator")
st.markdown(
    "Fill in the advert details below and hit **Predict** to see an estimated price (£)."
)

# user inputs
col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input("Mileage", 0, 500_000, value=50_000, step=1_000)
    vehicle_age = st.number_input("Vehicle age (years)", 0, 50, value=5)
    body_type = st.selectbox(
        "Body type",
        ["SUV", "Hatchback", "Saloon", "Estate", "Coupe", "Convertible", "MPV", "Other"],
    )
    fuel_type = st.selectbox(
        "Fuel type",
        ["Petrol", "Diesel", "Hybrid", "Electric", "Other"],
    )

with col2:
    makes = [""] + list_makes()        
    standard_make = st.selectbox("Make",makes,index=0,format_func=lambda x: "Select make…" if x == "" else x)
    models = list_models(standard_make) if standard_make else []
    models = [""] + models
    standard_model = st.selectbox("Model",models,index=0,format_func=lambda x: "Select model…" if x == "" else x)
    standard_colour = st.text_input("Colour", "Black")
    crossover_flag  = st.checkbox("Car-and-Van crossover?", False)

# assemble dataframe 
if st.button("Predict"):
    df = pd.DataFrame(
        {
            "mileage": [mileage],
            "vehicle_age": [vehicle_age],
            "mileage_to_age_ratio": [mileage / max(vehicle_age, 1)],
            "standard_colour": [standard_colour],
            "standard_make":   [standard_make],
            "standard_model":  [standard_model],
            "vehicle_condition": ["USED"],         
            "body_type": [body_type],
            "crossover_car_and_van": [crossover_flag],
            "fuel_type": [fuel_type],
        }
    )

    pred = pipe.predict(df)[0]
    st.success(f"Estimated price: **£{pred:,.0f}**")
