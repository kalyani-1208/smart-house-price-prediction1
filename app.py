
import streamlit as st
import pandas as pd
import joblib
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "joblib"
])

import os

# -----------------------------
# Paths
# -----------------------------

PROJECT_PATH = "."

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "house_price_model.pkl"
)

DATA_PATH = os.path.join(
    PROJECT_PATH,
    "cleaned_house_prices.csv"
)

# -----------------------------
# Load model and data
# -----------------------------

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Page
# -----------------------------

st.set_page_config(
    page_title="Smart House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI-Based Smart House Price Prediction")
st.subheader("Real Estate Analysis System")

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("🏠 Property Details")

state = st.sidebar.selectbox(
    "State",
    sorted(df["State"].unique())
)

city_list = sorted(
    df[df["State"] == state]["City"].unique()
)

city = st.sidebar.selectbox(
    "City",
    city_list
)

locality_list = sorted(
    df[
        (df["State"] == state) &
        (df["City"] == city)
    ]["Locality"].unique()
)

locality = st.sidebar.selectbox(
    "Locality",
    locality_list
)

property_type = st.sidebar.selectbox(
    "Property Type",
    sorted(df["Property_Type"].unique())
)

bhk = st.sidebar.number_input(
    "BHK",
    min_value=1,
    max_value=10,
    value=2
)

size = st.sidebar.number_input(
    "Size (Sq Ft)",
    min_value=300,
    max_value=10000,
    value=1200
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1950,
    max_value=2026,
    value=2018
)

furnished = st.sidebar.selectbox(
    "Furnished Status",
    sorted(df["Furnished_Status"].unique())
)

floor = st.sidebar.number_input(
    "Floor No",
    min_value=0,
    max_value=100,
    value=5
)

total_floors = st.sidebar.number_input(
    "Total Floors",
    min_value=1,
    max_value=100,
    value=10
)

nearby_schools = st.sidebar.number_input(
    "Nearby Schools",
    min_value=0,
    max_value=50,
    value=5
)

nearby_hospitals = st.sidebar.number_input(
    "Nearby Hospitals",
    min_value=0,
    max_value=50,
    value=3
)

transport = st.sidebar.selectbox(
    "Public Transport",
    sorted(df["Public_Transport_Accessibility"].unique())
)

parking = st.sidebar.selectbox(
    "Parking",
    sorted(df["Parking_Space"].unique())
)

security = st.sidebar.selectbox(
    "Security",
    sorted(df["Security"].unique())
)

facing = st.sidebar.selectbox(
    "Facing",
    sorted(df["Facing"].unique())
)

owner_type = st.sidebar.selectbox(
    "Owner Type",
    sorted(df["Owner_Type"].unique())
)

availability = st.sidebar.selectbox(
    "Availability Status",
    sorted(df["Availability_Status"].unique())
)

# -----------------------------
# Derived features
# -----------------------------

age = 2026 - year_built

floor_ratio = (
    floor / total_floors
    if total_floors > 0
    else 0
)

nearby_facilities = (
    nearby_schools +
    nearby_hospitals
)

amenities = st.sidebar.text_input(
    "Amenities",
    "Playground, Gym, Garden"
)

amenities_count = len([
    x for x in amenities.split(",")
    if x.strip()
])

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict House Price"):

    input_data = pd.DataFrame({
        "BHK": [bhk],
        "Size_in_SqFt": [size],
        "Year_Built": [year_built],
        "Floor_No": [floor],
        "Total_Floors": [total_floors],
        "Age_of_Property": [age],
        "Nearby_Schools": [nearby_schools],
        "Nearby_Hospitals": [nearby_hospitals],
        "Floor_Ratio": [floor_ratio],
        "Amenities_Count": [amenities_count],
        "Nearby_Facilities": [nearby_facilities],
        "State": [state],
        "City": [city],
        "Property_Type": [property_type],
        "Furnished_Status": [furnished],
        "Public_Transport_Accessibility": [transport],
        "Parking_Space": [parking],
        "Security": [security],
        "Facing": [facing],
        "Owner_Type": [owner_type],
        "Availability_Status": [availability]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"🏠 Estimated House Price: ₹{prediction:.2f} Lakhs"
    )

    st.info(
        f"Approximately ₹{prediction / 100:.2f} Crore"
    )

# -----------------------------
# Dashboard Metrics
# -----------------------------

st.markdown("---")

st.header("📊 Real Estate Analytics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Properties",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Average Price",
        f"₹{df['Price_in_Lakhs'].mean():.2f} L"
    )

with col3:
    st.metric(
        "Average Size",
        f"{df['Size_in_SqFt'].mean():.0f} Sq Ft"
    )

# -----------------------------
# City Analysis
# -----------------------------

st.subheader("🏙️ Average Price by City")

city_analysis = (
    df.groupby("City")["Price_in_Lakhs"]
      .mean()
      .sort_values(ascending=False)
      .head(15)
)

st.bar_chart(city_analysis)

# -----------------------------
# Property Type
# -----------------------------

st.subheader("🏡 Average Price by Property Type")

property_analysis = (
    df.groupby("Property_Type")["Price_in_Lakhs"]
      .mean()
      .sort_values(ascending=False)
)

st.bar_chart(property_analysis)

# -----------------------------
# BHK Analysis
# -----------------------------

st.subheader("🛏️ Average Price by BHK")

bhk_analysis = (
    df.groupby("BHK")["Price_in_Lakhs"]
      .mean()
      .sort_index()
)

st.line_chart(bhk_analysis)

# -----------------------------
# City Property Count
# -----------------------------

st.subheader("📍 Properties by City")

city_count = (
    df["City"]
    .value_counts()
    .head(15)
)

st.bar_chart(city_count)
