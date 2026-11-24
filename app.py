import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Bike Sharing Demand Predictor",
    page_icon="🚲",
    layout="wide"
)

# App title
st.title("🚲 Bike Sharing Demand Predictor")

st.write(
    "Predict hourly bike rental demand using a trained "
    "XGBoost machine learning model."
)

# Load trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_model",
    "xgboost_tuned_model.pkl"
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

st.success("Model loaded successfully!")

st.header("Prediction Settings")

st.write("Enter the date, time and weather conditions.")

# Date and time
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Date",
        value=pd.to_datetime("2012-07-01"),
        min_value=pd.to_datetime("2011-01-01"),
        max_value=pd.to_datetime("2012-12-31")
    )

with col2:
    selected_time = st.time_input(
        "Hour",
        value=pd.to_datetime("08:00").time()
    )

# Holiday
is_holiday = st.checkbox("Public Holiday")

# Weather conditions
st.subheader("Weather Conditions")

col3, col4 = st.columns(2)

with col3:
    weather = st.selectbox(
        "Weather",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear / Few Clouds",
            2: "Mist / Cloudy",
            3: "Light Rain / Snow",
            4: "Heavy Rain / Snow"
        }[x]
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-8.0,
        max_value=39.0,
        value=20.0
    )

    feels_like = st.number_input(
        "Feels-like Temperature (°C)",
        min_value=-16.0,
        max_value=50.0,
        value=20.0
    )

with col4:
    humidity = st.slider(
        "Humidity (%)",
        min_value=0,
        max_value=100,
        value=60
    )

    windspeed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=67.0,
        value=10.0
    )

# Convert user inputs into model features

hour = selected_time.hour
weekday = (selected_date.weekday() + 1) % 7
month = selected_date.month

# Encode year
yr = selected_date.year - 2011

# Encode season using the dataset's season mapping
season_mapping = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3,
    10: 4, 11: 4, 12: 4
}

season = season_mapping[month]

# Calculate working day
holiday = int(is_holiday)
workingday = int(weekday not in [0, 6] and holiday == 0)

# Normalize weather variables
normalized_temp = (temperature + 8) / 47
normalized_atemp = (feels_like + 16) / 66
normalized_hum = humidity / 100
normalized_windspeed = windspeed / 67

# Create model input
input_data = pd.DataFrame([{
    "hour_sin": np.sin(2 * np.pi * hour / 24),
    "hour_cos": np.cos(2 * np.pi * hour / 24),
    "weekday_sin": np.sin(2 * np.pi * weekday / 7),
    "weekday_cos": np.cos(2 * np.pi * weekday / 7),
    "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
    "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    "yr": yr,
    "holiday": holiday,
    "workingday": workingday,
    "season": season,
    "weathersit": weather,
    "temp": normalized_temp,
    "atemp": normalized_atemp,
    "hum": normalized_hum,
    "windspeed": normalized_windspeed
}])

# Ensure the feature order matches model training
input_data = input_data[[
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "yr", "holiday", "workingday",
    "season", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]]

st.subheader("Generated Model Features")
st.dataframe(input_data)

# Prediction
st.divider()

st.header("Bike Rental Demand Prediction")

if st.button("Predict Bike Rentals", type="primary"):

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Ensure prediction is non-negative
    prediction = max(0, prediction)

    # Display prediction
    st.subheader("Predicted Bike Rentals")

    st.metric(
        label="Estimated Rentals",
        value=f"{prediction:,.0f} bikes"
    )

    # Display demand level
    if prediction < 100:
        demand_level = "Low Demand"
    elif prediction < 300:
        demand_level = "Moderate Demand"
    elif prediction < 500:
        demand_level = "High Demand"
    else:
        demand_level = "Very High Demand"

    st.info(f"Demand Level: {demand_level}")

# Model Performance Dashboard

st.divider()

st.header("Model Performance")

st.write(
    "Performance of the selected XGBoost model "
    "on the held-out test dataset."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "43.726")

with col2:
    st.metric("RMSE", "66.231")

with col3:
    st.metric("R² Score", "0.910")

# Feature Importance

st.divider()

st.header("Feature Importance")

st.write(
    "The chart shows the relative importance of features "
    "used by the trained XGBoost model."
)

# Extract trained XGBoost model
fitted_preprocessor = model.named_steps["preprocessor"]
fitted_xgb = model.named_steps["model"]

# Get feature names and importance scores
feature_names = fitted_preprocessor.get_feature_names_out()
feature_importances = fitted_xgb.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
)

top_features = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(15)
    .sort_values("Importance", ascending=True)
)

# Create horizontal bar chart
fig = px.bar(
    top_features,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 15 Feature Importance - XGBoost Tuned",
    labels={
        "Importance": "Importance Score",
        "Feature": "Feature"
    }
)

fig.update_layout(
    yaxis=dict(
        categoryorder="array",
        categoryarray=top_features["Feature"].tolist()
    ),
    height=550
)

st.plotly_chart(fig, use_container_width=True)