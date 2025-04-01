# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Load models ---
minis_prophet = joblib.load("minis_prophet.pkl")
full_prophet = joblib.load("full_prophet.pkl")
minis_xgb = joblib.load("minis_xgb.pkl")
full_xgb = joblib.load("full_xgb.pkl")

# --- Helper functions ---
def forecast_range(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date)

def make_forecast(forecast_dates, blend_weights, recent_sales=None):
    df = pd.DataFrame({"Date": forecast_dates})
    df["dow"] = df["Date"].dt.weekday
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["is_closed"] = (df["dow"] == 6).astype(int)
    df["is_early_close"] = 0
    df["cap"] = 800

    prophet_input = df.rename(columns={"Date": "ds"})
    minis_prophet_pred = minis_prophet.predict(prophet_input)[["ds", "yhat"]].set_index("ds")["yhat"].clip(0, 350)
    full_prophet_pred = full_prophet.predict(prophet_input)[["ds", "yhat"]].set_index("ds")["yhat"].clip(0, 800)

    xgb_features = ["dow", "month", "day", "is_early_close", "is_closed"]
    minis_xgb_pred = minis_xgb.predict(df[xgb_features])
    full_xgb_pred = full_xgb.predict(df[xgb_features])

    df["Predicted Minis"] = blend_weights[0] * minis_prophet_pred.values + blend_weights[1] * minis_xgb_pred
    df["Predicted Full Size"] = blend_weights[2] * full_prophet_pred.values + blend_weights[3] * full_xgb_pred

    # Adjust predictions based on recent sales if provided
    if recent_sales is not None:
        # Calculate the average recent sales
        recent_minis_avg = np.mean([x[0] for x in recent_sales])
        recent_full_avg = np.mean([x[1] for x in recent_sales])
        
        # Calculate the ratio between recent and predicted sales for the first day
        minis_ratio = recent_minis_avg / df["Predicted Minis"].iloc[0] if df["Predicted Minis"].iloc[0] > 0 else 1
        full_ratio = recent_full_avg / df["Predicted Full Size"].iloc[0] if df["Predicted Full Size"].iloc[0] > 0 else 1
        
        # Apply the ratio to all predictions, with a gradual fade-out effect
        for i in range(len(df)):
            fade_factor = max(0, 1 - (i / 7))  # Fade out over a week
            df.loc[df.index[i], "Predicted Minis"] *= (1 + (minis_ratio - 1) * fade_factor)
            df.loc[df.index[i], "Predicted Full Size"] *= (1 + (full_ratio - 1) * fade_factor)

    df.loc[df["is_closed"] == 1, ["Predicted Minis", "Predicted Full Size"]] = 0
    
    # Round predictions to whole numbers
    df["Predicted Minis"] = df["Predicted Minis"].round().astype(int)
    df["Predicted Full Size"] = df["Predicted Full Size"].round().astype(int)
    
    return df

# --- Streamlit Interface ---
st.title("🍪 Gladstone Sales Forecast")
# Optional recent sales input
st.subheader("📊 Recent Sales (Optional)")
st.markdown("Enter sales data from the last three non-Sunday days to improve forecast accuracy.")

recent_sales = []
col1, col2 = st.columns(2)

# Get the last three non-Sunday days
current_date = datetime.today()
days_to_show = []
days_found = 0
days_back = 1  # Start from yesterday

while days_found < 3:
    check_date = current_date - timedelta(days=days_back)
    if check_date.weekday() != 6:  # Not Sunday
        days_to_show.append(check_date)
        days_found += 1
    days_back += 1

# Show inputs in reverse chronological order
for date in reversed(days_to_show):
    with col1:
        minis = st.number_input(f"{date.strftime('%A')} - Minis", min_value=0, value=0, key=f"mini_{date.strftime('%Y%m%d')}")
    with col2:
        full = st.number_input(f"{date.strftime('%A')} - Full Size", min_value=0, value=0, key=f"full_{date.strftime('%Y%m%d')}")
    recent_sales.append((minis, full))

# Date range selection
st.subheader("📅 Forecast Range")
st.markdown("Select your desired forecast date range to generate predictions.")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today())
with col2:
    end_date = st.date_input("End Date", value=datetime.today() + timedelta(days=14))

if st.button("📊 Generate Forecast"):
    forecast_dates = forecast_range(start_date, end_date)
    
    # Only use recent sales if at least one day has non-zero values
    recent_sales_to_use = recent_sales if any(x[0] > 0 or x[1] > 0 for x in recent_sales) else None
    
    # Generate forecast using the models
    blend_df = make_forecast(forecast_dates, blend_weights=(0.7, 0.3, 0.8, 0.2), recent_sales=recent_sales_to_use)

    st.subheader("🔮 Forecasted Sales")
    # Create a copy for display with formatted dates
    display_df = blend_df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%A, %Y-%m-%d")
    st.dataframe(display_df[["Date", "Predicted Minis", "Predicted Full Size"]].set_index("Date"))

    # Plot
    st.subheader("📈 Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(blend_df["Date"], blend_df["Predicted Minis"], label="Minis", marker='o')
    ax.plot(blend_df["Date"], blend_df["Predicted Full Size"], label="Full Size", marker='x')
    ax.set_title("Predicted Sales by Day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cookies")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Weekly totals
    blend_df["Week"] = blend_df["Date"].dt.to_period("W").apply(lambda x: x.start_time.strftime("%Y-%m-%d"))
    weekly_totals = blend_df.groupby("Week")[["Predicted Minis", "Predicted Full Size"]].sum().reset_index()
    st.subheader("📦 Total Cookies Forecasted Per Week")
    st.dataframe(weekly_totals.set_index("Week"))