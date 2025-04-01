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

def make_forecast(forecast_dates, blend_weights):
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

    df.loc[df["is_closed"] == 1, ["Predicted Minis", "Predicted Full Size"]] = 0
    
    # Round predictions to whole numbers
    df["Predicted Minis"] = df["Predicted Minis"].round().astype(int)
    df["Predicted Full Size"] = df["Predicted Full Size"].round().astype(int)
    
    return df

# --- Streamlit Interface ---
st.title("🍪 Gladstone Sales Forecast")
st.markdown("Select your desired forecast date range to generate predictions.")

# Date range selection
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today())
with col2:
    end_date = st.date_input("End Date", value=datetime.today() + timedelta(days=14))

if st.button("📊 Generate Forecast"):
    forecast_dates = forecast_range(start_date, end_date)
    
    # Generate forecast using the models
    blend_df = make_forecast(forecast_dates, blend_weights=(0.7, 0.3, 0.8, 0.2))

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