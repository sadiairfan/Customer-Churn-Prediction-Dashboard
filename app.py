import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and scaler
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Web App Dashboard Title and Description
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Enter the customer details below to predict the risk of churn.")

# User Inputs (English Labels)
age = st.slider("Customer Age", 18, 70, 35)
gender = st.selectbox("Customer Gender", ["Male", "Female"])
tenure = st.slider("Tenure (Months)", 1, 60, 12)
monthly_spending = st.number_input("Monthly Spending ($)", 20.0, 200.0, 50.0)
purchases = st.slider("Number of Purchases", 1, 15, 5)
support_requests = st.slider("Customer Support Requests", 0, 9, 2)
login_freq = st.slider("Login Frequency (Per Month)", 1, 30, 15)
satisfaction = st.slider("Customer Satisfaction Score (1-5)", 1, 5, 4)

# Dummy features to match dataset shape
city_and_sub_inputs = [0, 0, 0, 0, 0, 0]

# Prediction Button
if st.button("Predict Churn Risk"):
    gender_encoded = 1 if gender == "Male" else 0

    # Structure input data
    input_data = [age, gender_encoded, monthly_spending, tenure, purchases, support_requests, login_freq, satisfaction] + city_and_sub_inputs
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ High Churn Risk! This customer is likely to leave the business.")
    else:
        st.success("✅ Low Churn Risk! This customer is safe and likely to stay.")
