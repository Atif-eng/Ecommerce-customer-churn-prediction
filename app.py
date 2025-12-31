import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the trained model
model_filename = 'customer_churn_model.pkl'

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: The file '{model_filename}' was not found. Please upload it.")
    st.stop()

st.title("E-Commerce Churn Predictor (Simplified)")
st.write("Enter the 13 metrics below to predict customer churn.")

# --- USER INPUT FORM ---
st.sidebar.header("Customer Metrics")

def get_user_input():
    # 1. Engagement & Activity
    age = st.sidebar.number_input("Age", value=30)
    login_freq = st.sidebar.number_input("Login Frequency", value=15.0)
    session_avg = st.sidebar.number_input("Session Duration Avg", value=30.0)
    pages_session = st.sidebar.number_input("Pages Per Session", value=10.0)
    
    # 2. Shopping Habits
    cart_abandon = st.sidebar.number_input("Cart Abandonment Rate (%)", value=20.0)
    wishlist = st.sidebar.number_input("Wishlist Items", value=2.0)
    total_purchases = st.sidebar.number_input("Total Purchases", value=10.0)
    days_since_purchase = st.sidebar.number_input("Days Since Last Purchase", value=10.0)
    
    # 3. Interactions & Value
    discount_usage = st.sidebar.number_input("Discount Usage Rate (%)", value=10.0)
    email_open = st.sidebar.number_input("Email Open Rate (%)", value=30.0)
    cust_calls = st.sidebar.number_input("Customer Service Calls", value=1.0)
    reviews = st.sidebar.number_input("Product Reviews Written", value=1.0)
    lifetime_val = st.sidebar.number_input("Lifetime Value ($)", value=1000.0)

    # Store in a Dictionary (Order doesn't matter here, we fix it later)
    data = {
        'Age': age,
        'Login_Frequency': login_freq,
        'Session_Duration_Avg': session_avg,
        'Pages_Per_Session': pages_session,
        'Cart_Abandonment_Rate': cart_abandon,
        'Wishlist_Items': wishlist,
        'Total_Purchases': total_purchases,
        'Days_Since_Last_Purchase': days_since_purchase,
        'Discount_Usage_Rate': discount_usage,
        'Email_Open_Rate': email_open,
        'Customer_Service_Calls': cust_calls,
        'Product_Reviews_Written': reviews,
        'Lifetime_Value': lifetime_val
    }

    # Convert to DataFrame
    df = pd.DataFrame(data, index=[0])
    return df

# Get Inputs
input_df = get_user_input()

# --- FORCE EXACT COLUMN ORDER ---
# Based on your code: x = data[['Age', 'Login_Frequency', ...]]
expected_order = [
    'Age', 
    'Login_Frequency', 
    'Session_Duration_Avg', 
    'Pages_Per_Session',
    'Cart_Abandonment_Rate', 
    'Wishlist_Items', 
    'Total_Purchases',
    'Days_Since_Last_Purchase',
    'Discount_Usage_Rate',
    'Email_Open_Rate',
    'Customer_Service_Calls', 
    'Product_Reviews_Written',
    'Lifetime_Value'
]

# Reorder columns to match model training
input_df = input_df[expected_order]

# Display Summary
st.subheader("Input Data")
st.dataframe(input_df)

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)
        
        churn_prob = prob[0][1]
        
        st.subheader("Result")
        if prediction[0] == 1:
            st.error(f"CHURN PREDICTED (Probability: {churn_prob:.1%})")
        else:
            st.success(f"NO CHURN (Probability: {churn_prob:.1%})")
            
    except Exception as e:
        st.error(f"Error: {e}")