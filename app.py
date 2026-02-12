# 1 Good(Lower Risk)
# 0 Bad(Higher Risk)

import streamlit as st
import pandas as pd 
import joblib

model= joblib.load("xgb_credit_model.pkl")
encoders= {col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

st.title("CREDIT RISK PREDICTION")
st.write("Enter applicant information to predict if the credit risk is good or bad.")

age= st.number_input("Age", min_value=18, max_value=80, value=30)
sex= st.selectbox("Sex", ["male","female"])
job= st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing= st.selectbox("Housing", ["own","free","rent"])
saving_accounts= st.selectbox("Saving Accounts",["little","moderate","rich","quite rich"] )
checking_account= st.selectbox("Checking Accounts",["little","moderate","rich"] )
duration= st.number_input("Duration (months)", min_value=1, value=12)
credit_amount= st.number_input("Credit Amount", min_value=0, value=100)

input_df=pd.DataFrame({
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Age": [age],
    "Job": [job],
    "Housing":[encoders["Housing"].transform([housing])[0]],
    "Credit amount": [credit_amount],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Duration": [duration]
})

if st.button("Predict Risk"):
    pred= model.predict(input_df)[0]

    if pred==1:
        st.success("Predicted Credit Risk is: **GOOD**")
    if pred==0:
        st.error("Predicted Credit Risk is: **BAD**")