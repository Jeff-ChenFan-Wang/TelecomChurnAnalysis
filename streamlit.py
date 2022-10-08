import streamlit as st
import requests
import pandas as pd
import io
import json

st.title('Telecom churn')

# FastAPI endpoint
endpoint = 'http://localhost:8000/predict'

st.subheader('View Sample of Test Set')
st.write('Test 1')
gender = st.selectbox('Gender:', ['Male','Female'])
contract = st.selectbox('Contract Type:', ['Month-to-Month','One year','Two year'])
internetService = st.selectbox('Internet Service', ['DSL, Fiber optic, No'])
paymentMethod = st.selectbox('Payment Method',['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'])

tenure = st.number_input('Tenure:')
monthlyCharges = st.number_input('Monthly Charges:')
totalCharges = st.number_input('Total Charges:')

seniorCitizen = st.radio('Senior Citizen',('True','False'))
partner = st.radio('Has Partner',('True','False'))
dependents = st.radio('Has Dependents',('True','False'))
phoneService = st.radio('Has Phone Service',('True','False'))
multipleLines = st.radio('Has Multiple Lines',('True','False'))
onlineSecurity = st.radio('Subscribed to Online Security',('True','False'))
onlineBackup = st.radio('Subscribed to Online Backup',('True','False'))
deviceProtection = st.radio('Subscribed to Device Protection',('True','False'))
techSupport = st.radio('Subscribed to Tech Support',('True','False'))
streamingTv = st.radio('Subscribed to Streaming TV',('True','False'))
streamingMovies = st.radio('Subscribed to Streaming Movies',('True','False'))
paperlessBilling = st.radio('Subscribed to Paperless Biling',('True','False'))

if st.button('Start Prediction'):
    with st.spinner('Prediction in Progress. Please Wait...'):
        output = requests.post(endpoint, raw_data=, timeout=8000)

