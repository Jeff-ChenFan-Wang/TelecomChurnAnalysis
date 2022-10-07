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
carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)
