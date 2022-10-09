import streamlit as st
import requests


st.title('Telecom churn')

# FastAPI endpoint
endpoint = 'http://localhost:8000/predict'


st.subheader('Insert Customer Information to Find Out How Likely They Will Churn')
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

feature_names = ['gender', 'tenure', 'InternetService', 'Contract', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner',
       'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'PaperlessBilling']
var_names = [[gender],[tenure],[internetService],[contract],[paymentMethod],
             [monthlyCharges],[totalCharges],[bool(seniorCitizen)],
             [bool(partner)],[bool(dependents)],[bool(phoneService)],
             [bool(multipleLines)],[bool(onlineSecurity)],[bool(onlineBackup)],
             [bool(deviceProtection)],[bool(techSupport)],[bool(streamingTv)],
             [bool(streamingMovies)],[bool(paperlessBilling)]]

if st.button('Start Prediction'):
    with st.spinner('Prediction in Progress. Please Wait...'):
        out_json = dict(zip(feature_names,var_names))
        
        output = requests.post(
            url=endpoint, 
            json=out_json
        )
        
        churn_result = output.json()['churn_result']
        churn_prob = output.json()['churn_result']
        st.write(churn_result)
        st.write(churn_prob)

