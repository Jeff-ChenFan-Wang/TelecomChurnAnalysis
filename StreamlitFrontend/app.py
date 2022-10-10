import streamlit as st
import requests


st.title('Telecom Churn Prediction')

# FastAPI endpoint
endpoint = 'http://localhost:8000/predict'

with st.container():
    st.subheader('Fill out customer information to find out how likely they will churn:')
    
with st.container():
    col_misc, col_num = st.columns(2)
    with col_misc:
        gender = st.selectbox('Gender:', ['Male','Female'])
        contract = st.selectbox('Contract Type:', ['Month-to-Month','One year','Two year'])
        internetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        paymentMethod = st.selectbox('Payment Method',['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
            'Credit card (automatic)'])
    with col_num:
        tenure = st.number_input('Tenure with Company:',step=1)
        monthlyCharges = st.number_input('Monthly Charges:')
        totalCharges = st.number_input('Total Customer Lifetime Charges:')

with st.container():
    col_bool1, col_bool2, col_bool3 = st.columns(3)
    with col_bool1:
        seniorCitizen = st.radio('Is Senior Citizen',('True','False'))
        partner = st.radio('Has Partner',('True','False'))
        dependents = st.radio('Has Dependents',('True','False'))
        phoneService = st.radio('Has Phone Service',('True','False'))

    with col_bool2:
        multipleLines = st.radio('Has Multiple Lines',('True','False'))
        onlineSecurity = st.radio('Subscribed to Online Security',('True','False'))
        onlineBackup = st.radio('Subscribed to Online Backup',('True','False'))
        deviceProtection = st.radio('Subscribed to Device Protection',('True','False'))
    
    with col_bool3:
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

button_col1, button_col2, button_col3 = st.columns(3)
with button_col2:
    if st.button('Start Prediction'):
        with st.spinner('Prediction in Progress. Please Wait...'):
            out_json = dict(zip(feature_names,var_names))
            
            output = requests.post(
                url=endpoint, 
                json=out_json
            )
            
            churn_result = output.json()['churn_result']
            churn_prob = output.json()['churn_probability']
            if churn_result:
                st.error('Customer Likely to Churn', icon="🚨")
            else:
                st.success('Customer Likely to Stay', icon="✅")
            st.metric(label="Churn Probability", value=f'{round(churn_prob*100,2)}%')

