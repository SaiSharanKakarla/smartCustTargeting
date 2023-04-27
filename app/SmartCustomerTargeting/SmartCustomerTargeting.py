## Loading packages
#from pyexpat import features
from calendar import c
import code
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pickle

def styleing_cust(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

styleing_cust("css_styling.css")

## Loading requried data
data = pd.read_csv('bank-full.csv', sep = ';')
featureImp = pd.read_csv('rf_feature_imp.csv').sort_values(['Relative Importance'], ascending = False).head(10)
#FF4B4B


## Application Header
st.markdown('<div style = "background-color: rgb(0, 56, 120); text-align: center;"> <h5 style = "color: white; padding: 5px;"> Smart Banking Customer Targeting  </h5> </div>', unsafe_allow_html=True)


### Sidebar: For input selection
st.sidebar.markdown('''<b>Customer Details</b>''', unsafe_allow_html=True)

# Sidebar Feature Selection

## Job
job = st.sidebar.selectbox("Job", set(data['job']))
## Marital
marital = st.sidebar.selectbox("Martial Status", set(data['marital']))
## education
day = st.sidebar.selectbox("Day", set(data['day']))
## education
education = st.sidebar.selectbox("Education", set(data['education']))
## poutcome
poutcome = st.sidebar.selectbox("Poutcome", set(data['poutcome']))
    ## poutcome
contact = st.sidebar.selectbox("Contact", set(data['contact']))
    ## campaign
campaign = st.sidebar.selectbox("Campaign", set(data['campaign']))
## previous
previous = st.sidebar.selectbox("Previous", set(data['previous']))
## Balance
balance = int(st.sidebar.text_input("Account Balance (in $)", 1000))
## Duration
duration = int(st.sidebar.text_input("Select Call Duration (in Sec)", 60))
## pdays
pdays = int(st.sidebar.text_input("dpays?", 10))
## Housing
housing = st.sidebar.radio("Has Own House?", set(data['housing']),  horizontal=True)
## Loan
loan = st.sidebar.radio("Has Existing Loan?", set(data['loan']),  horizontal=True)
## default
default = st.sidebar.radio( "Loan Default?", set(data['default']),  horizontal=True)
## Age
age = st.sidebar.slider('Age', 0, 100, 25)

userInputObj = [{'job': job, 'marital': marital, 'day': day, 'poutcome': poutcome, 'campaign': campaign, 'previous': previous, 'contact':contact, 'education':education, 'balance': balance, 'duration': duration, 'pdays': pdays, 'housing': housing, 'loan': loan, 'default': default, 'age': age}]



## 
tab1, tab2, tab3 = st.tabs(['About', 'Recommendations', 'Key Drivers'])

#####################################
### Tab1: For Application Summary
#####################################

tab1.markdown('''
Traditionally businesses reach out to prospect customer for encashing potential opportunities (can be in terms of cross sell or upsell). Though this process of targeting customer for potential cross sell or upsell in observed in many industries, it is very wide and frequent seen in banking industry were a customer holding an with a bank will be targeted for cross selling opportunities like loans, fixed deposits and term deposits etc.,. Banks traditionally uses various channels to reach their customer and one of the major such channel is Telemarketing. 

Though telemarketing seems like a very easy way to reach a customer at the same time it is very costly. As per industry standard typical <b> Cost per Call (CPC) is around $2.7 to $5.6 </b>  (it might difference from business to business), based this statistic we can estimate the possible impact of targeting a wrong prospect and importance of accurate targeting strategy. In the current project we plan to address this problem of high operating cost due to inaccuracy customer targeting using machine learning. As part of the analysis, we will be using banking telemarketing call data for predicting the propensity of a customer opting for cross selling, which can later be used by banking businesses for making better call plan as well customer target list
''', unsafe_allow_html=True)

tab1.info('Current App developed with Machine Learning supported at backend. Generates realtime recommendation on customer targeting based on customer profile (details) selected.', icon="ℹ️")

#####################################
### Tab2: For recommendations
#####################################

## Function for predicting propensity

def predValue():
    #dfModel = pickle.load(open('decision_tree_24032023.sav', 'rb'))
    #rfModel = pickle.load(open('random_forest_24032023.sav', 'rb'))
    xgbModel = pickle.load(open('random_forest_24032023_lite.sav', 'rb'))

    ## Loading saved model
    dummyData = pd.get_dummies(pd.DataFrame(userInputObj))
    missingFeatures = list(set(list(xgbModel.feature_names_in_)) - set(list(dummyData.columns)))
    for i in missingFeatures:
        dummyData[i] = 0

    predictionValued = xgbModel.predict(dummyData[xgbModel.feature_names_in_])[0]
    predictionProb = xgbModel.predict_proba(dummyData[xgbModel.feature_names_in_])[0][1]

    return predictionValued, predictionProb




#data_canada = px.featureImp.gapminder().query("country == 'Canada'")
#fig = px.bar(featureImp, x='Relative Importance', y='Feature', orientation='h')
image = Image.open('Feature Importance.png')
tab3.image(image)


#####################################
### Tab3: For recommendations
#####################################
tab2.markdown('''<h4> Parameters Selected </h4>''', unsafe_allow_html=True)
tab2.table(userInputObj)
tab2.markdown('''<h4> Recommendations </h4>''', unsafe_allow_html=True)
col1, col2, col3 = tab2.columns(3)
## Priting Porbabiity
if st.sidebar.button('Generate Recommendations'):
    predictionValued, predictionProb = predValue()

    ## Checking positive or negative prediction
    if predictionValued == 1:
        image = Image.open('ok.png')
        tab2.success('Customer has %s Precentage probability to subscribe to term plan. Please go ahead and target!' %(np.round(predictionProb*100,1)), icon="✅")
        col2.image(image)
    else:
        image = Image.open('not ok.png')
        tab2.warning('Customer has very less propensity to purchase term plan. It is recommend not to target!', icon="⚠️")
        #col2.markdown('''<div> Recommendations </h4>''', unsafe_allow_html=True)
        col2.image(image)
