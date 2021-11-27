import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib



st.title('COVID-19 Prediction WebApp')

st.write('This app predicts the daily new deaths of COVID-19 deaths based on the given data. ')
st.subheader("Use the Sliders by the left to enter data.")
st.write("""The features used to predict the deaths is:
         
             
         1. date
         2. cumulative total cases
         3. daily new cases
         4. active cases
         5. cumulative total deaths""")
         
st.write('This WebApp uses ExtraTreesRegressor Machine Learning Model to predict the daily deaths.')
st.subheader("Analysis on past dataset")
st.image('image.png', caption='Analysis on past dataset')
def user_input_features():
    
    date = st.sidebar.date_input("Today's date", datetime.date.today())
    cm_total_cases = st.sidebar.slider("Cumulative Total Cases",0,100000,1000)
    daily_new_cases = st.sidebar.slider("Daily new cases", 0,100000,1000)
    active_cases = st.sidebar.slider('Active Cases', 0,100000,1000)
    cm_total_deaths = st.sidebar.slider("Cumulative total deaths", 0,100000,1000)
    data={'date':date,
          'cumulative_total_cases':cm_total_cases,
          'daily_new_cases':daily_new_cases,
          'active_cases':active_cases,
          'cumulative_total_deaths':cm_total_deaths
          }
    
    
    df = pd.DataFrame(data=data, index=[0])
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df.date.dt.day
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year
    
    df.drop('date', axis=1, inplace=True)

    return df
st.subheader('Given data:')
data = user_input_features()
st.table(data)
model = joblib.load('final_model_extra_tree_without_country.joblib')
preds = model.predict(data)
predss = preds.astype('int')
st.write(f'Based on the given data, daily new deaths of COVID-19 patients will be:  {predss}')
