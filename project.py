import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('malnutrition.csv', header= None)

st.write("Par 1 - Initial Data Exploration")
# show data set
st.write(data)
# description and summany of data

st.write(data.describe())
st.write(data.info())



st.write("Par 2 - Data Pre-processing and Cleaning") 


