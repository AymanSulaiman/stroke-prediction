import os
import pandas as pd
import numpy as np
import streamlit as st
import machine_learning_models

st.title('My first app?')
st.write("Here's our first attempt at using data to create a table:")

st.table(pd.read_csv(os.path.join('..','models','models.csv')))
input_placeholder = st.text_input("label goes here", 10)
gender = input_placeholder
age = input_placeholder
hypertension = input_placeholder
heart_disease = input_placeholder
married = input_placeholder
work_type = input_placeholder
residence = input_placeholder
avg_glucose_level = input_placeholder
bmi = input_placeholder
smoke = input_placeholder