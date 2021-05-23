import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st



model_df = pd.read_csv(os.path.join('..','models','models_cat.csv'))
st.write("""
# Top five models that work best with the dataset
""")
st.table(model_df[:5])







# Gender
gender_select = st.radio("Select Gender", ('Male', 'Female','Other'))
st.success(gender_select)

# Age
age = st.number_input("Enter your age")
st.success(age)

# Hypertension
hypertension_select = st.radio("Do you have Hypertension?", ('Yes', 'No'))
st.success(hypertension_select)

# Heart Disease
heart_disease_select = st.radio("Do you have Heart Disease?", ('Yes', 'No'))
st.success(heart_disease_select)

# Ever Married
married_select = st.radio("Have you ever been married?", ('Yes', 'No'))
st.success(married_select)

# Work Type
work_type_select = st.radio("What is your worktype", ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
st.success(work_type_select)

# Residence Type
Residence_type_select = st.radio("What is your residence type?", ('Urban', 'Rural'))
st.success(Residence_type_select)

# Avg Glucose Level
avg_glucose_level = st.number_input("Enter your glucose level")
st.success(avg_glucose_level)

# BMI
bmi = st.number_input("Enter your BMI (I won't tell anyone)")
st.success(bmi)

# Smoking Status
smoking_status_select = st.radio("What is your Smoking status", ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))
st.success(smoking_status_select)

user_df = pd.DataFrame(
    {
        'gender':[gender_select], 
        'age':[age], 
        'hypertension':[hypertension_select], 
        'heart_disease':[heart_disease_select], 
        'ever_married':[married_select],
        'work_type':[work_type_select], 
        'Residence_type':[Residence_type_select], 
        'avg_glucose_level':[avg_glucose_level], 
        'bmi':[bmi],
        'smoking_status':[smoking_status_select]

    }
)

st.table(user_df)

random_forest_model_path = os.path.join('..','models','random_forest_clf_cat.pkl')

loaded_model = joblib.load(random_forest_model_path)



# input_placeholder = st.text_input("label goes here", 10)
# gender = input_placeholder
# age = input_placeholder
# hypertension = input_placeholder
# heart_disease = input_placeholder
# married = input_placeholder
# work_type = input_placeholder
# residence = input_placeholder
# avg_glucose_level = input_placeholder
# bmi = input_placeholder
# smoke = input_placeholder