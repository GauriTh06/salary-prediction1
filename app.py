
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
try:
    with open('random_forest_regressor_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'random_forest_regressor_model.pkl' not found. Please ensure the model is saved.")
    st.stop()

# --- Streamlit App --- 
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Define mappings for categorical features (based on your training data encoding)
# NOTE: These mappings should be consistent with how LabelEncoder was applied during training.
# For a robust solution, you would save/load the LabelEncoders themselves.

gender_mapping = {'Female': 0, 'Male': 1}
education_level_mapping = {
    "Bachelor's Degree": 0,
    "Master's Degree": 1,
    "PhD": 2,
    "High School": 3,
    "Some College": 4,
    "Associate's Degree": 5,
    "Vocational School": 6
}

# This is a simplified example. In a real application, you'd likely map specific job titles.
# For demonstration, we'll use a numerical input for Job Title and advise the user.
# It's crucial that the numbers here correspond to the exact encoding used during training.
# A proper solution would involve storing and using the original LabelEncoder objects.
# For this example, let's just make it a number input and assume the user knows the encoding.

# Input fields
age = st.slider('Age', 20, 60, 30)
gender_option = st.selectbox('Gender', list(gender_mapping.keys()))
education_level_option = st.selectbox('Education Level', list(education_level_mapping.keys()))
job_title_input = st.number_input('Job Title (Encoded Numerical Value)', min_value=0, max_value=200, value=50)
years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0)

# Convert categorical inputs to their encoded numerical values
gender_encoded = gender_mapping[gender_option]
education_level_encoded = education_level_mapping[education_level_option]

# Prediction button
if st.button('Predict Salary'):
    # Create a DataFrame from the input features
    input_data = pd.DataFrame([[age, gender_encoded, education_level_encoded, job_title_input, years_of_experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ${prediction:,.2f}')
