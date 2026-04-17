
import pickle
import pandas as pd
import numpy as np

# Load the trained model
try:
    with open('random_forest_regressor_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'random_forest_regressor_model.pkl' not found. Please ensure the model is saved.")
    exit()

def predict_salary(age, gender, education_level, job_title, years_of_experience):
    """
    Predicts salary using the loaded Random Forest Regressor model.

    Args:
        age (float): Age of the individual.
        gender (int): Encoded gender (e.g., 0 for Female, 1 for Male).
        education_level (int): Encoded education level (e.g., 0 for Bachelor's, 1 for Master's, etc.).
        job_title (int): Encoded job title.
        years_of_experience (float): Years of experience.

    Returns:
        float: Predicted salary.
    """
    # Create a DataFrame from the input features
    input_data = pd.DataFrame([[age, gender, education_level, job_title, years_of_experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    return prediction

if __name__ == "__main__":
    print("\n--- Salary Prediction Example ---")

    # Example 1: Predict salary for a sample individual
    # NOTE: 'Gender', 'Education Level', and 'Job Title' must be numerically encoded
    # according to the LabelEncoder fitted during training. For a real application,
    # you would need to store and re-use the fitted LabelEncoders or create a mapping.
    sample_age = 30.0
    sample_gender = 1 # Assuming 1 for Male, 0 for Female based on training data's encoding
    sample_education_level = 0 # Assuming 0 for a specific education level based on training data's encoding
    sample_job_title = 50 # Assuming 50 for a specific job title based on training data's encoding
    sample_years_of_experience = 5.0

    predicted_salary = predict_salary(
        sample_age,
        sample_gender,
        sample_education_level,
        sample_job_title,
        sample_years_of_experience
    )

    print(f"Input features: Age={sample_age}, Gender={sample_gender}, "
          f"Education Level={sample_education_level}, Job Title={sample_job_title}, "
          f"Years of Experience={sample_years_of_experience}")
    print(f"Predicted Salary: ${predicted_salary:,.2f}")

    # Example 2: Another prediction
    sample_age_2 = 45.0
    sample_gender_2 = 0 # Female
    sample_education_level_2 = 3 # Another education level
    sample_job_title_2 = 120 # Another job title
    sample_years_of_experience_2 = 15.0

    predicted_salary_2 = predict_salary(
        sample_age_2,
        sample_gender_2,
        sample_education_level_2,
        sample_job_title_2,
        sample_years_of_experience_2
    )
    print(f"\nInput features: Age={sample_age_2}, Gender={sample_gender_2}, "
          f"Education Level={sample_education_level_2}, Job Title={sample_job_title_2}, "
          f"Years of Experience={sample_years_of_experience_2}")
    print(f"Predicted Salary: ${predicted_salary_2:,.2f}")
