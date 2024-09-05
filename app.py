import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the machine learning model
loaded_model = pickle.load(open(r'model.pkl','rb'))


# Title of the application
st.title("Diabetes Prediction")

# Create sliders and input fields for each feature
Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
Glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
BloodPressure = st.number_input('BloodPressure', min_value=0, max_value=140, value=70)
SkinThickness = st.number_input('SkinThickness', min_value=0, max_value=100, value=20)
Insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input('Age', min_value=0, max_value=120, value=30)

# Store the inputs into a dataframe
input_data = pd.DataFrame({
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
    'Age': [Age]
})

# Make prediction
if st.button('Predict'):
    prediction = loaded_model.predict(input_data)[0]
    st.write(f"The predicted outcome is: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    
    # Plot the prediction (binary output)
    fig, ax = plt.subplots()
    ax.bar(['No Diabetes', 'Diabetes'], [1 - prediction, prediction])
    ax.set_ylabel('Probability')
    st.pyplot(fig)