import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

# Load the model and label encoder
model = load('random_forest_model.joblib')
label_encoder = load('label_encoder.joblib')

# Custom CSS for a polished look with light baby blue background and black text color
st.markdown(
    """
    <style>
    .main {
        background-color: #e0f7fa;  /* Light baby blue background */
        color: #000000;  /* Black text color for better readability */
    }
    .sidebar .sidebar-content {
        background-color: #b2ebf2;  /* Slightly darker baby blue for sidebar */
        color: #000000;  /* Black text color for sidebar */
    }
    h1, h2, h3, p {
        color: #000000;  /* Black text color */
        font-size: 20px;  /* Font size 20px */
        font-family: 'Arial', sans-serif;  /* Clean font family */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title('Purchase Prediction App')

# Sidebar with color
st.sidebar.header('User Input')
st.sidebar.write("### Enter User Details")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 18, 100, 25)
    estimated_salary = st.sidebar.slider('Estimated Salary', 10000, 150000, 50000)
    return pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'EstimatedSalary': [estimated_salary]
    })

df = user_input_features()

# Encode gender
df['Gender'] = label_encoder.transform(df['Gender'])

# Make prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Main panel
st.write("## Prediction Results")

if prediction[0] == 1:
    st.markdown('<h2 style="color: #4caf50;">The user is likely to make a purchase.</h2>', unsafe_allow_html=True)  # Green
else:
    st.markdown('<h2 style="color: #f44336;">The user is not likely to make a purchase.</h2>', unsafe_allow_html=True)  # Red

# Plotting the bar chart with blue color
st.write("### Prediction Probability")
fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size if needed
categories = ['Purchase', 'No Purchase']
probs = prediction_proba[0]

ax.bar(categories, probs, color='#2196F3')  # Blue color
ax.set_ylim(0, 1)
ax.set_ylabel('Probability', fontsize=18, color='#000000')
ax.set_title('Probability of Purchase', fontsize=18, color='#000000')
ax.set_facecolor('#e0f7fa')  # Light baby blue background for the plot area
ax.tick_params(axis='both', colors='#000000')  # Black color for ticks and labels

st.pyplot(fig)

# Additional interactive elements
st.write("### User Input Data")
st.write(df)
