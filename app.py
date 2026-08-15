import streamlit as st
import pandas as pd
import joblib


# Load model
model = joblib.load("titanic_model.pkl")


st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger information to predict survival.")


# User inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=20.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["C", "Q", "S"]
)


# Convert Sex
sex_value = 1 if sex == "Female" else 0


# Convert Embarked
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0


# Create input DataFrame
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_value],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked_Q": [embarked_q],
    "Embarked_S": [embarked_s]
})


st.write("Input data:")
st.dataframe(input_data)


# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🟢 Passenger is predicted to survive.")
    else:
        st.error("🔴 Passenger is predicted not to survive.")