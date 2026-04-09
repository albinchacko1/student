import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("student_model.pkl")

st.title("Student Performance Prediction")
st.write("Enter student details below to predict whether the student will pass or fail.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=15, max_value=22, value=17)
    studytime = st.selectbox("Study Time", [1, 2, 3, 4])
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
    absences = st.number_input("Absences", min_value=0, max_value=100, value=4)
    internet = st.selectbox("Internet Access", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    schoolsup = st.selectbox("School Support", ["yes", "no"])
    G1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, value=10)
    G2 = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20, value=10)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([{
        "school": "GP",
        "sex": "F",
        "age": age,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": higher,
        "internet": internet,
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": absences,
        "G1": G1,
        "G2": G2
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("Prediction: Pass")
    else:
        st.error("Prediction: Fail")

    st.write(f"Probability of Fail: {probability[0]:.2f}")
    st.write(f"Probability of Pass: {probability[1]:.2f}")

    st.subheader("Entered Data")
    st.dataframe(input_data)