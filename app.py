import streamlit as st
import joblib
import numpy as np


model = joblib.load(r"C:\Users\balas\OneDrive\Desktop\Balu\Myprojects\vs\cancer\lung_cancer_model.pkl")

st.set_page_config(page_title="Lung Cancer Predictor", layout="centered")
st.title("🩺 Lung Cancer Risk Prediction")
st.markdown("Enter the details below to predict if the patient is at risk of **lung cancer**.")


gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

# Age
age = st.slider("Age", 20, 100, step=1)


def binary_input(feature_name):
    return st.radio(feature_name, ["No", "Yes"]) == "Yes"


smoking = int(binary_input("Do you smoke?"))
yellow_fingers = int(binary_input("Do you have yellow fingers?"))
anxiety = int(binary_input("Do you experience anxiety?"))
peer_pressure = int(binary_input("Are you under peer pressure?"))
chronic_disease = int(binary_input("Do you have any chronic diseases?"))
fatigue = int(binary_input("Do you feel fatigued?"))
allergy = int(binary_input("Do you have allergies?"))
wheezing = int(binary_input("Do you experience wheezing?"))
alcohol = int(binary_input("Do you consume alcohol?"))
coughing = int(binary_input("Are you coughing regularly?"))
short_breath = int(binary_input("Do you experience shortness of breath?"))
swallowing_diff = int(binary_input("Do you have difficulty swallowing?"))
chest_pain = int(binary_input("Do you feel chest pain?"))


input_data = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                        chronic_disease, fatigue, allergy, wheezing, alcohol,
                        coughing, short_breath, swallowing_diff, chest_pain]])


if st.button("🧠 Predict Lung Cancer Risk"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Lung Cancer Detected. Please consult a doctor!")
    else:
        st.success("✅ No Lung Cancer Detected. Stay healthy!")

st.markdown("---")
st.caption("Built by Bala Srivatsa | ML for Real Health")
