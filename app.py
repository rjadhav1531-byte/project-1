# Diabetes Prediction Streamlit App

import pickle
import numpy as np
import streamlit as st

# -------------------------------
# 1. Data Fetching & Model Setup
# -------------------------------
# Features used for prediction
features = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

# Load the trained model and scaler
model = pickle.load(open("trained_model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

# -------------------------------
# 2. Prediction Function
# -------------------------------
def prediction(data):
    data = np.asarray(data).reshape(1, -1)

    # Scale the data using the loaded scaler
    data_scaled = scaler.transform(data)

    # Predict (0 = No Diabetes, 1 = Diabetes)
    res = model.predict(data_scaled)[0]

    # Factors used for explanation
    glucose, bmi, insulin, age = data[0][1], data[0][5], data[0][4], data[0][7]

    explanation = {}
    if glucose > 140:
        explanation["Glucose Level"] = f"High glucose levels ({glucose} mg/dL) indicate a higher risk of diabetes."
    else:
        explanation["Glucose Level"] = f"Glucose levels ({glucose} mg/dL) are within normal range."

    if bmi > 30:
        explanation["BMI"] = f"A BMI of {bmi} indicates overweight, which increases diabetes risk."
    else:
        explanation["BMI"] = f"BMI of {bmi} is within a healthy range."

    if insulin > 150:
        explanation["Insulin Level"] = f"Insulin levels ({insulin} µU/mL) are high, suggesting possible insulin resistance."
    else:
        explanation["Insulin Level"] = f"Insulin levels ({insulin} µU/mL) are within normal range."

    if age > 45:
        explanation["Age"] = f"Age {age} is a significant risk factor for diabetes."
    else:
        explanation["Age"] = f"Age {age} is not a major risk factor for diabetes."

    return res, explanation

# -------------------------------
# 3. Main Application
# -------------------------------
def main():
    st.set_page_config(page_title="DiabetesPrediction", page_icon="⚕️", layout="centered")

    # Header
    st.markdown(
        '''
        <h2 align="center" style="color: lightblue; font-weight: bolder; margin-top: -60px;">Diabetes Prediction</h2><br>
        <p align="center" style="color: grey;">Fill in the details below and click Submit to know if you are at risk of Diabetes.</p>
        ''',
        unsafe_allow_html=True
    )

    # Input Form
    with st.form("patient-data", clear_on_submit=False):
        st.subheader("Enter Patient Data")

        # Two column layout
        c1, c2 = st.columns((0.5, 0.5), gap="medium")

        with c1:
            pregnancies = st.number_input("Pregnancy Count", min_value=0, help="Number of times pregnant")
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, help="Higher levels indicate a higher risk of diabetes.")
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, help="Normal: 90/60 to 120/80 mm Hg")
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, help="Thickness of the skin fold")

        with c2:
            insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, help="Higher insulin can indicate insulin resistance.")
            bmi = st.number_input("BMI", format="%.1f", help="Normal BMI is 18.5 to 24.9")
            dpf = st.number_input("Diabetes Pedigree Function", format="%.3f", help="Scores risk based on family history")
            age = st.number_input("Age", min_value=0, help="Age is a significant factor in diabetes risk")

        # Submit button
        submitted = st.form_submit_button("Submit")
        result_placeholder = st.empty()

    # -------------------------------
    # 4. Result Handling
    # -------------------------------
    if submitted:
        data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

        with st.spinner("Analyzing data..."):
            result, explanation = prediction(data)

        # Result display
        if result:
            result_message = "Diabetes suspected."
            color = "lightcoral"
            border_color = "#e50914"
        else:
            result_message = "Diabetes not suspected."
            color = "skyblue"
            border_color = "#3f79d7"

        result_placeholder.markdown(
            f'''<h4 align="center" style="color: {color}; margin-block: 15px; border: solid {border_color} 1px; border-radius: 10px; padding: 20px; margin-inline: auto; text-align: center;">{result_message}</h4>''',
            unsafe_allow_html=True
        )

        # Explanation
        st.markdown("<h4 style='color: grey;'>Detailed Explanation:</h4>", unsafe_allow_html=True)
        for key, value in explanation.items():
            st.write(f"**{key}:** {value}")

    else:
        result_placeholder.markdown(
            '''<h4 style="color: grey; margin-block: 15px; border: solid #1b3053 1px; border-radius: 10px; padding: 20px; margin-inline: auto; text-align: center;">Submit for results</h4>''',
            unsafe_allow_html=True
        )

# -------------------------------
# 5. Run App
# -------------------------------
if __name__ == "__main__":
    main()