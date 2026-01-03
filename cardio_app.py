#'import streamlit as st
# import joblib
# import numpy as np

# # ================= LOAD MODEL & SCALER =================
# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")

# st.title("❤️ Cardio Disease Prediction")

# # ================= USER INPUTS =================
# gender = st.selectbox("Gender", [1,2])
# height = st.number_input("Height (cm)", min_value=50.0)
# weight = st.number_input("Weight (kg)", min_value=10.0)
# ap_hi = st.number_input("Systolic BP", min_value=50.0)
# ap_lo = st.number_input("Diastolic BP", min_value=30.0)
# cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
# gluc = st.selectbox("Glucose Level", [1, 2, 3])
# smoke = st.selectbox("Smoking", [0, 1])
# alco = st.selectbox("Alcohol Intake", [0, 1])
# active = st.selectbox("Physically Active", [0, 1])
# calculated_age = st.number_input("Age", min_value=1)


# bmi = weight / ((height / 100) ** 2)

# # ================= PREDICT =================
# if st.button("Predict"):

#     # # -------- Gender One-Hot Encoding --------
#     # gender_male = 1 if gender == "Male" else 0
#     # gender_female = 1 if gender == "Female" else 0

#     # -------- BUILD FULL 12-FEATURE INPUT --------
#     # ⚠️ SAME ORDER AS TRAINING
#     X = np.array([[
#         gender,
#         height,
#         weight,
#         ap_hi,
#         ap_lo,      
#         cholesterol,
#         gluc,
#         smoke,
#         alco,
#         active,
#         calculated_age,
#         bmi
#     ]])

#     # -------- SCALE INPUT --------
#     X_scaled = scaler.transform(X)

#     # -------- MODEL PREDICTION --------
#     prediction = model.predict(X_scaled)

#     # -------- OUTPUT --------
#     if prediction[0] >= 0.5:
#         st.error("⚠️ High Risk of Cardio Disease")
#     else:
#         st.success("✅ Low Risk of Cardio Disease")

import streamlit as st

st.set_page_config(
    page_title="Cardiovascular Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-pkbazv {
        color: white;
    }
    
    .sidebar-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        margin-bottom: 1rem;
    }
    
    .main-title {
        text-align: center;
        color: #1e3a8a;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    
    .hero-card {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 3rem;
        border-radius: 15px;
        border-left: 8px solid #2563eb;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .cta-button {
        background-color: #2563eb;
        color: white;
        padding: 1rem 3rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        display: inline-block;
        text-decoration: none;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">❤️ CardioRisk AI</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="color: white; padding: 1rem; line-height: 1.8;">
            <p><strong>Clinical Decision Support System</strong></p>
            <p style="font-size: 0.9rem; opacity: 0.9;">
                Advanced machine learning-based cardiovascular risk assessment tool for healthcare professionals.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">❤️ Cardiovascular Disease Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Clinical Decision Support System</p>', unsafe_allow_html=True)

st.markdown("""
    <div class="hero-card">
        <h2 style="color: #1e3a8a; margin-top: 0;">Welcome to CardioRisk AI</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: #334155;">
            A state-of-the-art machine learning system designed to assist healthcare professionals 
            in assessing cardiovascular disease risk. This tool analyzes patient demographics, 
            vital signs, and lifestyle factors to provide evidence-based risk predictions.
        </p>
        <p style="font-size: 1.1rem; line-height: 1.8; color: #334155; margin-top: 1rem;">
            <strong>Navigate using the sidebar to:</strong>
        </p>
        <ul style="font-size: 1.05rem; line-height: 2; color: #475569;">
            <li><strong>Overview:</strong> Learn about cardiovascular disease and how this system works</li>
            <li><strong>Risk Prediction:</strong> Assess patient cardiovascular risk</li>
            <li><strong>Model Insights:</strong> Understand the AI model and its limitations</li>
            <li><strong>Medical Disclaimer:</strong> Review important legal and ethical information</li>
            <li><strong>About Project:</strong> Technical details and future development</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="color: #2563eb; margin-bottom: 1rem;">🎯 Accurate</h3>
            <p style="color: #64748b;">ML-powered risk assessment using validated clinical features</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="color: #059669; margin-bottom: 1rem;">⚡ Fast</h3>
            <p style="color: #64748b;">Instant predictions to support timely clinical decisions</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="color: #7c3aed; margin-bottom: 1rem;">🔒 Secure</h3>
            <p style="color: #64748b;">Designed with healthcare compliance and patient privacy in mind</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 2rem; border-top: 1px solid #e2e8f0;">
        CardioRisk AI v1.0 | For Healthcare Professional Use Only
        <br>
        © 2025 Healthcare AI Solutions
    </div>
""", unsafe_allow_html=True)


# import streamlit as st
# import joblib
# import pandas as pd

# # ================= LOAD PIPELINE =================
# pipeline = joblib.load("model.pkl")

# st.title("❤️ Cardio Disease Prediction")

# # ================= USER INPUTS =================
# gender = st.selectbox("Gender", [1, 2])
# height = st.number_input("Height (cm)", min_value=50.0)
# weight = st.number_input("Weight (kg)", min_value=10.0)
# ap_hi = st.number_input("Systolic BP", min_value=50.0)
# ap_lo = st.number_input("Diastolic BP", min_value=30.0)
# cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
# gluc = st.selectbox("Glucose Level", [1, 2, 3])
# smoke = st.selectbox("Smoking", [0, 1])
# alco = st.selectbox("Alcohol Intake", [0, 1])
# active = st.selectbox("Physically Active", [0, 1])
# calculated_age = st.number_input("Age", min_value=1)

# bmi = weight / ((height / 100) ** 2)

# # ================= PREDICT =================
# if st.button("Predict"):

#     # ✅ RAW INPUT AS DATAFRAME (MATCH TRAINING)
#     input_df = pd.DataFrame([{
#         "gender": gender,
#         "height": height,
#         "weight": weight,
#         "ap_hi": ap_hi,
#         "ap_lo": ap_lo,
#         "cholesterol": cholesterol,
#         "gluc": gluc,
#         "smoke": smoke,
#         "alco": alco,
#         "active": active,
#         "calculated_age": calculated_age,
#         "bmi": bmi
#     }])

#     # ✅ PIPELINE HANDLES SCALING + ENCODING
#     prediction = pipeline.predict(input_df)[0]
#     probability = pipeline.predict_proba(input_df)[0][1]

#     if prediction == 1:
#         st.error(f"⚠️ High Risk of Cardio Disease ({probability:.2%})")
#     else:
#         st.success(f"✅ Low Risk of Cardio Disease ({probability:.2%})")




import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))

