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
import joblib
import pandas as pd

# Set page config for a professional look
st.set_page_config(
    page_title="❤️ Cardio Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ULTIMATE CSS - FORCES ALL INPUT TEXT TO BLACK + CENTERED RESULT
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    .main {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab, #ff9ff3, #54a0ff);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 2rem 0;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* FORCE ALL INPUT TEXT TO BLACK */
    input, select, [role="combobox"], [data-baseweb="select"] *, 
    .stSelectbox span, .stNumberInput span,
    input[type="number"], input[type="text"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
    }
    
    [role="listbox"] *, [data-baseweb="menu"] * {
        color: #000000 !important;
        background: #FFFFFF !important;
    }
    
    .stSelectbox > div > div > div, 
    .stNumberInput > div > div > div {
        background: #FFFFFF !important;
        border-radius: 15px !important;
        border: 2px solid rgba(255,255,255,0.8) !important;
    }
    
    .stSelectbox label, .stNumberInput label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* CENTERED RESULT BOX */
    .result-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        padding: 2rem 0;
    }
    
    .result-box {
        background: linear-gradient(135deg, #FF4757, #FF3838, #C44569);
        color: white;
        padding: 4rem 3rem;
        border-radius: 30px;
        text-align: center;
        box-shadow: 0 30px 70px rgba(255,71,87,0.6);
        max-width: 600px;
        width: 90%;
        transform: scale(1.05);
        font-family: 'Poppins', sans-serif;
    }
    
    .result-box.low-risk {
        background: linear-gradient(135deg, #2ED573, #1ABC9C, #48D1CC);
        box-shadow: 0 30px 70px rgba(46,213,115,0.6);
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 1rem 0;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    
    .result-prob {
        font-size: 5rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 3px 3px 15px rgba(0,0,0,0.4);
        background: linear-gradient(45deg, white, rgba(255,255,255,0.8));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .result-label {
        font-size: 1.6rem;
        margin: 1rem 0;
        opacity: 0.95;
        font-weight: 600;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 25px 50px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53, #FECA57) !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 1rem 3rem !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        box-shadow: 0 12px 35px rgba(255,107,107,0.4) !important;
    }
    
    h1, h2, h3 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    .stMetric > label {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD PIPELINE =================
@st.cache_resource
def load_pipeline():
    return joblib.load("cardio_pipeline.pkl")

pipeline = load_pipeline()

# ================= HEADER =================
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
        <div class="glass-card">
            <h1 style='text-align: center;'>❤️ Cardio Disease Prediction</h1>
            <p style='text-align: center; font-size: 1.4rem; color: #000000;'>
                🚀 Enter your health metrics for AI-powered instant risk assessment
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="glass-card" style='background: linear-gradient(135deg, #667eea, #764ba2); color: white;'>
            <h3 style='margin: 0;'>🔬 AI Powered</h3>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

# ================= USER INPUTS =================
st.markdown("---")

st.markdown("""
    <div class="glass-card">
        <h2 style='text-align: center; margin-bottom: 2rem; color: #000000;'>📊 Enter Your Health Data</h2>
""", unsafe_allow_html=True)

# Input columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3>👤 Personal Info</h3>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", [1, 2], 
                         format_func=lambda x: "👨 Male" if x == 1 else "👩 Female")
    calculated_age = st.number_input("Age (years)", min_value=1, max_value=120, value=40)

with col2:
    st.markdown("<h3>📏 Measurements</h3>", unsafe_allow_html=True)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)

with col3:
    st.markdown("<h3>❤️ Blood Pressure</h3>", unsafe_allow_html=True)
    ap_hi = st.number_input("Systolic BP", min_value=50.0, max_value=250.0, value=120.0)
    ap_lo = st.number_input("Diastolic BP", min_value=30.0, max_value=150.0, value=80.0)

# Second row
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3>🩸 Blood Tests</h3>", unsafe_allow_html=True)
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3],
                              format_func=lambda x: ["✅ Normal", "⚠️ Above Normal", "❌ High"][x-1])
    gluc = st.selectbox("Glucose", [1, 2, 3],
                       format_func=lambda x: ["✅ Normal", "⚠️ Above Normal", "❌ High"][x-1])

with col2:
    st.markdown("<h3>🚬 Lifestyle</h3>", unsafe_allow_html=True)
    smoke = st.selectbox("Smoker", [0, 1], 
                        format_func=lambda x: "🚭 No" if x == 0 else "🚬 Yes")
    alco = st.selectbox("Alcohol", [0, 1], 
                       format_func=lambda x: "🍷 No" if x == 0 else "🍺 Yes")

with col3:
    st.markdown("<h3>⚡ Activity</h3>", unsafe_allow_html=True)
    active = st.selectbox("Physically Active", [0, 1],
                         format_func=lambda x: "💤 No" if x == 0 else "🏃 Yes")

st.markdown("</div>", unsafe_allow_html=True)

# BMI Display
bmi = weight / ((height / 100) ** 2)
col1, col2, col3, col4 = st.columns(4)
with col3:
    bmi_status = "✅ Healthy" if bmi < 25 else "⚠️ Overweight" if bmi < 30 else "❌ Obese"
    st.metric("📊 BMI", f"{bmi:.1f}", bmi_status)

# ================= PREDICT =================
if st.button("🔮 **PREDICT MY RISK NOW** 🚀", use_container_width=True):
    st.markdown("---")
    
    input_df = pd.DataFrame([{
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "calculated_age": calculated_age,
        "bmi": bmi
    }])

    with st.spinner("🔬 AI Analyzing your health data..."):
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1] * 100  # Convert to percentage

    # ✅ CENTERED RISK BOX WITH PROBABILITY %
    st.markdown("""
        <div class="result-container">
    """, unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
            <div class="result-box">
                <div class="result-title">⚠️🔥 HIGH RISK DETECTED</div>
                <div class="result-prob">{probability:.0f}%</div>
                <div class="result-label">Cardio Disease Risk Probability</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box low-risk">
                <div class="result-title">✅🛡️ LOW RISK</div>
                <div class="result-prob">{probability:.0f}%</div>
                <div class="result-label">Cardio Disease Risk Probability</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Recommendations
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #000000;'>💡 Health Recommendations</h2>", unsafe_allow_html=True)
    
    recs = {
        1: ["📞 **Doctor appointment ASAP**", "💊 **Daily BP monitoring**", "🏃‍♂️ **30min exercise**", "🚭 **Quit smoking**", "🥗 **Low-sodium diet**"],
        0: ["🥗 **Balanced diet**", "🏃 **30min exercise**", "💤 **7-8hrs sleep**", "🚭 **Stay smoke-free**", "📊 **Regular checkups**"]
    }
    
    st.markdown(f"""
        <div class="glass-card">
            <h3 style='color: #000000; text-align: center;'>
                { '🚨 URGENT ACTION REQUIRED' if prediction == 1 else '✅ MAINTAIN YOUR HEALTH' }
            </h3>
            <ul style='color: #000000; font-size: 1.2rem; padding-left: 2rem;'>
    """, unsafe_allow_html=True)
    
    for item in recs[prediction]:
        st.markdown(f"<li style='margin: 1rem 0; font-weight: 500;'>{item}</li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: white; padding: 2rem; 
#                 background: rgba(0,0,0,0.3); border-radius: 20px;'>
#         <h3 style='color: white;'>🔬 AI-Powered Health Assessment</h3>
#         <p style='color: white;'>⚕️ Educational tool only - consult your doctor</p>
#     </div>
# """, unsafe_allow_html=True)


# import streamlit as st
# import joblib
# import pandas as pd

# # ================= LOAD PIPELINE =================
# pipeline = joblib.load("cardio_pipeline.pkl")

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

