import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Risk Prediction - CardioRisk AI",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    .page-title {
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .section-header {
        color: #1e40af;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .high-risk {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #dc2626;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #059669;
    }
    
    .stButton > button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        width: 100%;
        font-size: 1.1rem;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1e40af;
    }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD PIPELINE =================
try:
    pipeline = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("⚠️ Model file 'model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

st.markdown('<h1 class="page-title">🩺 Cardiovascular Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown("### Patient Risk Assessment Tool")

st.markdown('<div class="info-box">Enter accurate patient data for reliable risk assessment. All fields are required.</div>', unsafe_allow_html=True)

st.markdown("---")

# ================= PATIENT INFORMATION =================
st.markdown('<p class="section-header">👤 Patient Information</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox(
        "Gender",
        options=[1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female",
        help="Biological sex of the patient"
    )

with col2:
    calculated_age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=30,
        help="Patient's age in years"
    )

with col3:
    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0,
        step=1.0,
        help="Height in centimeters"
    )

with col4:
    weight = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=70.0,
        step=0.5,
        help="Body weight in kilograms"
    )

# ================= CLINICAL MEASUREMENTS =================
st.markdown('<p class="section-header">🩺 Clinical Measurements</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    ap_hi = st.number_input(
        "Systolic BP (mmHg)",
        min_value=50.0,
        max_value=250.0,
        value=120.0,
        step=1.0,
        help="Upper number in blood pressure reading (normal: 90-120)"
    )

with col2:
    ap_lo = st.number_input(
        "Diastolic BP (mmHg)",
        min_value=30.0,
        max_value=150.0,
        value=80.0,
        step=1.0,
        help="Lower number in blood pressure reading (normal: 60-80)"
    )

with col3:
    cholesterol = st.selectbox(
        "Cholesterol Level",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
        help="1 = Normal, 2 = Above Normal, 3 = Well Above Normal"
    )

with col4:
    gluc = st.selectbox(
        "Glucose Level",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
        help="1 = Normal, 2 = Above Normal, 3 = Well Above Normal"
    )

# ================= LIFESTYLE FACTORS =================
st.markdown('<p class="section-header">🏃 Lifestyle Factors</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    smoke = st.selectbox(
        "Smoking Status",
        options=[0, 1],
        format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker",
        help="Current smoking status"
    )

with col2:
    alco = st.selectbox(
        "Alcohol Consumption",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Regular alcohol intake"
    )

with col3:
    active = st.selectbox(
        "Physical Activity",
        options=[0, 1],
        format_func=lambda x: "Inactive" if x == 0 else "Active",
        help="Regular physical activity or exercise"
    )

# Calculate BMI
bmi = weight / ((height / 100) ** 2)

st.markdown("---")

# Display calculated BMI
st.markdown("**📊 Calculated Body Mass Index (BMI)**")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "#3b82f6"
    elif 18.5 <= bmi < 25:
        bmi_category = "Normal"
        bmi_color = "#059669"
    elif 25 <= bmi < 30:
        bmi_category = "Overweight"
        bmi_color = "#f59e0b"
    else:
        bmi_category = "Obese"
        bmi_color = "#dc2626"
    
    st.markdown(f"""
        <div style="background-color: {bmi_color}15; border: 2px solid {bmi_color}; padding: 1rem; border-radius: 8px; text-align: center;">
            <h3 style="color: {bmi_color}; margin: 0;">BMI: {bmi:.1f}</h3>
            <p style="color: {bmi_color}; margin: 0.5rem 0 0 0; font-weight: 600;">Category: {bmi_category}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ================= PREDICTION BUTTON =================
st.markdown('<p class="section-header">🔍 Risk Assessment Result</p>', unsafe_allow_html=True)

predict_button = st.button("🔬 Analyze Cardiovascular Risk")

if predict_button:
    # Validation warnings
    if height < 100 or height > 220:
        st.warning("⚠️ Height value seems unusual. Please verify the measurement.")
    if weight < 30 or weight > 200:
        st.warning("⚠️ Weight value seems unusual. Please verify the measurement.")
    if ap_hi < 80 or ap_hi > 200:
        st.warning("⚠️ Systolic BP value seems unusual. Please verify the measurement.")
    if ap_lo < 50 or ap_lo > 130:
        st.warning("⚠️ Diastolic BP value seems unusual. Please verify the measurement.")
    
    # ✅ RAW INPUT AS DATAFRAME (MATCH TRAINING - NO CHANGES)
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
    
    # ✅ PIPELINE HANDLES SCALING + ENCODING (NO CHANGES)
    with st.spinner("Analyzing patient data..."):
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
    
    # ================= RESULTS DISPLAY =================
    if prediction == 1:
        st.markdown(f"""
            <div class="result-card high-risk">
                <h2 style="color: #dc2626; margin-bottom: 1rem;">⚠️ HIGH RISK DETECTED</h2>
                <p style="font-size: 1.5rem; font-weight: 600; color: #991b1b; margin-bottom: 0.5rem;">
                    Cardiovascular Disease Risk: {probability:.1%}
                </p>
                <p style="color: #7f1d1d; font-size: 1rem;">
                    This patient shows elevated risk factors for cardiovascular disease.
                    <br>Immediate medical consultation recommended.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Clinical Recommendations", expanded=True):
            st.markdown("""
                <div style="line-height: 1.8;">
                    <h4 style="color: #dc2626;">Immediate Actions</h4>
                    <p style="color: #991b1b;">
                        • Schedule comprehensive cardiovascular evaluation including ECG and echocardiogram<br>
                        • Order complete lipid panel, HbA1c, and renal function tests<br>
                        • Review current medications and consider cardiovascular prophylaxis
                    </p>
                    
                    <h4 style="color: #f59e0b; margin-top: 1.5rem;">Monitoring Protocol</h4>
                    <p style="color: #92400e;">
                        • Weekly blood pressure monitoring for first month<br>
                        • Monthly follow-up appointments for risk factor assessment<br>
                        • Quarterly lipid and glucose monitoring
                    </p>
                    
                    <h4 style="color: #f59e0b; margin-top: 1.5rem;">Lifestyle Modifications</h4>
                    <p style="color: #92400e;">
                        • Immediate smoking cessation if applicable<br>
                        • Structured cardiac rehabilitation program<br>
                        • DASH diet with sodium restriction (&lt;2000mg/day)<br>
                        • Supervised exercise program starting at 20-30 min/day
                    </p>
                    
                    <h4 style="color: #dc2626; margin-top: 1.5rem;">Follow-up</h4>
                    <p style="color: #991b1b;">
                        • Cardiology referral for specialist evaluation<br>
                        • Patient education on warning signs of acute events<br>
                        • Establish emergency action plan
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card low-risk">
                <h2 style="color: #059669; margin-bottom: 1rem;">✅ LOW RISK PROFILE</h2>
                <p style="font-size: 1.5rem; font-weight: 600; color: #047857; margin-bottom: 0.5rem;">
                    Cardiovascular Disease Risk: {probability:.1%}
                </p>
                <p style="color: #065f46; font-size: 1rem;">
                    Current health indicators suggest lower cardiovascular risk.
                    <br>Continue maintaining healthy lifestyle habits.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Prevention & Wellness Recommendations", expanded=True):
            st.markdown("""
                <div style="line-height: 1.8;">
                    <h4 style="color: #059669;">Maintain Healthy Practices</h4>
                    <p style="color: #047857;">
                        • Continue current healthy lifestyle and dietary habits<br>
                        • Maintain healthy weight within BMI range 18.5-24.9<br>
                        • Stay physically active with regular exercise routine
                    </p>
                    
                    <h4 style="color: #0891b2; margin-top: 1.5rem;">Regular Check-ups</h4>
                    <p style="color: #155e75;">
                        • Annual cardiovascular screening and physical examination<br>
                        • Yearly blood pressure, cholesterol, and glucose monitoring<br>
                        • Age-appropriate preventive screenings per guidelines
                    </p>
                    
                    <h4 style="color: #0891b2; margin-top: 1.5rem;">Heart-Healthy Diet</h4>
                    <p style="color: #155e75;">
                        • Mediterranean or DASH diet rich in fruits and vegetables<br>
                        • Omega-3 fatty acids from fish or supplements<br>
                        • Limit sodium intake to &lt;2300mg/day<br>
                        • Minimize saturated fats and avoid trans fats
                    </p>
                    
                    <h4 style="color: #059669; margin-top: 1.5rem;">Physical Activity</h4>
                    <p style="color: #047857;">
                        • 150 minutes/week of moderate-intensity aerobic activity<br>
                        • Strength training exercises twice weekly<br>
                        • Regular stretching and flexibility exercises
                    </p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem; border-top: 1px solid #e2e8f0;">
        CardioRisk AI v1.0 | For Healthcare Professional Use Only
    </div>
""", unsafe_allow_html=True)