import streamlit as st

st.set_page_config(
    page_title="Overview - CardioRisk AI",
    page_icon="ℹ️",
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
    
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border-left: 5px solid #3b82f6;
    }
    
    .section-title {
        color: #1e40af;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">ℹ️ System Overview</h1>', unsafe_allow_html=True)
st.markdown("### Understanding Cardiovascular Disease and This Assessment Tool")

st.markdown("---")

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">❤️ What is Cardiovascular Disease?</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem;">
            Cardiovascular disease (CVD) refers to conditions affecting the heart and blood vessels. 
            It is the leading cause of death globally, accounting for approximately 17.9 million deaths 
            annually according to the World Health Organization.
        </p>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem; margin-top: 1rem;">
            <strong>Common types include:</strong>
        </p>
        <ul style="line-height: 2; color: #475569; font-size: 1rem;">
            <li><strong>Coronary Artery Disease:</strong> Reduced blood flow to the heart muscle</li>
            <li><strong>Cerebrovascular Disease:</strong> Conditions affecting blood vessels supplying the brain</li>
            <li><strong>Peripheral Arterial Disease:</strong> Narrowing of blood vessels in limbs</li>
            <li><strong>Heart Failure:</strong> The heart's inability to pump blood effectively</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">🎯 The Problem We're Solving</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem;">
            Early detection and risk assessment are crucial for preventing cardiovascular disease. 
            However, traditional risk assessment methods can be:
        </p>
        <ul style="line-height: 2; color: #475569; font-size: 1rem; margin-top: 1rem;">
            <li>Time-consuming for healthcare professionals</li>
            <li>Limited by subjective interpretation</li>
            <li>Inconsistent across different practitioners</li>
            <li>Difficult to update with new research findings</li>
        </ul>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem; margin-top: 1rem;">
            <strong>Our Solution:</strong> An AI-powered clinical decision support system that provides 
            rapid, consistent, evidence-based cardiovascular risk assessments to assist healthcare 
            professionals in making informed decisions.
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="section-card">
            <h3 class="section-title">🔬 How It Works</h3>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem;">
                Our system uses machine learning to analyze patient data across three key categories:
            </p>
            <ol style="line-height: 2; color: #475569; font-size: 1rem; margin-top: 1rem;">
                <li><strong>Demographics:</strong> Age, gender, height, weight, BMI</li>
                <li><strong>Clinical Measurements:</strong> Blood pressure, cholesterol, glucose levels</li>
                <li><strong>Lifestyle Factors:</strong> Smoking, alcohol consumption, physical activity</li>
            </ol>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem; margin-top: 1rem;">
                The trained model processes these inputs and generates:
            </p>
            <ul style="line-height: 2; color: #475569; font-size: 1rem;">
                <li>Risk classification (High/Low)</li>
                <li>Probability score (0-100%)</li>
                <li>Identified risk factors</li>
                <li>Clinical recommendations</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="section-card">
            <h3 class="section-title">👥 Who Should Use This Tool?</h3>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem;">
                This system is designed for:
            </p>
            <ul style="line-height: 2; color: #475569; font-size: 1rem; margin-top: 1rem;">
                <li><strong>Cardiologists:</strong> Supplementary risk assessment tool</li>
                <li><strong>Primary Care Physicians:</strong> Screening and referral decisions</li>
                <li><strong>Nurse Practitioners:</strong> Patient triage and education</li>
                <li><strong>Healthcare Administrators:</strong> Population health management</li>
                <li><strong>Clinical Researchers:</strong> Study design and patient stratification</li>
            </ul>
            <div class="warning-box" style="margin-top: 1.5rem;">
                <strong>⚠️ Important:</strong> This tool is <strong>NOT</strong> intended for:
                <ul style="margin-top: 0.5rem;">
                    <li>Direct patient use without medical supervision</li>
                    <li>Emergency situations requiring immediate care</li>
                    <li>Replacing comprehensive clinical evaluation</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="info-box">
        <h4 style="color: #1e40af; margin-top: 0;">🚀 Ready to Get Started?</h4>
        <p style="color: #1e3a8a; font-size: 1rem; margin-bottom: 0.5rem;">
            Navigate to the <strong>Risk Prediction</strong> page using the sidebar to begin assessing patient risk.
        </p>
        <p style="color: #475569; font-size: 0.95rem; margin-top: 0.5rem;">
            We recommend reviewing the <strong>Model Insights</strong> and <strong>Medical Disclaimer</strong> 
            pages before your first assessment to understand the system's capabilities and limitations.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem; border-top: 1px solid #e2e8f0;">
        CardioRisk AI v1.0 | Clinical Decision Support System
    </div>
""", unsafe_allow_html=True)