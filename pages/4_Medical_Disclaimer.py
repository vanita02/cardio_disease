import streamlit as st

st.set_page_config(
    page_title="Medical Disclaimer - CardioRisk AI",
    page_icon="⚠️",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    .page-title {
        color: #dc2626;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .critical-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 3px solid #dc2626;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border-left: 5px solid #64748b;
    }
    
    .section-title {
        color: #1e40af;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .emergency-box {
        background: #7f1d1d;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        border: 3px solid #dc2626;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">⚠️ Medical Disclaimer & Legal Notice</h1>', unsafe_allow_html=True)
st.markdown("### Important Information for Users and Healthcare Providers")

st.markdown("---")

st.markdown("""
    <div class="critical-box">
        <h2 style="color: #dc2626; margin-top: 0; font-size: 1.8rem;">🚨 CRITICAL DISCLAIMER</h2>
        <p style="color: #991b1b; font-size: 1.2rem; line-height: 1.8; font-weight: 600;">
            This application is NOT a medical diagnostic tool and should NOT be used as a substitute 
            for professional medical advice, diagnosis, or treatment.
        </p>
        <p style="color: #7f1d1d; font-size: 1.05rem; line-height: 1.8; margin-top: 1rem;">
            CardioRisk AI is a <strong>clinical decision support system</strong> designed to assist 
            healthcare professionals in risk assessment. It does not diagnose cardiovascular disease, 
            nor does it provide medical treatment recommendations without proper clinical oversight.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="emergency-box">
        <h3 style="color: white; margin-top: 0; font-size: 1.6rem;">🚑 EMERGENCY SITUATIONS</h3>
        <p style="font-size: 1.1rem; line-height: 1.8; margin-bottom: 1rem;">
            <strong>If you or someone else is experiencing any of the following symptoms, 
            DO NOT use this tool. Call emergency services immediately (911 in the US):</strong>
        </p>
        <ul style="font-size: 1.05rem; line-height: 2; margin-left: 1.5rem;">
            <li>Chest pain or pressure</li>
            <li>Difficulty breathing or shortness of breath</li>
            <li>Pain radiating to arm, jaw, back, or neck</li>
            <li>Sudden severe headache</li>
            <li>Sudden weakness or numbness on one side of body</li>
            <li>Loss of consciousness or fainting</li>
            <li>Irregular or rapid heartbeat with symptoms</li>
            <li>Severe dizziness or confusion</li>
        </ul>
        <p style="font-size: 1.1rem; line-height: 1.8; margin-top: 1rem; font-weight: 600;">
            ⚠️ This tool is NOT designed for emergency situations and cannot replace emergency medical care.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">👨‍⚕️ For Healthcare Professionals</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem;">
            This system is intended as a <strong>supplementary tool</strong> to aid in clinical decision-making. 
            Healthcare providers using this system should:
        </p>
        <ul style="line-height: 2; color: #475569; font-size: 1rem; margin-top: 1rem;">
            <li><strong>Maintain professional judgment:</strong> Model predictions should inform, not dictate, 
            clinical decisions</li>
            <li><strong>Conduct comprehensive evaluation:</strong> Use standard diagnostic procedures and 
            complete patient assessments</li>
            <li><strong>Consider individual context:</strong> Account for patient history, comorbidities, 
            medications, and unique circumstances</li>
            <li><strong>Verify data accuracy:</strong> Ensure all input data is current, accurate, and 
            clinically appropriate</li>
            <li><strong>Document decisions:</strong> Record clinical reasoning for decisions made based on 
            or contrary to model predictions</li>
            <li><strong>Stay current:</strong> Keep informed about model updates, limitations, and 
            performance characteristics</li>
        </ul>
        <div class="warning-box">
            <p style="color: #92400e; font-size: 1rem; margin: 0;">
                <strong>⚠️ Professional Responsibility:</strong> Healthcare providers remain fully responsible 
                for all clinical decisions and patient care, regardless of model predictions. This tool does 
                not absolve practitioners of their professional duties or liability.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">🔐 Privacy & Data Security</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem;">
            <strong>Data Processing:</strong> This application processes patient data locally for risk 
            assessment purposes. Users and healthcare providers should:
        </p>
        <ul style="line-height: 2; color: #475569; font-size: 1rem; margin-top: 1rem;">
            <li>Ensure compliance with HIPAA, GDPR, and applicable data protection regulations</li>
            <li>Obtain appropriate patient consent before entering data</li>
            <li>Use secure networks and devices when accessing the system</li>
            <li>Never enter unnecessary personally identifiable information</li>
            <li>Follow institutional policies for electronic health record usage</li>
        </ul>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem; margin-top: 1rem;">
            <strong>Data Retention:</strong> Data entered into this system is not stored permanently. 
            However, users should be aware that session data may be temporarily cached by the browser.
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="section-card">
            <h3 class="section-title">📋 Limitations of Use</h3>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem;">
                This tool has important limitations:
            </p>
            <ul style="line-height: 2; color: #475569; font-size: 1rem; margin-top: 1rem;">
                <li>Based on statistical models that may not reflect individual patient circumstances</li>
                <li>Does not account for all risk factors or medical conditions</li>
                <li>Performance may vary across different populations</li>
                <li>Requires accurate, complete input data for reliable results</li>
                <li>Cannot predict specific cardiac events or timing</li>
                <li>Not validated for all patient populations or clinical settings</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="section-card">
            <h3 class="section-title">⚖️ Legal Notice</h3>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem;">
                <strong>No Warranties:</strong> This software is provided "as is" without warranties 
                of any kind, express or implied.
            </p>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem; margin-top: 1rem;">
                <strong>Limitation of Liability:</strong> The developers and distributors of this 
                tool are not liable for any clinical decisions, patient outcomes, or damages resulting 
                from use of this system.
            </p>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem; margin-top: 1rem;">
                <strong>Professional Consultation Required:</strong> Always seek the advice of qualified 
                healthcare providers with questions regarding medical conditions.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">🤖 Responsible AI Statement</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem;">
            This system is developed and maintained with a commitment to responsible artificial intelligence:
        </p>
        
        <h4 style="color: #2563eb; margin-top: 1.5rem; font-size: 1.2rem;">Fairness & Equity</h4>
        <p style="line-height: 1.8; color: #475569; font-size: 1rem;">
            We strive to ensure the model performs equitably across diverse patient populations. However, 
            users should be aware that algorithmic bias may exist and should monitor for disparities in 
            model performance across demographic groups.
        </p>
        
        <h4 style="color: #2563eb; margin-top: 1.5rem; font-size: 1.2rem;">Transparency</h4>
        <p style="line-height: 1.8; color: #475569; font-size: 1rem;">
            We provide clear information about features used, model limitations, and intended use cases. 
            The Model Insights page offers detailed information about how predictions are generated.
        </p>
        
        <h4 style="color: #2563eb; margin-top: 1.5rem; font-size: 1.2rem;">Accountability</h4>
        <p style="line-height: 1.8; color: #475569; font-size: 1rem;">
            Healthcare providers remain accountable for all clinical decisions. This tool is designed to 
            support, not replace, human judgment and expertise. Regular audits and performance monitoring 
            ensure the system functions as intended.
        </p>
        
        <h4 style="color: #2563eb; margin-top: 1.5rem; font-size: 1.2rem;">Safety</h4>
        <p style="line-height: 1.8; color: #475569; font-size: 1rem;">
            Patient safety is paramount. The system includes validation checks, clear disclaimers, and 
            emphasis on human oversight. Users should report any safety concerns or unexpected model behavior 
            to system administrators immediately.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="warning-box">
        <h4 style="color: #92400e; margin-top: 0; font-size: 1.3rem;">📞 Questions or Concerns?</h4>
        <p style="color: #78350f; line-height: 1.8; font-size: 1rem;">
            If you have questions about this tool's use, limitations, or appropriate application in 
            clinical practice, please consult with your institution's medical informatics team, risk 
            management department, or seek guidance from appropriate clinical leadership.
        </p>
        <p style="color: #78350f; line-height: 1.8; font-size: 1rem; margin-top: 1rem;">
            For technical issues or to report unexpected model behavior, contact your system administrator 
            or the development team through appropriate institutional channels.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background: #1e3a8a; color: white; padding: 2rem; border-radius: 10px; margin-top: 2rem; text-align: center;">
        <h3 style="color: white; margin-top: 0;">By using this system, you acknowledge that you have read, 
        understood, and agree to these terms.</h3>
        <p style="font-size: 1.05rem; margin-bottom: 0;">
            This disclaimer was last updated: January 2025
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem; border-top: 1px solid #e2e8f0;">
        CardioRisk AI v1.0 | For Healthcare Professional Use Only
    </div>
""", unsafe_allow_html=True)