import streamlit as st

st.set_page_config(
    page_title="About Project - CardioRisk AI",
    page_icon="📄",
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
    
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #059669;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">📄 About This Project</h1>', unsafe_allow_html=True)
st.markdown("### Project Documentation & Technical Overview")

st.markdown("---")

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">💡 Project Motivation</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem;">
            Cardiovascular disease remains the leading cause of death globally, claiming millions of lives 
            annually. Early detection and risk assessment are critical for prevention and intervention, yet 
            traditional screening methods face several challenges:
        </p>
        <ul style="line-height: 2; color: #475569; font-size: 1rem; margin-top: 1rem;">
            <li>Limited healthcare resources and specialist availability in many regions</li>
            <li>Time constraints in busy clinical settings</li>
            <li>Variability in risk assessment approaches across providers</li>
            <li>Need for rapid, consistent screening in population health programs</li>
        </ul>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem; margin-top: 1rem;">
            <strong>CardioRisk AI</strong> was developed to address these challenges by providing healthcare 
            professionals with a rapid, consistent, and evidence-based tool for cardiovascular risk assessment. 
            By leveraging machine learning, we aim to enhance clinical decision-making while maintaining the 
            essential role of human medical expertise.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">🎯 Problem Statement</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem;">
            <strong>Primary Challenge:</strong> How can we assist healthcare professionals in efficiently 
            identifying patients at elevated cardiovascular risk while maintaining accuracy and clinical 
            relevance?
        </p>
        <div class="highlight-box" style="margin-top: 1.5rem;">
            <h4 style="color: #047857; margin-top: 0; font-size: 1.2rem;">Our Approach</h4>
            <p style="color: #065f46; line-height: 1.8; font-size: 1rem;">
                Develop an AI-powered clinical decision support system that:
            </p>
            <ul style="color: #065f46; line-height: 2; font-size: 1rem;">
                <li>Analyzes readily available patient data (demographics, vital signs, lifestyle factors)</li>
                <li>Provides instant risk assessments with probability scores</li>
                <li>Identifies specific risk factors requiring attention</li>
                <li>Offers evidence-based recommendations for risk management</li>
                <li>Integrates seamlessly into existing clinical workflows</li>
            </ul>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="section-card">
            <h3 class="section-title">🔬 Machine Learning Pipeline</h3>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem;">
                The system employs a supervised machine learning approach with the following architecture:
            </p>
            
            <h4 style="color: #2563eb; margin-top: 1.5rem; font-size: 1.1rem;">Data Processing</h4>
            <ul style="line-height: 2; color: #475569; font-size: 0.95rem;">
                <li>Feature engineering (BMI calculation)</li>
                <li>Data standardization and normalization</li>
                <li>Categorical encoding for ordinal features</li>
                <li>Handling of missing values and outliers</li>
            </ul>
            
            <h4 style="color: #2563eb; margin-top: 1.5rem; font-size: 1.1rem;">Model Training</h4>
            <ul style="line-height: 2; color: #475569; font-size: 0.95rem;">
                <li>Binary classification task (CVD risk: Yes/No)</li>
                <li>Train-test split for validation</li>
                <li>Cross-validation for robustness</li>
                <li>Hyperparameter optimization</li>
                <li>Performance evaluation metrics</li>
            </ul>
            
            <h4 style="color: #2563eb; margin-top: 1.5rem; font-size: 1.1rem;">Deployment</h4>
            <ul style="line-height: 2; color: #475569; font-size: 0.95rem;">
                <li>Serialized pipeline using joblib</li>
                <li>Real-time inference capabilities</li>
                <li>Probability calibration for risk scores</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="section-card">
            <h3 class="section-title">💻 Technology Stack</h3>
            <p style="line-height: 1.8; color: #334155; font-size: 1rem; margin-bottom: 1rem;">
                Built with modern, production-ready technologies:
            </p>
            
            <h4 style="color: #059669; margin-top: 1.5rem; font-size: 1.1rem;">Frontend & UI</h4>
            <div style="margin-bottom: 1rem;">
                <span class="tech-badge">Streamlit</span>
                <span class="tech-badge">HTML/CSS</span>
                <span class="tech-badge">Responsive Design</span>
            </div>
            
            <h4 style="color: #059669; margin-top: 1.5rem; font-size: 1.1rem;">Backend & ML</h4>
            <div style="margin-bottom: 1rem;">
                <span class="tech-badge">Python 3.8+</span>
                <span class="tech-badge">Scikit-learn</span>
                <span class="tech-badge">Pandas</span>
                <span class="tech-badge">NumPy</span>
                <span class="tech-badge">Joblib</span>
            </div>
            
            <h4 style="color: #059669; margin-top: 1.5rem; font-size: 1.1rem;">Development</h4>
            <div style="margin-bottom: 1rem;">
                <span class="tech-badge">Git</span>
                <span class="tech-badge">Jupyter Notebooks</span>
                <span class="tech-badge">VS Code</span>
            </div>
            
            <div class="info-box" style="margin-top: 1.5rem;">
                <p style="color: #1e3a8a; font-size: 0.95rem; margin: 0;">
                    <strong>🔒 Security:</strong> All data processing occurs locally. No patient 
                    information is transmitted to external servers or stored persistently.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">📊 Dataset Information</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem;">
            The model was trained on a comprehensive cardiovascular disease dataset containing patient 
            health records with the following characteristics:
        </p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem;">
            <div style="background: #f1f5f9; padding: 1.5rem; border-radius: 8px;">
                <h4 style="color: #334155; margin-top: 0; font-size: 1.1rem;">📈 Dataset Scope</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.8;">
                    Large-scale clinical dataset with demographic, physiological, and lifestyle features 
                    collected from diverse patient populations. Data includes both patients with and without 
                    cardiovascular disease for balanced model training.
                </p>
            </div>
            
            <div style="background: #f1f5f9; padding: 1.5rem; border-radius: 8px;">
                <h4 style="color: #334155; margin-top: 0; font-size: 1.1rem;">🔬 Feature Categories</h4>
                <ul style="color: #475569; font-size: 0.95rem; line-height: 2;">
                    <li>Demographics (age, gender)</li>
                    <li>Anthropometric (height, weight, BMI)</li>
                    <li>Clinical (BP, cholesterol, glucose)</li>
                    <li>Behavioral (smoking, alcohol, activity)</li>
                </ul>
            </div>
        </div>
        
        <div class="info-box" style="margin-top: 1.5rem;">
            <p style="color: #1e3a8a; font-size: 1rem; margin: 0;">
                <strong>📝 Note:</strong> Dataset details, preprocessing steps, and model performance 
                metrics are documented in the technical documentation. The model undergoes regular 
                validation to ensure continued accuracy and relevance.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">🚀 Future Improvements</h3>
        <p style="line-height: 1.8; color: #334155; font-size: 1.05rem; margin-bottom: 1.5rem;">
            Planned enhancements to expand functionality and improve clinical utility:
        </p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="background: #fef3c7; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <h4 style="color: #92400e; margin-top: 0; font-size: 1.1rem;">🎯 Short-term Goals</h4>
                <ul style="color: #78350f; line-height: 2; font-size: 0.95rem;">
                    <li>Integration with electronic health records (EHR)</li>
                    <li>Batch processing for population screening</li>
                    <li>Enhanced visualization of risk factors</li>
                    <li>Multi-language support</li>
                    <li>PDF report generation</li>
                    <li>User authentication and role-based access</li>
                </ul>
            </div>
            
            <div style="background: #dbeafe; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <h4 style="color: #1e40af; margin-top: 0; font-size: 1.1rem;">🔮 Long-term Vision</h4>
                <ul style="color: #1e3a8a; line-height: 2; font-size: 0.95rem;">
                    <li>Incorporation of genetic risk factors</li>
                    <li>Temporal risk tracking over time</li>
                    <li>Deep learning models for improved accuracy</li>
                    <li>Explainable AI (XAI) for feature importance</li>
                    <li>Mobile application development</li>
                    <li>Real-time continuous monitoring integration</li>
                </ul>
            </div>
        </div>
        
        <div style="background: #f0fdf4; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #059669; margin-top: 1rem;">
            <h4 style="color: #047857; margin-top: 0; font-size: 1.1rem;">🔬 Research Directions</h4>
            <p style="color: #065f46; line-height: 1.8; font-size: 0.95rem;">
                Ongoing research to enhance model fairness, reduce bias, improve performance across 
                diverse populations, and validate effectiveness in real-world clinical settings through 
                prospective studies and clinical trials.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="section-card">
        <h3 class="section-title">📜 Version Information</h3>
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #cbd5e1;">
                    <th style="text-align: left; padding: 0.75rem; color: #334155;">Component</th>
                    <th style="text-align: left; padding: 0.75rem; color: #334155;">Version</th>
                    <th style="text-align: left; padding: 0.75rem; color: #334155;">Release Date</th>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem; color: #475569;">CardioRisk AI Application</td>
                    <td style="padding: 0.75rem; color: #475569; font-weight: 600;">v1.0.0</td>
                    <td style="padding: 0.75rem; color: #475569;">January 2025</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem; color: #475569;">ML Model Pipeline</td>
                    <td style="padding: 0.75rem; color: #475569; font-weight: 600;">v1.0.0</td>
                    <td style="padding: 0.75rem; color: #475569;">January 2025</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem; color: #475569;">UI Framework</td>
                    <td style="padding: 0.75rem; color: #475569; font-weight: 600;">Streamlit</td>
                    <td style="padding: 0.75rem; color: #475569;">—</td>
                </tr>
                <tr>
                    <td style="padding: 0.75rem; color: #475569;">Python Environment</td>
                    <td style="padding: 0.75rem; color: #475569; font-weight: 600;">3.8+</td>
                    <td style="padding: 0.75rem; color: #475569;">—</td>
                </tr>
            </table>
        </div>
        
        <div class="info-box" style="margin-top: 1.5rem;">
            <h4 style="color: #1e40af; margin-top: 0; font-size: 1.1rem;">📝 Release Notes</h4>
            <p style="color: #1e3a8a; font-size: 0.95rem; line-height: 1.8;">
                <strong>v1.0.0 (January 2025)</strong>
            </p>
            <ul style="color: #334155; font-size: 0.95rem; line-height: 2;">
                <li>Initial production release</li>
                <li>Core prediction engine with 12-feature model</li>
                <li>Multi-page Streamlit interface</li>
                <li>Clinical recommendations system</li>
                <li>Comprehensive documentation and disclaimers</li>
                <li>Risk factor identification and analysis</li>
            </ul>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="highlight-box">
        <h3 style="color: #047857; margin-top: 0; font-size: 1.4rem;">🎓 Academic & Portfolio Value</h3>
        <p style="line-height: 1.8; color: #065f46; font-size: 1.05rem;">
            This project demonstrates:
        </p>
        <ul style="line-height: 2; color: #065f46; font-size: 1rem;">
            <li><strong>End-to-End ML Engineering:</strong> From data processing to production deployment</li>
            <li><strong>Healthcare AI Application:</strong> Domain-specific knowledge and clinical relevance</li>
            <li><strong>User Experience Design:</strong> Professional, hospital-grade interface development</li>
            <li><strong>Responsible AI Practices:</strong> Ethical considerations, transparency, and human oversight</li>
            <li><strong>Technical Documentation:</strong> Comprehensive system documentation and user guides</li>
            <li><strong>Production Readiness:</strong> Error handling, validation, and deployment considerations</li>
        </ul>
        <p style="line-height: 1.8; color: #065f46; font-size: 1.05rem; margin-top: 1rem;">
            <strong>Perfect for:</strong> Data science portfolios, healthcare informatics projects, 
            ML engineering demonstrations, and academic research in clinical AI applications.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem; border-top: 1px solid #e2e8f0;">
        CardioRisk AI v1.0 | Developed for Healthcare Professional Use
        <br>
        © 2025 Healthcare AI Solutions | MIT License
    </div>
""", unsafe_allow_html=True)