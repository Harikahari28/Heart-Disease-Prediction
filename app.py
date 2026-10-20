import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import tempfile
from video_model.inference import load_echo_model, predict_video

from multimodal.tabular_ensemble import ensemble_tabular
from multimodal.late_fusion import late_fusion
from video_model.inference import predict_video
from explainability.shap_tabular import explain_tabular, explain_tabular_table
from explainability.gradcam_video import gradcam_for_video
from explainability.chatbot_engine import get_chatbot_response


@st.cache_resource
def load_echo_cached():
    return load_echo_model()

echo_model = load_echo_cached()

if hasattr(st, "fragment"):
    fragment = st.fragment
else:
    def fragment(func):
        return func

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Professional Styling
# ===============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Make Streamlit Top Header Bar Transparent and hide default decorations & menu buttons */
    header, [data-testid="stHeader"], .stAppHeader {
        background: transparent !important;
        background-color: transparent !important;
        border-bottom: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stDecoration"] {
        display: none !important;
        height: 0px !important;
    }
    
    /* Hide Deploy and Menu buttons but keep sidebar toggle visible */
    [data-testid="stDeployButton"], [data-testid="stMainMenu"] {
        display: none !important;
    }
    
    /* Style collapse/open sidebar button so it is clearly visible */
    [data-testid="collapsedControl"], .stSidebarCollapseButton {
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
    }
    
    /* Main background and font */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0A0D14 !important;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* Custom titles */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 16px;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
        border-bottom: 2px solid #3182CE;
        padding-bottom: 8px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(49, 130, 206, 0.4);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4), 0 0 20px rgba(49, 130, 206, 0.1);
    }
    
    /* Cards for predictions */
    .pred-card {
        background: rgba(26, 32, 44, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid #3182CE;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .model-name {
        font-family: 'Outfit', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #CBD5E0;
        margin-bottom: 8px;
    }
    .prob-text {
        font-size: 16px;
        color: #E2E8F0;
        margin: 6px 0;
    }
    .risk-high {
        color: #FF5252;
        font-weight: 700;
        font-size: 18px;
    }
    .risk-low {
        color: #48BB78;
        font-weight: 700;
        font-size: 18px;
    }
    .risk-medium {
        color: #F6C244;
        font-weight: 700;
        font-size: 18px;
    }
    
    /* Highest risk highlight */
    .final-risk {
        background: rgba(45, 55, 72, 0.6);
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0E14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Style the radio menu itself */
    div[data-testid="stRadio"] {
        background: rgba(22, 27, 34, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px;
    }
    div[data-testid="stRadio"] label {
        padding: 10px 14px;
        border-radius: 8px;
        margin: 4px 0;
        transition: all 0.2s ease;
        cursor: pointer;
        color: #E2E8F0 !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: rgba(49, 130, 206, 0.15) !important;
        border-left: 3px solid #3182CE;
        color: #3182CE !important;
        font-weight: 600;
    }
    
    /* Premium Streamlit Buttons styling */
    div.stButton > button {
        background: linear-gradient(135deg, #3182CE 0%, #2B6CB0 100%);
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 10px rgba(49, 130, 206, 0.2);
        margin: 4px 0;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4299E1 0%, #3182CE 100%);
        box-shadow: 0 6px 15px rgba(49, 130, 206, 0.4);
        transform: translateY(-2px);
        border: none;
        color: #FFFFFF !important;
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metrics panel */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        margin-top: 30px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    .metric-box {
        flex: 1;
        min-width: 200px;
        background: rgba(22, 27, 34, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        border-color: rgba(49, 130, 206, 0.3);
        background: rgba(22, 27, 34, 0.6);
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 32px;
        font-weight: 800;
        color: #3182CE;
        margin-bottom: 4px;
        font-family: 'Outfit', sans-serif;
    }
    .metric-label {
        font-size: 13px;
        color: #A0AEC0;
        font-weight: 500;
    }
    
    /* Live Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(72, 187, 120, 0.08);
        border: 1px solid rgba(72, 187, 120, 0.2);
        color: #48BB78;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #48BB78;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px #48BB78;
        animation: pulseStatus 1.8s infinite;
    }
    @keyframes pulseStatus {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(72, 187, 120, 0.5); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(72, 187, 120, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(72, 187, 120, 0); }
    }
    
    /* 3D Holographic Heart Container */
    .heart-container {
        position: relative;
        width: 100%;
        height: 350px;
        display: flex;
        justify-content: center;
        align-items: center;
        perspective: 800px;
        background: radial-gradient(circle, rgba(49, 130, 206, 0.05) 0%, transparent 70%);
    }
    .grid-back {
        position: absolute;
        width: 280px;
        height: 280px;
        background-image: 
            linear-gradient(rgba(49, 130, 206, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(49, 130, 206, 0.05) 1px, transparent 1px);
        background-size: 20px 20px;
        background-position: center;
        border-radius: 50%;
        border: 1px solid rgba(49, 130, 206, 0.1);
        transform: rotateX(60deg) translateZ(-40px);
        animation: spinGrid 30s linear infinite;
    }
    .heart-wrapper {
        transform-style: preserve-3d;
        animation: rotateHeart 20s linear infinite;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hologram-heart {
        width: 160px;
        height: 160px;
        animation: beat 1.3s ease-in-out infinite;
        filter: drop-shadow(0 0 15px rgba(239, 83, 80, 0.6));
    }
    .scan-line {
        position: absolute;
        width: 80%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #3182CE, transparent);
        box-shadow: 0 0 8px #3182CE;
        animation: scan 4s ease-in-out infinite;
    }
    
    @keyframes rotateHeart {
        0% { transform: rotateY(0deg) rotateX(10deg); }
        100% { transform: rotateY(360deg) rotateX(10deg); }
    }
    @keyframes beat {
        0% { transform: scale(1); }
        25% { transform: scale(1.15); }
        40% { transform: scale(1.02); }
        60% { transform: scale(1.08); }
        100% { transform: scale(1); }
    }
    @keyframes spinGrid {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes scan {
        0% { top: 15%; opacity: 0.1; }
        50% { top: 85%; opacity: 0.8; }
        100% { top: 15%; opacity: 0.1; }
    }
    
    /* Fade In Animations */
    .fade-in {
        animation: fadeIn 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Clean streamlit layout spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Fix Chat Input Coloring (Premium Dark Blue matching website theme) */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] div,
    [data-testid="stChatInput"] textarea {
        background-color: #121824 !important;
        color: #FAFAFA !important;
        -webkit-text-fill-color: #FAFAFA !important;
    }
    [data-testid="stChatInput"] {
        border: 1px solid rgba(49, 130, 206, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
    }
    /* Placeholder styling */
    [data-testid="stChatInput"] p {
        color: #A0AEC0 !important;
    }
    
    /* Fix Chat Input Wrapper surrounding background to be dark slate */
    [data-testid="stBottomBlockContainer"], 
    [data-testid="stBottom"],
    .stBottomBlockContainer,
    .stBottomBlockContainer > div {
        background-color: #0A0D14 !important;
        background: #0A0D14 !important;
        border-top: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Session State
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for current page if not already present
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home / Dashboard"

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 10px; margin-top: -30px;'>
    <h2 style='color: #3182CE; font-weight: 800; margin-bottom: 0; font-size: 26px; font-family: "Outfit", sans-serif; letter-spacing: -0.5px;'>PULSE AI</h2>
    <span style='color: #718096; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Diagnostic Console</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

options = [
    "🏠 Home / Dashboard",
    "📋 Tabular Risk Factors",
    "🫀 ECHO Video Analysis",
    "🔗 Multimodal Prediction",
    "💬 Chat Assistant"
]

default_index = options.index(st.session_state.current_page)

selected_mode = st.sidebar.radio(
    "Navigation Menu",
    options,
    index=default_index
)

if selected_mode != st.session_state.current_page:
    st.session_state.current_page = selected_mode
    st.rerun()

mode = st.session_state.current_page

# ===============================
# Load Tabular Models
# ===============================
@st.cache_resource
def load_tabular_models():
    model_paths = {
        "Logistic Regression": "utils/Logistic Regression_model.pkl",
        "XGBoost": "utils/XGBoost_model.pkl",
        "Random Forest": "utils/Random Forest_model.pkl",
        "SVM": "utils/SVM_model.pkl"
    }
    models = {}
    for name, path in model_paths.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models

models = load_tabular_models()
training_columns = joblib.load("utils/training_columns.pkl")

# ===============================
# Subpage Header (only shown for diagnostic modes)
# ===============================
if mode != "🏠 Home / Dashboard":
    st.markdown("<div class='main-title'>Heart Disease Prediction System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Clinical decision support using patient risk factors and echocardiography analysis</div>",
        unsafe_allow_html=True
    )

# ===============================
# HOME / DASHBOARD MODE
# ===============================
if mode == "🏠 Home / Dashboard":
    # Hero Section
    hero_col1, hero_col2 = st.columns([1.2, 0.8])
    with hero_col1:
        st.markdown("""
        <div class='fade-in' style='margin-top: 15px;'>
            <div class='status-badge' style='margin-bottom: 25px;'>
                <span class='status-dot'></span>
                AI Core Status: Online & Validated
            </div>
            <h1 style='font-size: 48px; font-weight: 800; line-height: 1.1; margin: 0 0 18px 0; background: linear-gradient(135deg, #FFFFFF 0%, #A0AEC0 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Next-Generation<br>Multimodal Cardiac Diagnostic Engine
            </h1>
            <p style='color: #A0AEC0; font-size: 17px; line-height: 1.6; margin: 0 0 32px 0;'>
                A state-of-the-art decision support system built for clinical practitioners. By combining 
                clinical risk factors with deep-learning-based apical 4-chamber echocardiography video analysis, 
                Pulse AI leverages performance-weighted Late Fusion for maximum diagnostic reliability.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Begin Multimodal Assessment 🚀", key="btn_hero_start", use_container_width=True):
            st.session_state.current_page = "🔗 Multimodal Prediction"
            st.rerun()
            
    with hero_col2:
        st.markdown("""
        <div class='heart-container fade-in'>
            <div class='grid-back'></div>
            <div class='heart-wrapper'>
                <svg viewBox="0 0 24 24" class="hologram-heart">
                    <defs>
                        <linearGradient id="heartGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#FF5252" />
                            <stop offset="50%" stop-color="#FF1744" />
                            <stop offset="100%" stop-color="#D50000" />
                        </linearGradient>
                    </defs>
                    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" fill="url(#heartGrad)" opacity="0.85"/>
                    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" fill="none" stroke="#FFFFFF" stroke-width="0.3" stroke-dasharray="1 1" opacity="0.5"/>
                </svg>
            </div>
            <div class='scan-line'></div>
            <div style='position: absolute; bottom: 12px; left: 12px; font-family: monospace; font-size: 11px; color: #48BB78; opacity: 0.8;'>
                SYS.MODE: STABLE_3.13<br>
                CORE: ECHONET_ACTIVE
            </div>
            <div style='position: absolute; top: 12px; right: 12px; font-family: monospace; font-size: 11px; color: #3182CE; opacity: 0.8;'>
                TELEMETRY: VERIFIED<br>
                MODEL_LATENCY: &lt; 2.4s
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Metrics Section
    st.markdown("""
    <div class='metric-container fade-in'>
        <div class='metric-box'>
            <div class='metric-val'>89.1%</div>
            <div class='metric-label'>XGBoost Test Accuracy</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>82.6%</div>
            <div class='metric-label'>XGBoost CV Accuracy</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>0.91</div>
            <div class='metric-label'>EchoNet Area Under ROC</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>&lt; 2.4s</div>
            <div class='metric-label'>Mean Inference Speed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Assessment Modules Section
    st.markdown("<div class='section-header fade-in'>🧠 Core Diagnostic Modalities</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='glass-card fade-in' style='height: 280px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <div style='font-size: 28px; margin-bottom: 12px;'>📋</div>
                <h3 style='margin: 0 0 8px 0; color: #FFFFFF;'>Clinical Risk Factors</h3>
                <p style='color: #A0AEC0; font-size: 14px; line-height: 1.5; margin: 0;'>
                    Assesses cardiovascular health using clinical measurements such as blood pressure, 
                    cholesterol, and electrocardiogram (ECG) data. Evaluates patient risk across 
                    four ML models (XGBoost, RF, SVM, LR) and visualizes features using SHAP explainability.
                </p>
            </div>
            <div style='margin-top: auto;'></div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Tabular Analyzer 🔬", key="btn_tab_link", use_container_width=True):
            st.session_state.current_page = "📋 Tabular Risk Factors"
            st.rerun()

    with col2:
        st.markdown("""
        <div class='glass-card fade-in' style='height: 280px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <div style='font-size: 28px; margin-bottom: 12px;'>🫀</div>
                <h3 style='margin: 0 0 8px 0; color: #FFFFFF;'>Echocardiography DL</h3>
                <p style='color: #A0AEC0; font-size: 14px; line-height: 1.5; margin: 0;'>
                    Performs automated ejection fraction and cardiac wall motion classification from 
                    apical 4-chamber echocardiography videos. Uses the EchoNet-Dynamic 3D Spatiotemporal Convolutional 
                    model, displaying saliency maps via Grad-CAM.
                </p>
            </div>
            <div style='margin-top: auto;'></div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Echo Video Analyzer 🎬", key="btn_echo_link", use_container_width=True):
            st.session_state.current_page = "🫀 ECHO Video Analysis"
            st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class='glass-card fade-in' style='height: 280px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <div style='font-size: 28px; margin-bottom: 12px;'>🔗</div>
                <h3 style='margin: 0 0 8px 0; color: #FFFFFF;'>Late-Fusion Multimodal AI</h3>
                <p style='color: #A0AEC0; font-size: 14px; line-height: 1.5; margin: 0;'>
                    The primary project module. Combines patient tabular clinical risk factors and spatial-temporal video 
                    representations using a performance-weighted late-fusion classifier (Echo 60% • Clinical 40%). 
                    Generates joint predictions and dual explainability diagnostics.
                </p>
            </div>
            <div style='margin-top: auto;'></div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Multimodal Assessment 🔗", key="btn_multi_link", use_container_width=True):
            st.session_state.current_page = "🔗 Multimodal Prediction"
            st.rerun()

    with col4:
        st.markdown("""
        <div class='glass-card fade-in' style='height: 280px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <div style='font-size: 28px; margin-bottom: 12px;'>💬</div>
                <h3 style='margin: 0 0 8px 0; color: #FFFFFF;'>Clinical AI Assistant</h3>
                <p style='color: #A0AEC0; font-size: 14px; line-height: 1.5; margin: 0;'>
                    An interactive chat companion designed to help clinicians interpret diagnostic results, 
                    review model confidence levels, explain SHAP/Grad-CAM parameters, and cross-reference 
                    medical guidelines for heart failure risk management.
                </p>
            </div>
            <div style='margin-top: auto;'></div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Access Chat Companion 💬", key="btn_chat_link", use_container_width=True):
            st.session_state.current_page = "💬 Chat Assistant"
            st.rerun()

    # Flowchart / System Pipeline
    st.markdown("<div class='section-header fade-in'>⚙️ Deep Learning & Fusion Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""<div class='fade-in'><p style='color: #A0AEC0; font-size: 14px; margin-bottom: 15px;'>The diagnostic pipeline routes raw data inputs to separate specialized architectures before unifying them in the Late Fusion Layer.</p><svg viewBox="0 0 800 240" class="glass-card" style="width: 100%; border-radius: 12px; margin: 0;"><defs><linearGradient id="glowGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#3182CE"/><stop offset="100%" stop-color="#319795"/></linearGradient><filter id="shadow"><feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#3182CE" flood-opacity="0.2"/></filter></defs><line x1="200" y1="70" x2="320" y2="70" stroke="rgba(49, 130, 206, 0.3)" stroke-width="2" stroke-dasharray="4 4"/><line x1="200" y1="170" x2="320" y2="170" stroke="rgba(49, 130, 206, 0.3)" stroke-width="2" stroke-dasharray="4 4"/><line x1="500" y1="70" x2="560" y2="120" stroke="rgba(49, 130, 206, 0.3)" stroke-width="2"/><line x1="500" y1="170" x2="560" y2="120" stroke="rgba(49, 130, 206, 0.3)" stroke-width="2"/><line x1="680" y1="120" x2="700" y2="120" stroke="rgba(49, 130, 206, 0.3)" stroke-width="2" stroke-dasharray="4 4"/><rect x="20" y="40" width="180" height="60" rx="10" fill="rgba(22, 27, 34, 0.8)" stroke="#3182CE" stroke-width="1.5" filter="url(#shadow)"/><text x="35" y="65" fill="#FFFFFF" font-family="'Outfit', sans-serif" font-weight="600" font-size="13">Clinical Risk Factors</text><text x="35" y="82" fill="#A0AEC0" font-family="sans-serif" font-size="11">Age, BP, ECG, ST-Slope...</text><rect x="20" y="140" width="180" height="60" rx="10" fill="rgba(22, 27, 34, 0.8)" stroke="#319795" stroke-width="1.5" filter="url(#shadow)"/><text x="35" y="165" fill="#FFFFFF" font-family="'Outfit', sans-serif" font-weight="600" font-size="13">Echocardiography Video</text><text x="35" y="182" fill="#A0AEC0" font-family="sans-serif" font-size="11">Apical 4-Chamber (.mp4)</text><rect x="320" y="40" width="180" height="60" rx="10" fill="rgba(22, 27, 34, 0.8)" stroke="#3182CE" stroke-width="1.5" filter="url(#shadow)"/><text x="335" y="65" fill="#FFFFFF" font-family="'Outfit', sans-serif" font-weight="600" font-size="13">XGBoost & SVM Ensemble</text><text x="335" y="82" fill="#A0AEC0" font-family="sans-serif" font-size="11">Optimized tabular classification</text><rect x="320" y="140" width="180" height="60" rx="10" fill="rgba(22, 27, 34, 0.8)" stroke="#319795" stroke-width="1.5" filter="url(#shadow)"/><text x="335" y="165" fill="#FFFFFF" font-family="'Outfit', sans-serif" font-weight="600" font-size="13">EchoNet 3D-ResNet</text><text x="335" y="182" fill="#A0AEC0" font-family="sans-serif" font-size="11">EF assessment & function checks</text><rect x="540" y="90" width="140" height="60" rx="10" fill="url(#glowGrad)" stroke="none" filter="url(#shadow)"/><text x="560" y="118" fill="#FFFFFF" font-family="'Outfit', sans-serif" font-weight="700" font-size="15">Late Fusion</text><text x="560" y="135" fill="rgba(255,255,255,0.8)" font-family="sans-serif" font-size="11">Clinically Weighted</text><rect x="700" y="90" width="90" height="60" rx="10" fill="rgba(22, 27, 34, 0.8)" stroke="#E2E8F0" stroke-width="1.5" filter="url(#shadow)"/><text x="712" y="118" fill="#FFFFFF" font-family="'Outfit', sans-serif" font-weight="600" font-size="12">SHAP + CAM</text><text x="712" y="135" fill="#A0AEC0" font-family="sans-serif" font-size="11">Explainable AI</text></svg></div>""", unsafe_allow_html=True)

# ===============================
# TABULAR MODE
# ===============================
elif mode == "📋 Tabular Risk Factors":
    st.markdown("<div class='section-header'>📋 Patient Clinical Data</div>", unsafe_allow_html=True)
    st.markdown("Enter patient information to assess cardiovascular risk.")

    with st.form("tabular_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", options=["M", "F"], index=0)
            chest_pain = st.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"])
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=300, value=140)
            cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "Yes" if x else "No")

        with col2:
            resting_ecg = st.selectbox("Resting ECG Result", options=["Normal", "ST", "LVH"])
            max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=220, value=150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", options=["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
            oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
            st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

        submitted = st.form_submit_button("🔬 Run Risk Assessment", use_container_width=True)

    if submitted:
        # Prepare input
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        encoded = pd.get_dummies(input_df)
        encoded = encoded.reindex(columns=training_columns, fill_value=0)

        best_prob = -1
        best_model_name = None
        preds_list = []

        for name, model in models.items():
            prob = model.predict_proba(encoded)[0][1]
            healthy_prob = 1 - prob

            risk_class = "risk-high" if prob > 0.45 else "risk-low"
            risk_text = "High Risk of Heart Disease" if prob > 0.45 else "Low Risk"

            preds_list.append({
                "name": name,
                "prob": prob,
                "healthy_prob": healthy_prob,
                "risk_class": risk_class,
                "risk_text": risk_text
            })

            if prob > best_prob:
                best_prob = prob
                best_model_name = name

        st.session_state.tabular_result = {
            "preds_list": preds_list,
            "best_prob": best_prob,
            "best_model_name": best_model_name,
            "input_df": input_df,
            "encoded": encoded
        }

    if "tabular_result" in st.session_state:
        res = st.session_state.tabular_result
        st.markdown("<div class='section-header'>📈 Model Predictions</div>", unsafe_allow_html=True)
        st.info("*Most reliable model- XGBOOST (Acc- 0.891304       F1- 0.902913*")

        for pred in res["preds_list"]:
            with st.container():
                st.markdown(f"""
                <div class='pred-card'>
                    <div class='model-name'>{pred["name"]}</div>
                    <div class='prob-text'>Disease Probability: <strong>{pred["prob"]*100:.2f}%</strong></div>
                    <div class='prob-text'>Healthy Probability: <strong>{pred["healthy_prob"]*100:.2f}%</strong></div>
                    <div style='margin-top:12px; font-size:18px;'>
                        Assessment: <span class='{pred["risk_class"]}'>{pred["risk_text"]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div class='final-risk'>
            <strong>Highest Estimated Risk: {res["best_prob"]*100:.2f}%</strong><br>
            <small>From model: <strong>{res["best_model_name"]}</strong></small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>🧩 Input Feature Summary</div>", unsafe_allow_html=True)
        st.dataframe(res["input_df"])

        # Tabular explainability
        with st.expander("🧭 Explainability: Tabular SHAP"):
            st.markdown("<div class='section-header'>SHAP Feature Importance</div>", unsafe_allow_html=True)
            try:
                with st.spinner("Generating SHAP explanations..."):
                    shap_fig = explain_tabular(models, res["encoded"], nsamples=50)
                    st.pyplot(shap_fig)
                    shap_table = explain_tabular_table(models, res["encoded"], nsamples=50)
                    st.dataframe(shap_table)
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

# ===============================
# ECHO VIDEO MODE
# ===============================
elif mode == "🫀 ECHO Video Analysis":
    st.markdown("<div class='section-header'>🫀 Echocardiography Video Analysis</div>", unsafe_allow_html=True)
    st.markdown("Upload an apical 4-chamber echocardiogram video for automated cardiac function assessment.")

    uploaded_video = st.file_uploader("Choose an ECHO video file (.mp4 recommended)", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        file_key = f"echo_file_{uploaded_video.name}_{uploaded_video.size}"
        if "echo_file_key" not in st.session_state or st.session_state.echo_file_key != file_key:
            st.session_state.echo_file_key = file_key
            if "echo_result" in st.session_state:
                del st.session_state.echo_result

        st.video(uploaded_video)
        
        if st.button("🔬 Analyze Video", use_container_width=True):
            with st.spinner("Processing video and running EchoNet-Dynamic inference..."):
                uploaded_video.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name

                prob = predict_video(video_path, echo_model)
                st.session_state.echo_result = {
                    "prob": prob,
                    "video_path": video_path
                }

        if "echo_result" in st.session_state:
            res = st.session_state.echo_result
            prob = res["prob"]
            video_path = res["video_path"]

            st.markdown("<div class='section-header'>📊 Echo Analysis Result</div>", unsafe_allow_html=True)
            
            risk_class = "risk-high" if prob > 0.45 else "risk-low"
            risk_text = "High Risk based on echocardiography" if prob > 0.45 else "Low Risk based on echocardiography"

            st.markdown(f"""
            <div class='pred-card'>
                <div class='model-name'>Model Prediction</div>
                <div class='prob-text'>Probability of abnormal cardiac function: <strong>{prob*100:.2f}%</strong></div>
                <div style='margin-top:12px; font-size:18px;'>
                    Assessment: <span class='{risk_class}'>{risk_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🧭 Explainability: Echo Grad-CAM"):
                st.subheader("Best Frame with Convergent Heatmap")
                try:
                    cam_img = gradcam_for_video(video_path, echo_model)
                    st.image(cam_img, use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM image failed: {e}")

# ===============================
# MULTIMODAL PLACEHOLDER
# ===============================
elif mode == "🔗 Multimodal Prediction":
    st.markdown("<div class='section-header'>🔗 Multimodal Heart Disease Assessment</div>", unsafe_allow_html=True)
    st.markdown(
        "This module combines **clinical risk factors** and **echocardiography video analysis** "
        "using a **late-fusion, probability-level ensemble**."
    )

    with st.form("multimodal_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", ["M", "F"])
            chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 0, 300, 140)
            cholesterol = st.number_input("Serum Cholesterol (mg/dL)", 0, 600, 200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

        with col2:
            resting_ecg = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
            max_hr = st.number_input("Maximum Heart Rate Achieved", 0, 220, 150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
            oldpeak = st.number_input("Oldpeak (ST Depression)", -3.0, 7.0, 1.0)
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        uploaded_video = st.file_uploader(
            "Upload Echocardiography Video (.mp4)", type=["mp4", "avi", "mov"]
        )
        if uploaded_video is not None:
            st.video(uploaded_video)
        submitted = st.form_submit_button("🔬 Run Multimodal Assessment", use_container_width=True)

    if uploaded_video is not None:
        file_key = f"multi_file_{uploaded_video.name}_{uploaded_video.size}"
        if "multi_file_key" not in st.session_state or st.session_state.multi_file_key != file_key:
            st.session_state.multi_file_key = file_key
            if "multi_result" in st.session_state:
                del st.session_state.multi_result

    if submitted:
        if uploaded_video is None:
            st.error("Please upload an echocardiography video for multimodal assessment.")
        else:
            with st.spinner("Running multimodal analysis..."):
                # ------------------------------
                # TABULAR INPUT PREPARATION
                # ------------------------------
                input_df = pd.DataFrame([{
                    "Age": age,
                    "Sex": sex,
                    "ChestPainType": chest_pain,
                    "RestingBP": resting_bp,
                    "Cholesterol": cholesterol,
                    "FastingBS": fasting_bs,
                    "RestingECG": resting_ecg,
                    "MaxHR": max_hr,
                    "ExerciseAngina": exercise_angina,
                    "Oldpeak": oldpeak,
                    "ST_Slope": st_slope
                }])

                encoded = pd.get_dummies(input_df)
                encoded = encoded.reindex(columns=training_columns, fill_value=0)

                # ------------------------------
                # TABULAR ENSEMBLE (PERFORMANCE-WEIGHTED)
                # ------------------------------
                P_tab, tab_details = ensemble_tabular(models, encoded)

                # ------------------------------
                # ECHO MODEL INFERENCE
                # ------------------------------
                uploaded_video.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name

                P_echo = predict_video(video_path, echo_model)

                # ------------------------------
                # LATE FUSION (CLINICALLY WEIGHTED)
                # ------------------------------
                P_multi = late_fusion(P_tab, P_echo, alpha=0.6)
                P_multi_cat = late_fusion(P_tab, P_echo, alpha=0.6, as_category=True)

                st.session_state.multi_result = {
                    "P_tab": P_tab,
                    "P_echo": P_echo,
                    "P_multi": P_multi,
                    "P_multi_cat": P_multi_cat,
                    "encoded": encoded,
                    "video_path": video_path,
                    "input_df": input_df
                }

    if "multi_result" in st.session_state:
        res = st.session_state.multi_result
        P_tab = res["P_tab"]
        P_echo = res["P_echo"]
        P_multi = res["P_multi"]
        P_multi_cat = res["P_multi_cat"]
        encoded = res["encoded"]
        video_path = res["video_path"]
        input_df = res["input_df"]

        # ------------------------------
        # RESULTS
        # ------------------------------
        st.markdown("<div class='section-header'>📊 Multimodal Result</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='pred-card'>
            <div class='model-name'>Tabular Risk Ensemble</div>
            <div class='prob-text'>Probability of Heart Failure: <strong>{P_tab*100:.2f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='pred-card'>
            <div class='model-name'>Echocardiography Model</div>
            <div class='prob-text'>Probability of disease due to Reduced EF: <strong>{P_echo*100:.2f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Map category to styling and human-readable text
        if P_multi_cat == "low":
            risk_class = "risk-low"
            risk_text = "Low Risk of Heart Failure"
        elif P_multi_cat == "medium":
            risk_class = "risk-medium"
            risk_text = "Moderate Risk of Heart Failure"
        else:
            risk_class = "risk-high"
            risk_text = "High Risk of Heart Failure"

        st.markdown(f"""
        <div class='final-risk'>
            <strong>Final Multimodal Probability: {P_multi*100:.2f}%</strong><br>
            <span class='{risk_class}'>{risk_text} ({P_multi_cat.title()})</span><br>
            <small>Fusion Strategy: Late fusion (Echo 60% • Clinical 40%)</small>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🧭 Explainability: SHAP + Grad-CAM"):
            st.subheader("Tabular SHAP Waterfall Plot")
            try:
                with st.spinner("Generating SHAP waterfall plot..."):
                    shap_fig = explain_tabular(models, encoded, nsamples=50)
                    st.pyplot(shap_fig)
            except Exception as e:
                st.error(f"SHAP waterfall plot failed: {e}")

            st.subheader("Tabular SHAP Table")
            try:
                with st.spinner("Generating SHAP explanations..."):
                    shap_table = explain_tabular_table(models, encoded, nsamples=50)
                    st.dataframe(shap_table)
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

            st.subheader("Best Frame with Convergent Heatmap")
            try:
                cam_img = gradcam_for_video(video_path, echo_model)
                st.image(cam_img, use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")


# ===============================
# CHAT ASSISTANT
# ===============================
else:  # Chat Assistant
    st.markdown("<div class='section-header'>💬 Clinical Chat Assistant</div>", unsafe_allow_html=True)
    st.markdown("Ask questions about cardiovascular risk, interpretation of results, or guideline recommendations.")

    # Resolve Gemini API Key behind the scenes (secrets.toml, environment variables, etc.)
    DEFAULT_GEMINI_API_KEY = ""
    api_key = DEFAULT_GEMINI_API_KEY.strip() if DEFAULT_GEMINI_API_KEY.strip() else ""
    
    if not api_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
            
    if not api_key:
        import os
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    user_input = st.chat_input("Type your question here...")

    if user_input:
        response = get_chatbot_response(user_input, api_key=api_key, chat_history=st.session_state.chat_history)
        st.session_state.chat_history.append({"user": user_input, "bot": response})

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"], unsafe_allow_html=True)