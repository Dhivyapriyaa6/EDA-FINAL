import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import warnings
import hashlib
import time

# Import MongoDB functions
from db import (
    create_user, get_user, get_all_users, update_user_login,
    user_exists, email_exists, log_activity, log_search,
    save_forecast, get_user_forecasts, get_user_statistics,
    log_download, get_user_searches, get_user_activities,
    get_ist_time
)

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Advanced Flood Forecasting System",
    page_icon="🌧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
    }
    
    .login-container {
        max-width: 480px;
        margin: 60px auto;
        padding: 60px 50px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        box-shadow: 0 25px 80px rgba(0,0,0,0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .login-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    .floating-element {
        animation: float 3s ease-in-out infinite;
    }
    
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 15px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 35px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    .dashboard-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 30px;
        margin: 15px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 15px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card h4 {
        font-size: 1rem;
        font-weight: 600;
        opacity: 0.9;
        margin: 0;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .header-container h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .welcome-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
        animation: slideInRight 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .welcome-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 2s infinite;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    
    .interactive-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .interactive-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        animation: fadeInUp 0.5s ease-out;
    }
    
    .stError {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        animation: fadeInUp 0.5s ease-out;
    }
    
    .alert-card {
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .alert-high {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.1) 0%, rgba(192, 57, 43, 0.1) 100%);
        border-color: #e74c3c;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.1) 0%, rgba(230, 126, 34, 0.1) 100%);
        border-color: #f39c12;
    }
    
    .alert-low {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(39, 174, 96, 0.1) 100%);
        border-color: #2ecc71;
    }
    
    @media (max-width: 768px) {
        .login-container {
            margin: 20px;
            padding: 40px 30px;
        }
        
        .header-container h1 {
            font-size: 2rem;
        }
        
        .metric-card h2 {
            font-size: 2rem;
        }
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    </style>
""", unsafe_allow_html=True)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False

def auth_page():
    st.markdown("""
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: -1;">
            <div style="position: absolute; width: 4px; height: 4px; background: rgba(255,255,255,0.3); border-radius: 50%; animation: float 6s ease-in-out infinite; top: 20%; left: 10%;"></div>
            <div style="position: absolute; width: 6px; height: 6px; background: rgba(102, 126, 234, 0.4); border-radius: 50%; animation: float 8s ease-in-out infinite; top: 60%; left: 80%;"></div>
            <div style="position: absolute; width: 3px; height: 3px; background: rgba(118, 75, 162, 0.5); border-radius: 50%; animation: float 7s ease-in-out infinite; top: 30%; left: 70%;"></div>
            <div style="position: absolute; width: 5px; height: 5px; background: rgba(240, 147, 251, 0.3); border-radius: 50%; animation: float 9s ease-in-out infinite; top: 80%; left: 20%;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class="floating-element" style="text-align: center; margin-bottom: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); padding: 35px; border-radius: 25px; box-shadow: 0 20px 60px rgba(0,0,0,0.2); position: relative; overflow: hidden;">
                <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); animation: pulse 3s infinite;"></div>
                <h1 style="color: white; margin: 15px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.3); font-size: 2.5rem; font-weight: 800; position: relative; z-index: 1;">
                    🌧 Advanced Flood Forecasting System
                </h1>
                <p style="color: rgba(255,255,255,0.95); font-size: 20px; font-weight: 500; margin: 10px 0; position: relative; z-index: 1;">
                    🚀 Real-Time AI-Powered Rainfall Prediction & Early Warning
                </p>
                <div style="margin-top: 20px; position: relative; z-index: 1;">
                    <span style="display: inline-block; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 5px; font-size: 14px; font-weight: 600;">
                        🤖 Machine Learning
                    </span>
                    <span style="display: inline-block; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 5px; font-size: 14px; font-weight: 600;">
                        📊 Real-time Analytics
                    </span>
                    <span style="display: inline-block; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 5px; font-size: 14px; font-weight: 600;">
                        🌾 AgriTech Solutions
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.show_signup:
            st.markdown("""
                <div style="text-align: center; margin-bottom: 30px;">
                    <h2 style="color: #667eea; font-size: 2rem; font-weight: 700; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        Welcome Back! 👋
                    </h2>
                    <p style="color: #7f8c8d; font-size: 16px; margin: 10px 0; font-weight: 500;">
                        Login to access your personalized dashboard
                    </p>
                    <div style="width: 60px; height: 4px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 15px auto; border-radius: 2px;"></div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                st.markdown('<div style="margin-bottom: 25px;"><label style="display: block; color: #2c3e50; font-weight: 600; margin-bottom: 8px; font-size: 14px;">👤 Username</label>', unsafe_allow_html=True)
                username = st.text_input("", placeholder="Enter your username", label_visibility="collapsed")
                
                st.markdown('</div><div style="margin-bottom: 25px;"><label style="display: block; color: #2c3e50; font-weight: 600; margin-bottom: 8px; font-size: 14px;">🔒 Password</label>', unsafe_allow_html=True)
                password = st.text_input("", type="password", placeholder="Enter your password", label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_btn = st.form_submit_button("🔐 Login", use_container_width=True)
                with col_b:
                    guest_btn = st.form_submit_button("👤 Guest Access", use_container_width=True)
                
                if login_btn:
                    user = get_user(username)
                    if user:
                        if user["password"] == hash_password(password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_name = user["name"]
                            update_user_login(username)
                            log_activity(username, "login")
                            st.success("✅ Login successful!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Incorrect password!")
                    else:
                        st.error("❌ Username not found!")
                
                if guest_btn:
                    st.session_state.logged_in = True
                    st.session_state.username = "guest"
                    st.session_state.user_name = "Guest User"
                    log_activity("guest", "guest_login")
                    st.rerun()
            
            st.markdown("""
                <div style="text-align: center; margin: 40px 0;">
                    <div style="width: 100%; height: 1px; background: linear-gradient(90deg, transparent, #667eea, transparent); margin: 20px 0;"></div>
                    <p style="color: #7f8c8d; font-size: 16px; margin: 15px 0; font-weight: 500;">Don't have an account yet?</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("✨ Create New Account", use_container_width=True):
                st.session_state.show_signup = True
                st.rerun()
        
        else:
            st.markdown("""
                <div style="text-align: center; margin-bottom: 30px;">
                    <h2 style="color: #667eea; font-size: 2rem; font-weight: 700; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        Join Our Community! 🚀
                    </h2>
                    <p style="color: #7f8c8d; font-size: 16px; margin: 10px 0; font-weight: 500;">
                        Create your account to access advanced AI forecasting
                    </p>
                    <div style="width: 60px; height: 4px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 15px auto; border-radius: 2px;"></div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.form("signup_form"):
                st.markdown('<div style="margin-bottom: 20px;"><label style="display: block; color: #2c3e50; font-weight: 600; margin-bottom: 8px; font-size: 14px;">👤 Full Name</label>', unsafe_allow_html=True)
                new_name = st.text_input("", placeholder="Enter your full name", label_visibility="collapsed", key="signup_name")
                
                st.markdown('</div><div style="margin-bottom: 20px;"><label style="display: block; color: #2c3e50; font-weight: 600; margin-bottom: 8px; font-size: 14px;">📧 Email Address</label>', unsafe_allow_html=True)
                new_email = st.text_input("", placeholder="Enter your email", label_visibility="collapsed", key="signup_email")
                
                st.markdown('</div><div style="margin-bottom: 20px;"><label style="display: block; color: #2c3e50; font-weight: 600; margin-bottom: 8px; font-size: 14px;">🔑 Username</label>', unsafe_allow_html=True)
                new_username = st.text_input("", placeholder="Choose a unique username", label_visibility="collapsed", key="signup_username")
                
                st.markdown('</div><div style="margin-bottom: 20px;"><label style="display: block; color: #2c3e50; font-weight: 600; margin-bottom: 8px; font-size: 14px;">🔒 Password</label>', unsafe_allow_html=True)
                new_password = st.text_input("", type="password", placeholder="Create a strong password", label_visibility="collapsed", key="signup_password")
                
                st.markdown('</div><div style="margin-bottom: 25px;"><label style="display: block; color: #2c3e50; font-weight: 600; margin-bottom: 8px; font-size: 14px;">🔒 Confirm Password</label>', unsafe_allow_html=True)
                confirm_password = st.text_input("", type="password", placeholder="Re-enter your password", label_visibility="collapsed", key="signup_confirm")
                st.markdown("</div>", unsafe_allow_html=True)
                
                signup_btn = st.form_submit_button("✨ Create My Account", use_container_width=True)
                
                if signup_btn:
                    if not all([new_name, new_email, new_username, new_password, confirm_password]):
                        st.error("⚠️ Please fill all fields!")
                    elif new_password != confirm_password:
                        st.error("⚠️ Passwords do not match!")
                    elif len(new_password) < 6:
                        st.error("⚠️ Password must be at least 6 characters!")
                    elif user_exists(new_username):
                        st.error("⚠️ Username already exists!")
                    elif email_exists(new_email):
                        st.error("⚠️ Email already registered!")
                    else:
                        user_id = create_user(new_username, hash_password(new_password), new_email, new_name)
                        if user_id:
                            st.success("✅ Account created successfully! Please login.")
                            log_activity(new_username, "account_created")
                            st.session_state.show_signup = False
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Error creating account. Please try again.")
            
            st.markdown("""
                <div style="text-align: center; margin: 40px 0;">
                    <div style="width: 100%; height: 1px; background: linear-gradient(90deg, transparent, #667eea, transparent); margin: 20px 0;"></div>
                    <p style="color: #7f8c8d; font-size: 16px; margin: 15px 0; font-weight: 500;">Already have an account?</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔐 Back to Login", use_container_width=True):
                st.session_state.show_signup = False
                st.rerun()

@st.cache_data
def load_data():
    try:
        rainfall_df = pd.read_csv("data/rainfall in india 1901-2015.csv")
        district_df = pd.read_csv("data/district wise rainfall normal.csv")
        flood_df = pd.read_csv("data/flood_risk_dataset_india.csv")
        crops_df = pd.read_csv("data/Custom_Crops_yield_Historical_Dataset.csv")
        
        rainfall_df.columns = rainfall_df.columns.str.lower().str.strip()
        district_df.columns = district_df.columns.str.upper().str.strip()
        flood_df.columns = flood_df.columns.str.lower().str.strip()
        crops_df.columns = crops_df.columns.str.lower().str.strip()
        
        return rainfall_df, district_df, flood_df, crops_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def get_top_5_crops(state, district, crops_df):
    try:
        district_crops = crops_df[
            (crops_df['state name'].str.lower() == state.lower()) & 
            (crops_df['dist name'].str.lower() == district.lower())
        ]
        if not district_crops.empty:
            top_crops = district_crops.nlargest(5, 'yield_kg_per_ha')[['crop', 'yield_kg_per_ha', 'area_ha']].drop_duplicates('crop')
            return top_crops.reset_index(drop=True)
    except Exception as e:
        st.warning(f"Error: {e}")
    return None

def get_crop_details(crop_name, crops_df, state, district):
    try:
        crop_data = crops_df[
            (crops_df['crop'].str.lower() == crop_name.lower()) &
            (crops_df['state name'].str.lower() == state.lower()) &
            (crops_df['dist name'].str.lower() == district.lower())
        ]
        if not crop_data.empty:
            return {
                'avg_yield': crop_data['yield_kg_per_ha'].mean(),
                'n_req': crop_data['n_req_kg_per_ha'].mean(),
                'p_req': crop_data['p_req_kg_per_ha'].mean(),
                'k_req': crop_data['k_req_kg_per_ha'].mean(),
                'temp': crop_data['temperature_c'].mean(),
                'humidity': crop_data['humidity_%'].mean(),
                'rainfall': crop_data['rainfall_mm'].mean(),
                'ph': crop_data['ph'].mean(),
                'area': crop_data['area_ha'].mean()
            }
    except Exception as e:
        st.warning(f"Error: {e}")
    return None

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_lstm_data(data, lookback=12):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

@st.cache_resource
def train_lstm_model(district_data, epochs=100):
    try:
        monthly_cols = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        monthly_data = district_data[monthly_cols].values.flatten().astype(float)
        extended_data = np.tile(monthly_data, 25)
        
        lookback = 12
        X, y, scaler = prepare_lstm_data(extended_data, lookback)
        
        if len(X) == 0:
            return None, None, None
        
        model = create_lstm_model((lookback, 1))
        model.fit(X, y, epochs=epochs, batch_size=16, verbose=0, validation_split=0.2)
        
        return model, scaler, extended_data
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

def forecast_rainfall_lstm(model, scaler, last_sequence, months_ahead=12):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(months_ahead):
        X_input = current_sequence[-12:].reshape(1, 12, 1)
        y_pred = model.predict(X_input, verbose=0)
        predictions.append(y_pred[0, 0])
        current_sequence = np.append(current_sequence, y_pred[0, 0])
    
    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return forecast

def generate_7day_forecast(annual_rainfall):
    base_daily = annual_rainfall / 365
    daily_forecast = []
    
    for day in range(7):
        daily_val = base_daily * np.random.uniform(0.5, 1.5)
        daily_forecast.append(max(0, daily_val))
    
    return daily_forecast

def calculate_flood_risk(rainfall, threshold=200):
    if rainfall > threshold * 1.5:
        return "Very High", "#e74c3c", "🔴", "alert-high"
    elif rainfall > threshold * 1.2:
        return "High", "#f39c12", "🟠", "alert-medium"
    elif rainfall > threshold * 0.8:
        return "Medium", "#f1c40f", "🟡", "alert-medium"
    else:
        return "Low", "#2ecc71", "🟢", "alert-low"

def dashboard_page():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"""
            <div class='header-container'>
                <div style="position: relative; z-index: 2;">
                    <h1 style='color: white; margin: 0; font-size: 2.8rem; font-weight: 800; text-shadow: 0 4px 8px rgba(0,0,0,0.3);'>
                        🌧 Advanced Flood Forecasting & Climate Intelligence
                    </h1>
                    <p style='color: rgba(255,255,255,0.95); font-size: 18px; margin: 15px 0 0 0; font-weight: 500;'>
                        🚀 Real-Time AI-Powered Rainfall Prediction & Early Warning System
                    </p>
                    <div style="margin-top: 20px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
                        <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 600;">
                            🤖 LSTM Neural Networks
                        </span>
                        <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 600;">
                            📊 Real-time Analytics
                        </span>
                        <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 600;">
                            🌾 AgriTech Solutions
                        </span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.write("")
        st.write("")
        st.write("")
        if st.button("🚪 Logout", use_container_width=True):
            log_activity(st.session_state.username, "logout")
            st.session_state.logged_in = False
            st.rerun()
    
    user_stats = get_user_statistics(st.session_state.username)
    
    ist_now = get_ist_time()

    st.markdown(f"""
        <div class='welcome-banner'>
            <div style="position: relative; z-index: 2;">
                <h3 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 700;'>
                    👋 Welcome back, {st.session_state.user_name}!
                </h3>
                <p style='color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 16px; font-weight: 500;'>
                    📅 {ist_now.strftime("%A, %B %d, %Y")} | 🕐 {ist_now.strftime("%I:%M %p IST")}
                </p>
                <div style="margin-top: 15px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                    <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; font-size: 12px; font-weight: 600;">
                        🔍 {user_stats['total_searches'] if user_stats else 0} Searches
                    </span>
                    <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; font-size: 12px; font-weight: 600;">
                        📈 {user_stats['total_forecasts'] if user_stats else 0} Forecasts
                    </span>
                    <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; font-size: 12px; font-weight: 600;">
                        🔐 Login #{user_stats['login_count'] if user_stats else 0}
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    rainfall_df, district_df, flood_df, crops_df = load_data()
    
    if any(x is None for x in [rainfall_df, district_df, flood_df, crops_df]):
        st.error("Please ensure all CSV files are in the 'data' folder")
        return
    
    with st.sidebar:
        st.markdown("### 🎛 Control Panel")
        st.markdown("---")
        
        st.markdown("#### 📍 Location Selection")
        available_states = sorted(district_df["STATE_UT_NAME"].unique())
        selected_state = st.selectbox("State", available_states, index=0)
        
        state_districts = district_df[district_df["STATE_UT_NAME"] == selected_state]["DISTRICT"].unique()
        selected_district = st.selectbox("District", sorted(state_districts), index=0)
        
        if 'last_search' not in st.session_state or st.session_state.last_search != (selected_state, selected_district):
            log_search(st.session_state.username, selected_state, selected_district)
            st.session_state.last_search = (selected_state, selected_district)
        
        st.markdown("---")
        st.markdown("#### ⚙ Forecast Configuration")
        forecast_year = st.number_input("Target Year", min_value=2025, max_value=2050, value=2027, step=1)
        forecast_months = st.slider("Forecast Duration (months)", min_value=6, max_value=24, value=12, step=6)
        lstm_epochs = st.slider("Training Epochs", min_value=50, max_value=200, value=100, step=50)
        
        st.markdown("---")
        if st.button("🚀 Generate AI Forecast", type="primary", use_container_width=True):
            st.session_state.forecast_ready = True
            st.session_state.current_district = selected_district
            st.session_state.current_state = selected_state
            st.session_state.forecast_year = forecast_year
            st.session_state.forecast_months = forecast_months
            st.session_state.lstm_epochs = lstm_epochs
        
        st.markdown("---")
        st.markdown("#### 📊 Quick Stats")
        st.metric("Total States", len(available_states))
        st.metric("Total Districts", len(district_df))
        if user_stats:
            st.metric("Your Searches", user_stats['total_searches'])
            st.metric("Your Forecasts", user_stats['total_forecasts'])
    
    district_info = district_df[
        (district_df["STATE_UT_NAME"] == selected_state) & 
        (district_df["DISTRICT"] == selected_district)
    ].iloc[0]
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Real-Time Dashboard",
        "🌤 7-Day Forecast",
        "🤖 AI Predictions",
        "🌾 AgriTech & Water Management"
    ])
    
    with tab1:
        st.markdown(f"### 📍 Location: {selected_district}, {selected_state}")
        
        annual = float(district_info['ANNUAL'])
        monsoon = float(district_info[['JUN', 'JUL', 'AUG', 'SEP']].sum())
        winter = float(district_info[['JAN', 'FEB']].sum())
        summer = float(district_info[['MAR', 'APR', 'MAY']].sum())
        post_monsoon = float(district_info[['OCT', 'NOV', 'DEC']].sum())
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
                <div class='metric-card interactive-card'>
                    <div style="position: relative; z-index: 2;">
                        <h4 style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>🌧 Annual Rainfall</h4>
                        <h2 style='margin: 15px 0; font-size: 2.2rem; font-weight: 800;'>{annual:.1f}</h2>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.8rem; font-weight: 600;'>mm/year</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='metric-card interactive-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                    <div style="position: relative; z-index: 2;">
                        <h4 style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>🌦 Monsoon</h4>
                        <h2 style='margin: 15px 0; font-size: 2.2rem; font-weight: 800;'>{monsoon:.1f}</h2>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.8rem; font-weight: 600;'>mm</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='metric-card interactive-card' style='background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);'>
                    <div style="position: relative; z-index: 2;">
                        <h4 style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>❄ Winter</h4>
                        <h2 style='margin: 15px 0; font-size: 2.2rem; font-weight: 800;'>{winter:.1f}</h2>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.8rem; font-weight: 600;'>mm</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class='metric-card interactive-card' style='background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);'>
                    <div style="position: relative; z-index: 2;">
                        <h4 style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>☀ Summer</h4>
                        <h2 style='margin: 15px 0; font-size: 2.2rem; font-weight: 800;'>{summer:.1f}</h2>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.8rem; font-weight: 600;'>mm</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            risk, color, icon, _ = calculate_flood_risk(annual)
            st.markdown(f"""
                <div class='metric-card interactive-card' style='background: {color};'>
                    <div style="position: relative; z-index: 2;">
                        <h4 style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>⚠ Risk Level</h4>
                        <h2 style='margin: 15px 0; font-size: 2.2rem; font-weight: 800;'>{icon} {risk}</h2>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.8rem; font-weight: 600;'>Status</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Monthly Rainfall Pattern")
            months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
            monthly_values = [float(district_info[m]) for m in months]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=months, y=monthly_values,
                marker=dict(color=monthly_values, colorscale='Viridis', showscale=True),
                text=[f"{val:.1f}" for val in monthly_values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Rainfall: %{y:.1f} mm<extra></extra>'
            ))
            fig.update_layout(xaxis_title="Month", yaxis_title="Rainfall (mm)", height=500, template='plotly_white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Seasonal Distribution")
            seasonal_data = {
                'Winter': winter, 'Pre-Monsoon': summer,
                'Monsoon': monsoon, 'Post-Monsoon': post_monsoon
            }
            fig = go.Figure(data=[go.Pie(
                labels=list(seasonal_data.keys()), values=list(seasonal_data.values()), hole=0.5,
                marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']),
                textinfo='label+percent', textposition='outside',
                hovertemplate='<b>%{label}</b><br>Rainfall: %{value:.1f} mm<extra></extra>'
            )])
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Rainfall Distribution by Month")
            try:
                months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
                monthly_values = [float(district_info[m]) for m in months]
                
                fig = go.Figure(data=[go.Box(y=monthly_values, name='Monthly Distribution',
                    marker=dict(color='#667eea'), boxmean='sd')])
                fig.update_layout(yaxis_title="Rainfall (mm)", height=500, template='plotly_white',
                    showlegend=False, xaxis_title="Distribution Statistics")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Box plot error: {e}")
        
        with col2:
            st.markdown("##### Comparative Risk Analysis")
            try:
                # Calculate averages for different metrics
                months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
                
                # Current district metrics
                current_annual = float(district_info['ANNUAL'])
                current_monsoon = float(district_info[['JUN', 'JUL', 'AUG', 'SEP']].sum())
                current_winter = float(district_info[['JAN', 'FEB']].sum())
                current_summer = float(district_info[['MAR', 'APR', 'MAY']].sum())
                current_post_monsoon = float(district_info[['OCT', 'NOV', 'DEC']].sum())
                
                # State averages
                state_districts = district_df[district_df["STATE_UT_NAME"] == selected_state]
                state_annual = state_districts['ANNUAL'].mean()
                state_monsoon = state_districts[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1).mean()
                state_winter = state_districts[['JAN', 'FEB']].sum(axis=1).mean()
                state_summer = state_districts[['MAR', 'APR', 'MAY']].sum(axis=1).mean()
                state_post_monsoon = state_districts[['OCT', 'NOV', 'DEC']].sum(axis=1).mean()
                
                # National averages
                national_annual = district_df['ANNUAL'].mean()
                national_monsoon = district_df[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1).mean()
                national_winter = district_df[['JAN', 'FEB']].sum(axis=1).mean()
                national_summer = district_df[['MAR', 'APR', 'MAY']].sum(axis=1).mean()
                national_post_monsoon = district_df[['OCT', 'NOV', 'DEC']].sum(axis=1).mean()
                
                # Create radar chart
                categories = ['Annual', 'Monsoon', 'Winter', 'Summer', 'Post-Monsoon']
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=[current_annual, current_monsoon, current_winter, current_summer, current_post_monsoon],
                    theta=categories,
                    fill='toself',
                    name='Current District',
                    line=dict(color="#f80915", width=2),
                    fillcolor='rgba(102, 126, 234, 0.3)'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=[state_annual, state_monsoon, state_winter, state_summer, state_post_monsoon],
                    theta=categories,
                    fill='toself',
                    name='State Average',
                    line=dict(color='#f39c12', width=2),
                    fillcolor='rgba(243, 156, 18, 0.2)'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=[national_annual, national_monsoon, national_winter, national_summer, national_post_monsoon],
                    theta=categories,
                    fill='toself',
                    name='National Average',
                    line=dict(color='#2ecc71', width=2),
                    fillcolor='rgba(46, 204, 113, 0.2)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(current_annual, state_annual, national_annual) * 1.1])
                    ),
                    showlegend=True,
                    height=500,
                    template='plotly_white',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Comparison chart error: {e}")
        
        st.markdown("---")
        st.markdown("#### Flood Risk Zones & District Metrics")
        
        try:
            if not flood_df.empty and 'latitude' in flood_df.columns and 'longitude' in flood_df.columns:
                flood_df_copy = flood_df.copy()
                flood_df_copy['risk_score'] = (flood_df_copy['rainfall_mm'] / flood_df_copy['rainfall_mm'].max() * 100) if 'rainfall_mm' in flood_df.columns else 50
                fig = px.scatter_mapbox(flood_df_copy.head(500), lat='latitude', lon='longitude', color='risk_score', size='risk_score',
                    color_continuous_scale='RdYlGn_r', zoom=4, height=1000, title="Flood Risk Heat Map - India")
                fig.update_layout(mapbox_style="open-street-map", template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Map error: {e}")
    
    with tab2:
        st.markdown("### 🌤 7-Day Real-Time Forecast & Alert System")
        
        daily_forecast = generate_7day_forecast(annual)
        dates = [(datetime.now() + timedelta(days=i)).strftime("%a, %b %d") for i in range(7)]
        
        st.markdown("#### 📅 Daily Rainfall Forecast (Next 7 Days)")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        for idx, (col, date, rainfall) in enumerate(zip([col1, col2, col3, col4, col5, col6, col7], dates, daily_forecast)):
            risk, color, icon, _ = calculate_flood_risk(rainfall)
            with col:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {color} 0%, rgba(0,0,0,0.1) 100%); 
                    padding: 15px; border-radius: 12px; text-align: center; color: white;'>
                        <p style='margin: 0; font-size: 0.85rem; font-weight: 600;'>{date}</p>
                        <h3 style='margin: 10px 0;'>{icon}</h3>
                        <p style='margin: 5px 0; font-size: 1.1rem; font-weight: 700;'>{rainfall:.1f} mm</p>
                        <p style='margin: 0; font-size: 0.8rem;'>{risk}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=daily_forecast, mode='lines+markers',
            name='Daily Forecast', line=dict(color='#667eea', width=3),
            marker=dict(size=12, color=daily_forecast, colorscale='RdYlGn_r', showscale=True),
            fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)'))
        fig.update_layout(xaxis_title="Date", yaxis_title="Rainfall (mm)", height=400, template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### ⚠ Active Alerts & Warnings")
        
        high_risk_days = sum(1 for r in daily_forecast if r > 200)
        medium_risk_days = sum(1 for r in daily_forecast if 100 < r <= 200)
        low_risk_days = sum(1 for r in daily_forecast if r <= 100)
        
        if high_risk_days > 0:
            st.markdown(f"""
                <div class='alert-card alert-high'>
                    <h4 style='margin: 0; color: #e74c3c;'>🔴 HIGH FLOOD ALERT</h4>
                    <p style='margin: 10px 0 0 0; color: #c0392b;'><strong>{high_risk_days} days</strong> with rainfall >200mm expected</p>
                    <p style='margin: 5px 0 0 0; font-size: 0.9rem; color: #555;'>Immediate action required. Prepare evacuation routes and emergency supplies.</p>
                </div>
            """, unsafe_allow_html=True)
        
        if medium_risk_days > 0:
            st.markdown(f"""
                <div class='alert-card alert-medium'>
                    <h4 style='margin: 0; color: #f39c12;'>🟠 MODERATE FLOOD WARNING</h4>
                    <p style='margin: 10px 0 0 0; color: #d68910;'><strong>{medium_risk_days} days</strong> with rainfall 100-200mm expected</p>
                    <p style='margin: 5px 0 0 0; font-size: 0.9rem; color: #555;'>Monitor water levels and stay alert. Activate emergency protocols if needed.</p>
                </div>
            """, unsafe_allow_html=True)
        
        if low_risk_days >= 5:
            st.markdown(f"""
                <div class='alert-card alert-low'>
                    <h4 style='margin: 0; color: #2ecc71;'>🟢 LOW RISK CONDITIONS</h4>
                    <p style='margin: 10px 0 0 0; color: #27ae60;'><strong>{low_risk_days} days</strong> with minimal rainfall expected</p>
                    <p style='margin: 5px 0 0 0; font-size: 0.9rem; color: #555;'>Conditions favorable for outdoor activities and agricultural operations.</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Advisory & Action Items")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Recommended Actions**")
            if high_risk_days > 0:
                st.warning("⚠️ Issue evacuation alerts to vulnerable areas")
                st.warning("⚠️ Deploy rescue and relief teams")
                st.warning("⚠️ Activate emergency operation centers")
            elif medium_risk_days > 0:
                st.info("ℹ️ Increase monitoring and surveillance")
                st.info("ℹ️ Position emergency resources on standby")
                st.info("ℹ️ Brief all stakeholders on situation")
            else:
                st.success("✅ Continue routine monitoring")
                st.success("✅ Maintain preparedness levels")
        
        with col2:
            st.markdown("**📊 Forecast Summary**")
            st.metric("Peak Daily Rainfall", f"{max(daily_forecast):.1f} mm", "Next 7 days")
            st.metric("Average Daily Rainfall", f"{np.mean(daily_forecast):.1f} mm", "Next 7 days")
            st.metric("Total Expected", f"{sum(daily_forecast):.1f} mm", "Next 7 days")
    
    with tab3:
        st.markdown("### AI-Powered LSTM Forecasting Engine")
        
        if 'forecast_ready' in st.session_state and st.session_state.forecast_ready:
            try:
                district_data = district_df[
                    (district_df["STATE_UT_NAME"] == st.session_state.current_state) & 
                    (district_df["DISTRICT"] == st.session_state.current_district)
                ]
                
                if not district_data.empty:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Initializing LSTM neural network...")
                    progress_bar.progress(20)
                    
                    model, scaler, extended_data = train_lstm_model(
                        district_data.iloc[0],
                        st.session_state.lstm_epochs
                    )
                    
                    progress_bar.progress(70)
                    status_text.text("📈 Generating long-term predictions...")
                    
                    if model is not None:
                        scaled_data = scaler.transform(extended_data.reshape(-1, 1))
                        forecast = forecast_rainfall_lstm(
                            model, scaler, scaled_data.flatten(), st.session_state.forecast_months
                        )
                        
                        forecast_id = save_forecast(
                            st.session_state.username,
                            st.session_state.current_state,
                            st.session_state.current_district,
                            forecast.tolist(),
                            st.session_state.forecast_year,
                            st.session_state.forecast_months
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Forecast complete!")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"✅ Successfully generated {st.session_state.forecast_months}-month forecast")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 📈 Long-Term Rainfall Forecast")
                        
                        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        forecast_labels = (month_names * ((st.session_state.forecast_months // 12) + 1))[:st.session_state.forecast_months]
                        
                        months_cols = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
                        historical = district_data.iloc[0][months_cols].values.astype(float)
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=month_names, y=historical, mode='lines+markers', name='Historical Average',
                            line=dict(color='#667eea', width=3), marker=dict(size=10, symbol='circle'),
                            fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_labels, y=forecast, mode='lines+markers',
                            name=f'AI Forecast {st.session_state.forecast_year}',
                            line=dict(color='#e74c3c', width=3, dash='dash'),
                            marker=dict(size=10, symbol='diamond')
                        ))
                        
                        upper_bound = forecast * 1.15
                        lower_bound = forecast * 0.85
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_labels + forecast_labels[::-1],
                            y=np.concatenate([upper_bound, lower_bound[::-1]]),
                            fill='toself', fillcolor='rgba(231, 76, 60, 0.15)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval (±15%)', showlegend=True
                        ))
                        
                        fig.update_layout(
                            xaxis_title="Month", yaxis_title="Rainfall (mm)", height=500,
                            template='plotly_white', hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 12px; color: white; text-align: center;'>
                                    <h4 style='margin: 0; opacity: 0.9;'>Total Forecast</h4>
                                    <h2 style='margin: 10px 0;'>{forecast.max():.1f}</h2>
                                    <p style='margin: 0; opacity: 0.8;'>mm</p>
                                </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                padding: 20px; border-radius: 12px; color: white; text-align: center;'>
                                    <h4 style='margin: 0; opacity: 0.9;'>Avg Monthly</h4>
                                    <h2 style='margin: 10px 0;'>{forecast.mean():.1f}</h2>
                                    <p style='margin: 0; opacity: 0.8;'>mm</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); 
                                padding: 20px; border-radius: 12px; color: white; text-align: center;'>
                                    <h4 style='margin: 0; opacity: 0.9;'>Peak Month</h4>
                                    <h2 style='margin: 10px 0;'>{forecast.max():.1f}</h2>
                                    <p style='margin: 0; opacity: 0.8;'>mm</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            high_risk = np.sum(forecast > 200)
                            risk_color = "#e74c3c" if high_risk > 3 else "#2ecc71"
                            st.markdown(f"""
                                <div style='background: {risk_color}; padding: 20px; border-radius: 12px; 
                                color: white; text-align: center;'>
                                    <h4 style='margin: 0; opacity: 0.9;'>High Risk Months</h4>
                                    <h2 style='margin: 10px 0;'>{high_risk}</h2>
                                    <p style='margin: 0; opacity: 0.8;'>months</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        forecast_df = pd.DataFrame({
                            'Month': forecast_labels,
                            'Predicted Rainfall (mm)': np.round(forecast, 2),
                            'Risk Level': [calculate_flood_risk(val)[0] for val in forecast],
                            'Status': [calculate_flood_risk(val)[2] for val in forecast]
                        })
                        
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown("#### Monthly Forecast Table")
                            forecast_df = pd.DataFrame({
                                'Month': forecast_labels,
                                'Predicted Rainfall (mm)': np.round(forecast, 2),
                                'Risk Level': [calculate_flood_risk(val)[0] for val in forecast],
                                'Status': [calculate_flood_risk(val)[2] for val in forecast]
                            })
                            
                            st.dataframe(
                                forecast_df.style.background_gradient(subset=['Predicted Rainfall (mm)'], cmap='RdYlGn_r'),
                                use_container_width=True, height=400
                            )
                        
                        with col2:
                            st.markdown("#### Risk Distribution")
                            high_risk_months = np.sum(forecast > 200)
                            medium_risk_months = np.sum((forecast > 100) & (forecast <= 200))
                            low_risk_months = np.sum(forecast <= 100)
                            total_months = len(forecast)
                            
                            risk_data = {
                                'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
                                'Count': [high_risk_months, medium_risk_months, low_risk_months],
                                'Percentage': [
                                    f"{(high_risk_months/total_months*100):.1f}%",
                                    f"{(medium_risk_months/total_months*100):.1f}%",
                                    f"{(low_risk_months/total_months*100):.1f}%"
                                ]
                            }
                            
                            risk_df = pd.DataFrame(risk_data)
                            st.dataframe(risk_df, use_container_width=True, hide_index=True)
                            
                            if high_risk_months > 3:
                                st.error("🔴 HIGH FLOOD RISK DETECTED!")
                            elif high_risk_months > 0:
                                st.warning("🟠 MODERATE FLOOD RISK")
                            else:
                                st.success("🟢 LOW FLOOD RISK")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                            
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            csv = forecast_df.to_csv(index=False)
                            if st.download_button("📥 Download CSV", csv, 
                                f"forecast_{st.session_state.current_district}.csv", 
                                "text/csv", use_container_width=True):
                                log_download(st.session_state.username, "csv", 
                                    st.session_state.current_state, st.session_state.current_district, str(forecast_id))
                        
                        with col2:
                            json_data = forecast_df.to_json(orient='records', indent=2)
                            if st.download_button("📥 Download JSON", json_data,
                                f"forecast_{st.session_state.current_district}.json",
                                "application/json", use_container_width=True):
                                log_download(st.session_state.username, "json",
                                    st.session_state.current_state, st.session_state.current_district, str(forecast_id))
                        
                        with col3:
                            report = f"""RAINFALL FORECAST REPORT
========================
Location: {st.session_state.current_district}, {st.session_state.current_state}
Forecast Year: {st.session_state.forecast_year}
Duration: {st.session_state.forecast_months} months
Generated: {get_ist_time().strftime("%Y-%m-%d %H:%M:%S IST")}
User: {st.session_state.username}
Forecast ID: {forecast_id}

SUMMARY STATISTICS
------------------
Total Rainfall: {forecast.sum():.2f} mm
Average Monthly: {forecast.mean():.2f} mm
Peak Month: {forecast.max():.2f} mm
Minimum Month: {forecast.min():.2f} mm

RISK ASSESSMENT
---------------
High Risk Months: {high_risk_months}
Medium Risk Months: {medium_risk_months}
Low Risk Months: {low_risk_months}

MONTHLY FORECAST
----------------
{forecast_df.to_string(index=False)}"""
                            if st.download_button("📥 Download Report", report,
                                f"report_{st.session_state.current_district}.txt",
                                "text/plain", use_container_width=True):
                                log_download(st.session_state.username, "report",
                                    st.session_state.current_state, st.session_state.current_district, str(forecast_id))
                    else:
                        st.error("❌ Failed to train LSTM model")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.info("📌 Click 'Generate AI Forecast' in sidebar to generate long-term predictions")
            
            st.markdown("#### 📂 Your Previous Forecasts")
            user_forecasts = get_user_forecasts(st.session_state.username, limit=5)
            
            if user_forecasts:
                for fc in user_forecasts:
                    with st.expander(f"📊 {fc['district']}, {fc['state']} - {fc['created_at'].strftime('%Y-%m-%d %H:%M')}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Forecast Year", fc['forecast_year'])
                        with col2:
                            st.metric("Duration", f"{fc['forecast_months']} months")
                        with col3:
                            st.metric("Total Rainfall", f"{fc['metadata']['total_rainfall']:.1f} mm")
            else:
                st.info("No previous forecasts found. Generate your first forecast!")
    
    with tab4:
        st.markdown("### 🌾 AgriTech Intelligence & Water Management")
        
        top_crops = get_top_5_crops(selected_state, selected_district, crops_df)
        annual_rainfall = float(district_info["ANNUAL"])
        
        if top_crops is not None and not top_crops.empty:
            col_left, col_right = st.columns([2, 1], gap="large")
            
            with col_left:
                if annual_rainfall > 2000:
                    zone_label = "High Rainfall Zone (>2000mm)"
                    zone_color = "#f39c12"
                    zone_emoji = "🌊"
                    crops_list = {
                        "Rice": "Water-intensive, ideal for high rainfall",
                        "Jute": "Thrives in humid conditions",
                        "Tea": "Requires abundant rainfall",
                        "Coconut": "Tropical climate suitable",
                        "Rubber": "High rainfall zones"
                    }
                elif annual_rainfall > 1000:
                    zone_label = "Moderate Rainfall Zone (1000-2000mm)"
                    zone_color = "#f39c12"
                    zone_emoji = "🌾"
                    crops_list = {}
                    for idx, row in top_crops.iterrows():
                        crop_name = row['crop']
                        details = get_crop_details(crop_name, crops_df, selected_state, selected_district)
                        if details and idx < 5:
                            crops_list[crop_name.upper()] = f"Yield: {details['avg_yield']:.0f} kg/ha"
                else:
                    zone_label = "Low Rainfall Zone (<1000mm)"
                    zone_color = "#e74c3c"
                    zone_emoji = "☀"
                    crops_list = {
                        "Bajra": "Drought-resistant cereal",
                        "Jowar": "Low water requirement",
                        "Pulses": "Suitable for arid zones",
                        "Groundnut": "Sandy soil adaptation",
                        "Castor": "Semi-arid regions"
                    }
                
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {zone_color} 0%, 
                    {"#e67e22" if zone_color=="#f39c12" else "#c0392b" if zone_color=="#e74c3c" else "#d4af37"} 100%); 
                    color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px; 
                    text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.2);'>
                        <h3 style='margin: 0; font-size: 1.3rem;'>{zone_emoji} {zone_label}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 🌱 Recommended Crops")
                for idx, (crop, desc) in enumerate(crops_list.items(), 1):
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                        padding: 15px; border-radius: 12px; margin-bottom: 12px; border-left: 4px solid {zone_color}; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);'>
                            <h4 style='margin: 0; color: {zone_color};'>#{idx} {crop}</h4>
                            <p style='margin: 5px 0 0 0; color: #666; font-size: 0.9rem;'>{desc}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🔍 Detailed Crop Analysis")
                
                for idx, row in top_crops.iterrows():
                    crop_name = row['crop']
                    details = get_crop_details(crop_name, crops_df, selected_state, selected_district)
                    
                    if details:
                        with st.expander(f"🌾 {crop_name.upper()} - Yield: {details['avg_yield']:.0f} kg/ha", expanded=(idx==0)):
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 15px; border-radius: 10px; color: white; text-align: center;'>
                                        <h4 style='margin: 0; opacity: 0.9;'>Nitrogen</h4>
                                        <h2 style='margin: 10px 0;'>{details['n_req']:.0f}</h2>
                                        <p style='margin: 0;'>kg/ha</p>
                                    </div>
                                    <div style='background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                                    padding: 15px; border-radius: 10px; color: white; text-align: center; margin-top: 10px;'>
                                        <h4 style='margin: 0; opacity: 0.9;'>Temperature</h4>
                                        <h2 style='margin: 10px 0;'>{details['temp']:.1f}</h2>
                                        <p style='margin: 0;'>°C</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col_b:
                                st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                    padding: 15px; border-radius: 10px; color: white; text-align: center;'>
                                        <h4 style='margin: 0; opacity: 0.9;'>Phosphorus</h4>
                                        <h2 style='margin: 10px 0;'>{details['p_req']:.0f}</h2>
                                        <p style='margin: 0;'>kg/ha</p>
                                    </div>
                                    <div style='background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); 
                                    padding: 15px; border-radius: 10px; color: white; text-align: center; margin-top: 10px;'>
                                        <h4 style='margin: 0; opacity: 0.9;'>Humidity</h4>
                                        <h2 style='margin: 10px 0;'>{details['humidity']:.0f}</h2>
                                        <p style='margin: 0;'>%</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col_c:
                                st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                                    padding: 15px; border-radius: 10px; color: white; text-align: center;'>
                                        <h4 style='margin: 0; opacity: 0.9;'>Potassium</h4>
                                        <h2 style='margin: 10px 0;'>{details['k_req']:.0f}</h2>
                                        <p style='margin: 0;'>kg/ha</p>
                                    </div>
                                    <div style='background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); 
                                    padding: 15px; border-radius: 10px; color: white; text-align: center; margin-top: 10px;'>
                                        <h4 style='margin: 0; opacity: 0.9;'>pH Level</h4>
                                        <h2 style='margin: 10px 0;'>{details['ph']:.2f}</h2>
                                        <p style='margin: 0;'>Optimal</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%); 
                                padding: 15px; border-radius: 10px; color: white; text-align: center; margin-top: 10px;'>
                                    <h4 style='margin: 0; opacity: 0.9;'>Ideal Rainfall</h4>
                                    <h2 style='margin: 10px 0;'>{details['rainfall']:.0f}</h2>
                                    <p style='margin: 0;'>mm/year</p>
                                </div>
                            """, unsafe_allow_html=True)
            
            with col_right:
                risk_level, risk_color, risk_icon, _ = calculate_flood_risk(annual_rainfall)
                
                st.markdown(f"""
                    <div style='background: {risk_color}; color: white; padding: 20px; border-radius: 15px; 
                    text-align: center; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.2);'>
                        <h2 style='margin: 0; font-size: 2.5rem;'>{risk_icon}</h2>
                        <h3 style='margin: 10px 0 0 0;'>{risk_level} Risk</h3>
                        <p style='margin: 5px 0 0 0;'>Current Status</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 💧 Water Management")
                
                if annual_rainfall > 1500:
                    strategies = [
                        ("Drainage Systems", "Install subsurface drainage", "#e74c3c"),
                        ("Raised Beds", "Elevate fields 15-30cm", "#e67e22"),
                        ("Flood-Resistant", "Use tolerant varieties", "#f39c12"),
                        ("Warning System", "IoT-based monitoring", "#c0392b")
                    ]
                elif annual_rainfall > 800:
                    strategies = [
                        ("Drip Irrigation", "30-60% water saving", "#11998e"),
                        ("Rainwater Harvest", "Build farm ponds", "#3498db"),
                        ("Mulching", "50-70% evaporation reduction", "#2ecc71"),
                        ("Scheduling", "Weather-based timing", "#1abc9c")
                    ]
                else:
                    strategies = [
                        ("Well Recharge", "6m depth wells", "#9b59b6"),
                        ("Dry Farming", "Mulching essential", "#e74c3c"),
                        ("Early Sowing", "Catch monsoon rains", "#f39c12"),
                        ("Crop Selection", "Drought varieties", "#27ae60")
                    ]
                
                for name, desc, color in strategies:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {color} 0%, rgba(0,0,0,0.1) 100%); 
                        padding: 12px; border-radius: 10px; color: white; margin-bottom: 10px;'>
                            <h5 style='margin: 0;'>{name}</h5>
                            <p style='margin: 5px 0 0 0; font-size: 0.85rem;'>{desc}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("#### 📊 Yield Ranking")
                crop_names = [c.upper()[:10] for c in top_crops['crop'].values]
                yields = top_crops['yield_kg_per_ha'].values
                
                fig = go.Figure(data=[go.Bar(
                    y=crop_names, x=yields, orientation='h',
                    marker=dict(color=yields, colorscale='Viridis', showscale=False),
                    text=[f"{y:.0f}" for y in yields], textposition='outside'
                )])
                fig.update_layout(height=400, template='plotly_white', showlegend=False, 
                    xaxis_title="Yield (kg/ha)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("### 🌾 Agricultural Recommendations & Suggestions")
            
            # Get rainfall zone information
            if annual_rainfall > 2000:
                zone_label = "High Rainfall Zone (>2000mm)"
                zone_color = "#f39c12"
                zone_emoji = "🌊"
                zone_description = "This region receives abundant rainfall, making it ideal for water-intensive crops."
                suggested_crops = {
                    "Rice": "Primary staple crop, requires 1000-2000mm rainfall",
                    "Jute": "Fiber crop, thrives in humid conditions with high rainfall",
                    "Tea": "Cash crop, requires consistent rainfall and humidity",
                    "Coconut": "Tropical crop, suitable for coastal high-rainfall areas",
                    "Rubber": "Plantation crop, needs high rainfall and humidity",
                    "Sugarcane": "Water-intensive crop, ideal for high rainfall zones",
                    "Banana": "Tropical fruit, requires consistent moisture"
                }
                water_strategies = [
                    ("Drainage Management", "Install proper drainage to prevent waterlogging", "#e74c3c"),
                    ("Raised Bed Farming", "Elevate fields 15-30cm above ground level", "#e67e22"),
                    ("Flood-Resistant Varieties", "Use crops tolerant to excess water", "#f39c12"),
                    ("Water Harvesting", "Store excess water for dry periods", "#3498db")
                ]
            elif annual_rainfall > 1000:
                zone_label = "Moderate Rainfall Zone (1000-2000mm)"
                zone_color = "#27ae60"
                zone_emoji = "🌾"
                zone_description = "This region has balanced rainfall, suitable for diverse crop cultivation."
                suggested_crops = {
                    "Wheat": "Winter crop, requires 400-600mm rainfall",
                    "Maize": "Summer crop, needs 500-800mm rainfall",
                    "Cotton": "Cash crop, requires 600-1000mm rainfall",
                    "Soybean": "Oilseed crop, needs 500-700mm rainfall",
                    "Groundnut": "Oilseed crop, suitable for moderate rainfall",
                    "Pulses": "Various types, drought-tolerant varieties available",
                    "Vegetables": "Diverse options with proper irrigation"
                }
                water_strategies = [
                    ("Drip Irrigation", "Efficient water delivery, 30-60% water saving", "#11998e"),
                    ("Rainwater Harvesting", "Build farm ponds and check dams", "#3498db"),
                    ("Mulching", "Reduce evaporation by 50-70%", "#2ecc71"),
                    ("Crop Rotation", "Improve soil moisture retention", "#1abc9c")
                ]
            else:
                zone_label = "Low Rainfall Zone (<1000mm)"
                zone_color = "#e74c3c"
                zone_emoji = "☀️"
                zone_description = "This region has limited rainfall, requiring drought-resistant crops and water conservation."
                suggested_crops = {
                    "Bajra (Pearl Millet)": "Highly drought-resistant cereal crop",
                    "Jowar (Sorghum)": "Drought-tolerant grain crop",
                    "Pulses": "Chickpea, pigeon pea, and other drought-resistant varieties",
                    "Groundnut": "Oilseed crop adapted to sandy soils",
                    "Castor": "Industrial crop for semi-arid regions",
                    "Moth Bean": "Drought-resistant pulse crop",
                    "Cluster Bean": "Drought-tolerant vegetable crop"
                }
                water_strategies = [
                    ("Well Recharge", "Dig wells 6m deep for groundwater", "#9b59b6"),
                    ("Dry Farming", "Use mulching and moisture conservation", "#e74c3c"),
                    ("Early Sowing", "Plant crops to catch monsoon rains", "#f39c12"),
                    ("Drought Varieties", "Use specially bred drought-resistant seeds", "#27ae60")
                ]
            
            # Display zone information
            col_left, col_right = st.columns([2, 1], gap="large")
            
            with col_left:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {zone_color} 0%, 
                    {"#e67e22" if zone_color=="#f39c12" else "#c0392b" if zone_color=="#e74c3c" else "#2ecc71"} 100%); 
                    color: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; 
                    text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.2);'>
                        <h2 style='margin: 0; font-size: 2.5rem;'>{zone_emoji}</h2>
                        <h3 style='margin: 10px 0; font-size: 1.5rem;'>{zone_label}</h3>
                        <p style='margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.9;'>{zone_description}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 🌱 Recommended Crops for Your Region")
                for idx, (crop, desc) in enumerate(suggested_crops.items(), 1):
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                        padding: 18px; border-radius: 12px; margin-bottom: 15px; border-left: 5px solid {zone_color}; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);'>
                            <h4 style='margin: 0; color: {zone_color}; font-size: 1.1rem;'>#{idx} {crop}</h4>
                            <p style='margin: 8px 0 0 0; color: #666; font-size: 0.95rem; line-height: 1.4;'>{desc}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col_right:
                # Risk assessment
                risk_level, risk_color, risk_icon, _ = calculate_flood_risk(annual_rainfall)
                
                st.markdown(f"""
                    <div style='background: {risk_color}; color: white; padding: 20px; border-radius: 15px; 
                    text-align: center; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.2);'>
                        <h2 style='margin: 0; font-size: 2.5rem;'>{risk_icon}</h2>
                        <h3 style='margin: 10px 0 0 0;'>{risk_level} Risk</h3>
                        <p style='margin: 5px 0 0 0;'>Current Status</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 💧 Water Management Strategies")
                for name, desc, color in water_strategies:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {color} 0%, rgba(0,0,0,0.1) 100%); 
                        padding: 25px; border-radius: 15px; color: white; margin-bottom: 18px; 
                        box-shadow: 0 6px 15px rgba(0,0,0,0.2); border: 2px solid rgba(255,255,255,0.1);'>
                            <h4 style='margin: 0; font-size: 1.3rem; font-weight: bold; text-shadow: 0 2px 4px rgba(0,0,0,0.3);'>{name}</h4>
                            <p style='margin: 12px 0 0 0; font-size: 1rem; line-height: 1.5; opacity: 0.95;'>{desc}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Additional recommendations section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 General Agricultural Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 12px; color: white; text-align: center;'>
                        <h4 style='margin: 0; opacity: 0.9;'>🌱 Soil Preparation</h4>
                        <p style='margin: 10px 0 0 0; font-size: 0.9rem;'>Test soil pH and nutrient levels before planting. Add organic matter to improve soil structure.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 20px; border-radius: 12px; color: white; text-align: center;'>
                        <h4 style='margin: 0; opacity: 0.9;'>🌾 Crop Planning</h4>
                        <p style='margin: 10px 0 0 0; font-size: 0.9rem;'>Plan crop rotation to maintain soil fertility. Consider intercropping for better resource utilization.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); 
                    padding: 20px; border-radius: 12px; color: white; text-align: center;'>
                        <h4 style='margin: 0; opacity: 0.9;'>📊 Market Research</h4>
                        <p style='margin: 10px 0 0 0; font-size: 0.9rem;'>Research local market demand and prices before selecting crops. Consider value-added products.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Contact information for further assistance
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🆘 Need More Specific Guidance?")
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%); 
                padding: 20px; border-radius: 12px; color: white; text-align: center;'>
                    <h4 style='margin: 0;'>📞 Contact Agricultural Extension Services</h4>
                    <p style='margin: 10px 0 0 0; font-size: 0.95rem;'>For district-specific crop recommendations and technical guidance, contact your local agricultural extension office.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 **Note**: These are general recommendations based on rainfall patterns. For specific crop varieties and cultivation practices suitable for your exact location, consult with local agricultural experts.")
    

import time
def main():
    if not st.session_state.logged_in:
        auth_page()
    else:
        dashboard_page()

if __name__ == "__main__":
    main()