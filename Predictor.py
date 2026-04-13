#!/usr/bin/env python
# coding: utf-8

# # Prediction of shear strength in RC columns

# In[8]:

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 不需要显式导入 xgboost，但环境中必须已安装
# joblib 加载时会自动依赖 xgboost

st.set_page_config(
    page_title="Predictor of RC Column Shear Strength",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Predictor of RC Column Shear Strength")
st.markdown("""
This application is based on the Ph3-XGBR model to predict the shear strength $V_u$ (kN) of RC columns.
Please enter the following 8 feature values and click the predict button to obtain the result.
""")

with st.sidebar:
    st.header("📌 Model Information")
    st.markdown("""
     - **Input**: 8 features  
     - **Output**: Shear strength $V_u$ (kN)  
     - **Algorithm**: XGBoost Regressor
    """)
    st.divider()
    st.caption("Model file: Prediction of shear strength in RC columns.pkl")

# 检查 xgboost 是否可用（可选，提升用户体验）
try:
    import xgboost
    st.sidebar.success("✅ XGBoost backend ready")
except ImportError:
    st.sidebar.error("❌ XGBoost not installed. Please add 'xgboost' to requirements.txt")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load('Prediction of shear strength in RC columns.pkl')

try:
    model = load_model()
    st.success("✅ Model loaded successfully!", icon="🎉")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}\nEnsure the model file exists and xgboost is installed.")
    st.stop()

st.header("📊 Input features")
col1, col2 = st.columns(2)

with col1:
    L = st.number_input("**L** - column height / mm", min_value=80.0, max_value=1600.0, value=1000.0)
    fc = st.number_input("**fc** - concrete compressive strength / MPa", min_value=4.0, max_value=216.0, value=25.0)
    ρs = st.number_input("**ρs** - transverse reinforcement ratio (%)", min_value=0.0, max_value=2.0, value=0.02, format="%.4f")
    P = st.number_input("**P** - axial load / kN", min_value=0.0, max_value=18400.0, value=500.0)

with col2:
    Vc = st.number_input("**Vc** - shear contribution of concrete:fc^0.5Ag/(L/h) / kN", min_value=32.0, max_value=2830.0, value=200.0)
    Vs = st.number_input("**Vs** - shear contribution of transverse reinforcement:Asfysh/s / kN", min_value=0.0, max_value=3580.0, value=150.0)
    Vl = st.number_input("**Vl** - shear contribution of longitudinal reinforcement:Alfyl / kN", min_value=25.0, max_value=11400.0, value=300.0)
    Vp = st.number_input("**Vp** - shear contribution of axial force:P(h-c)/2L / kN", min_value=0.0, max_value=2520.0, value=0.0)

input_data = pd.DataFrame({
    'L': [L], 'fc': [fc], 'ρs': [ρs], 'P': [P],
    'Vc': [Vc], 'Vs': [Vs], 'Vl': [Vl], 'Vp': [Vp]
})

st.markdown("---")
col_btn, col_res = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔮 Start predicting", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Prediction in progress, please wait..."):
        prediction = model.predict(input_data)[0]
    with col_res:
        st.success("### Prediction result")
        st.metric(label="📐 Shear strength $V_u$ (kN)", value=f"{prediction:.2f} kN")


