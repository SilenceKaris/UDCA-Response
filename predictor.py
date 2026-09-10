import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
STAGE_MAPPING = {1: "Preclinical Stage", 2: "Asymptomatic Stage", 3: "Symptomatic Stage", 4: "Decompensated Stage"}

# Key: keep the same order as during training (ALBI is automatically calculated)
MODEL_FEATURE_ORDER = ['ALP', 'Stages', 'ALBI', 'GGT', 'C3', 'D_D', 'DBIL']
SHAP_LABELS = ['ALP', 'Stages', 'ALBI', 'GGT', 'C3', 'D_D', 'DBIL']

# ==================== Page Setup ====================
st.set_page_config(page_title="Incomplete UDCA Response Prediction", layout="wide")
st.title("Risk Prediction of Incomplete UDCA Response")

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    try:
        with open('lgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()
if model is None:
    st.error(f"Failed to load model: {error}")
    st.stop()

# ==================== Input Interface ====================
st.subheader("📋 Clinical Parameter Input")
col1, col2 = st.columns(2)

with col1:
    # Total bilirubin and albumin input (used to automatically calculate ALBI)
    tbil = st.number_input("Total Bilirubin TBIL (μmol/L)", 
                          min_value=1.0, max_value=1000.0, 
                          value=15.0, step=0.1, key='tbil')

    alb = st.number_input("Albumin ALB (g/L)", 
                         min_value=1.0, max_value=100.0, 
                         value=40.0, step=0.1, key='alb')

    # Automatically calculate ALBI score and display it
    # (Formula: ALBI = 0.66×log10(TBIL) - 0.085×ALB)
    albi_score = 0.66 * np.log10(tbil) - 0.085 * alb
    st.metric("ALBI Score (Automatically Calculated)", f"{albi_score:.3f}", 
              help="Calculation formula: ALBI = 0.66×log₁₀(TBIL) - 0.085×ALB")

    # Other parameters
    ggt = st.number_input("GGT (U/L)", 
                         min_value=0.0, max_value=2000.0, 
                         value=120.0, step=1.0, key='ggt')

    dbil = st.number_input("DBIL (μmol/L)", 
                          min_value=0.0, max_value=500.0, 
                          value=15.0, step=0.1, key='dbil')

with col2:
    alp = st.number_input("ALP (U/L)", 
                         min_value=0.0, max_value=3000.0, 
                         value=200.0, step=1.0, key='alp')

    c3 = st.number_input("Complement C3 (g/L)", 
                        min_value=0.0, max_value=5.0, 
                        value=0.9, step=0.01, key='c3')

    d_d = st.number_input("D-Dimer (mg/L)", 
                         min_value=0.0, max_value=100.0, 
                         value=0.8, step=0.01, key='d_d')

    stages = st.selectbox("Natural History Stage", 
                         options=list(STAGE_MAPPING.keys()),
                         format_func=lambda x: f"Stage {x} - {STAGE_MAPPING[x]}",
                         key='stages')

# Build input dictionary (in the same order as model training)
inputs = {
    'ALP': alp,
    'Stages': stages,
    'ALBI': albi_score,  # Use the automatically calculated value
    'GGT': ggt,
    'C3': c3,
    'D_D': d_d,
    'DBIL': dbil
}

with st.expander("ℹ️ Data Privacy & Disclaimer"):
    st.markdown("""
- All input data are processed locally and in real time for prediction only; **no data are collected, stored, or transmitted**, and all entries are discarded upon closing the page.
- All values shown are simulated demonstration data and do not correspond to any actual patient.
- This tool is intended **solely for academic research and demonstration purposes** and must not be used as a basis for clinical diagnosis or treatment decisions.
""")
# ==================== Prediction ====================
st.divider()
if st.button("🔍 Predict", type="primary"):
    feature_values = [inputs[col] for col in MODEL_FEATURE_ORDER]
    X = pd.DataFrame([feature_values], columns=MODEL_FEATURE_ORDER)

    try:
        proba = model.predict_proba(X)[0][1]
        risk_pct = proba * 100

        col_res, col_shap = st.columns([1, 2])

        with col_res:
            st.subheader("Prediction Result")
            st.metric("Probability of Poor UDCA Response", f"{risk_pct:.1f}%")
            st.progress(min(float(proba), 1.0))

            # Threshold remains 0.5 (50%)
            if proba >= 0.5:
                st.error("High Risk")
            else:
                st.success("Low Risk")

            # Feature reference table
            with st.expander("📊 Feature Reference"):
                st.markdown("""
                | Feature Label | Description |
                |---------------|-------------|
                | ALP | Alkaline Phosphatase |
                | Stages | Natural History Stage |
                | ALBI | ALBI Score (automatically calculated from TBIL and ALB) |
                | GGT | Gamma-Glutamyl Transferase |
                | C3 | Complement C3 |
                | D_D | D-Dimer |
                | DBIL | Direct Bilirubin |
                """)

        with col_shap:
            st.subheader("SHAP Feature Contribution")
            with st.spinner("Calculating..."):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)

                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                    base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                else:
                    sv = shap_vals[0]
                    base = explainer.expected_value

                # SHAP still shows ALBI (as a single composite indicator)
                fig, ax = plt.subplots(figsize=(10, 6))
                exp = shap.Explanation(sv, base, X.iloc[0].values, SHAP_LABELS)
                shap.plots.waterfall(exp, max_display=8, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.error(f"Prediction error: {e}")
