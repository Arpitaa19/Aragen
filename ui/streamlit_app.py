import streamlit as st

# Page config
st.set_page_config(
    page_title="BioInsight Lite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("BioInsight Lite")

    st.markdown("---")
    st.caption("Hackathon Submission")

# Main Content
st.title("BioInsight Lite")
st.subheader("Data Explorer & Predictor")

st.markdown("---")

st.markdown("""
### Project Goal
Build a mini application to explore chemical/biological datasets and predict compound properties using ML models.

### Key Components
1.  **Data Ingestion & Cleaning:** ChEMBL (v36) Dataset
2.  **Exploratory Data Analysis (EDA):** Visualization & Statistics
3.  **Model Development:** Logistic Regression vs XGBoost
4.  **Evaluation & Explainability:** ROC-AUC metrics & SHAP
5.  **Deployment:** Streamlit Application

### Get Started
Select a module from the sidebar to begin.
""")
