import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory (if needed for direct run, though module import handles it in app)
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_loader import get_model_loader
from utils.preprocessing import preprocess_data

st.set_page_config(page_title="Dataset Search", layout="wide")
st.title("Guided Dataset Search")

# Load Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/sample_bioactivity.csv')
    except:
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.error("No data found.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Search Filters")

# Filter: Molecular Weight
min_mw, max_mw = int(df['mw_freebase'].min()), int(df['mw_freebase'].max())
mw_range = st.sidebar.slider("Molecular Weight", min_mw, max_mw, (min_mw, max_mw))

# Filter: LogP
min_logp, max_logp = float(df['alogp'].min()), float(df['alogp'].max())
logp_range = st.sidebar.slider("LogP (Lipophilicity)", min_logp, max_logp, (min_logp, max_logp))

# Filter: Activity
activity_filter = st.sidebar.multiselect("Activity Status (Ground Truth)", ["Active", "Inactive"], default=["Active", "Inactive"])

# Apply Filters
filtered_df = df[
    (df['mw_freebase'].between(mw_range[0], mw_range[1])) &
    (df['alogp'].between(logp_range[0], logp_range[1]))
]

if "Active" not in activity_filter:
    filtered_df = filtered_df[filtered_df['is_active'] == 0]
if "Inactive" not in activity_filter:
    filtered_df = filtered_df[filtered_df['is_active'] == 1]

# ==========================================
# PREDICTIONS
# ==========================================

try:
    loader = get_model_loader()
    
    if not filtered_df.empty:
        # Use robust centralized preprocessing
        # This fixes 'feature mismatch' by filling defaults and engineering interactions
        X = preprocess_data(filtered_df, loader.scaler, loader.encoders)
        
        # Predict
        probs = loader.xgboost.predict_proba(X)[:, 1]
        
        # Insert prediction columns
        # Use .copy() to avoid SettingWithCopy warning on filtered view
        filtered_df = filtered_df.copy()
        filtered_df.insert(2, "Predicted_Prob", probs)
        filtered_df['Model_Prediction'] = ['Active' if p > 0.5 else 'Inactive' for p in probs]
        
except Exception as e:
    st.warning(f"Could not generate predictions: {e}")

# Display
st.subheader(f"Search Results ({len(filtered_df)} compounds)")

# LIMIT DISPLAY for performance and styling limits
max_display = 1000
if len(filtered_df) > max_display:
    st.warning(f"Displaying top {max_display} results. Download CSV to see all.")
    display_df = filtered_df.head(max_display)
else:
    display_df = filtered_df

# Safe display without styling if still too huge, or minimal styling
try:
    st.dataframe(
        display_df.style.background_gradient(subset=['Predicted_Prob'], cmap='Greens'),
        use_container_width=True
    )
except:
    # Fallback if styling fails
    st.dataframe(display_df, use_container_width=True)

# Download Full Results
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Results (CSV)", csv, "search_results.csv", "text/csv")
