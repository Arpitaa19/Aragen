import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Exploratory Data Analysis", page_icon=None, layout="wide")
st.title("Exploratory Data Analysis")

# Load Data
try:
    df = pd.read_csv('data/sample_bioactivity.csv')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Tabs for Organized Exploration
tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Correlations"])

with tab1:
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Compounds", f"{len(df):,}")
    col2.metric("Active", f"{(df.is_active==1).sum():,} ({100*(df.is_active==1).sum()/len(df):.1f}%)")
    col3.metric("Inactive", f"{(df.is_active==0).sum():,} ({100*(df.is_active==0).sum()/len(df):.1f}%)")
    
    st.markdown("#### Sample Data")
    st.dataframe(df.head(50), use_container_width=True)





with tab2:
    st.markdown("### Target Distribution")
    fig_pie = px.pie(
        df, 
        names=df['is_active'].map({1: 'Active', 0: 'Inactive'}), 
        title='Active vs Inactive Ratio',
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Feature Distributions")
    feature = st.selectbox("Select Feature to Visualize", ['mw_freebase', 'alogp', 'psa', 'hba', 'hbd', 'rtb', 'aromatic_rings', 'qed_weighted'])
    
    # Histogram with Activity overlay
    fig_hist = px.histogram(
        df, x=feature, color=df['is_active'].map({1: 'Active', 0: 'Inactive'}),
        nbins=50, barmode='overlay', opacity=0.7,
        title=f"Distribution of {feature} by Activity",
        color_discrete_map={'Active': '#2ecc71', 'Inactive': '#e74c3c'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.markdown("### Correlations & Relationships")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        x_axis = st.selectbox("X-Axis", ['mw_freebase', 'alogp', 'psa'], index=0)
        y_axis = st.selectbox("Y-Axis", ['alogp', 'psa', 'mw_freebase'], index=1)
        
    with col2:
        # Scatter Plot
        fig_scatter = px.scatter(
            df.sample(min(1000, len(df))), 
            x=x_axis, y=y_axis,
            color=df['is_active'].sample(min(1000, len(df))).map({1: 'Active', 0: 'Inactive'}),
            title=f"{x_axis} vs {y_axis}",
            color_discrete_map={'Active': '#2ecc71', 'Inactive': '#e74c3c'},
            hover_data=['compound_id'] if 'compound_id' in df.columns else None
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    if st.checkbox("Show Correlation Matrix"):
        corr_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Limit to key columns for readability
        key_cols = ['mw_freebase', 'alogp', 'psa', 'hba', 'hbd', 'rtb', 'aromatic_rings', 'is_active']
        corr_cols = [c for c in key_cols if c in corr_cols]
        
        corr = df[corr_cols].corr()
        fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation Matrix")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Feature Correlation with Bioactivity (Target)")
    # Univariate Feature Importance
    corr_target = df.select_dtypes(include=[np.number]).corr()['is_active'].drop('is_active').sort_values()
    # Filter to interesting ones
    fig_imp = px.bar(
        x=corr_target.values, 
        y=corr_target.index, 
        orientation='h',
        title="Univariate Feature Importance (Correlation with Active)",
        labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
        color=corr_target.values,
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig_imp, use_container_width=True)

