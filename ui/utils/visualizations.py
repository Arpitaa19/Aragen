"""
Visualization utilities using Plotly
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_class_distribution(df, target_col='is_active'):
    """Plot active/inactive distribution"""
    counts = df[target_col].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Active', 'Inactive'],
        values=[counts.get(1, 0), counts.get(0, 0)],
        marker_colors=['#2ecc71', '#e74c3c'],
        hole=0.4
    )])
    
    fig.update_layout(
        title="Activity Distribution",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_feature_distribution(df, feature, bins=50):
    """Plot histogram of a feature"""
    fig = px.histogram(
        df, 
        x=feature,
        nbins=bins,
        color='is_active' if 'is_active' in df.columns else None,
        color_discrete_map={1: '#2ecc71', 0: '#e74c3c'},
        labels={'is_active': 'Activity'},
        title=f'Distribution of {feature}'
    )
    
    fig.update_layout(height=400, showlegend=True)
    return fig


def plot_scatter(df, x_col, y_col, color_col='is_active'):
    """Plot scatter plot"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map={1: '#2ecc71', 0: '#e74c3c'},
        labels={color_col: 'Activity'},
        title=f'{x_col} vs {y_col}',
        opacity=0.6
    )
    
    fig.update_layout(height=500)
    return fig


def plot_correlation_heatmap(df, features):
    """Plot correlation heatmap"""
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        width=700
    )
    
    return fig


def plot_shap_waterfall(shap_values, feature_names, feature_values, base_value):
    """
    Create waterfall chart for SHAP values
    
    Args:
        shap_values: Array of SHAP values
        feature_names: List of feature names
        feature_values: Array of feature values
        base_value: Base prediction value
    """
    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_values = shap_values[indices]
    sorted_feature_vals = feature_values[indices]
    
    # Create labels
    labels = [f"{feat}<br>= {val:.2f}" for feat, val in zip(sorted_features, sorted_feature_vals)]
    
    # Build waterfall
    cumsum = np.cumsum([base_value] + list(sorted_values))
    
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(sorted_values) + ["total"],
        x=labels + ["Final"],
        y=list(sorted_values) + [cumsum[-1]],
        text=[f"{v:+.3f}" for v in sorted_values] + [f"{cumsum[-1]:.3f}"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#3498db"}}
    ))
    
    fig.update_layout(
        title="SHAP Feature Contributions",
        showlegend=False,
        height=500,
        xaxis_title="Features",
        yaxis_title="SHAP Value"
    )
    
    return fig
