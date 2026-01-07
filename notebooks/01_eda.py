"""
Exploratory Data Analysis (EDA)
BioInsight Lite - Compound Bioactivity Dataset
"""

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create results directory if not exists
os.makedirs('results', exist_ok=True)

# Load data
print("📥 Loading data...")
try:
    df = pd.read_csv('data/sample_bioactivity.csv')
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features")
except FileNotFoundError:
    print("❌ Data file not found! Run from project root.")
    exit()

# Display first rows
print(df.head())

# ==========================================
# 2. DATASET OVERVIEW
# ==========================================

print("\n" + "="*60)
print("📊 DATASET OVERVIEW")
print("="*60)

# Basic info
print(f"\n📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Data types
print("\n📋 Data Types:")
print(df.dtypes.value_counts())

# Summary statistics
print("\n📈 Summary Statistics:")
print(df.describe().T)

# ==========================================
# 3. MISSING VALUE ANALYSIS
# ==========================================

print("\n" + "="*60)
print("🔍 MISSING VALUE ANALYSIS")
print("="*60)

# Calculate missing values
missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
    'Dtype': df.dtypes
}).sort_values('Missing_Percentage', ascending=False)

# Display missing values
print("\n📊 Missing Values by Feature:")
print(missing_df[missing_df['Missing_Count'] > 0])

# Visualize missing values - Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    df.isnull(),
    yticklabels=False,
    cbar=True,
    cmap='viridis'
)
plt.title('Missing Values Heatmap (Yellow = Missing)', fontsize=16, pad=20)
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('results/eda_missing_heatmap.png', dpi=300, bbox_inches='tight')
# plt.show() # Commented out for script execution

# Missing value patterns
if df.isnull().sum().sum() > 0:
    # Create missing value bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=missing_df['Column'],
            y=missing_df['Missing_Percentage'],
            text=missing_df['Missing_Percentage'].round(1),
            textposition='auto',
            marker_color='indianred'
        )
    ])
    
    fig.update_layout(
        title='Missing Value Percentage by Feature',
        xaxis_title='Feature',
        yaxis_title='Missing %',
        height=500,
        xaxis={'tickangle': -45}
    )
    
    fig.write_html('results/eda_missing_bar.html')
    # fig.show()

# ==========================================
# 4. TARGET VARIABLE ANALYSIS
# ==========================================

print("\n" + "="*60)
print("🎯 TARGET VARIABLE ANALYSIS")
print("="*60)

target_col = 'is_active'

if target_col in df.columns:
    # Class distribution
    target_counts = df[target_col].value_counts()
    target_pct = df[target_col].value_counts(normalize=True) * 100
    
    print("\n📊 Class Distribution:")
    print(f"Active (1):   {target_counts.get(1, 0):,} ({target_pct.get(1, 0):.2f}%)")
    print(f"Inactive (0): {target_counts.get(0, 0):,} ({target_pct.get(0, 0):.2f}%)")
    print(f"\n⚖️ Imbalance Ratio: {target_counts.max() / target_counts.min():.2f}:1")
    
    # Visualization - Pie Chart
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type':'pie'}, {'type':'bar'}]],
        subplot_titles=('Class Distribution', 'Class Counts')
    )
    
    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=['Active', 'Inactive'],
            values=[target_counts.get(1, 0), target_counts.get(0, 0)],
            marker_colors=['#2ecc71', '#e74c3c'],
            hole=0.4
        ),
        row=1, col=1
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=['Active', 'Inactive'],
            y=[target_counts.get(1, 0), target_counts.get(0, 0)],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f"{target_counts.get(1, 0):,}", f"{target_counts.get(0, 0):,}"],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Target Variable Analysis"
    )
    
    fig.write_html('results/eda_target_distribution.html')
    # fig.show()

# ==========================================
# 5. NUMERICAL FEATURES ANALYSIS
# ==========================================

print("\n" + "="*60)
print("📊 NUMERICAL FEATURES ANALYSIS")
print("="*60)

# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

# Remove ID columns
numerical_cols = [col for col in numerical_cols if not col.endswith('_id')]

print(f"\n📏 Found {len(numerical_cols)} numerical features")

# Summary statistics
summary_stats = df[numerical_cols].describe().T
summary_stats['skewness'] = df[numerical_cols].skew()
summary_stats['kurtosis'] = df[numerical_cols].kurtosis()

print("\n📈 Extended Summary Statistics:")
print(summary_stats.round(2))

# Distribution plots for key features
key_features = [
    'mw_freebase', 'alogp', 'hba', 'hbd', 'psa', 
    'rtb', 'aromatic_rings', 'num_ro5_violations', 'qed_weighted'
]
key_features = [f for f in key_features if f in numerical_cols]

# Create distribution plots
n_cols = 3
n_rows = (len(key_features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    # Histogram with KDE
    if target_col in df.columns:
        df[df[target_col]==1][feature].hist(
            ax=ax, bins=50, alpha=0.6, label='Active', color='#2ecc71', density=True
        )
        df[df[target_col]==0][feature].hist(
            ax=ax, bins=50, alpha=0.6, label='Inactive', color='#e74c3c', density=True
        )
        ax.legend()
    else:
        df[feature].hist(ax=ax, bins=50, alpha=0.7, color='#3498db', density=True)
    
    # KDE overlay
    try:
        df[feature].plot(kind='kde', ax=ax, secondary_y=True, linewidth=2, color='black')
    except:
        pass
    
    ax.set_title(f'{feature}\n(Skew: {df[feature].skew():.2f})', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.3)

# Remove empty subplots
for idx in range(len(key_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('results/eda_distributions.png', dpi=300, bbox_inches='tight')
# plt.show()

# ==========================================
# 6. BOX PLOTS (Outlier Detection)
# ==========================================

print("\n" + "="*60)
print("📦 OUTLIER DETECTION (Box Plots)")
print("="*60)

# Box plots for key features
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    if target_col in df.columns:
        df.boxplot(
            column=feature,
            by=target_col,
            ax=ax,
            patch_artist=True
        )
        ax.set_title(f'{feature} by Activity')
    else:
        df.boxplot(column=feature, ax=ax, patch_artist=True)
        ax.set_title(feature)
    
    ax.set_xlabel('')
    plt.sca(ax)
    plt.xticks([1, 2], ['Inactive', 'Active'] if target_col in df.columns else [''])

# Remove empty subplots
for idx in range(len(key_features), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig('results/eda_boxplots.png', dpi=300, bbox_inches='tight')
# plt.show()

# Outlier statistics
print("\n📊 Outlier Detection (IQR Method):")
for feature in key_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    outlier_pct = len(outliers) / len(df) * 100
    
    print(f"{feature:20s}: {len(outliers):6,} outliers ({outlier_pct:5.2f}%)")

# ==========================================
# 7. CORRELATION ANALYSIS
# ==========================================

print("\n" + "="*60)
print("🔗 CORRELATION ANALYSIS")
print("="*60)

# Calculate correlation matrix
corr_features = key_features.copy()
if target_col in df.columns and target_col not in corr_features:
    corr_features.append(target_col)

corr_matrix = df[corr_features].corr()

# Correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)

plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('results/eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()

# Top correlations with target
if target_col in corr_matrix.columns:
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    target_corr = target_corr[target_corr.index != target_col]
    
    print("\n🎯 Top 10 Correlations with Target:")
    print(target_corr.head(10))
    
    # Visualize
    fig = go.Figure(data=[
        go.Bar(
            x=target_corr.head(10).values,
            y=target_corr.head(10).index,
            orientation='h',
            marker_color='#3498db',
            text=target_corr.head(10).values.round(3),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Top 10 Features Correlated with Activity',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Feature',
        height=500
    )
    
    fig.write_html('results/eda_target_correlation.html')
    # fig.show()

# High correlation pairs (multicollinearity check)
print("\n⚠️ High Correlation Pairs (|r| > 0.7):")
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
    print(high_corr_df)
else:
    print("No high correlation pairs found")

# ==========================================
# 8. BIVARIATE ANALYSIS (Scatter Plots)
# ==========================================

print("\n" + "="*60)
print("🔬 BIVARIATE ANALYSIS")
print("="*60)

# Key feature pairs
feature_pairs = [
    ('mw_freebase', 'alogp'),
    ('mw_freebase', 'psa'),
    ('hba', 'hbd'),
    ('rtb', 'aromatic_rings')
]

feature_pairs = [(f1, f2) for f1, f2 in feature_pairs if f1 in df.columns and f2 in df.columns]

# Create scatter plots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'{f1} vs {f2}' for f1, f2 in feature_pairs]
)

for idx, (feat1, feat2) in enumerate(feature_pairs):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    if target_col in df.columns:
        # Separate by activity
        active = df[df[target_col] == 1]
        inactive = df[df[target_col] == 0]
        
        # Sample for performance
        sample_size = min(5000, len(active), len(inactive))
        active_sample = active.sample(n=sample_size, random_state=42)
        inactive_sample = inactive.sample(n=sample_size, random_state=42)
        
        fig.add_trace(
            go.Scatter(
                x=inactive_sample[feat1],
                y=inactive_sample[feat2],
                mode='markers',
                name='Inactive',
                marker=dict(color='#e74c3c', size=4, opacity=0.5),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=active_sample[feat1],
                y=active_sample[feat2],
                mode='markers',
                name='Active',
                marker=dict(color='#2ecc71', size=4, opacity=0.5),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
    else:
        # All data
        sample_df = df.sample(n=min(5000, len(df)), random_state=42)
        fig.add_trace(
            go.Scatter(
                x=sample_df[feat1],
                y=sample_df[feat2],
                mode='markers',
                marker=dict(color='#3498db', size=4, opacity=0.5)
            ),
            row=row, col=col
        )
    
    fig.update_xaxes(title_text=feat1, row=row, col=col)
    fig.update_yaxes(title_text=feat2, row=row, col=col)

fig.update_layout(
    height=800,
    title_text="Bivariate Feature Relationships",
    showlegend=True
)

fig.write_html('results/eda_scatter_plots.html')
# fig.show()

# ==========================================
# 9. CATEGORICAL FEATURES ANALYSIS
# ==========================================

print("\n" + "="*60)
print("📋 CATEGORICAL FEATURES ANALYSIS")
print("="*60)

# Select categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if df[col].nunique() < 50]

if categorical_cols:
    print(f"\n📏 Found {len(categorical_cols)} categorical features")
    
    for col in categorical_cols[:5]:  # Top 5 categorical
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))
        
        # Bar chart
        value_counts = df[col].value_counts().head(15)
        
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color='#3498db',
                text=value_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f'Distribution of {col} (Top 15)',
            xaxis_title=col,
            yaxis_title='Count',
            height=400,
            xaxis={'tickangle': -45}
        )
        
        fig.write_html(f'results/eda_categorical_{col}.html')
        # fig.show()
else:
    print("No categorical features found")

# ==========================================
# 10. FEATURE NORMALIZATION COMPARISON
# ==========================================

print("\n" + "="*60)
print("⚖️ FEATURE NORMALIZATION COMPARISON")
print("="*60)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Select features to normalize
norm_features = ['mw_freebase', 'alogp', 'psa', 'qed_weighted']
norm_features = [f for f in norm_features if f in df.columns]

# Original distributions
original_data = df[norm_features].copy()

# Apply different scalers
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

standard_scaled = pd.DataFrame(
    standard_scaler.fit_transform(original_data.fillna(original_data.mean())),
    columns=original_data.columns
)

minmax_scaled = pd.DataFrame(
    minmax_scaler.fit_transform(original_data.fillna(original_data.mean())),
    columns=original_data.columns
)

robust_scaled = pd.DataFrame(
    robust_scaler.fit_transform(original_data.fillna(original_data.mean())),
    columns=original_data.columns
)

# Compare distributions
fig = make_subplots(
    rows=len(norm_features), cols=4,
    subplot_titles=['Original', 'StandardScaler', 'MinMaxScaler', 'RobustScaler'] * len(norm_features),
    vertical_spacing=0.1
)

for idx, feature in enumerate(norm_features):
    row = idx + 1
    
    # Original
    fig.add_trace(
        go.Histogram(x=original_data[feature], nbinsx=50, name='Original', showlegend=(idx==0)),
        row=row, col=1
    )
    
    # Standard scaled
    fig.add_trace(
        go.Histogram(x=standard_scaled[feature], nbinsx=50, name='Standard', showlegend=(idx==0)),
        row=row, col=2
    )
    
    # MinMax scaled
    fig.add_trace(
        go.Histogram(x=minmax_scaled[feature], nbinsx=50, name='MinMax', showlegend=(idx==0)),
        row=row, col=3
    )
    
    # Robust scaled
    fig.add_trace(
        go.Histogram(x=robust_scaled[feature], nbinsx=50, name='Robust', showlegend=(idx==0)),
        row=row, col=4
    )
    
    fig.update_yaxes(title_text=feature, row=row, col=1)

fig.update_layout(
    height=300*len(norm_features),
    title_text="Feature Normalization Comparison",
    showlegend=True
)

fig.write_html('results/eda_normalization_comparison.html')
# fig.show()

# Statistics comparison
print("\n📊 Normalization Statistics Comparison:")
for feature in norm_features:
    print(f"\n{feature}:")
    print(f"  Original:  mean={original_data[feature].mean():.2f}, std={original_data[feature].std():.2f}")
    print(f"  Standard:  mean={standard_scaled[feature].mean():.2f}, std={standard_scaled[feature].std():.2f}")
    print(f"  MinMax:    mean={minmax_scaled[feature].mean():.2f}, std={minmax_scaled[feature].std():.2f}")
    print(f"  Robust:    mean={robust_scaled[feature].mean():.2f}, std={robust_scaled[feature].std():.2f}")

# ==========================================
# 11. ACTIVITY BY FEATURE RANGES
# ==========================================

print("\n" + "="*60)
print("🎯 ACTIVITY RATE BY FEATURE RANGES")
print("="*60)

if target_col in df.columns:
    # Bin features and calculate activity rate
    for feature in ['mw_freebase', 'alogp', 'psa']:
        if feature not in df.columns:
            continue
        
        # Create bins
        df[f'{feature}_bin'] = pd.cut(df[feature], bins=10)
        
        # Calculate activity rate per bin
        activity_by_bin = df.groupby(f'{feature}_bin', observed=True)[target_col].agg(['mean', 'count'])
        activity_by_bin['mean'] *= 100  # Convert to percentage
        
        print(f"\n{feature} Activity Rate:")
        print(activity_by_bin)
        
        # Visualize
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[str(x) for x in activity_by_bin.index],
            y=activity_by_bin['mean'],
            name='Activity Rate %',
            marker_color='#2ecc71',
            text=activity_by_bin['mean'].round(1),
            textposition='auto'
        ))
        
        fig.add_trace(go.Scatter(
            x=[str(x) for x in activity_by_bin.index],
            y=activity_by_bin['count'],
            name='Sample Count',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig.update_layout(
            title=f'Activity Rate by {feature} Range',
            xaxis_title=feature,
            yaxis_title='Activity Rate (%)',
            yaxis2=dict(
                title='Sample Count',
                overlaying='y',
                side='right'
            ),
            height=500,
            xaxis={'tickangle': -45}
        )
        
        fig.write_html(f'results/eda_activity_by_{feature}.html')
        # fig.show()
        
        # Clean up
        df.drop(f'{feature}_bin', axis=1, inplace=True)

# ==========================================
# 12. SUMMARY REPORT
# ==========================================

print("\n" + "="*60)
print("📝 EDA SUMMARY REPORT")
print("="*60)

print("✅ EDA COMPLETE!")
print("📁 All visualizations saved to 'results/' directory")
