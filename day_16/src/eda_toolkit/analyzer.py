"""
EDA Toolkit - Automated exploratory data analysis functions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any




def generate_numeric_summary(df: pd.DataFrame, numeric_cols: List[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive summary of numeric columns.
    
    Args:
        df: DataFrame to analyze
        numeric_cols: List of numeric column names
        
    Returns:
        Dictionary with 'figure' and 'outliers' keys
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not col.lower().endswith('_id')]

    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    outliers_dict = {}

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]

        ax.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')

        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.2f}')
        
        # Detect outliers (2 sigma rule)
        std_val = df[col].std()
        outliers = df[(df[col] > mean_val + 2*std_val) | (df[col] < mean_val - 2*std_val)]
        outliers_dict[col] = len(outliers)
        
        ax.set_title(f'{col} Distribution')
        ax.legend()
    
    # Hide extra subplots if grid has empty spaces
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    return {
        'figure': fig,
        'outliers': outliers_dict
    }



def generate_relationship_analysis(df: pd.DataFrame, numeric_cols: List[str] = None) -> Dict[str, Any]:
    """
    Analyze relationships between numeric features.
    
    Args:
        df: DataFrame to analyze
        numeric_cols: List of numeric column names
        
    Returns:
        Dictionary with 'heatmap', 'pairplot', and 'top_correlations' keys
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not col.lower().endswith('_id')]
    
    corr = df[numeric_cols].corr()

    fig_heatmap, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')

    fig_pairplot = sns.pairplot(df[numeric_cols], diag_kind='kde')
    fig_pairplot.figure.suptitle('Pairwise Relationships', y=1.01)

    corr_pairs = corr.unstack()

    corr_pairs = corr_pairs[corr_pairs != 1.0]

    top_corr = corr_pairs.abs().sort_values(ascending=False).head(5)

    top_correlations = [
        (pair[0], pair[1], corr_pairs[pair])
        for pair in top_corr.index
    ]

    return {
        'heatmap': fig_heatmap,
        'pairplot': fig_pairplot.figure,
        'top_correlations': top_correlations
    }






def generate_categorical_analysis(df: pd.DataFrame, 
                                  categorical_cols: List[str] = None,
                                  target_col: str = None) -> Dict[str, Any]:
    """
    Analyze categorical features and their relationship to target.
    
    Args:
        df: DataFrame to analyze
        categorical_cols: List of categorical column names
        target_col: Target variable column name
        
    Returns:
        Dictionary with 'count_plots' and 'box_plots' keys
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if not col.lower().endswith('_id')]

    if target_col is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['amount', 'price', 'value', 'target']):
                target_col = col
                break
    
    if target_col is None and len(numeric_cols) > 0:
        target_col = numeric_cols[0]

    n_cats = len(categorical_cols)
    n_rows = (n_cats + 1) // 2

    fig_counts, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))

    if n_cats == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(f'{col} Distribution')
        ax.tick_params(axis='x', rotation=45)

    for idx in range(n_cats, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    fig_boxes, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))

    if n_cats == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        sns.boxplot(data=df, x=col, y=target_col, ax=ax)
        ax.set_title(f'{target_col} by {col}')
        ax.tick_params(axis='x', rotation=45)

    for idx in range(n_cats, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    warnings = []
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        small_categories = value_counts[value_counts < 5]
        if len(small_categories) > 0:
            warnings.append(f"{col} has {len(small_categories)} categories with <5 samples")

    return {
        'count_plots': fig_counts,
        'box_plots': fig_boxes,
        'warnings': warnings
    }

    



# df = pd.read_csv('data/sales_data.csv')
# # result = generate_relationship_analysis(df)

# # print("Top 5 Correlations:")
# # for feat1, feat2, corr_val in result['top_correlations']:
# #     print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")

# # result['heatmap'].savefig('output/correlation_heatmap.png')
# # result['pairplot'].savefig('output/pairplot.png')

# result = generate_categorical_analysis(df, target_col='purchase_amount')

# print("Warnings:", result['warnings'])

# result['count_plots'].savefig('output/categorical_counts.png')
# result['box_plots'].savefig('output/categorical_boxplots.png')