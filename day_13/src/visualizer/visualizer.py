import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_amount_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Makes a figure that has a histogram showing the outliers of amount

    Args:
        df: pd.DataFrame of transactions

    Returns:
        plt.Figure of a histogram with outliers colored red
    """
    df_copy = df.copy()
    median = df_copy['amount'].median()
    mean = df_copy['amount'].mean()
    std = df_copy['amount'].std()
    df_copy['is_outlier'] = (df_copy['amount'] < (mean - 2 * std)) | \
        (df_copy['amount'] > (mean + 2 * std))
    
    
    fig, ax = plt.subplots()
    
    outliers = df_copy[df_copy['is_outlier']]['amount']
    regular = df_copy[~df_copy['is_outlier']]['amount']

    

    ax.hist(regular, bins=30, edgecolor='black', alpha=0.7, label='Normal')

    ax.hist(outliers, bins=30,color='red', alpha=0.7, label='Outlier')
    ax.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')

    ax.set_xlabel('Amount')
    ax.set_ylabel('Frequency')
    ax.set_title('Tranaction Amount Distribution')
    ax.legend()

    return fig

def plot_time_series(df: pd.DataFrame) -> plt.Figure:
    """
    Makes a figure that will show a line plot of dates

    Args:
        df: pd.DataFrame of transactions

    Returns:
        plt.Figure of line plot showing missing dates
    """ 
    df['date'] = pd.to_datetime(df['date'])
    full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
    missing_dates = full_date_range.difference(df['date'])
    
    fig, ax = plt.subplots()

    ax.plot(df['date'], df['amount'])

    ax.axvspan(missing_dates.min(), missing_dates.max(), alpha=0.3, color='red', label='Missing Data')

    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    ax.set_title('Transaction Amounts Over Time')
    return fig

def plot_age_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Make a figure to highlight impossible ages

    Args:
        df: pd.DataFrame of transactions

    Returns:
        plt.Figure of histogram showing impossible ages
    """
    real_ages_mask = (df['customer_age'] > 0) & (df['customer_age'] < 150)
    fake_ages_mask = ~real_ages_mask

    real_ages = df[real_ages_mask]['customer_age']
    fake_ages = df[fake_ages_mask]['customer_age']

    fig, ax = plt.subplots()
    ax.boxplot([real_ages, fake_ages], labels=['Real Ages', 'Impossible Ages'])
    ax.set_ylabel('Age')
    ax.set_title('Customer Age Distribution')
    return fig

def plot_category_balance(df: pd.DataFrame) -> plt.Figure:
    """
    Make a figure to highlight impossible ages

    Args:
        df: pd.DataFrame of transactions

    Returns:
        plt.Figure of histogram showing impossible ages
    """

    category_counts = df['category'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(category_counts.index, category_counts.values)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Category Distribution')
    
    return fig

def create_dashboard(df: pd.DataFrame) -> plt.Figure:
    """Create 2x2 dashboard with all visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ============ TOP-LEFT: Amount Distribution ============
    ax = axes[0, 0]
    median = df['amount'].median()
    mean = df['amount'].mean()
    std = df['amount'].std()
    
    outliers_mask = (df['amount'] < (mean - 2 * std)) | (df['amount'] > (mean + 2 * std))
    outliers = df[outliers_mask]['amount']
    regular = df[~outliers_mask]['amount']
    
    ax.hist(regular, bins=30, edgecolor='black', alpha=0.7, label='Normal')
    ax.hist(outliers, bins=30, color='red', alpha=0.7, label='Outlier')
    ax.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    ax.set_xlabel('Amount')
    ax.set_ylabel('Frequency')
    ax.set_title('Transaction Amount Distribution')
    ax.legend()
    
    # ============ TOP-RIGHT: Time Series ============
    ax = axes[0, 1]
    df['date'] = pd.to_datetime(df['date'])
    ax.plot(df['date'], df['amount'], marker='o', markersize=3, label='Transactions')
    ax.axvspan('2024-02-15', '2024-02-24', alpha=0.3, color='red', label='Missing Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    ax.set_title('Transaction Amounts Over Time')
    ax.legend()
    
    # ============ BOTTOM-LEFT: Age Distribution ============
    ax = axes[1, 0]

    
    real_ages_mask = (df['customer_age'] > 0) & (df['customer_age'] < 150)
    fake_ages_mask = ~real_ages_mask

    real_ages = df[real_ages_mask]['customer_age']
    fake_ages = df[fake_ages_mask]['customer_age']

    ax.boxplot([real_ages, fake_ages], labels=['Real Ages', 'Impossible Ages'])
    ax.set_ylabel('Age')
    ax.set_title('Customer Age Distribution')
    ax.legend()
    
    # ============ BOTTOM-RIGHT: Category Balance ============
    ax = axes[1, 1]
    
    category_counts = df['category'].value_counts()
    ax.bar(category_counts.index, category_counts.values)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Category Distribution')
    
    return fig


    
    
df = pd.read_csv('transactions_with_issues.csv')
create_dashboard(df)
plt.show()