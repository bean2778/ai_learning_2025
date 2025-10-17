import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_age_distribution(df) -> plt.Figure:
    sns.histplot(data=df, x='customer_age', kde=True)
    plt.title('Customer Age Distribution')
    plt.axvline(df['customer_age'].mean(), color='red', linestyle='--')

    return plt.gcf()

def plot_income_by_segment(df) -> plt.Figure:
    sns.histplot(data=df, x='annual_income', hue='customer_segment', multiple='dodge')
    plt.title('Income By Segment')
    plt.axvline(df['annual_income'].mean(), color='red', linestyle='--')

    return plt.gcf()

def plot_purchase_distribution(df) -> plt.Figure:
    sns.histplot(data=df, x='purchase_amount', kde=True)
    plt.title('Purchase Amount Distribution')
    # plt.axvline(df['purchase_amount'].quantiler(0.25), color='black', linestyle='--', label='.25')

    return plt.gcf()

def plot_income_vs_purchase(df) -> plt.Figure:
    sns.scatterplot(data=df, x='annual_income', y='purchase_amount', hue='customer_segment')
    sns.regplot(data=df, x='annual_income', y='purchase_amount', scatter=False)

    return plt.gcf()

def plot_age_vs_days_since_signup(df) -> plt.Figure:
    sns.jointplot(data=df, x='customer_age', y='days_since_signup', hue='customer_segment')
    plt.title('Age vs Days Since Signup')

    return plt.gcf()

def plot_purchase_by_category(df) -> plt.Figure:
    sns.boxplot(data=df, x='product_category', y='purchase_amount')
    plt.title('Purchase by Category')

    return plt.gcf()

def plot_income_by_segment_violin(df) -> plt.Figure:
    sns.violinplot(data=df, x='customer_segment', y='annual_income')
    plt.title('Income by Segment')

    return plt.gcf()

def plot_correlation_matrix(df) -> plt.Figure:
    numeric_cols = ['customer_age', 'annual_income', 'purchase_amount', 'days_since_signup']
    corr = df[numeric_cols].corr()  # ✅ Add .corr() here!
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    return plt.gcf()



df = pd.read_csv('data/sales_data.csv')
fig = plot_correlation_matrix(df)

plt.show()