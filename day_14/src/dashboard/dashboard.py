import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src directory to path so we can import visualizer
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from visualizer.visualizer import (
    plot_amount_distribution,
    plot_time_series,
    plot_age_distribution,
    plot_category_balance
)


def find_missing(df: pd.DataFrame) -> int:
    return df.isnull().sum().sum()

def find_outliers(df: pd.DataFrame) -> int:
    if 'amount' in df.columns:
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        outlier_mask = (df['amount'] > mean_amount + 2*std_amount) | (df['amount'] < mean_amount - 2*std_amount)
        return outlier_mask.sum()
    return 0

def find_fake_ages(df:pd.DataFrame) -> int:
    if 'customer_age' in df.columns:
        fake_ages = (df['customer_age'] > 120) | (df["customer_age"] < 0)
        return fake_ages.sum()
    return 0

def find_negative_amounts(df:pd.DataFrame) -> int:
    if 'amount' in df.columns:
        neg_mask = df['amount'] < 0
        return neg_mask.sum()
    return 0

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Data Quality Explorer",
    page_icon="📊",
    layout="wide"  # Use full width
)

st.title("📊 Data Quality Explorer")
st.write("Upload a CSV file to analyze data quality issues")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    st.header("📋 Data Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(10))  # Interactive table
    
    with col2:
        st.subheader("Dataset Info")
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        st.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("⚙️ Analysis Settings")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)

    st.header("📊 Data Quality Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Amount Distribution", 
        "📈 Time Series", 
        "👤 Age Distribution", 
        "📊 Category Balance"
    ])

    with tab1:
        # 1. Add a subheader
        st.subheader("Transaction Amount distribution")
        # 2. Call plot_amount_distribution(df) and store in variable
        fig = plot_amount_distribution(df)
        # 3. Use st.pyplot() to display it
        st.pyplot(fig)
        # 4. Add a caption explaining what the red bars mean
        st.caption("Red bars indicate outliers")

    with tab2:
        st.subheader("Transaction Over Time")
        fig = plot_time_series(df)
        st.pyplot(fig)
        st.caption("Red shaded areas indicate missing data periods")

    with tab3:
        st.subheader("Age Distribution")
        fig = plot_age_distribution(df)
        st.pyplot(fig)
        st.caption("Box plot shows valid vs impossible ages")

    with tab4:
        st.subheader("Category Balance")
        fig = plot_category_balance(df)
        st.pyplot(fig)
        st.caption("Shows class imbalance that could affect ML models")

    # Summary Statistics Section
    st.header("📈 Summary Statistics & Quality Report")

    missing_count = find_missing(df)
    outlier_count = find_outliers(df)
    fake_age_count = find_fake_ages(df)
    neg_count = find_negative_amounts(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Numeric Columns")
        st.dataframe(df.describe())
        
    with col2:
        st.subheader("Data Quality Issues")
        st.metric("Missing Values", missing_count)
        st.metric("Outlier (amount)", outlier_count)
        st.metric("Invalid Ages", fake_age_count)
        st.metric("Negative Amounts", neg_count)
        
    with col3:
        st.subheader("Recommendations")
        # TODO: Suggest fixes based on issues found
        if missing_count > 0:
            st.warning(f"⚠️ {missing_count} missing values detected. Consider imputation or removal.")

        if outlier_count > 0:
            st.warning(f"⚠️ {outlier_count} outliers found. Investigate if these are errors or valid extreme values.")
        
        if fake_age_count > 0:
            st.warning(f"⚠️ {fake_age_count} invalid ages found. Ages should be between 0-120.")
        
        if neg_count > 0:
            st.warning(f"⚠️ {neg_count} negative amounts found. Check for data entry errors.")
        
        # If everything is clean
        if missing_count + outlier_count + fake_age_count + neg_count == 0:
            st.success("✅ No major data quality issues detected!")

    st.header("📥 Export Report")

    # Create a summary text report
    report = f"""
    Data Quality Report
    ==================
    Dataset: {uploaded_file.name}
    Rows: {len(df)}
    Columns: {len(df.columns)}

    Issues Found:
    - Missing Values: {missing_count}
    - Outliers: {outlier_count}
    - Invalid Ages: {fake_age_count}
    - Negative Amounts: {neg_count}
    """

    st.download_button(
        label="Download Report",
        data=report,
        file_name="data_quality_report.txt",
        mime="text/plain"
    )
        

    
else:
    st.info("👆 Upload a CSV file to get started")

