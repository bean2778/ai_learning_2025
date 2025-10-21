import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import os

# Get the directory where dashboard.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to day_19/
parent_dir = os.path.dirname(current_dir)

model = joblib.load(os.path.join(parent_dir, 'storage_model.pkl'))
scaler = joblib.load(os.path.join(parent_dir, 'storage_scaler.pkl'))
feature_names = joblib.load(os.path.join(parent_dir, 'feature_names.pkl'))

st.title("Renewable Energy Storage Investment Model")
candidates = pd.read_csv(os.path.join(parent_dir, 'data_storage_candidates.csv'))

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Country Analysis", "Predictions", "Technical Details"])

with tab1:
    st.header("Model Overview")

    st.subheader("Problem Statement")
    st.text("Energy storage companies and infrastructure investors"
            " need to identify which countries represent the best " 
            "opportunities for battery storage investments based on "
            "renewable penetration, growth trajectory, and energy mix complexity.")

    st.subheader("Top Investment Opportunities")

    top_countries = candidates.nlargest(10, 'storage_score')


    st.dataframe(top_countries[['country', 'storage_score', 'avg_renewable_share', 'growth_rate_pct']])
    column_config={
        'country': 'Country',
        'storage_score': 'Storage Score',
        'avg_renewable_share': 'Renewable Share',
        'growth_rate_pct': 'Growth Rate %'
    }

    fig = px.bar(
        top_countries,
        x='country',
        y='storage_score',
        title = 'Top 10 Countries for Storage Investment',
        labels={'storage_score': 'Storage Need Score', 'country': 'Country'},
        color='storage_score',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig)

with tab2:
    st.header("Country Analysis")

    all_countries = candidates['country'].tolist()

    selected_countries = st.multiselect(
        "Select countries to compare:",
        options=all_countries,
        default=['Netherlands', 'Iceland', 'Norway', 'Denmark']
    )

    # Filter selected countries
    filtered_data = candidates[candidates['country'].isin(selected_countries)]

    # Metric
    metric = st.selectbox(
        "Choose metric to visualize:",
        options=['storage_score', 'avg_renewable_share', 'growth_rate_pct', 'total_production']
    )

    # Bar chart comparison
    fig = px.bar(
        filtered_data,
        x='country',
        y=metric,
        title=f'{metric.replace("_", " ").title()} Comparison',
        color='country'
    )
    st.plotly_chart(fig)

with tab3:
    st.header("Storage Need Prediction")
    
    st.write("Adjust renewable energy mix percentages to predict storage need score:")
    
    # Input sliders
    resource_type = {}
    resource_type['hydro'] = st.slider('Hydro %', 0.0, 100.0, 20.0, 1.0)
    resource_type['wind'] = st.slider('Wind %', 0.0, 100.0, 30.0, 1.0)
    resource_type['solar'] = st.slider('Solar %', 0.0, 100.0, 25.0, 1.0)
    resource_type['geo'] = st.slider('Geothermal %', 0.0, 100.0, 5.0, 1.0)
    resource_type['other'] = st.slider('Other Renewables %', 0.0, 100.0, 20.0, 1.0)
    
    # Show total
    total_pct = sum(resource_type.values())
    st.metric("Total Percentage", f"{total_pct:.1f}%")
    
    if abs(total_pct - 100.0) > 1.0:
        st.warning("⚠️ Percentages should add up to 100%")
    
    # Create input array (must match feature_names order)
    input_data = np.array([[
        resource_type['hydro'],
        resource_type['wind'],
        resource_type['solar'],
        resource_type['geo'],
        resource_type['other']
    ]])
    
    # IMPORTANT: Scale the input using the same scaler from training
    input_scaled = scaler.transform(input_data)
    
    # Make prediction on SCALED data
    prediction = model.predict(input_scaled)[0]
    
    # Display prediction
    st.subheader("Predicted Storage Need Score")
    st.metric(
        label="Storage Score",
        value=f"{prediction:.2f}",
        help="Higher scores indicate greater need for energy storage infrastructure"
    )
    
    # Show comparison to countries
    st.write("**How does this compare to actual countries?**")
    # candidates = pd.read_csv('data_storage_candidates.csv')
    similar_countries = candidates.iloc[(candidates['storage_score'] - prediction).abs().argsort()[:3]]
    st.dataframe(similar_countries[['country', 'storage_score']])

with tab4:
    st.header("Technical Details")
    
    st.subheader("Data Sources")
    st.write("""
    - **Dataset:** Global Renewable Energy Production Statistics (2010-2022)
    - **Size:** 181,915 rows × 12 columns
    - **Coverage:** 52 countries, 6 renewable energy types, 156 months
    - **Source:** Energy statistics aggregated by country, product, and time period
    """)
    
    st.subheader("Business Problem")
    st.write("""
    Energy storage companies and infrastructure investors need to identify which 
    countries represent the best opportunities for battery storage investments based 
    on renewable penetration, growth trajectory, and energy mix complexity.
    """)
    
    st.subheader("Methodology")
    
    st.write("**Storage Score Calculation:**")
    st.latex(r"Storage\_Score = 0.4 \times Share_{norm} + 0.4 \times Growth_{norm} + 0.2 \times Diversity_{norm}")
    
    st.write("""
    Where:
    - **Share_norm:** Normalized average renewable energy share (0-1)
    - **Growth_norm:** Normalized growth rate from 2010 to 2022 (0-1)
    - **Diversity_norm:** Number of renewable types present (normalized)
    """)
    
    st.subheader("Model Performance")
    
    st.write("**Model Type:** Random Forest Regressor")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", "0.948")
    with col2:
        st.metric("Test Set Size", "20%")
    
    st.write("""
    **Model Features:**
    - Hydro percentage
    - Wind percentage
    - Solar percentage
    - Geothermal percentage
    - Other renewables percentage
    
    **Target Variable:** Storage need score (0-100 scale)
    """)
    
    st.subheader("Preprocessing Pipeline")
    st.write("""
    1. **Data Cleaning:** Convert time strings to datetime objects, handle missing values
    2. **Filtering:** Remove aggregate categories, keep only discrete renewable sources
    3. **Aggregation:** Calculate country-level totals and time ranges
    4. **Feature Engineering:** Calculate renewable mix percentages and composite scores
    5. **Normalization:** Min-max scaling of all metrics to 0-1 range
    """)
    
    st.subheader("Deployment Architecture")
    st.write("""
    **Three deployment patterns implemented:**
    
    1. **Batch Processing:** 8,033 predictions/second for large-scale analysis
    2. **REST API:** Real-time predictions via FastAPI with automatic documentation
    3. **Edge Deployment:** 89.4 KB ONNX model for browser/mobile deployment
    
    **Current Dashboard:** Deployed on Streamlit Cloud with interactive visualizations
    """)
    
    st.subheader("Key Insights")
    st.write("""
    - **Top Investment Opportunities:** Netherlands, Iceland, Norway show highest storage scores
    - **Growth Leaders:** Countries with 400%+ renewable growth (Netherlands: 838%)
    - **High Penetration:** Nordic countries lead with 15-30% renewable share
    - **Emerging Markets:** Eastern Europe showing rapid growth rates
    """)
    
    st.subheader("GitHub Repository")
    st.write("Full code, data, and documentation available at: [Add your GitHub link]")