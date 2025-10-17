import plotly.express as px
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport


df = pd.read_csv('data/product_pricing.csv')
st.title("EDA Dashboard")

tab1, tab2, tab3, tab4=  st.tabs(["Overview", "Distributions", "Relationships", "Auto Report"])

with tab1:
    st.header("Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"**Columns:** {list(df.columns)}")
    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])  # Only show columns with missing values

# Tab 2: You'll build this
with tab2:
    st.header("Distributions")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns")
    else:
        selected_col = st.selectbox("Select a numeric column for histogram:", numeric_cols)

        if selected_col:
            st.subheader(f"Histogram of {selected_col}")
            
            fig = px.histogram(df, x=selected_col, title=f'Distibution of {selected_col}')

            st.plotly_chart(fig)
    
            st.subheader(f"Summary Statistics for {selected_col}")
            st.dataframe(df[selected_col].describe())


# Tab 3: You'll build this  
with tab3:
    st.header("Relationships")

    col1 = st.selectbox("Select first column:", df.columns, key='col1')
    col2 = st.selectbox("Select second column:", df.columns, key='col2')

    if col1 == col2:
        st.warning("Please select two different columns")
    else:
        # Both numeric: scatter plot + correlation
        if df[col1].dtype in ['float64', 'int64'] and df[col2].dtype in ['float64', 'int64']:
            fig = px.scatter(df, x=col1, y=col2, title=f'{col1} vs {col2}')
            st.plotly_chart(fig)
            
            corr = df[col1].corr(df[col2])
            st.metric("Correlation", f"{corr:.3f}")
        
        # Both categorical: grouped bar chart
        elif df[col1].dtype == 'object' and df[col2].dtype == 'object':
            fig = px.histogram(df, x=col1, color=col2, barmode='group', 
                             title=f'{col1} grouped by {col2}')
            st.plotly_chart(fig)
        
        # Mixed: box plot (numeric by categorical)
        else:
            # Figure out which is numeric and which is categorical
            if df[col1].dtype in ['float64', 'int64']:
                numeric_col = col1
                cat_col = col2
            else:
                numeric_col = col2
                cat_col = col1
            
            fig = px.box(df, x=cat_col, y=numeric_col, 
                        title=f'{numeric_col} by {cat_col}')
            st.plotly_chart(fig)




# Tab 4: You'll build this
with tab4:
    st.header("Automated Profiling Report")
    
    if st.button("Generate Report"):
        with st.spinner("Generating profile report... this may take a minute"):
            profile = ProfileReport(df, title="Profile Report", minimal=True)
            st.components.v1.html(profile.to_html(), height=800, scrolling=True)
