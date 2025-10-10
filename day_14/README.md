# Data Quality Explorer 📊

An interactive web application for automated data quality assessment and visualization. Built with Streamlit and Python, this tool helps data scientists and ML engineers quickly identify data quality issues before model training.

## Features

### 🔍 Automated Quality Detection
- **Missing Values**: Identifies and counts null/NaN values across all columns
- **Outliers**: Detects statistical outliers using 2-standard-deviation rule
- **Invalid Ages**: Flags impossible age values (< 0 or > 120)
- **Negative Amounts**: Catches negative transaction amounts

### 📊 Interactive Visualizations
- **Amount Distribution**: Histogram with outliers highlighted in red
- **Time Series**: Transaction trends with missing data periods marked
- **Age Distribution**: Box plots comparing valid vs invalid ages
- **Category Balance**: Bar chart showing class distribution

### 📈 Summary Statistics
- Descriptive statistics for all numeric columns
- Issue counts with severity indicators
- Actionable recommendations for data cleaning

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd day_14

# Install dependencies
pip install -e .

# Or install manually
pip install streamlit pandas numpy matplotlib
```

## Usage

### Running the Dashboard

```bash
# From the day_14 directory
streamlit run src/dashboard/dashboard.py

# Or specify a port
streamlit run src/dashboard/dashboard.py --server.port 8502
```

The dashboard will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Upload Data**: Click "Browse files" and select a CSV file
2. **Review Preview**: Check the first 10 rows and basic dataset info
3. **Explore Visualizations**: Click through the tabs to see different quality issues
4. **Read Recommendations**: Review suggested actions in the quality report

## Sample Dataset

A sample dataset with deliberate quality issues is included: `data/transactions_with_issues.csv`

**Known issues in sample data:**
- 3 amount outliers (1000, -50, 500)
- 2 impossible ages (150, -5)
- 1 10-day gap in transaction dates (Feb 15-24, 2024)
- Class imbalance in categories

## Project Structure

```
day_14/
├── src/
│   ├── dashboard/
│   │   └── dashboard.py          # Main Streamlit application
│   └── visualizer/
│       └── visualizer.py         # Matplotlib plotting functions
├── data/
│   └── transactions_with_issues.csv
├── pyproject.toml                # Package configuration
└── README.md
```

## Technical Details

### Built With
- **Streamlit 1.28+**: Interactive web framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Static visualizations

### Data Quality Checks

**Outlier Detection (2σ Rule)**
```python
outliers = values > (mean + 2*std) OR values < (mean - 2*std)
```

**Age Validation**
```python
invalid = age < 0 OR age > 120
```

**Missing Data**
```python
missing = df.isnull().sum().sum()  # Total across all columns
```

## Use Cases

### Healthcare ML Pipelines
- Validate patient vital signs data before training diagnostic models
- Detect impossible values in medical records
- Identify data collection gaps

### Financial Analytics
- Catch negative transaction amounts (data entry errors)
- Detect fraudulent outliers in spending patterns
- Ensure complete time series data

### IoT/Sustainability Systems
- Validate sensor readings (temperature, humidity, etc.)
- Detect equipment malfunctions through outliers
- Identify communication failures (missing timestamps)

## Learning Context

This project was built as part of a 12-month AI/ML learning roadmap, demonstrating:
- **Week 2 Skills**: NumPy, pandas, data cleaning, validation, optimization
- **Production Mindset**: Quality checks before ML model training
- **Interactive Dashboards**: Making data insights accessible to non-technical stakeholders

## Extending the Dashboard

### Add New Quality Checks

Add a detection function:
```python
def find_duplicates(df: pd.DataFrame) -> int:
    return df.duplicated().sum()
```

Use it in the metrics section:
```python
duplicate_count = find_duplicates(df)
st.metric("Duplicate Rows", duplicate_count)
```

### Add New Visualizations

Create visualization in `visualizer/visualizer.py`:
```python
def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    # ... plotting code ...
    return fig
```

Add tab in `dashboard.py`:
```python
with tab5:
    st.subheader("Correlation Matrix")
    fig = plot_correlation_matrix(df)
    st.pyplot(fig)
```

## Future Enhancements

- [ ] Export quality report to PDF
- [ ] Compare multiple datasets side-by-side
- [ ] Custom outlier threshold selection
- [ ] Automated data cleaning suggestions
- [ ] Integration with data validation frameworks (Great Expectations)
- [ ] Support for other file formats (Excel, JSON, Parquet)

## Contributing

This is a learning project, but suggestions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - feel free to use this for your own projects

## Acknowledgments

- Built following industry-focused ML engineering practices
- Visualization patterns inspired by production ML pipelines
- Part of a comprehensive ML systems engineering learning journey

## Contact

Dave Bean  
bean2778@gmail.com
[\[Portfolio Link\]](https://github.com/bean2778/ai_learning_2025)

---

**Week 2 Complete!** 🎉  
This dashboard demonstrates the complete data preprocessing pipeline essential for production ML systems.
