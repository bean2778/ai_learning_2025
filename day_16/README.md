# EDA Toolkit - Automated Exploratory Data Analysis

Comprehensive EDA automation functions for machine learning projects. Generate complete exploratory analysis reports with single function calls.

## Overview

This toolkit provides three core functions that automate the exploratory data analysis phase of ML projects:

1. **`generate_numeric_summary()`** - Analyze all numeric features
2. **`generate_relationship_analysis()`** - Explore feature correlations
3. **`generate_categorical_analysis()`** - Understand categorical patterns

## Features

- ✅ **Auto-detection** - Works on any CSV without knowing schema
- ✅ **Manual override** - Specify columns explicitly when needed
- ✅ **Comprehensive analysis** - Multiple visualizations per function
- ✅ **Actionable insights** - Outliers, correlations, warnings included
- ✅ **Production-ready** - Returns structured dictionaries for automation

## Installation

```bash
# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from eda_toolkit.analyzer import (
    generate_numeric_summary,
    generate_relationship_analysis,
    generate_categorical_analysis
)

# Load your data
df = pd.read_csv('your_data.csv')

# Run complete EDA workflow
numeric_result = generate_numeric_summary(df)
relationship_result = generate_relationship_analysis(df)
categorical_result = generate_categorical_analysis(df, target_col='purchase_amount')

# Save outputs
numeric_result['figure'].savefig('reports/numeric_summary.png')
relationship_result['heatmap'].savefig('reports/correlations.png')
relationship_result['pairplot'].savefig('reports/pairplot.png')
categorical_result['count_plots'].savefig('reports/categories.png')

# Print insights
print("Outliers detected:", numeric_result['outliers'])
print("\nTop 5 Correlations:")
for feat1, feat2, corr in relationship_result['top_correlations']:
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")
print("\nWarnings:", categorical_result['warnings'])
```

## Function Documentation

### `generate_numeric_summary(df, numeric_cols=None)`

Analyzes all numeric columns in the dataset.

**Parameters:**
- `df` (pd.DataFrame): Dataset to analyze
- `numeric_cols` (List[str], optional): Specific columns to analyze. If None, auto-detects all numeric columns and filters out ID columns.

**Returns:**
Dictionary with:
- `'figure'` (plt.Figure): Multi-panel histogram plot with KDE curves
- `'outliers'` (dict): Count of outliers per column (using 2σ rule)

**What it shows:**
- Distribution of each numeric feature
- Mean and median reference lines
- Outliers highlighted in analysis

**Example:**
```python
result = generate_numeric_summary(df)

# Access outputs
fig = result['figure']
outlier_counts = result['outliers']

# Save and inspect
fig.savefig('numeric_summary.png')
print("Outliers per column:", outlier_counts)
```

---

### `generate_relationship_analysis(df, numeric_cols=None)`

Explores relationships between all numeric features.

**Parameters:**
- `df` (pd.DataFrame): Dataset to analyze
- `numeric_cols` (List[str], optional): Specific columns to analyze. If None, auto-detects.

**Returns:**
Dictionary with:
- `'heatmap'` (plt.Figure): Correlation matrix heatmap
- `'pairplot'` (plt.Figure): Pairwise scatter plots with distributions
- `'top_correlations'` (List[Tuple]): Top 5 strongest correlations as (feature1, feature2, correlation)

**What it shows:**
- Correlation strength between all feature pairs
- Scatter plots showing relationship patterns
- Distribution of individual features (diagonal of pair plot)

**Example:**
```python
result = generate_relationship_analysis(df)

# Print top correlations
print("Top 5 Correlations:")
for feat1, feat2, corr in result['top_correlations']:
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")

# Save visualizations
result['heatmap'].savefig('correlation_heatmap.png')
result['pairplot'].savefig('pairplot.png')
```

---

### `generate_categorical_analysis(df, categorical_cols=None, target_col=None)`

Analyzes categorical features and their relationship to target variable.

**Parameters:**
- `df` (pd.DataFrame): Dataset to analyze
- `categorical_cols` (List[str], optional): Specific columns to analyze. If None, auto-detects object/category columns.
- `target_col` (str, optional): Target variable name. If None, infers from keywords ('amount', 'price', 'value', 'target') or uses first numeric column.

**Returns:**
Dictionary with:
- `'count_plots'` (plt.Figure): Frequency distribution of each category
- `'box_plots'` (plt.Figure): Target distribution per category
- `'warnings'` (List[str]): Warnings about small categories (<5 samples)

**What it shows:**
- Frequency of each category value
- How target variable differs across categories
- Potential data quality issues (rare categories)

**Example:**
```python
result = generate_categorical_analysis(df, target_col='purchase_amount')

# Check warnings
if result['warnings']:
    print("Data quality issues:")
    for warning in result['warnings']:
        print(f"  - {warning}")

# Save visualizations
result['count_plots'].savefig('category_frequencies.png')
result['box_plots'].savefig('target_by_category.png')
```

---

## Usage Patterns

### Auto-Detection Mode (Recommended for exploration)

```python
# Just pass the DataFrame - functions figure out what to analyze
numeric_result = generate_numeric_summary(df)
relationship_result = generate_relationship_analysis(df)
categorical_result = generate_categorical_analysis(df)
```

### Manual Specification Mode (When you know what you want)

```python
# Explicitly specify columns
numeric_result = generate_numeric_summary(
    df, 
    numeric_cols=['age', 'income', 'purchase_amount']
)

relationship_result = generate_relationship_analysis(
    df,
    numeric_cols=['age', 'income', 'purchase_amount', 'days_since_signup']
)

categorical_result = generate_categorical_analysis(
    df,
    categorical_cols=['customer_segment', 'product_category'],
    target_col='purchase_amount'
)
```

### Complete EDA Pipeline

```python
import pandas as pd
from eda_toolkit.analyzer import *

# Load data
df = pd.read_csv('data.csv')

# Run complete analysis
results = {}
results['numeric'] = generate_numeric_summary(df)
results['relationships'] = generate_relationship_analysis(df)
results['categorical'] = generate_categorical_analysis(df)

# Save all outputs
results['numeric']['figure'].savefig('reports/1_numeric_summary.png', dpi=300, bbox_inches='tight')
results['relationships']['heatmap'].savefig('reports/2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
results['relationships']['pairplot'].savefig('reports/3_pairplot.png', dpi=300, bbox_inches='tight')
results['categorical']['count_plots'].savefig('reports/4_category_frequencies.png', dpi=300, bbox_inches='tight')
results['categorical']['box_plots'].savefig('reports/5_target_by_category.png', dpi=300, bbox_inches='tight')

# Generate text report
with open('reports/eda_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("## OUTLIER DETECTION\n")
    for col, count in results['numeric']['outliers'].items():
        f.write(f"  {col}: {count} outliers detected\n")
    
    f.write("\n## TOP CORRELATIONS\n")
    for feat1, feat2, corr in results['relationships']['top_correlations']:
        f.write(f"  {feat1} <-> {feat2}: {corr:.3f}\n")
    
    f.write("\n## DATA QUALITY WARNINGS\n")
    if results['categorical']['warnings']:
        for warning in results['categorical']['warnings']:
            f.write(f"  - {warning}\n")
    else:
        f.write("  No warnings\n")

print("Complete EDA report generated in reports/ directory")
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/test_analyzer.py -v

# With output
pytest tests/test_analyzer.py -v -s

# Specific test
pytest tests/test_analyzer.py::test_numeric_summary_returns_dict -v
```

## Project Structure

```
eda_toolkit/
├── src/
│   └── eda_toolkit/
│       ├── __init__.py          # Package exports
│       └── analyzer.py          # Main EDA functions
├── tests/
│   └── test_analyzer.py         # Comprehensive test suite
├── data/
│   └── sales_data.csv           # Example dataset
├── output/                       # Generated plots saved here
├── pyproject.toml               # Package configuration
└── README.md                    # This file
```

## Real-World Applications

### Healthcare ML Pipeline
```python
# Analyze patient data before training diagnostic model
patient_df = pd.read_csv('patient_records.csv')

# Quick EDA to identify data issues
numeric_result = generate_numeric_summary(patient_df)
print("Outliers (potential data errors):", numeric_result['outliers'])

# Find which features correlate with readmission risk
relationship_result = generate_relationship_analysis(patient_df)
print("Key predictors:", relationship_result['top_correlations'][:3])
```

### Financial Services
```python
# Analyze transaction data for fraud detection
transaction_df = pd.read_csv('transactions.csv')

# Understand normal transaction patterns
categorical_result = generate_categorical_analysis(
    transaction_df, 
    target_col='transaction_amount'
)

# Identify suspicious merchant categories
if categorical_result['warnings']:
    print("Categories with few samples (potential fraud):", 
          categorical_result['warnings'])
```

### IoT Sensor Analysis
```python
# Analyze wind turbine sensor data
sensor_df = pd.read_csv('turbine_sensors.csv')

# Detect sensor malfunctions (outliers)
numeric_result = generate_numeric_summary(sensor_df)
failing_sensors = [col for col, count in numeric_result['outliers'].items() 
                   if count > 10]
print("Sensors requiring maintenance:", failing_sensors)

# Identify correlated sensors (redundancy)
relationship_result = generate_relationship_analysis(sensor_df)
high_corr = [(f1, f2, c) for f1, f2, c in relationship_result['top_correlations'] 
             if abs(c) > 0.9]
print("Redundant sensors:", high_corr)
```

## Design Philosophy

### Why Auto-Detection?

Production ML systems often process user-uploaded data with unknown schemas. Auto-detection allows:
- One-line analysis of any dataset
- No need to inspect column names first
- Works in automated pipelines

### Why Return Dictionaries?

Structured returns enable:
- Programmatic access to insights
- Integration with automated reporting
- Clear, self-documenting outputs

### Why Multiple Visualizations?

Each function provides multiple related views because:
- Different plots reveal different patterns
- Comprehensive analysis prevents missed insights
- Single function call = complete picture

## Performance Considerations

**Suitable for:** Datasets up to ~100,000 rows

**For larger datasets:**
- Use manual column specification (analyze subset of features)
- Sample data before analysis
- Consider using Dask for distributed computation

**Pair plot performance:** Pair plots scale as O(n²) with number of columns. For >10 numeric columns, consider specifying a subset.

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Contributing

This is a learning project demonstrating production ML patterns. Feel free to:
- Add new analysis functions
- Improve auto-detection logic
- Add statistical tests
- Create interactive plot options

## License

MIT License - See LICENSE file for details

## Development Journey

This toolkit was built as part of Day 16 of a 6-month AI/ML learning roadmap, focusing on:
- Production-ready code patterns
- Automated EDA workflows
- Discovery-based learning approach

**Key learnings:**
- Auto-detection vs manual specification trade-offs
- When to use seaborn vs matplotlib
- How professional EDA libraries work
- Importance of structured returns for automation

## Acknowledgments

Built following industry best practices from:
- pandas-profiling architecture patterns
- sweetviz design philosophy
- Production ML system requirements

---

**Status:** Production-ready EDA toolkit ✅  
**Created:** Day 16 of ML Learning Journey  
**Purpose:** Automate repetitive exploratory analysis tasks