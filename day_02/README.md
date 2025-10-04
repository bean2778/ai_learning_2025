# Data Quality Framework

A production-minded data quality assessment tool built as part of a 6-month AI/ML learning journey. This framework provides comprehensive quality checks for numeric, categorical, and temporal data - essential for reliable machine learning pipelines.

## Project Context

This is the **Weekend Project** from Week 1 of my [AI/ML Learning Roadmap](https://github.com/bean2778/ai_learning_2025). The goal was to build a professional data quality tool that demonstrates production-ready habits from day one:

- ✅ Proper error handling and logging
- ✅ Comprehensive testing (unit, integration, performance, property-based)
- ✅ Modern Python packaging with `pyproject.toml`
- ✅ Type hints and documentation throughout
- ✅ Modular, extensible architecture

**Learning Focus:** Building production ML tools, not just running tutorials.

## Features

### Multi-Type Data Analysis

**Numeric Data:**
- Statistical summaries (mean, std dev, min, max)
- Outlier detection using 2-standard deviation rule
- Missing value tracking
- Type validation

**Categorical Data:**
- Cardinality and frequency distribution
- Most/least common categories
- Mixed type handling (strings, integers, categorical floats)
- Empty string vs None distinction

**Temporal Data:**
- Date range identification
- Gap detection in time series
- Frequency/granularity analysis (daily, hourly, irregular)
- Invalid date handling

### Smart Type Detection

Automatic data type inference using statistical sampling:
- Uses Yamane's formula for optimal sample size
- Intelligent precedence rules: temporal > numeric > categorical
- Handles ambiguous cases (e.g., Unix timestamps vs regular integers)

### Production-Ready Features

- **Graceful error handling**: Returns informative error dictionaries instead of crashing
- **Transparent data quality**: Reports what was excluded and why
- **Warning system**: Guides users toward appropriate analyzers
- **Human-readable reports**: Clean formatting for quick insights

## Installation

```bash
# Clone the repository
git clone https://github.com/bean2778/ai_learning_2025.git
cd ai_learning_2025/day_02

# Install in development mode
pip install -e .
```

## Quick Start

### Automatic Type Detection

```python
from data_checker import check_data, generate_report

# The dispatcher automatically detects data type
data = [1, 2, 3, 4, 5, 99, 100]
result = check_data(data)
print(generate_report(result))
```

**Output:**
```
Data Quality Report
==================================================
Data Type: NUMERIC

Total Values: 7
Valid Data Points: 7

Statistical Summary:
------------------------------
  Mean: 30.57
  Std Dev: 42.67
  Min: 1.00
  Max: 100.00

Outliers Detected: 2
  Values: [99, 100]
...
```

### Direct Analysis (When You Know Your Data Type)

```python
from data_checker import check_data_quality, check_categorical_quality, check_temporal_quality

# Numeric analysis
numeric_data = [10, 20, 30, None, 40, 50]
result = check_data_quality(numeric_data)

# Categorical analysis
categories = ['red', 'blue', 'green', 'red', 'blue', 'red']
result = check_categorical_quality(categories)

# Temporal analysis
dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-05']
result = check_temporal_quality(dates)
```

### CSV Integration

```python
from data_checker import analyze_csv_file

# Analyze data directly from CSV
result = analyze_csv_file('sensor_data.csv')
print(generate_report(result))
```

## API Reference

### Core Functions

#### `check_data(data: List[Any]) -> Dict[str, Any]`

Automatically detects data type and routes to appropriate analyzer.

**Parameters:**
- `data`: List of values to analyze

**Returns:**
- Dictionary containing analysis results with `data_type` field

---

#### `check_data_quality(data: List[float]) -> Dict[str, Any]`

Performs numeric data quality checks.

**Returns Dictionary Keys:**
- `data_type`: 'numeric'
- `mean`, `std`, `min`, `max`: Statistical measures
- `outliers`: List of outlier values
- `valid data points`: Count of valid numeric values
- `excluded_non_numeric`: List of non-numeric values found

---

#### `check_categorical_quality(data: List[str]) -> Dict[str, Any]`

Performs categorical data quality checks.

**Returns Dictionary Keys:**
- `data_type`: 'categorical'
- `unique count`: Number of distinct categories
- `frequency distribution`: Dict of category → count
- `most common`, `least common`: Category names
- `warnings`: List of potential issues (e.g., mostly numeric data)

---

#### `check_temporal_quality(data: List[str]) -> Dict[str, Any]`

Performs temporal data quality checks.

**Returns Dictionary Keys:**
- `data_type`: 'temporal'
- `earliest`, `latest`: Date range boundaries
- `detected_frequency`: Most common time delta (e.g., "1 day")
- `gaps_detected`: Number of gaps found
- `gap_after_dates`: List of dates where gaps occur
- `pattern`: 'regular' or 'irregular'

---

#### `generate_report(analysis_results: Dict[str, Any]) -> str`

Generates human-readable report from analysis results.

**Parameters:**
- `analysis_results`: Output from any analyzer function

**Returns:**
- Formatted string report with sections for statistics, quality issues, and warnings

---

#### `analyze_csv_file(filepath: str) -> Dict[str, Any]`

End-to-end CSV file analysis.

**Parameters:**
- `filepath`: Path to CSV file (analyzes first column)

**Returns:**
- Analysis results with additional `source_file` and `total_rows` fields

## Architecture

### Design Patterns

**Strategy Pattern:** Separate analyzer for each data type, allowing:
- Independent evolution of analysis logic
- Clear separation of concerns
- Easy testing and maintenance
- Extensibility for new data types

**Dispatcher Pattern:** Smart routing based on statistical sampling:
1. Uses Yamane's formula to determine sample size
2. Tests sample for numeric, categorical, and temporal validity
3. Routes to analyzer with highest confidence
4. Applies precedence rules for ties

### Project Structure

```
day_02/
├── src/
│   └── data_checker/
│       ├── __init__.py          # Package exports
│       └── data_checker.py      # Core analysis functions
├── tests/
│   └── test_data_checker.py    # Comprehensive test suite
├── pyproject.toml              # Modern Python packaging
├── README.md                   # This file
└── LICENSE                     # GPL-3.0-or-later
```

## Testing

The framework includes comprehensive tests:
- **Unit tests**: Individual function validation
- **Parameterized tests**: Systematic edge case coverage
- **Property-based tests**: Mathematical invariant verification (using Hypothesis)
- **Integration tests**: End-to-end CSV workflow
- **Performance benchmarks**: Scaling analysis (using pytest-benchmark)

```bash
# Run all tests
pytest tests/test_data_checker.py -v

# Run with coverage
pytest tests/test_data_checker.py --cov=data_checker

# Run benchmarks
pytest tests/test_data_checker.py --benchmark-only

# Run specific test
pytest tests/test_data_checker.py::test_dispatcher_detects_numeric -v
```

## Real-World Applications

This framework addresses data quality issues critical for production ML systems:

**Healthcare:**
- Validate patient data completeness before model training
- Detect temporal gaps in medical monitoring data
- Identify outliers in vital signs that may indicate data errors

**Sustainability:**
- Check sensor data quality from IoT devices
- Detect missing readings in environmental monitoring
- Validate temporal consistency for time series forecasting

**Financial Services:**
- Ensure transaction data integrity
- Detect anomalous values in risk metrics
- Validate categorical consistency (account types, statuses)

## Technical Highlights

### Pandas Series Mastery

Leverages pandas for efficient categorical and temporal analysis:

```python
# Frequency distribution in one line
series.value_counts().to_dict()

# Instant cardinality
series.nunique()

# Date parsing with error handling
pd.to_datetime(series, errors='coerce')
```

### Robust Type Handling

Distinguishes between ambiguous data types:
- Small integers (1, 2, 3) → numeric
- Large integers (1704067200) → Unix timestamps → temporal
- Strings ('2024-01-01') → temporal
- Mixed strings/numbers → categorical with warnings

### Property-Based Testing

Uses Hypothesis to discover edge cases:
- Found floating-point precision issues with large numbers
- Discovered numerical overflow with extreme values
- Verified statistical properties across thousands of random inputs

## Dependencies

- **numpy** (>= 1.21.0): Numerical operations
- **pandas** (>= 1.3.0): Data manipulation and analysis

**Development dependencies:**
- pytest: Testing framework
- pytest-benchmark: Performance testing
- hypothesis: Property-based testing

## Development Roadmap

**Current Status:** Core functionality complete ✅

**Potential Enhancements:**
- [ ] Add data type: text (NLP preprocessing checks)
- [ ] Enhanced outlier detection (IQR, isolation forest)
- [ ] Correlation analysis for multi-column datasets
- [ ] Data quality scoring system
- [ ] Export reports to JSON/HTML/PDF
- [ ] Configuration file support for custom thresholds
- [ ] CLI interface for command-line usage

## Learning Journey

This project demonstrates skills developed in Week 1:

**Day 1:** Environment setup, basic data checker with error handling  
**Day 2:** Module development, packaging with pyproject.toml  
**Day 3:** Production error handling and logging  
**Day 4:** Comprehensive testing with pytest and Hypothesis  
**Day 5:** Advanced testing patterns, integration tests, performance benchmarks  
**Day 6-7:** Weekend project - Multi-type analysis framework

**Key Skills Applied:**
- Modern Python packaging
- Production error handling patterns
- Comprehensive testing strategies
- Type hints and documentation
- Modular architecture design
- Real-world data quality considerations

## Contributing

This is a learning project, but suggestions and feedback are welcome! Feel free to:
- Open issues for bugs or enhancement ideas
- Submit PRs with improvements
- Share how you've used or adapted the framework

## License

GPL-3.0-or-later - See LICENSE file for details

## Author

**bean2778** - Senior C++ systems engineer learning ML/AI

Following a 12-month industry-focused roadmap to transition into ML Systems Engineering. This project represents the first major milestone: building production-ready ML tools with proper engineering practices from day one.

Connect: [GitHub](https://github.com/bean2778)

---

*Part of the [AI/ML Learning Roadmap 2025](https://github.com/bean2778/ai_learning_2025) - Building production ML skills through hands-on projects.*
