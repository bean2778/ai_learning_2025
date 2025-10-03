import pytest
import pandas as pd
import math
import tempfile
import os
import numpy.testing as npt
from data_checker import check_data_quality, analyze_csv_file, check_categorical_quality, check_temporal_quality, check_data
from hypothesis import given, strategies as st

# Group 1: basic calculations
@pytest.mark.parametrize("data, expected_mean, expected_valid_count", [
    ([1, 2, 3, 4, 5], 3.0, 5),
    ([42], 42.0, 1),
    ([1, 2, 3, 4, 5, 6, 7], 4.0, 7),
], ids=["simple_sequence", "single_value", "longer_sequence"])
def test_statistical_calculations(data, expected_mean, expected_valid_count):
    result = check_data_quality(data)
    npt.assert_allclose(result['mean'], expected_mean, rtol=1e-7)
    assert result['valid data points'] == expected_valid_count


# Group 2: outlier detection
@pytest.mark.parametrize("data, expected_outliers", [
    ([1, 2, 3, 4, 6, 88, 7, 8, 10, 99], [99]),
    ([1, 2, 3, 4, 6, 7], []),   
], ids=["has_outliers", "no_outliers"])
def test_outlier_detection(data, expected_outliers):
    result = check_data_quality(data)
    assert result['outliers'] == expected_outliers

# Group 3: Tests that verify error handling
@pytest.mark.parametrize("data,expected_error", [
    ([], 'No valid data points to analyze'),
], ids=["empty_data"])
def test_error_cases(data, expected_error):
    result = check_data_quality(data)
    
    assert 'error' in result
    assert result['error'] == expected_error
    assert result['valid data points'] == 0

# Group 4: Tests that verify data cleaning/exclusion
@pytest.mark.parametrize("data,expected_excluded_non_numeric,expected_excluded_nan_none", [
    ([1, 2, 'abc', None, 3.5], ['abc'], [None]),
], ids=["mixed_types"])
def test_data_exclusion(data, expected_excluded_non_numeric, expected_excluded_nan_none):
    result = check_data_quality(data)
    
    assert result['excluded_non_numeric'] == expected_excluded_non_numeric
    assert result['excluded_nan_none'] == expected_excluded_nan_none



def test_returns_error_dict_when_appropriate():
    data = []
    result = check_data_quality(data)

    assert('error' in result)

def test_no_crashes_on_bad_input():
    with pytest.raises(ValueError):
        data = ['a', [1, 2, 3], 4]
        result = check_data_quality(data)

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100), min_size=1))
def test_mean_always_within_range(data):
    result = check_data_quality(data)
    
    # Skip if we got an error result (no valid data)
    if 'error' in result:
        return
    
    # Get the clean data bounds
    valid_data = [x for x in data if x is not None and not pd.isna(x)]
    if len(valid_data) == 0:
        return
        
    min_val = min(valid_data)
    max_val = max(valid_data)
    mean_val = result['mean']
    
    # The mathematical property that should ALWAYS be true
    assert min_val <= mean_val <= max_val or math.isclose(mean_val, min_val) or math.isclose(mean_val, max_val)

def test_mean_calculation_precise():
    data = [1.0, 2.0, 3.0]
    result = check_data_quality(data)
    
    # Instead of: assert result['mean'] == 2.0
    npt.assert_allclose(result['mean'], 2.0, rtol=1e-7, atol=1e-8)

@pytest.mark.parametrize("size", [10, 100, 1000, 10000])
def test_performance_scaling(benchmark, size):
    """Measure how check_data_quality scales with data size"""
    data = list(range(size))
    result = benchmark(check_data_quality, data)
    assert result['valid data points'] == size
    assert 'mean' in result

@pytest.mark.parametrize("data_characteristics,data_generator", [
    ("clean", lambda: list(range(1000))),
    ("with_nans", lambda: [1, 2, None, 4] * 250),
    ("with_outliers", lambda: list(range(900)) + [999999] * 100),
], ids=["clean", "with_nans", "with_outliers"])
def test_performance_by_data_type(benchmark, data_characteristics, data_generator):
    """See if data quality affects performance"""
    data = data_generator()
    result = benchmark(check_data_quality, data)
    assert result is not None

def test_complete_csv_workflow():
    """Integration test: file → analysis → report"""
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write('values\n')
        f.write('1\n2\n3\n4\n5\n')
        temp_path = f.name
    
    try:
        # Test the complete workflow
        result = analyze_csv_file(temp_path)
        
        # Verify end-to-end correctness
        assert result['mean'] == 3.0
        assert result['valid data points'] == 5
        assert result['source_file'] == temp_path
        assert result['total_rows'] == 5
        
    finally:
        # Cleanup
        os.unlink(temp_path)

def test_workflow_handles_missing_file():
    """Integration test: verify error handling works end-to-end"""
    result = analyze_csv_file('nonexistent.csv')
    
    assert 'error' in result
    assert 'not found' in result['error'].lower()

def test_workflow_handles_bad_data():
    """Integration test: messy CSV data handled gracefully"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write('values\n')
        f.write('1\n2\nabc\n4\n5\n')  # Bad data in CSV
        temp_path = f.name
    
    try:
        result = analyze_csv_file(temp_path)
        
        # Should handle bad data gracefully
        assert result['valid data points'] == 4  # abc excluded
        assert 'excluded_non_numeric' in result
        
    finally:
        os.unlink(temp_path)

def test_debug_what_happens():
    """Debug: trace the exact flow"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write('values\n')
        f.write('1\n2\nabc\n4\n5\n')
        temp_path = f.name
    
    try:
        df = pd.read_csv(temp_path)
        print(f"\n1. DataFrame:\n{df}")
        print(f"   Column dtype: {df['values'].dtype}")
        
        raw_data = df.iloc[:, 0].tolist()
        print(f"\n2. Raw data: {raw_data}")
        print(f"   Types: {[type(x) for x in raw_data]}")
        
        numeric_data = pd.to_numeric(df.iloc[:, 0], errors='coerce').tolist()
        print(f"\n3. After to_numeric: {numeric_data}")
        print(f"   Types: {[type(x) for x in numeric_data]}")
        
        result = check_data_quality(numeric_data)
        print(f"\n4. Final result: {result}")
        
    finally:
        os.unlink(temp_path)


# categorical tests

def test_basic_cat():
    data = ['a', 'b', 'b', 'b', 'c', 'c']
    result = check_categorical_quality(data)
    assert result['unique count'] == 3
    assert result['most common'] == 'b'
    assert result['least common'] == 'a'
    assert result['frequency distribution']['b'] == 3

@pytest.mark.parametrize("data", [
    ([1.3, 2.3, 3.4, 4.5, 5.7]),
    ([None, None, None]),
    ([]),
], ids=["floats", "Nones", "empty"])
def test_errors(data):
    result = check_categorical_quality(data)
    print(result)
    assert result['error'] == 'no valid data points'

def test_mixed_types_warning():
    """Test that mixing types triggers appropriate warnings"""
    data = [1, 2, 3, 'a', 'b']  # Mostly numbers
    result = check_categorical_quality(data)
    
    assert 'warning' in result or 'warnings' in result
    assert result['valid_categories'] == 5

# ============================================================================
# Temporal Data Tests
# ============================================================================

def test_temporal_basic():
    """Test basic temporal analysis with daily data"""
    dates = [
        '2024-01-01',
        '2024-01-02', 
        '2024-01-03',
        '2024-01-05',  # Gap!
        '2024-01-06'
    ]
    result = check_temporal_quality(dates)
    
    assert result['analysis'] == 'temporal'
    assert result['total values'] == 5
    assert result['total valid values'] == 5
    assert result['earliest'] == pd.to_datetime('2024-01-01')
    assert result['latest'] == pd.to_datetime('2024-01-06')
    assert result['gaps_detected'] == 1


def test_temporal_with_invalid_dates():
    """Test temporal data with some invalid entries"""
    dates = [
        '2024-01-01',
        'not-a-date',
        '2024-01-03',
        None,
        '2024-01-04'
    ]
    result = check_temporal_quality(dates)
    
    assert result['analysis'] == 'temporal'
    assert result['total values'] == 5
    assert result['total valid values'] == 3
    assert result['invalid values'] == 2


def test_temporal_irregular_pattern():
    """Test temporal data with irregular spacing (event data)"""
    dates = [
        '2024-01-01',
        '2024-01-15',
        '2024-03-20',
        '2024-06-30'
    ]
    result = check_temporal_quality(dates)
    
    assert result['analysis'] == 'temporal'
    assert result['pattern'] == 'irregular'
    assert 'gaps' in result
    assert result['gaps'] == 'N/A - no regular pattern'


def test_temporal_hourly_data():
    """Test temporal data with hourly frequency"""
    dates = [
        '2024-01-01 10:00:00',
        '2024-01-01 11:00:00',
        '2024-01-01 12:00:00',
        '2024-01-01 14:00:00',  # Missing 13:00
        '2024-01-01 15:00:00'
    ]
    result = check_temporal_quality(dates)
    
    assert result['analysis'] == 'temporal'
    assert result['gaps_detected'] >= 1


def test_temporal_no_gaps():
    """Test temporal data with no gaps (perfect sequence)"""
    dates = [
        '2024-01-01',
        '2024-01-02',
        '2024-01-03',
        '2024-01-04',
        '2024-01-05'
    ]
    result = check_temporal_quality(dates)
    
    assert result['analysis'] == 'temporal'
    assert result['gaps_detected'] == 0


def test_temporal_all_invalid():
    """Test temporal data with all invalid entries"""
    dates = ['not-a-date', 'also-not-a-date', None, '']
    result = check_temporal_quality(dates)
    
    assert 'error' in result
    assert result['error'] == 'no valid data'


def test_temporal_single_date():
    """Test temporal data with single valid date"""
    dates = ['2024-01-01']
    result = check_temporal_quality(dates)
    
    assert result['analysis'] == 'temporal'
    assert result['total valid values'] == 1
    # Should handle gracefully (no gaps with only one date)


# ============================================================================
# Dispatcher Tests
# ============================================================================

def test_dispatcher_detects_numeric():
    """Test dispatcher correctly routes numeric data"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = check_data(data)
    
    assert result['analysis'] == 'numerical'
    assert 'mean' in result
    assert 'std' in result


def test_dispatcher_detects_categorical():
    """Test dispatcher correctly routes categorical data"""
    data = ['red', 'blue', 'green', 'red', 'blue', 'red']
    result = check_data(data)
    
    assert result['analysis'] == 'categorical'
    assert 'unique count' in result
    assert 'frequency distribution' in result


def test_dispatcher_detects_temporal():
    """Test dispatcher correctly routes temporal data"""
    data = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
    result = check_data(data)
    
    assert result['analysis'] == 'temporal'
    assert 'earliest' in result
    assert 'latest' in result


def test_dispatcher_precedence_temporal_over_numeric():
    """Test that temporal takes precedence over numeric when ambiguous"""
    # Dates that could be parsed as numbers
    data = ['2024-01-01', '2024-01-02', '2024-01-03']
    result = check_data(data)
    
    assert result['analysis'] == 'temporal'


def test_dispatcher_precedence_numeric_over_categorical():
    """Test that numeric takes precedence over categorical when ambiguous"""
    # Integers are valid for both numeric and categorical
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    result = check_data(data)
    
    assert result['analysis'] == 'numerical'


def test_dispatcher_mixed_with_nulls():
    """Test dispatcher handles mixed data with many nulls"""
    data = [1, 2, None, 3, None, 4, 5, None]
    result = check_data(data)
    
    assert result['analysis'] == 'numerical'
    assert 'mean' in result


def test_dispatcher_small_dataset():
    """Test dispatcher with very small dataset (< sample size)"""
    data = [1, 2, 3]
    result = check_data(data)
    
    # Should still work and check all elements
    assert 'analysis' in result


def test_dispatcher_large_dataset():
    """Test dispatcher with large dataset (uses sampling)"""
    data = list(range(1000))
    result = check_data(data)
    
    assert result['analysis'] == 'numerical'
    assert result['valid data points'] == 1000


def test_dispatcher_empty_dataset():
    """Test dispatcher handles empty list"""
    data = []
    result = check_data(data)
    
    assert 'error' in result
    assert result['error'] == 'empty dataset'


def test_dispatcher_all_none():
    """Test dispatcher with all None values"""
    data = [None, None, None, None]
    result = check_data(data)
    
    # Should route somewhere and that analyzer should handle it
    assert 'error' in result or 'data_type' in result


def test_dispatcher_mixed_types_categorical():
    """Test dispatcher routes mixed strings/numbers to categorical"""
    data = ['apple', 'banana', 1, 2, 'cherry']
    result = check_data(data)
    
    assert result['analysis'] == 'categorical'


def test_dispatcher_yamane_sampling():
    """Test that Yamane's formula produces reasonable sample sizes"""
    # With 100 items and e=0.1, should get reasonable sample
    data = list(range(100))
    result = check_data(data)
    
    # Just verify it works and produces correct type
    assert result['analysis'] == 'numerical'
    assert result['valid data points'] == 100