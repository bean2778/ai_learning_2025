# tests/test_validators.py

import pytest
import pandas as pd
import numpy as np
from validators.validators import validate_schema, validate_business_rules, detect_anomalies

# ============ Schema Validation Tests ============

def test_validate_schema_perfect_match():
    """All columns present with correct types"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    schema = {'customer_id': 'int64', 'name': 'object'}
    
    result = validate_schema(df, schema)
    
    assert result['valid'] == True
    assert len(result['missing_columns']) == 0
    assert len(result['extra_columns']) == 0
    assert len(result['type_mismatches']) == 0

def test_validate_schema_missing_columns():
    """Schema requires columns not in DataFrame"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3]
    })
    schema = {'customer_id': 'int64', 'name': 'object', 'email': 'object'}
    
    result = validate_schema(df, schema)
    
    assert result['valid'] == False
    assert 'name' in result['missing_columns']
    assert 'email' in result['missing_columns']

def test_validate_schema_extra_columns():
    """DataFrame has columns not in schema"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'extra_col': [100, 200, 300]
    })
    schema = {'customer_id': 'int64', 'name': 'object'}
    
    result = validate_schema(df, schema)
    
    assert result['valid'] == False
    assert 'extra_col' in result['extra_columns']

def test_validate_schema_type_mismatch():
    """Column exists but wrong dtype"""
    df = pd.DataFrame({
        'customer_id': ['1', '2', '3'],  # Strings instead of ints
        'name': ['Alice', 'Bob', 'Charlie']
    })
    schema = {'customer_id': 'int64', 'name': 'object'}
    
    result = validate_schema(df, schema)
    
    assert result['valid'] == False
    assert len(result['type_mismatches']) == 1
    assert result['type_mismatches'][0][0] == 'customer_id'
    assert result['type_mismatches'][0][1] == 'int64'  # expected
    assert 'object' in result['type_mismatches'][0][2]  # actual

# ============ Business Rules Tests ============

def test_validate_business_rules_all_valid():
    """All business rules pass"""
    df = pd.DataFrame({
        'amount': [100, 200, 300],
        'age': [25, 45, 60],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
        'signup_date': ['2024-01-01', '2024-02-01', '2024-03-01']
    })
    
    result = validate_business_rules(df)
    
    assert result['valid'] == True
    assert result['violations']['negative_amounts'] == 0
    assert result['violations']['invalid_ages'] == 0
    assert result['violations']['invalid_emails'] == 0
    assert result['violations']['future_dates'] == 0

def test_validate_business_rules_negative_amounts():
    """Detects negative amounts"""
    df = pd.DataFrame({
        'amount': [100, -50, 200, -30]
    })
    
    result = validate_business_rules(df)
    
    assert result['valid'] == False
    assert result['violations']['negative_amounts'] == 2
    assert len(result['violation_details']['negative_amounts']) == 2

def test_validate_business_rules_invalid_ages():
    """Detects ages outside 1-120 range"""
    df = pd.DataFrame({
        'age': [25, 0, 45, 150, 60]
    })
    
    result = validate_business_rules(df)
    
    assert result['valid'] == False
    assert result['violations']['invalid_ages'] == 2
    assert len(result['violation_details']['invalid_ages']) == 2

def test_validate_business_rules_invalid_emails():
    """Detects emails without @"""
    df = pd.DataFrame({
        'email': ['alice@test.com', 'badmail.com', 'bob@test.com', 'nope']
    })
    
    result = validate_business_rules(df)
    
    assert result['valid'] == False
    assert result['violations']['invalid_emails'] == 2

def test_validate_business_rules_future_dates():
    """Detects dates in the future"""
    future_date = (pd.Timestamp.now() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    df = pd.DataFrame({
        'signup_date': ['2024-01-01', future_date, '2024-03-01']
    })
    
    result = validate_business_rules(df)
    
    assert result['valid'] == False
    assert result['violations']['future_dates'] == 1

def test_validate_business_rules_missing_columns():
    """Handles missing columns gracefully"""
    df = pd.DataFrame({
        'amount': [100, 200]
    })
    
    result = validate_business_rules(df)
    
    # Should only have violations key for amount
    assert 'negative_amounts' in result['violations']
    assert 'invalid_ages' not in result['violations']
    assert 'invalid_emails' not in result['violations']

# ============ Anomaly Detection Tests ============

def test_detect_anomalies_finds_high_outlier():
    """Detects values above upper bound"""
    df = pd.DataFrame({
        'amount': [100, 150, 120, 140, 130, 900, 110]
    })
    
    result = detect_anomalies(df, 'amount')
    
    assert len(result) == 1
    assert result.iloc[0]['value'] == 900
    assert 'above' in result.iloc[0]['reason']

def test_detect_anomalies_finds_low_outlier():
    """Detects values below lower bound"""
    df = pd.DataFrame({
        'price': [50, 55, 52, 48, 5, 51, 53]
    })
    
    result = detect_anomalies(df, 'price')
    
    assert len(result) == 1
    assert result.iloc[0]['value'] == 5
    assert 'below' in result.iloc[0]['reason']

def test_detect_anomalies_finds_both():
    """Detects both high and low outliers"""
    df = pd.DataFrame({
        'value': [50, 55, 52, 48, 5, 51, 200]
    })
    
    result = detect_anomalies(df, 'value')
    
    assert len(result) == 2
    assert 5 in result['value'].values
    assert 200 in result['value'].values

def test_detect_anomalies_none_found():
    """Returns empty DataFrame when no outliers"""
    df = pd.DataFrame({
        'amount': [100, 110, 120, 130, 140]
    })
    
    result = detect_anomalies(df, 'amount')
    
    assert len(result) == 0
    assert 'row_index' in result.columns
    assert 'value' in result.columns
    assert 'reason' in result.columns

def test_detect_anomalies_preserves_other_columns():
    """Result includes all original columns"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'amount': [100, 120, 900, 130, 110]
    })
    
    result = detect_anomalies(df, 'amount')
    
    assert 'customer_id' in result.columns
    assert result.iloc[0]['customer_id'] == 3

def test_detect_anomalies_row_index():
    """Correctly captures original row index"""
    df = pd.DataFrame({
        'amount': [100, 120, 900, 130]
    })
    
    result = detect_anomalies(df, 'amount')
    
    assert result.iloc[0]['row_index'] == 2  # Original index of 900