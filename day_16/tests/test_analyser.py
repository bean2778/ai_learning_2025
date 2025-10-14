"""
Tests for EDA toolkit functions
"""
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from eda_toolkit.analyzer import (
    generate_numeric_summary,
    generate_relationship_analysis,
    generate_categorical_analysis
)


@pytest.fixture
def sample_data():
    """Load sample sales data"""
    return pd.read_csv('data/sales_data.csv')


def test_numeric_summary_returns_dict(sample_data):
    """Test numeric summary returns proper structure"""
    result = generate_numeric_summary(sample_data)
    
    assert result is not None
    assert isinstance(result, dict)
    assert 'figure' in result
    assert 'outliers' in result
    
    # Check outliers is dict
    assert isinstance(result['outliers'], dict)
    
    # Check figure is matplotlib figure
    assert isinstance(result['figure'], plt.Figure)
    
    plt.close('all')


def test_numeric_summary_auto_detect(sample_data):
    """Test auto-detection of numeric columns"""
    result = generate_numeric_summary(sample_data)
    
    # Should have detected multiple columns
    assert len(result['outliers']) > 0
    
    plt.close('all')


def test_numeric_summary_manual_cols(sample_data):
    """Test with manually specified columns"""
    result = generate_numeric_summary(sample_data, ['customer_age', 'annual_income'])
    
    # Should only have outliers for specified columns
    assert 'customer_age' in result['outliers']
    assert 'annual_income' in result['outliers']
    
    plt.close('all')


def test_relationship_analysis_returns_dict(sample_data):
    """Test relationship analysis returns proper structure"""
    result = generate_relationship_analysis(sample_data)
    
    assert result is not None
    assert isinstance(result, dict)
    assert 'heatmap' in result
    assert 'pairplot' in result
    assert 'top_correlations' in result
    
    # Check types
    assert isinstance(result['heatmap'], plt.Figure)
    assert isinstance(result['pairplot'], plt.Figure)
    assert isinstance(result['top_correlations'], list)
    
    plt.close('all')


def test_relationship_analysis_correlations(sample_data):
    """Test correlation extraction"""
    result = generate_relationship_analysis(sample_data)
    
    # Should have top correlations
    assert len(result['top_correlations']) > 0
    
    # Each correlation should be tuple of (feat1, feat2, value)
    for corr in result['top_correlations']:
        assert len(corr) == 3
        assert isinstance(corr[0], str)  # feature 1 name
        assert isinstance(corr[1], str)  # feature 2 name
        assert isinstance(corr[2], (float, int))  # correlation value
    
    plt.close('all')


def test_categorical_analysis_returns_dict(sample_data):
    """Test categorical analysis returns proper structure"""
    result = generate_categorical_analysis(sample_data, target_col='purchase_amount')
    
    assert result is not None
    assert isinstance(result, dict)
    assert 'count_plots' in result
    assert 'box_plots' in result
    assert 'warnings' in result
    
    # Check types
    assert isinstance(result['count_plots'], plt.Figure)
    assert isinstance(result['box_plots'], plt.Figure)
    assert isinstance(result['warnings'], list)
    
    plt.close('all')


def test_categorical_analysis_auto_detect(sample_data):
    """Test auto-detection of categorical columns and target"""
    result = generate_categorical_analysis(sample_data)
    
    # Should auto-detect and return valid results
    assert result is not None
    assert 'count_plots' in result
    
    plt.close('all')


def test_categorical_analysis_warnings(sample_data):
    """Test small category warnings"""
    result = generate_categorical_analysis(sample_data, target_col='purchase_amount')
    
    # Warnings should be a list
    assert isinstance(result['warnings'], list)
    
    plt.close('all')


def test_all_functions_with_save(sample_data):
    """Integration test - run all functions and save outputs"""
    
    # Numeric summary
    result1 = generate_numeric_summary(sample_data)
    result1['figure'].savefig('output/test_numeric_summary.png', dpi=100, bbox_inches='tight')
    
    # Relationship analysis
    result2 = generate_relationship_analysis(sample_data)
    result2['heatmap'].savefig('output/test_heatmap.png', dpi=100, bbox_inches='tight')
    result2['pairplot'].savefig('output/test_pairplot.png', dpi=100, bbox_inches='tight')
    
    print("\nTop 5 Correlations:")
    for feat1, feat2, corr_val in result2['top_correlations']:
        print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
    
    # Categorical analysis
    result3 = generate_categorical_analysis(sample_data, target_col='purchase_amount')
    result3['count_plots'].savefig('output/test_count_plots.png', dpi=100, bbox_inches='tight')
    result3['box_plots'].savefig('output/test_box_plots.png', dpi=100, bbox_inches='tight')
    
    print("\nWarnings:")
    for warning in result3['warnings']:
        print(f"  {warning}")
    
    plt.close('all')
    
    # All should complete without errors
    assert True