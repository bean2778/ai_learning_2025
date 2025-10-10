# tests/test_visualizer.py

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from visualizer.visualizer import (
    plot_amount_distribution,
    plot_time_series,
    plot_age_distribution,
    plot_category_balance,
    create_dashboard
)

@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=50),
        'amount': [100, 200, 150, 1000, -50] + [100] * 45,
        'customer_age': [25, 30, 150, -5, 40] + [35] * 45,
        'category': ['A'] * 30 + ['B'] * 15 + ['C'] * 3 + ['D'] * 2
    })

def test_plot_amount_distribution(sample_data):
    """Test amount distribution plot runs without error"""
    fig = plot_amount_distribution(sample_data)
    assert fig is not None
    plt.close(fig)

def test_plot_time_series(sample_data):
    """Test time series plot runs without error"""
    fig = plot_time_series(sample_data)
    assert fig is not None
    plt.close(fig)

def test_plot_age_distribution(sample_data):
    """Test age distribution plot runs without error"""
    fig = plot_age_distribution(sample_data)
    assert fig is not None
    plt.close(fig)

def test_plot_category_balance(sample_data):
    """Test category balance plot runs without error"""
    fig = plot_category_balance(sample_data)
    assert fig is not None
    plt.close(fig)

def test_create_dashboard(sample_data):
    """Test dashboard creation runs without error"""
    fig = create_dashboard(sample_data)
    assert fig is not None
    plt.close(fig)

def test_handles_empty_data():
    """Test functions handle empty DataFrame"""
    empty_df = pd.DataFrame({
        'date': [],
        'amount': [],
        'customer_age': [],
        'category': []
    })
    
    # Should not crash
    fig = plot_amount_distribution(empty_df)
    assert fig is not None
    plt.close(fig)