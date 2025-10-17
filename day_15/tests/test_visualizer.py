import pytest
import pandas as pd
import matplotlib.pyplot as plt
from viz_stats.visualizer import (
    plot_age_distribution,
    plot_income_by_segment,
    plot_purchase_distribution,
    plot_income_vs_purchase,
    plot_age_vs_days_since_signup,
    plot_purchase_by_category,
    plot_income_by_segment_violin,
    plot_correlation_matrix
)

@pytest.fixture
def sample_data():
    """Load sample data for testing"""
    return pd.read_csv('data/sales_data.csv')

def test_age_distribution(sample_data):
    """Test age distribution plot doesn't crash"""
    fig = plot_age_distribution(sample_data)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_income_by_segment(sample_data):
    """Test income by segment plot doesn't crash"""
    fig = plot_income_by_segment(sample_data)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_purchase_distribution(sample_data):
    """Test purchase distribution plot doesn't crash"""
    fig = plot_purchase_distribution(sample_data)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_income_vs_purchase(sample_data):
    """Test income vs purchase scatter plot doesn't crash"""
    fig = plot_income_vs_purchase(sample_data)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_age_vs_days_since_signup(sample_data):
    """Test age vs days since signup plot doesn't crash"""
    fig = plot_age_vs_days_since_signup(sample_data)
    assert fig is not None
    # jointplot returns JointGrid, not Figure directly
    plt.close('all')

def test_purchase_by_category(sample_data):
    """Test purchase by category box plot doesn't crash"""
    fig = plot_purchase_by_category(sample_data)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_income_by_segment_violin(sample_data):
    """Test income by segment violin plot doesn't crash"""
    fig = plot_income_by_segment_violin(sample_data)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_correlation_matrix(sample_data):
    """Test correlation matrix heatmap doesn't crash"""
    fig = plot_correlation_matrix(sample_data)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)