import pytest
import pandas as pd
import numpy as np
from data_analysis.analysis import (
    get_customer_summary,
    get_country_analysis,
    get_monthly_trends,
    calculate_customer_lifetime_value
)

# Fixtures for test data
@pytest.fixture
def sample_customers():
    """Create sample customer data"""
    return pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'country': ['USA', 'USA', 'UK'],
        'signup_date': ['2023-01-15', '2023-02-20', '2023-01-10'],
        'customer_segment': ['Premium', 'Standard', 'Premium']
    })

@pytest.fixture
def sample_transactions():
    """Create sample transaction data"""
    return pd.DataFrame({
        'transaction_id': [101, 102, 103, 104, 105],
        'customer_id': [1, 1, 2, 3, 1],
        'product_category': ['electronics', 'clothing', 'electronics', 'home', 'electronics'],
        'amount': [100.0, 50.0, 200.0, 75.0, 150.0],
        'quantity': [1, 2, 1, 3, 1],
        'transaction_date': ['2024-01-15', '2024-01-20', '2024-01-16', '2024-02-17', '2024-02-10']
    })

# Tests for get_customer_summary
def test_customer_summary_basic(sample_transactions):
    """Test customer summary returns correct shape and columns"""
    result = get_customer_summary(sample_transactions)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # 3 unique customers
    assert 'total_spent' in result.columns
    assert 'num_transactions' in result.columns
    assert 'avg_transaction' in result.columns
    assert 'most_purchased_category' in result.columns

def test_customer_summary_totals(sample_transactions):
    """Test total spent calculations are correct"""
    result = get_customer_summary(sample_transactions)
    
    # Customer 1 has 3 transactions: 100 + 50 + 150 = 300
    assert result.loc[1, 'total_spent'] == 300.0
    # Customer 2 has 1 transaction: 200
    assert result.loc[2, 'total_spent'] == 200.0
    # Customer 3 has 1 transaction: 75
    assert result.loc[3, 'total_spent'] == 75.0

def test_customer_summary_transaction_counts(sample_transactions):
    """Test transaction counts are correct"""
    result = get_customer_summary(sample_transactions)
    
    assert result.loc[1, 'num_transactions'] == 3
    assert result.loc[2, 'num_transactions'] == 1
    assert result.loc[3, 'num_transactions'] == 1

def test_customer_summary_averages(sample_transactions):
    """Test average transaction amounts"""
    result = get_customer_summary(sample_transactions)
    
    # Customer 1: (100 + 50 + 150) / 3 = 100
    assert result.loc[1, 'avg_transaction'] == 100.0

def test_customer_summary_most_purchased_category(sample_transactions):
    """Test most purchased category identification"""
    result = get_customer_summary(sample_transactions)
    
    # Customer 1 bought electronics 2 times, clothing 1 time
    assert result.loc[1, 'most_purchased_category'] == (1, 'electronics')

# Tests for get_country_analysis
def test_country_analysis_basic(sample_customers, sample_transactions):
    """Test country analysis returns correct structure"""
    result = get_country_analysis(sample_customers, sample_transactions)
    
    assert isinstance(result, pd.DataFrame)
    assert 'total_revenue' in result.columns
    assert 'number_customers' in result.columns
    assert 'average_rev_per_customer' in result.columns
    assert 'number_transactions' in result.columns

def test_country_analysis_revenue(sample_customers, sample_transactions):
    """Test total revenue per country"""
    result = get_country_analysis(sample_customers, sample_transactions)
    
    # USA: customers 1 and 2 = 300 + 200 = 500
    assert result.loc['USA', 'total_revenue'] == 500.0
    # UK: customer 3 = 75
    assert result.loc['UK', 'total_revenue'] == 75.0

def test_country_analysis_customer_count(sample_customers, sample_transactions):
    """Test number of customers per country"""
    result = get_country_analysis(sample_customers, sample_transactions)
    
    assert result.loc['USA', 'number_customers'] == 2
    assert result.loc['UK', 'number_customers'] == 1

def test_country_analysis_avg_revenue(sample_customers, sample_transactions):
    """Test average revenue per customer calculation"""
    result = get_country_analysis(sample_customers, sample_transactions)
    
    # USA: 500 total / 2 customers = 250
    assert result.loc['USA', 'average_rev_per_customer'] == 250.0
    # UK: 75 total / 1 customer = 75
    assert result.loc['UK', 'average_rev_per_customer'] == 75.0

# Tests for get_monthly_trends
def test_monthly_trends_basic(sample_transactions):
    """Test monthly trends returns correct structure"""
    result = get_monthly_trends(sample_transactions)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # 2 months: Jan 2024 and Feb 2024

def test_monthly_trends_columns(sample_transactions):
    """Test monthly trends has correct multi-level columns"""
    result = get_monthly_trends(sample_transactions)
    
    # Should have multi-level columns from agg
    assert ('amount', 'sum') in result.columns
    assert ('amount', 'count') in result.columns
    assert ('amount', 'mean') in result.columns

def test_monthly_trends_january(sample_transactions):
    """Test January 2024 calculations"""
    result = get_monthly_trends(sample_transactions)
    
    # January has 3 transactions: 100, 50, 200 = 350 total
    jan_2024 = pd.Period('2024-01', 'M')
    assert result.loc[jan_2024, ('amount', 'sum')] == 350.0
    assert result.loc[jan_2024, ('amount', 'count')] == 3

def test_monthly_trends_february(sample_transactions):
    """Test February 2024 calculations"""
    result = get_monthly_trends(sample_transactions)
    
    # February has 2 transactions: 75, 150 = 225 total
    feb_2024 = pd.Period('2024-02', 'M')
    assert result.loc[feb_2024, ('amount', 'sum')] == 225.0
    assert result.loc[feb_2024, ('amount', 'count')] == 2

# Tests for calculate_customer_lifetime_value
def test_lifetime_value_basic(sample_customers, sample_transactions):
    """Test lifetime value returns correct structure"""
    result = calculate_customer_lifetime_value(sample_customers, sample_transactions)
    
    assert isinstance(result, pd.DataFrame)
    assert 'name' in result.columns
    assert 'country' in result.columns
    assert 'segment' in result.columns
    assert 'total_spent' in result.columns
    assert 'days_since_first' in result.columns
    assert 'days_since_last' in result.columns

def test_lifetime_value_customer_info(sample_customers, sample_transactions):
    """Test customer information is correctly merged"""
    result = calculate_customer_lifetime_value(sample_customers, sample_transactions)
    
    assert result.loc[1, 'name'] == 'Alice'
    assert result.loc[1, 'country'] == 'USA'
    assert result.loc[1, 'segment'] == 'Premium'

def test_lifetime_value_total_spent(sample_customers, sample_transactions):
    """Test total spent matches customer summary"""
    result = calculate_customer_lifetime_value(sample_customers, sample_transactions)
    
    assert result.loc[1, 'total_spent'] == 300.0
    assert result.loc[2, 'total_spent'] == 200.0
    assert result.loc[3, 'total_spent'] == 75.0

def test_lifetime_value_days_since(sample_customers, sample_transactions):
    """Test days since calculations are positive integers"""
    result = calculate_customer_lifetime_value(sample_customers, sample_transactions)
    
    # Days since should be positive (transactions are in past)
    assert result.loc[1, 'days_since_first'] > 0
    assert result.loc[1, 'days_since_last'] > 0
    
    # First purchase should be longer ago than last purchase
    assert result.loc[1, 'days_since_first'] >= result.loc[1, 'days_since_last']

# Edge case tests
def test_empty_transactions():
    """Test functions handle empty DataFrames gracefully"""
    empty_trans = pd.DataFrame(columns=['transaction_id', 'customer_id', 'product_category', 
                                        'amount', 'quantity', 'transaction_date'])
    
    result = get_customer_summary(empty_trans)
    assert len(result) == 0

def test_single_customer(sample_customers):
    """Test with single customer having multiple transactions"""
    transactions = pd.DataFrame({
        'transaction_id': [1, 2, 3],
        'customer_id': [1, 1, 1],
        'product_category': ['electronics', 'electronics', 'clothing'],
        'amount': [100.0, 200.0, 50.0],
        'quantity': [1, 1, 1],
        'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-03']
    })
    
    result = get_customer_summary(transactions)
    assert len(result) == 1
    assert result.loc[1, 'total_spent'] == 350.0