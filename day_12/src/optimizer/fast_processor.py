import pandas as pd
import numpy as np

def process_transactions_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slow implementation - uses loops and .apply()
    
    Processing steps:
    1. Calculate total (price * quantity)
    2. Apply discount based on quantity
    3. Categorize transactions (small/medium/large)
    4. Flag high-value customers (>$1000 total spend)
    """
    result = df.copy()
    
    # Step 1: Calculate total
    result['total'] = result['price'] * result['quantity']
    
    # Step 2: Apply discount
    conditions = [result['quantity'] > 10, result['quantity'] > 5, result['quantity'] <= 5]
    choices = [0.15, 0.10, 0.05]
    result['discount'] = np.select(conditions, choices)
    result['final_price'] = result['total'] * (1 - result['discount'])
    
    # Step 3: Categorize
    cat_con = [
        result['final_price'] < 50, 
        (result['final_price'] >= 50) & (result['final_price'] < 200),
        result['final_price'] >= 200
    ]
    cat_choice = ['small', 'medium', 'large']
    result['category'] = np.select(cat_con, cat_choice, default='unknown')

    # Step 4: Flag high-value customers
    customer_totals = result.groupby('customer_id')['final_price'].sum()
    result['high_value'] = result['customer_id'].map(customer_totals) > 1000
    
    return result


def sample_data():
    """Load sample data"""
    return pd.read_csv('sample_data.csv')

process_transactions_fast(sample_data())