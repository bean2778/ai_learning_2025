import pandas as pd
import numpy as np

def process_transactions_slow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slow implementation - uses loops and .apply()
    
    Processing steps:
    1. Calculate total (price * quantity)
    2. Apply discount based on quantity
    3. Categorize transactions (small/medium/large)
    4. Flag high-value customers (>$1000 total spend)
    """
    result = df.copy()
    
    # Step 1: Calculate total using .apply() - SLOW
    result['total'] = result.apply(
        lambda row: row['price'] * row['quantity'], 
        axis=1
    )
    
    # Step 2: Apply discount using .apply() - SLOW
    def calculate_discount(row):
        if row['quantity'] > 10:
            return 0.15
        elif row['quantity'] > 5:
            return 0.10
        else:
            return 0.05
    
    result['discount'] = result.apply(calculate_discount, axis=1)
    result['final_price'] = result['total'] * (1 - result['discount'])
    
    # Step 3: Categorize using .apply() - SLOW
    def categorize(row):
        if row['final_price'] < 50:
            return 'small'
        elif row['final_price'] < 200:
            return 'medium'
        else:
            return 'large'
    
    result['category'] = result.apply(categorize, axis=1)
    
    # Step 4: Flag high-value customers using loop - VERY SLOW
    customer_totals = {}
    for idx, row in result.iterrows():
        customer_id = row['customer_id']
        if customer_id not in customer_totals:
            customer_totals[customer_id] = 0
        customer_totals[customer_id] += row['final_price']
    
    result['high_value'] = result['customer_id'].apply(
        lambda x: customer_totals[x] > 1000
    )
    
    return result
