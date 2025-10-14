"""
Generate realistic customer sales data with varied correlation strengths
"""
import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 200

# Primary predictor: Income (strong correlation with purchases)
annual_income = np.random.normal(75000, 25000, n_samples)
annual_income = np.clip(annual_income, 30000, 150000)

# Age (moderate correlation with income, weak with purchases)
customer_age = np.random.normal(45, 15, n_samples)
customer_age = np.clip(customer_age, 18, 80)
# Add slight correlation with income (older = higher income, but noisy)
customer_age = customer_age + (annual_income - 75000) / 2000
customer_age = np.clip(customer_age, 18, 80)

# Days since signup (weakly correlated with age, independent of purchases)
days_since_signup = np.random.normal(500, 300, n_samples)
days_since_signup = np.clip(days_since_signup, 30, 1500)
# Older customers slightly longer tenure
days_since_signup = days_since_signup + (customer_age - 45) * 5
days_since_signup = np.clip(days_since_signup, 30, 1500)

# Product category (affects purchase independent of income)
categories = np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_samples)

# Category base prices (different price ranges)
category_multipliers = {
    'Electronics': 1.5,
    'Clothing': 0.7,
    'Home': 1.0,
    'Sports': 0.9
}
category_effect = np.array([category_multipliers[cat] for cat in categories])

# Purchase amount: Strong income correlation + category effect + noise
base_purchase = (annual_income / 1000) * 1.2  # 0.7-0.8 correlation
category_purchase = base_purchase * category_effect
noise = np.random.normal(0, 30, n_samples)
purchase_amount = category_purchase + noise
purchase_amount = np.clip(purchase_amount, 20, 500)

# Customer segments based on income (useful categorical feature)
segments = pd.cut(annual_income, 
                  bins=[0, 50000, 80000, 200000],
                  labels=['Bronze', 'Silver', 'Gold'])

# Create DataFrame
df = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'customer_age': customer_age.astype(int),
    'annual_income': annual_income.round(2),
    'days_since_signup': days_since_signup.astype(int),
    'customer_segment': segments,
    'product_category': categories,
    'purchase_amount': purchase_amount.round(2)
})

# Add some missing values (realistic)
missing_indices = np.random.choice(n_samples, size=10, replace=False)
df.loc[missing_indices[:5], 'customer_age'] = np.nan
df.loc[missing_indices[5:], 'days_since_signup'] = np.nan

# Add a few outliers
outlier_indices = np.random.choice(n_samples, size=5, replace=False)
df.loc[outlier_indices, 'purchase_amount'] = np.random.uniform(600, 1000, 5)

df.to_csv('data/sales_data.csv', index=False)
print("Created realistic sales data with:")
print(f"- {n_samples} samples")
print(f"- {df.isnull().sum().sum()} missing values")
print(f"- Expected correlations:")
print("  - Income → Purchase: ~0.70-0.75 (strong)")
print("  - Age → Income: ~0.40-0.50 (moderate)")
print("  - Age → Purchase: ~0.30-0.40 (weak)")
print("  - Days_since_signup → Purchase: ~0.05-0.15 (very weak)")
print("\nSaved to: day_16/data/sales_data.csv")