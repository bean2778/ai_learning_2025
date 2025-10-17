"""
Generate three sample datasets for ML problem formulation practice.
VERSION 3: Strong, clear signal

Datasets:
1. customer_purchases.csv - Binary classification (imbalanced, clear signal)
2. product_pricing.csv - Regression with outliers
3. customer_segments.csv - Clustering with natural groups
"""

import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Create data directory
os.makedirs('data', exist_ok=True)

print("Generating sample datasets (v3 - strong clear signal)...\n")

# ============================================================================
# Dataset 1: Customer Purchase Prediction (Classification)
# VERSION 3: Strong correlations, clear patterns
# ============================================================================
print("1. Creating customer_purchases.csv...")

n_samples = 1000

# Generate base features
age = np.random.normal(40, 12, n_samples).clip(18, 75)
income = np.random.normal(60000, 25000, n_samples).clip(20000, 150000)
previous_purchases = np.random.poisson(3, n_samples).clip(0, 15)
time_on_site = np.random.gamma(2, 3, n_samples).clip(0, 25)
email_opens = np.random.binomial(12, 0.35, n_samples)

# Normalize to 0-1 for clear weighting
income_scaled = (income - 20000) / (150000 - 20000)
prev_scaled = previous_purchases / 15
time_scaled = time_on_site / 25
email_scaled = email_opens / 12

# Create purchase probability with strong signal
# Each feature contributes clearly to purchase decision
base_prob = 0.05  # Base 5% purchase rate

purchase_prob = (
    base_prob +
    0.25 * income_scaled +           # Higher income = more likely to buy
    0.30 * prev_scaled +              # Past buyers buy again (strongest signal)
    0.15 * email_scaled +             # Engaged customers buy more
    0.12 * time_scaled                # Time on site indicates interest
)

# Add some noise but keep signal strong
noise = np.random.normal(0, 0.08, n_samples)
purchase_prob = (purchase_prob + noise).clip(0, 0.95)

# Generate purchased labels with threshold for ~20% purchase rate
threshold = np.percentile(purchase_prob, 80)  # Top 20% purchase
purchased = (purchase_prob >= threshold).astype(int)

# Create dataframe
df1 = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'age': age.round(0).astype(int),
    'income': income.round(2),
    'previous_purchases': previous_purchases.astype(int),
    'time_on_site_minutes': time_on_site.round(2),
    'email_opens_last_month': email_opens.astype(int),
    'purchased': purchased
})

df1.to_csv('data/customer_purchases.csv', index=False)
print(f"   ✓ {len(df1)} rows")
print(f"   ✓ Target distribution: {purchased.sum()} purchased ({100*purchased.mean():.1f}%)")
print(f"   ✓ Features: {len(df1.columns)-2} (excluding customer_id and target)")

# Show correlations
correlations = df1.drop('customer_id', axis=1).corr()['purchased'].drop('purchased').sort_values(ascending=False)
print(f"\n   Correlations with 'purchased':")
for col, corr in correlations.items():
    print(f"     {col:30s}: {corr:+.3f}")

# ============================================================================
# Dataset 2: Product Pricing (Regression)
# Strong predictive features
# ============================================================================
print("\n2. Creating product_pricing.csv...")

n_products = 800

# Features with clear relationships to price
manufacturing_cost = np.random.uniform(15, 100, n_products)
competitor_price = manufacturing_cost * np.random.uniform(1.8, 2.8, n_products)
brand_strength = np.random.choice([1, 2, 3, 4, 5], n_products, p=[0.1, 0.2, 0.4, 0.2, 0.1])
market_demand = np.random.uniform(0.3, 1.0, n_products)
seasonality = np.random.choice(['low', 'medium', 'high'], n_products, p=[0.25, 0.5, 0.25])

# Price formula with strong relationships
base_price = manufacturing_cost * 1.8  # Base markup

# Brand effect (stronger)
brand_multiplier = 0.8 + (brand_strength / 5) * 0.6  # Range: 0.8 to 1.4

# Demand effect
demand_multiplier = 0.85 + market_demand * 0.3  # Range: 0.85 to 1.15

# Season effect
season_map = {'low': 0.85, 'medium': 1.0, 'high': 1.20}
season_multiplier = np.array([season_map[s] for s in seasonality])

# Calculate optimal price
optimal_price = base_price * brand_multiplier * demand_multiplier * season_multiplier

# Add realistic noise
noise = np.random.normal(0, 8, n_products)
optimal_price = optimal_price + noise

# Add a few outliers (pricing mistakes)
outlier_indices = np.random.choice(n_products, size=25, replace=False)
outlier_adjustments = np.random.choice([-40, -30, 40, 60], size=25)
optimal_price[outlier_indices] += outlier_adjustments

optimal_price = optimal_price.clip(20, 400)

df2 = pd.DataFrame({
    'product_id': range(1, n_products + 1),
    'manufacturing_cost': manufacturing_cost.round(2),
    'competitor_avg_price': competitor_price.round(2),
    'brand_strength': brand_strength,
    'market_demand_score': market_demand.round(3),
    'seasonality': seasonality,
    'optimal_price': optimal_price.round(2)
})

df2.to_csv('data/product_pricing.csv', index=False)
print(f"   ✓ {len(df2)} rows")
print(f"   ✓ Target range: ${optimal_price.min():.2f} - ${optimal_price.max():.2f}")
print(f"   ✓ Mean price: ${optimal_price.mean():.2f}")

# Show correlations for numeric features
numeric_df = df2.select_dtypes(include=[np.number]).drop('product_id', axis=1)
price_corr = numeric_df.corr()['optimal_price'].drop('optimal_price').sort_values(ascending=False)
print(f"\n   Correlations with 'optimal_price':")
for col, corr in price_corr.items():
    print(f"     {col:30s}: {corr:+.3f}")

# ============================================================================
# Dataset 3: Customer Segmentation (Clustering - No Target)
# Three distinct, well-separated clusters
# ============================================================================
print("\n3. Creating customer_segments.csv...")

n_customers = 600
segment_sizes = [200, 250, 150]
segments = []

# Segment 1: Budget-conscious families (low spend, high frequency, large baskets)
seg1 = pd.DataFrame({
    'avg_purchase_amount': np.random.normal(45, 10, segment_sizes[0]).clip(25, 75),
    'purchase_frequency_per_month': np.random.normal(12, 2, segment_sizes[0]).clip(8, 18).astype(int),
    'avg_basket_size': np.random.normal(28, 4, segment_sizes[0]).clip(18, 40),
    'premium_product_ratio': np.random.beta(2, 8, segment_sizes[0]).clip(0, 0.25)
})
segments.append(seg1)

# Segment 2: Young professionals (medium-high spend, very high frequency, small baskets)
seg2 = pd.DataFrame({
    'avg_purchase_amount': np.random.normal(130, 25, segment_sizes[1]).clip(80, 220),
    'purchase_frequency_per_month': np.random.normal(18, 3, segment_sizes[1]).clip(12, 26).astype(int),
    'avg_basket_size': np.random.normal(12, 3, segment_sizes[1]).clip(5, 20),
    'premium_product_ratio': np.random.beta(5, 5, segment_sizes[1]).clip(0.25, 0.75)
})
segments.append(seg2)

# Segment 3: Luxury shoppers (very high spend, low frequency, small baskets, high premium)
seg3 = pd.DataFrame({
    'avg_purchase_amount': np.random.normal(320, 60, segment_sizes[2]).clip(200, 500),
    'purchase_frequency_per_month': np.random.normal(4, 1.5, segment_sizes[2]).clip(1, 8).astype(int),
    'avg_basket_size': np.random.normal(6, 1.5, segment_sizes[2]).clip(3, 12),
    'premium_product_ratio': np.random.beta(8, 2, segment_sizes[2]).clip(0.65, 0.98)
})
segments.append(seg3)

# Combine and shuffle
df3 = pd.concat(segments, ignore_index=True)
df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)
df3.insert(0, 'customer_id', range(1, n_customers + 1))

# Round for cleanliness
df3['avg_purchase_amount'] = df3['avg_purchase_amount'].round(2)
df3['avg_basket_size'] = df3['avg_basket_size'].round(1)
df3['premium_product_ratio'] = df3['premium_product_ratio'].round(3)

df3.to_csv('data/customer_segments.csv', index=False)
print(f"   ✓ {len(df3)} rows")
print(f"   ✓ NO TARGET - find natural segments!")
print(f"   ✓ Features: {len(df3.columns)-1}")
print(f"   ✓ Hidden segments: 3 (well-separated clusters)")
print(f"     - Budget families: low spend, high freq, large baskets")
print(f"     - Young professionals: medium spend, very high freq, small baskets")
print(f"     - Luxury: high spend, low freq, premium products")

print("\n✅ All datasets created in data/ folder!")
print("\nDataset summary:")
print("  • customer_purchases.csv  → Classification (20% purchase rate, clear signal)")
print("  • product_pricing.csv     → Regression (strong cost/brand relationships)")
print("  • customer_segments.csv   → Clustering (3 distinct groups)")