"""
Train a model to predict renewable energy storage investment potential
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def train_storage_model():
    """
    Train model to predict storage_score from country metrics
    """
    # Load preprocessed data
    df = pd.read_csv('data_storage_candidates.csv')
    
    print(f"Loaded {len(df)} countries")
    print(f"Columns: {df.columns.tolist()}")
    
    # Features: things we can measure about a country
    feature_cols = [
        'avg_renewable_share',
        'growth_rate_pct', 
        'num_renewable_types',
        'total_production',
        'production_2022'
    ]
    
    # Target: storage investment score
    target_col = 'storage_score'
    
    # Remove rows with missing values
    df_clean = df[feature_cols + [target_col]].dropna()
    print(f"Clean data: {len(df_clean)} countries")
    
    # Split features and target
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Scale features (important for production models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    print("\nTraining model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²:  {test_score:.3f}")
    
    # Save model and scaler
    joblib.dump(model, 'storage_model.pkl')
    joblib.dump(scaler, 'storage_scaler.pkl')
    joblib.dump(feature_cols, 'feature_names.pkl')
    
    print(f"\nSaved:")
    print(f"  - storage_model.pkl")
    print(f"  - storage_scaler.pkl") 
    print(f"  - feature_names.pkl")
    
    return model, scaler, feature_cols

if __name__ == "__main__":
    train_storage_model()