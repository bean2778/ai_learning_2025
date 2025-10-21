"""
Batch prediction: Score all countries for weekly investment report
"""

import pandas as pd
import joblib
import time

def batch_predict():
    """
    Load model and predict storage scores for all countries
    """
    # Load model artifacts
    print("Loading model...")
    model = joblib.load('storage_model.pkl')
    scaler = joblib.load('storage_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    # Load data (in production: fresh data from database)
    print("Loading country data...")
    df = pd.read_csv('data_storage_candidates.csv')
    
    # Prepare features
    X = df[feature_names].dropna()
    countries = df.loc[X.index, 'country'].values
    
    print(f"\nProcessing {len(X)} countries...")
    
    # Time the batch prediction
    start = time.time()
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    
    elapsed = time.time() - start
    
    # Create results dataframe
    results = pd.DataFrame({
        'country': countries,
        'predicted_storage_score': predictions,
        'actual_storage_score': df.loc[X.index, 'storage_score'].values
    })
    
    # Sort by predicted score
    results = results.sort_values('predicted_storage_score', ascending=False)
    
    # Save results
    results.to_csv('batch_predictions.csv', index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PREDICTION COMPLETE")
    print(f"{'='*60}")
    print(f"Countries processed: {len(X)}")
    print(f"Time elapsed: {elapsed:.3f} seconds")
    print(f"Throughput: {len(X)/elapsed:.1f} predictions/second")
    print(f"\nTop 5 Investment Opportunities:")
    print(results.head()[['country', 'predicted_storage_score']])
    print(f"\nResults saved to: batch_predictions.csv")

if __name__ == "__main__":
    batch_predict()