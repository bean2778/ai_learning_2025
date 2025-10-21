"""
Real-time API: Get instant storage score predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Dict

# Initialize FastAPI app
app = FastAPI(
    title="Renewable Energy Storage Investment API",
    description="Predict storage investment scores for countries",
    version="1.0.0"
)

# Load model artifacts at startup
print("Loading model artifacts...")
model = joblib.load('storage_model.pkl')
scaler = joblib.load('storage_scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
print(f"Model loaded. Features: {feature_names}")

class CountryProfile(BaseModel):
    """Input schema for prediction request"""
    avg_renewable_share: float = Field(..., description="Average renewable energy share (%)")
    growth_rate_pct: float = Field(..., description="Growth rate 2010-2022 (%)")
    num_renewable_types: int = Field(..., ge=1, le=6, description="Number of renewable types (1-6)")
    total_production: float = Field(..., gt=0, description="Total renewable production (GWh)")
    production_2022: float = Field(..., gt=0, description="Production in 2022 (GWh)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "avg_renewable_share": 45.5,
                "growth_rate_pct": 25.0,
                "num_renewable_types": 4,
                "total_production": 500000.0,
                "production_2022": 50000.0
            }
        }

class PredictionResponse(BaseModel):
    """Output schema for prediction response"""
    storage_score: float
    investment_category: str
    features_used: Dict[str, float]

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "RandomForestRegressor",
        "features": feature_names
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(profile: CountryProfile):
    """
    Predict storage investment score for a country profile
    """
    try:
        # Convert input to feature array (correct order!)
        features = np.array([[
            profile.avg_renewable_share,
            profile.growth_rate_pct,
            profile.num_renewable_types,
            profile.total_production,
            profile.production_2022
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        storage_score = float(model.predict(features_scaled)[0])
        
        # Categorize investment potential
        if storage_score >= 70:
            category = "High Priority"
        elif storage_score >= 50:
            category = "Medium Priority"
        else:
            category = "Low Priority"
        
        return PredictionResponse(
            storage_score=round(storage_score, 2),
            investment_category=category,
            features_used={
                "avg_renewable_share": profile.avg_renewable_share,
                "growth_rate_pct": profile.growth_rate_pct,
                "num_renewable_types": profile.num_renewable_types,
                "total_production": profile.total_production,
                "production_2022": profile.production_2022
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features": feature_names
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)