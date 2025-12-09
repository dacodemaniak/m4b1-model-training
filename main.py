# main.py
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

# ---Settings and artefacts loading ---
MODEL_DIR = "./model_artefacts"
MODEL_PATH = os.path.join(MODEL_DIR, "best_regressor.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.joblib")

# Globals for model and scaler
model = None
scaler = None
FEATURE_COLUMNS = [
    "crim", "zn", "indus", "chas", "nox", "rm", "age", 
    "dis", "rad", "tax", "ptratio", "b", "lstat"
]

# Load model on API startup
def load_artefacts():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and Scaler successfully loaded")
    except Exception as e:
        print(f"Something went wrong loading artefacts : {e}")
        raise RuntimeError("Model file unavalailable")

# --- Data scheme settings (Pydantic) ---

# Validation query
class HousingFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int   # Binary must be float or integer
    nox: float
    rm: float
    age: float
    dis: float
    rad: float
    tax: float
    ptratio: float
    b: float
    lstat: float
    
    # JSON sample
    class Config:
        schema_extra = {
            "example": {
                "crim": 0.00632, "zn": 18, "indus": 2.31, "chas": 0, "nox": 0.538, 
                "rm": 6.575, "age": 65.2, "dis": 4.09, "rad": 1, "tax": 296, 
                "ptratio": 15.3, "b": 396.9, "lstat": 4.98
            }
        }

# Repsonse body settings
class PredictionResponse(BaseModel):
    predicted_medv: float
    model_version: str = "v1.0"


# --- FastAPI Application ---
app = FastAPI(
    title="Boston Housing Price Predictor",
    version="1.0"
)

# API start event to load models
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    load_artefacts()
    yield
    # --- shutdown (optionnel) ---
    # clean up here if needed

# --- prediction endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_housing_price(features: HousingFeatures):
    """
    Got input feature returns price prediction (MEDV).
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        # 1. Convert pydantic to pandas dataframe
        data_df = pd.DataFrame([features.model_dump()])
        data_df = data_df[FEATURE_COLUMNS]
        
        # 2. Pre-process : Scaling
        scaled_features = scaler.transform(data_df)
        
        # 3. Prediction
        prediction = model.predict(scaled_features)[0]
        
        # 4. Return response
        return PredictionResponse(predicted_medv=float(prediction))
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing prediction")

# --- Health check endpoint ---
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}