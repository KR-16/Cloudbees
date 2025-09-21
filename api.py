"""
FastAPI Deployment Wrapper for Iris Classification Model
This module provides a REST API for model inference with observability features.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="MLOps-enabled API for iris flower classification",
    version="1.0.0"
)

# Global variables for model and scaler
model = None
scaler = None
model_metadata = None
prediction_history = []

class IrisFeatures(BaseModel):
    """Pydantic model for input validation"""
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="Predicted class (0: setosa, 1: versicolor, 2: virginica)")
    prediction_proba: List[float] = Field(..., description="Prediction probabilities for each class")
    confidence: float = Field(..., description="Confidence score (max probability)")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str = None
    total_predictions: int

def load_latest_model():
    """Load the latest trained model and scaler"""
    global model, scaler, model_metadata
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Get the latest model version
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("iris_classifier")[0]
        
        # Load model from MLflow
        model_uri = f"models:/iris_classifier/{latest_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Try to load scaler from artifacts
        artifacts_dir = f"artifacts/model_v{latest_version.version}"
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            logger.warning("Scaler not found, using raw features")
            scaler = None
        
        # Load metadata
        metadata_path = os.path.join(artifacts_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {"model_version": str(latest_version.version)}
        
        logger.info(f"Model loaded successfully: version {latest_version.version}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up Iris Classification API...")
    success = load_latest_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Iris Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model status"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=str(model_metadata.get("model_version")) if model_metadata and model_metadata.get("model_version") else None,
        total_predictions=len(prediction_history)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """Make a prediction for iris classification"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        input_data = np.array([
            [features.sepal_length, features.sepal_width, 
             features.petal_length, features.petal_width]
        ])
        
        # Apply scaling if scaler is available
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0].tolist()
        confidence = max(prediction_proba)
        
        # Create response
        response = PredictionResponse(
            prediction=int(prediction),
            prediction_proba=prediction_proba,
            confidence=float(confidence),
            model_version=model_metadata.get("model_version", "unknown"),
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction for observability
        prediction_log = {
            "timestamp": response.timestamp,
            "input": features.dict(),
            "prediction": int(prediction),
            "confidence": float(confidence),
            "model_version": response.model_version
        }
        prediction_history.append(prediction_log)
        
        # Keep only last 1000 predictions in memory
        if len(prediction_history) > 1000:
            prediction_history.pop(0)
        
        logger.info(f"Prediction made: class {prediction} with confidence {confidence:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get recent prediction history for observability"""
    if limit > 1000:
        limit = 1000
    
    return {
        "total_predictions": len(prediction_history),
        "recent_predictions": prediction_history[-limit:],
        "model_version": model_metadata.get("model_version") if model_metadata else None
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information and metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_metadata": model_metadata,
        "model_type": type(model).__name__,
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target_classes": ["setosa", "versicolor", "virginica"]
    }

@app.get("/metrics")
async def get_metrics():
    """Get basic metrics for observability"""
    if not prediction_history:
        return {
            "total_predictions": 0,
            "average_confidence": 0,
            "predictions_by_class": {},
            "recent_activity": []
        }
    
    # Calculate metrics
    total_predictions = len(prediction_history)
    confidences = [p["confidence"] for p in prediction_history]
    average_confidence = sum(confidences) / len(confidences)
    
    # Count predictions by class
    class_counts = {}
    for p in prediction_history:
        class_name = ["setosa", "versicolor", "virginica"][p["prediction"]]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Recent activity (last 10 predictions)
    recent_activity = prediction_history[-10:] if len(prediction_history) >= 10 else prediction_history
    
    return {
        "total_predictions": total_predictions,
        "average_confidence": round(average_confidence, 3),
        "predictions_by_class": class_counts,
        "recent_activity": recent_activity
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
