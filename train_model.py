"""
MLOps Training Pipeline with MLflow Integration
This script demonstrates MLOps best practices including:
- Automated model versioning and artifact logging
- Experiment tracking with MLflow
- Model evaluation and metrics logging
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import json

def load_and_preprocess_data():
    """Load and preprocess the iris dataset"""
    print("Loading iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create DataFrame for better handling
    feature_names = iris.feature_names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some basic preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    return X_scaled, y, feature_names, scaler

def train_model(X, y, n_estimators=100, random_state=42):
    """Train a Random Forest classifier"""
    print(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=3
    )
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_test, y_test, cv=5)
    
    metrics = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores.tolist()
    }
    
    return metrics, y_pred

def log_model_artifacts(model, scaler, metrics, feature_names, model_version):
    """Log model artifacts and metadata"""
    # Create artifacts directory
    artifacts_dir = f"artifacts/model_v{model_version}"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save model and scaler
    model_path = os.path.join(artifacts_dir, "model.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'model_version': model_version,
        'timestamp': datetime.now().isoformat(),
        'feature_names': feature_names,
        'metrics': metrics,
        'model_type': 'RandomForestClassifier',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth
    }
    
    metadata_path = os.path.join(artifacts_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return artifacts_dir, metadata

def main():
    """Main training pipeline"""
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("iris_classification")
    
    # Generate model version based on timestamp
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with mlflow.start_run(run_name=f"iris_rf_v{model_version}"):
        print(f"Starting MLflow run: iris_rf_v{model_version}")
        
        # Load and preprocess data
        X, y, feature_names, scaler = load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Log parameters
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("cv_mean", metrics['cv_mean'])
        mlflow.log_metric("cv_std", metrics['cv_std'])
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="iris_classifier"
        )
        
        # Log artifacts
        artifacts_dir, metadata = log_model_artifacts(
            model, scaler, metrics, feature_names, model_version
        )
        mlflow.log_artifacts(artifacts_dir, "artifacts")
        
        # Log feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        mlflow.log_dict(feature_importance, "feature_importance.json")
        
        print(f"Model training completed successfully!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Cross-validation mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"Model artifacts saved to: {artifacts_dir}")
        print(f"MLflow run completed: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
