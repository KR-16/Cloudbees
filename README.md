# MLOps Takehome Assessment - Iris Classification Pipeline

## Overview

This project demonstrates a complete MLOps workflow for iris flower classification, implementing best practices around pipeline design, deployment, and observability. The solution includes automated model versioning, experiment tracking with MLflow, a deployment-ready FastAPI service, and comprehensive monitoring capabilities.

## What I Implemented and Why

I chose to build a comprehensive MLOps pipeline that addresses multiple aspects of the assessment requirements:

1. **ML Pipeline Extension**: Created an automated training pipeline with MLflow integration that handles data preprocessing, model training, evaluation, and artifact logging
2. **Deployment-Ready API**: Built a FastAPI service with Docker containerization for production-ready model serving
3. **Observability & Metrics**: Implemented comprehensive monitoring including prediction tracking, confidence scoring, and performance metrics
4. **Automated Versioning**: Integrated MLflow for experiment tracking, model versioning, and artifact management

This approach demonstrates end-to-end MLOps capabilities while keeping the scope focused and practical.

## Project Structure

```
├── train_model.py          # ML training pipeline with MLflow integration
├── api.py                  # FastAPI deployment wrapper
├── test_api.py            # API testing script
├── run_pipeline.py        # End-to-end pipeline runner
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── README.md            # This file
```

## Features Implemented

### 1. ML Pipeline with MLflow Integration
- **Automated Training**: Random Forest classifier with hyperparameter tracking
- **Data Preprocessing**: StandardScaler with artifact persistence
- **Model Evaluation**: Cross-validation and comprehensive metrics
- **Experiment Tracking**: MLflow integration for reproducibility
- **Artifact Management**: Model, scaler, and metadata versioning

### 2. Production-Ready API
- **FastAPI Framework**: High-performance async API with automatic documentation
- **Input Validation**: Pydantic models for robust data validation
- **Error Handling**: Comprehensive error handling and logging
- **Health Checks**: Built-in health monitoring endpoints
- **Auto-reload**: Model loading from MLflow registry

### 3. Observability & Monitoring
- **Prediction Tracking**: Complete prediction history with timestamps
- **Confidence Scoring**: Model confidence metrics for each prediction
- **Performance Metrics**: Real-time API performance monitoring
- **Model Metadata**: Version tracking and feature importance logging

### 4. Containerization
- **Docker Support**: Production-ready containerization
- **Docker Compose**: Easy deployment and orchestration
- **Health Checks**: Container-level health monitoring
- **Volume Mounting**: Persistent storage for MLflow and artifacts

## How to Run

### Option 1: Complete Pipeline (Recommended)
```bash
# Run the complete MLOps pipeline
python run_pipeline.py
```

This will:
1. Install dependencies
2. Train the model with MLflow tracking
3. Start the API server
4. Run comprehensive tests
5. Display all endpoints and monitoring URLs

### Option 2: Step-by-Step Execution

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Train the Model
```bash
python train_model.py
```

#### 3. Start the API Server
```bash
python api.py
```

#### 4. Test the API
```bash
python test_api.py
```

### Option 3: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t iris-classifier .
docker run -p 8000:8000 -v $(pwd)/mlruns:/app/mlruns -v $(pwd)/artifacts:/app/artifacts iris-classifier
```

## API Endpoints

Once the server is running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Info**: http://localhost:8000/model/info
- **Metrics**: http://localhost:8000/metrics
- **Prediction History**: http://localhost:8000/predictions/history

### Example API Usage

```python
import requests

# Make a prediction
response = requests.post("http://localhost:8000/predict", json={
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
})

print(response.json())
# {
#   "prediction": 0,
#   "prediction_proba": [0.95, 0.03, 0.02],
#   "confidence": 0.95,
#   "model_version": "20241201_143022",
#   "timestamp": "2024-12-01T14:30:25.123456"
# }
```

## MLflow Integration

The project includes comprehensive MLflow integration:

```bash
# View MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Access at http://localhost:5000
```

MLflow tracks:
- Model parameters and hyperparameters
- Training and validation metrics
- Model artifacts and metadata
- Feature importance
- Model versions and lineage

## Monitoring & Observability

The API provides several monitoring endpoints:

- **`/health`**: Service health and model status
- **`/metrics`**: Real-time performance metrics
- **`/predictions/history`**: Complete prediction audit trail
- **`/model/info`**: Model metadata and configuration

## Assumptions and Limitations

### Assumptions
- Python 3.9+ environment
- Standard iris dataset (4 features, 3 classes)
- Local MLflow tracking (file-based)
- Single model deployment (no A/B testing)
- In-memory prediction history (not persistent)

### Limitations
- Prediction history limited to 1000 entries in memory
- No database persistence for production scale
- Single model version serving (no model routing)
- Basic error handling (no retry mechanisms)
- Local deployment only (no cloud integration)

### Production Considerations
For production deployment, consider:
- Database integration for prediction persistence
- Model versioning and A/B testing
- Distributed logging and monitoring
- Cloud deployment and scaling
- Security and authentication
- Rate limiting and request validation

## Testing

The project includes comprehensive testing:

```bash
# Run API tests
python test_api.py

# Test specific endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## Dependencies

Key dependencies:
- **scikit-learn**: Machine learning framework
- **fastapi**: Web framework for API
- **mlflow**: ML lifecycle management
- **pandas/numpy**: Data processing
- **docker**: Containerization
- **uvicorn**: ASGI server

## Next Steps

This implementation provides a solid foundation for MLOps workflows. Potential enhancements include:

1. **Database Integration**: Persistent storage for predictions and metrics
2. **Model Monitoring**: Drift detection and performance monitoring
3. **CI/CD Pipeline**: Automated testing and deployment
4. **Cloud Integration**: AWS/GCP/Azure deployment
5. **Advanced Monitoring**: Prometheus/Grafana integration
6. **Security**: Authentication and authorization
7. **Scaling**: Horizontal scaling and load balancing

---

*This project demonstrates MLOps best practices in a focused, end-to-end implementation suitable for assessment and learning purposes.*
