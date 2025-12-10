# Customer Churn Prediction API
A machine learning system for predicting customer churn in telecommunications, featuring multiple ML models, automated training pipelines, and a FastAPI-based REST API server for single and batch-predictions.

## Overview

This project implements an end-to-end customer churn prediction system with:

- **5 Machine Learning Models**: XGBoost, Logistic Regression, Random Forest, SVM, and Neural Network
- **Automated Training Pipeline**: Hyperparameter tuning with cross-validation
- **REST API**: FastAPI-based service for single and batch predictions
- **Model Registry**: Dynamic model loading and version management
- **Preprocessing Pipeline**: Consistent data transformation with target encoding

## Project Structure

```
├── data/
│   └── raw/                          # dataset
├── model_store/                      # Saved models and preprocessor
├── reports/
│   └── model_report.md              # Auto-generated performance report
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI application
│   │   ├── model_registry.py        # Model management
│   │   ├── predictor.py             # Prediction logic
│   │   ├── preprocessor_wrapper.py  # Preprocessing wrapper
│   │   └── schemas.py               # Pydantic models
│   ├── data_processing/
│   │   ├── load_data.py             # Data loading utilities
│   │   └── preprocessing.py         # Preprocessing pipeline
│   └── models/
│       ├── base_model.py            # Abstract base model
│       ├── xgboost_model.py         # XGBoost
│       ├── logistic_regression.py   # Logistic Regression
│       ├── random_forest.py         # Random Forest
│       ├── svm_model.py             # Support Vector Machine
│       └── neural_network.py        # Neural Network (MLP)
├── train.py                         # Training script
└── requirements.txt                 # Python dependencies
```

## Features
### Data Preprocessing
The preprocessing pipeline handles:

- **Numeric features**: StandardScaler normalization
- **Categorical features**: Target encoding for better model performance
- **Missing values**: Median imputation for numeric features
- **Feature engineering**: Automatic handling of "SeniorCitizen" conversion

### Model Training
- **Automated hyperparameter tuning** using RandomizedSearchCV for efficieny 
- **Cross-validation** with ROC-AUC scoring
- **Parameter compatibility filtering** to prevent training errors
- **Model persistence** with joblib serialization
- **Performance reporting** with classification metrics and confusion matrices

### API Capabilities
- **Single prediction**: Real-time predictions for individual customers
- **Batch prediction**: Efficient processing of multiple customers
- **Model selection**: Choose specific models or use the default (XGBoost)
- **Feature importance**: Extract model interpretability insights
- **Health checks**: Monitor API and model status
- **CORS enabled**: Ready for web application integration

### Model Performance

After training, check `reports/model_report.md` for detailed performance metrics:

- ROC-AUC scores
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- Best hyperparameters
- Cross-validation scores

**Current best performers:**
- **XGBoost**: ROC-AUC 0.8463
- **Neural Network**: ROC-AUC 0.8452
- **Random Forest**: ROC-AUC 0.8443

## Setup

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Antarctic-Oscillation/customer-churn-prediction
cd customer-churn-prediction
```

2. **Install UV**
guide can be found here: [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

3. **Install dependencies**
```bash
uv add -r requirements.txt
```

## Usage

### Training Models

**Train all models:**
```bash
uv run train.py --all
```
**Train default model (XGBoost):**
```bash
uv run train.py
```

**Train a specific model:**
```bash
uv run train.py --model xgboost
uv run train.py --model logistic
uv run train.py --model random_forest
uv run train.py --model svm
uv run train.py --model neural
```


Training outputs:
- Saved models in `model_store/`
- Preprocessor pipeline in `model_store/preprocessor.pkl`
- Performance report in `reports/model_report.md`

### Running the API

**Start the API server:**
```bash
uv run src.api.main
```

Or using uvicorn directly:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/models` | GET | List models |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch prediction |
| `/model/{name}/feature_importance` | GET | Feature importance |

# API Curl Request Examples

Complete collection of curl commands for testing all API endpoints.

### Table of Contents
- [Root Endpoint](#root-endpoint)
- [Health Check](#health-check)
- [List Models](#list-models)
- [Single Prediction](#single-prediction)
- [Batch Prediction](#batch-prediction)
- [Feature Importance](#feature-importance)

---

### Root Endpoint

Get API information and available models.

```bash
curl -X GET http://localhost:8000/
```

---

### Health Check

Check API health status and model loading status.

```bash
curl -X GET http://localhost:8000/health
```

---

### List Models

Get all loaded models with their versions and metadata.

```bash
curl -X GET http://localhost:8000/models
```

---

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "1234-ABCD",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.85,
    "TotalCharges": 1078.20
  }'
```

### Prediction with Specific Model
**Using XGBoost:**
```bash
curl -X POST "http://localhost:8000/predict?model_name=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "1234-ABCD",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.85,
    "TotalCharges": 1078.20
  }'
```

---

### Batch Prediction

#### Small Batch (2-5 Customers)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "customerID": "1234-ABCD",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 1078.20
      },
      {
        "customerID": "5678-EFGH",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 34,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 56.95,
        "TotalCharges": 1889.50
      }
    ]
  }'
```

---

### Feature Importance

```bash
curl -X GET http://localhost:8000/model/xgboost/feature_importance
```

---

### Save Response to File

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @batch_customers.json \
  -o batch_results.json
```