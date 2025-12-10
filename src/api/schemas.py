from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class CustomerData(BaseModel):
    customerID: str = Field(..., description="Unique customer identifier")
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[int] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None


class PredictionResponse(BaseModel):
    customerID: str
    churn_probability: float
    churn_prediction: int
    churn_prediction_label: str
    confidence: str
    model_version: str
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, float]
