from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from .schemas import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from .model_registry import ModelRegistry
from .preprocessor_wrapper import PreprocessorWrapper
from .predictor import predict_batch, summarize_predictions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATHS = {
    "logistic_regression": "model_store/logistic_regression_model.pkl",
    "random_forest": "model_store/random_forest_model.pkl",
    "xgboost": "model_store/xgboost_model.pkl",
    "svm": "model_store/svm_model.pkl",
    "neural_network": "model_store/neural_network_model.pkl",
}
PREPROCESSOR_PATH = "model_store/preprocessor.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Customer Churn Prediction API (enterprise variant)...")

    app.state.model_registry = ModelRegistry(MODEL_PATHS)
    app.state.model_registry.load_models()
    app.state.preprocessor = PreprocessorWrapper(PREPROCESSOR_PATH)

    app.state.default_model = app.state.model_registry.default_model()
    logger.info(f"Default model set to: {app.state.default_model}")

    yield

    # Shutdown: cleanup
    logger.info("Shutting down API...")
    app.state.model_registry = None
    app.state.preprocessor = None
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Customer Churn Prediction API (Enterprise)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Customer Churn Prediction API (Enterprise)",
        "version": app.version,
        "status": "active",
        "available_models": (
            list(app.state.model_registry.registry.keys())
            if app.state.model_registry
            else []
        ),
        "default_model": app.state.default_model,
    }


@app.get("/health")
async def health():
    registry = app.state.model_registry
    pre = app.state.preprocessor
    return {
        "status": "healthy" if registry and registry.registry else "degraded",
        "models_loaded": len(registry.registry) if registry else 0,
        "preprocessor_loaded": pre.preprocessor is not None if pre else False,
        "timestamp": datetime.utcnow(),
    }


@app.get("/models")
async def list_models():
    return app.state.model_registry.list_models()

def probability_to_confidence(p: float) -> str:
    if p > 0.8:
        return "Very High"
    if p > 0.6:
        return "High"
    if p > 0.4:
        return "Medium"
    return "Low"

# unified prediction function used by endpoints
def _prepare_and_predict(df: pd.DataFrame, model_key: str):
    pre = app.state.preprocessor
    if pre and pre.preprocessor is not None:
        X = pre.transform(df)
    else:
        X = df.select_dtypes(include=[np.number]).fillna(0)

    model_entry = app.state.model_registry.get(model_key)
    if model_entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")

    model = model_entry["model"]
    probs, preds = predict_batch(model, X)
    return probs, preds

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData, model_name: Optional[str] = None):
    if not app.state.model_registry or not app.state.model_registry.registry:
        raise HTTPException(status_code=503, detail="No models loaded")

    model_key = model_name or app.state.default_model
    if model_key not in app.state.model_registry.registry:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")

    df = pd.DataFrame([customer.dict()])
    try:
        probs, preds = _prepare_and_predict(df, model_key)
        prob = float(probs[0])
        pred = int(preds[0])
        return PredictionResponse(
            customerID=customer.customerID,
            churn_probability=prob,
            churn_prediction=pred,
            churn_prediction_label="Yes" if pred == 1 else "No",
            confidence=probability_to_confidence(prob),
            model_version=app.state.model_registry.get(model_key)["version"],
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.exception(f"predict_single error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_endpoint(
    request: BatchPredictionRequest, background_tasks: BackgroundTasks
):
    if not app.state.model_registry or not app.state.model_registry.registry:
        raise HTTPException(status_code=503, detail="No models loaded")

    model_key = app.state.default_model
    if model_key not in app.state.model_registry.registry:
        raise HTTPException(
            status_code=404, detail=f"Default model {model_key} not found"
        )

    df = pd.DataFrame([c.dict() for c in request.customers])
    try:
        probs, preds = _prepare_and_predict(df, model_key)
        model_version = app.state.model_registry.get(model_key)["version"]
        ts = datetime.now()
        responses = []
        for cid, p, pr in zip(df["customerID"].tolist(), probs, preds):
            responses.append(
                {
                    "customerID": cid,
                    "churn_probability": float(p),
                    "churn_prediction": int(pr),
                    "churn_prediction_label": "Yes" if pr == 1 else "No",
                    "confidence": probability_to_confidence(float(p)),
                    "model_version": model_version,
                    "timestamp": ts,
                }
            )

        summary = summarize_predictions(np.array(preds), np.array(probs))
        background_tasks.add_task(
            logger.info, f"Batch predicted {len(responses)} rows, summary={summary}"
        )

        return BatchPredictionResponse(predictions=responses, summary=summary)
    except Exception as e:
        logger.exception(f"predict_batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/{model_name}/feature_importance")
async def feature_importance(model_name: str):
    entry = app.state.model_registry.get(model_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found")

    model = entry["model"]

    pre = app.state.preprocessor
    feature_names = pre.feature_names if pre else None

    try:
        if hasattr(model, "feature_importances_"):
            import numpy as np

            imp = model.feature_importances_
            if feature_names and len(feature_names) == len(imp):
                fmap = dict(zip(feature_names, imp.tolist()))
            else:
                fmap = {f"f_{i}": float(v) for i, v in enumerate(imp.tolist())}
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_[0]).tolist()
            if feature_names and len(feature_names) == len(coef):
                fmap = dict(zip(feature_names, coef))
            else:
                fmap = {f"f_{i}": float(v) for i, v in enumerate(coef)}
        else:
            raise HTTPException(
                status_code=400, detail="Model does not expose feature importance"
            )
        top = sorted(fmap.items(), key=lambda x: x[1], reverse=True)[:20]
        return {"model": model_name, "feature_importance": fmap, "top_features": top}
    except Exception as e:
        logger.exception(f"feature_importance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
