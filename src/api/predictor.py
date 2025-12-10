import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

def predict_batch(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Some models accept DataFrame directly
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # fallback: use decision_function then apply sigmoid
        if hasattr(model, "decision_function"):
            df = model.decision_function(X)
            probs = 1 / (1 + np.exp(-df))
        else:
            # As last resort, use predict and map to 0/1 probabilities
            preds = model.predict(X)
            probs = np.array(preds, dtype=float)
    preds = (probs > 0.5).astype(int)
    return probs, preds

def summarize_predictions(preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    total = len(preds)
    high_risk = int((preds == 1).sum())
    avg_prob = float(np.mean(probs)) if total > 0 else 0.0
    return {
        "total_customers": total,
        "high_risk": high_risk,
        "low_risk": int(total - high_risk),
        "avg_churn_probability": avg_prob
    }
