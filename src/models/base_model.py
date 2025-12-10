from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseChurnModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.best_params = None
        self.cv_scores = None
        self.training_history = {}
        
    @abstractmethod
    def build_model(self, params: dict = None):
        pass
    
    @abstractmethod
    def get_param_grid(self) -> dict:
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        logger.info(f"Training {self.model_name}...")
        
        param_grid = self.get_param_grid()
        if param_grid:
            # using RandomizedSearchCV instead of GridSearchCV for the sake of effeciency
            grid_search = RandomizedSearchCV(
                self.model, param_grid, cv=3, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f"Best parameters: {self.best_params}")
        else:
            self.model.fit(X_train, y_train)
        
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring='roc_auc'
        )
        logger.info(f"CV ROC-AUC: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std() * 2:.4f})")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        results = {
            'model_name': self.model_name,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'cv_scores_mean': self.cv_scores.mean() if self.cv_scores is not None else None,
            'best_params': self.best_params
        }
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'training_history': self.training_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.best_params = saved_data['best_params']
        self.cv_scores = saved_data['cv_scores']
        self.training_history = saved_data['training_history']
        logger.info(f"Model loaded from {filepath}")