from xgboost import XGBClassifier
from .base_model import BaseChurnModel

class XGBoostChurnModel(BaseChurnModel):
    def __init__(self):
        super().__init__("XGBoost")
        
    def build_model(self, params: dict = None):
        default_params = {
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        
        self.model = XGBClassifier(**default_params)
        return self.model
    
    def get_param_grid(self) -> dict:
        """Return XGBoost hyperparameter grid based on latest research"""
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1]
        }