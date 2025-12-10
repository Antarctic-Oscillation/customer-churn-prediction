import numpy as np
from sklearn.linear_model import LogisticRegression
from .base_model import BaseChurnModel
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogisticRegressionChurnModel(BaseChurnModel):
    def __init__(self):
        super().__init__("Logistic Regression")
        
    def build_model(self, params: dict = None):
        default_params = {
            'random_state': 42,
            'max_iter': 500,
            'class_weight': 'balanced'
        }
        if params:
            filtered_params = self._filter_params(params)
            default_params.update(filtered_params)
        
        self.model = LogisticRegression(**default_params)
        return self.model
    
    def _filter_params(self, params: dict) -> dict:
        filtered = params.copy()
        
        # If penalty is not elasticnet, remove l1_ratio
        if 'penalty' in filtered and filtered['penalty'] != 'elasticnet':
            if 'l1_ratio' in filtered:
                del filtered['l1_ratio']
                warnings.warn(f"Removed l1_ratio for penalty='{filtered['penalty']}'. "
                            f"l1_ratio is only used with penalty='elasticnet'")
        
        # If penalty is elasticnet, ensure solver is saga
        if filtered.get('penalty') == 'elasticnet':
            filtered['solver'] = 'saga'
            if 'l1_ratio' not in filtered:
                filtered['l1_ratio'] = 0.5
        
        return filtered
    
    def get_param_grid(self) -> dict:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [1000],
            'class_weight': ['balanced']
        }
        
        return param_grid
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        param_grid = self.get_param_grid()
        
        best_score = -np.inf
        best_params = None
        
        for C in param_grid['C']:
            for penalty in param_grid['penalty']:
                for solver in param_grid['solver']:
                    if penalty == 'elasticnet' and solver != 'saga':
                        continue
                    if penalty in ['l1', 'elasticnet'] and solver not in ['liblinear', 'saga']:
                        continue
                    
                    params = {
                        'C': C,
                        'penalty': penalty,
                        'solver': solver,
                        'max_iter': 1000,
                        'class_weight': 'balanced',
                        'random_state': 42
                    }
                    
                    # Add l1_ratio only for elasticnet
                    if penalty == 'elasticnet':
                        for l1_ratio in param_grid['l1_ratio']:
                            params['l1_ratio'] = l1_ratio
                            score = self._evaluate_params(X_train, y_train, params)
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
                    else:
                        score = self._evaluate_params(X_train, y_train, params)
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
        
        self.build_model(best_params)
        self.model.fit(X_train, y_train)
        self.best_params = best_params
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
    
    def _evaluate_params(self, X_train, y_train, params: dict) -> float:
        from sklearn.model_selection import cross_val_score
        
        model = LogisticRegression(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        return scores.mean()