import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from .base_model import BaseChurnModel
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVMChurnModel(BaseChurnModel):
    def __init__(self):
        super().__init__("SVM")
        
    def build_model(self, params: dict = None):
        """Build SVM model with filtered parameters"""
        default_params = {
            'probability': True,
            'random_state': 42
        }
        if params:
            filtered_params = self._filter_params(params)
            default_params.update(filtered_params)
        
        self.model = SVC(**default_params)
        return self.model
    
    def _filter_params(self, params: dict) -> dict:
        """Filter SVM parameters to avoid compatibility issues"""
        filtered = params.copy()
        
        kernel = filtered.get('kernel', 'rbf')
        
        if 'degree' in filtered and kernel != 'poly':
            del filtered['degree']
            warnings.warn(f"Removed degree parameter for kernel='{kernel}'. "
                        f"degree is only used with kernel='poly'")
        
        if 'coef0' in filtered and kernel not in ['poly', 'sigmoid']:
            del filtered['coef0']
            warnings.warn(f"Removed coef0 parameter for kernel='{kernel}'. "
                        f"coef0 is only used with kernel='poly' or 'sigmoid'")
        
        if 'gamma' in filtered:
            gamma = filtered['gamma']
            if kernel == 'linear':
                # gamma is ignored for linear kernel, but we can keep it for consistency
                pass
            elif isinstance(gamma, str) and gamma not in ['scale', 'auto']:
                filtered['gamma'] = 'scale'
                warnings.warn(f"Converted invalid gamma='{gamma}' to 'scale'")
        
        
        return filtered
    
    def get_param_grid(self) -> list:
        param_grids = [
            # RBF kernel
            {
                'kernel': ['rbf'],
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'class_weight': ['balanced', None]
            },
            # Linear kernel
            {
                'kernel': ['linear'],
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'class_weight': ['balanced', None]
            },
            # Polynomial kernel
            {
                'kernel': ['poly'],
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'degree': [2, 3, 4, 5],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'coef0': [0.0, 0.1, 0.5, 1.0],
                'class_weight': ['balanced', None]
            },
            # Sigmoid kernel
            {
                'kernel': ['sigmoid'],
                'C': [0.001, 0.01, 0.1, 1.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'coef0': [0.0, 0.1, 0.5, 1.0],
                'class_weight': ['balanced', None]
            }
        ]
        
        return param_grids
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train with kernel-specific parameter filtering"""
        
        param_grids = self.get_param_grid()
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for param_grid in param_grids:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Create base model with current parameter grid
                    base_model = self.model.__class__(probability=True, random_state=42)
                    grid_search = RandomizedSearchCV(
                        base_model,
                        param_grid,
                        cv=3,
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(X_train, y_train)
                    
                    if grid_search.best_score_ > best_score:
                        best_score = grid_search.best_score_
                        best_params = grid_search.best_params_
                        best_model = grid_search.best_estimator_
                        
            except Exception as e:
                logger.warning(f"Grid search failed for {param_grid['kernel'][0]} kernel: {str(e)}")
                continue
        
        if best_model is not None:
            self.model = best_model
            self.best_params = best_params
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {best_score:.4f}")