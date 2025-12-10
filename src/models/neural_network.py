import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from .base_model import BaseChurnModel
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralNetworkChurnModel(BaseChurnModel):
    def __init__(self):
        super().__init__("Neural Network")
        
    def build_model(self, params: dict = None):
        """Build Neural Network model with filtered parameters"""
        default_params = {
            'random_state': 42,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        if params:
            filtered_params = self._filter_params(params)
            default_params.update(filtered_params)
        
        self.model = MLPClassifier(**default_params)
        return self.model
    
    def _filter_params(self, params: dict) -> dict:
        """Filter Neural Network parameters to avoid compatibility issues"""
        filtered = params.copy()
        
        solver = filtered.get('solver', 'adam')
        
        if 'learning_rate' in filtered:
            learning_rate = filtered['learning_rate']
            
            if 'learning_rate_init' in filtered and learning_rate == 'constant':
                if filtered['learning_rate_init'] <= 0:
                    del filtered['learning_rate_init']
                    warnings.warn("Removed invalid learning_rate_init <= 0")
            
            if 'power_t' in filtered and learning_rate != 'invscaling':
                del filtered['power_t']
                warnings.warn(f"Removed power_t for learning_rate='{learning_rate}'. "
                            f"power_t is only used with learning_rate='invscaling'")
        
        if 'momentum' in filtered and solver != 'sgd':
            del filtered['momentum']
            warnings.warn(f"Removed momentum for solver='{solver}'. "
                        f"momentum is only used with solver='sgd'")
        
        if 'nesterovs_momentum' in filtered and solver != 'sgd':
            del filtered['nesterovs_momentum']
            warnings.warn(f"Removed nesterovs_momentum for solver='{solver}'. "
                        f"nesterovs_momentum is only used with solver='sgd'")
        
        if 'beta_1' in filtered and solver != 'adam':
            del filtered['beta_1']
            warnings.warn(f"Removed beta_1 for solver='{solver}'. "
                        f"beta_1 is only used with solver='adam'")
        
        if 'beta_2' in filtered and solver != 'adam':
            del filtered['beta_2']
            warnings.warn(f"Removed beta_2 for solver='{solver}'. "
                        f"beta_2 is only used with solver='adam'")
        
        if 'epsilon' in filtered and solver != 'adam':
            del filtered['epsilon']
            warnings.warn(f"Removed epsilon for solver='{solver}'. "
                        f"epsilon is only used with solver='adam'")
        
        if 'batch_size' in filtered:
            batch_size = filtered['batch_size']
            if solver == 'lbfgs' and batch_size != 'auto':
                # lbfgs doesn't use mini-batches, so batch_size should be auto
                filtered['batch_size'] = 'auto'
                warnings.warn("Set batch_size='auto' for lbfgs solver")
        
        if 'hidden_layer_sizes' in filtered:
            hidden_sizes = filtered['hidden_layer_sizes']
            if isinstance(hidden_sizes, (list, tuple)):
                if any(size <= 0 for size in hidden_sizes):
                    valid_sizes = [size for size in hidden_sizes if size > 0]
                    if valid_sizes:
                        filtered['hidden_layer_sizes'] = tuple(valid_sizes)
                        warnings.warn(f"Filtered hidden_layer_sizes to {valid_sizes}")
                    else:
                        del filtered['hidden_layer_sizes']
                        warnings.warn("Removed invalid hidden_layer_sizes")
        
        return filtered
    
    def get_param_grid(self) -> list:
        """Return parameter grids with solver-specific combinations"""
        param_grids = [
            # Adam solver
            {
                'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128, 64), (100,)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.001, 0.01],
                'beta_1': [0.9],
                'beta_2': [0.999],
                'epsilon': [1e-08],
                'early_stopping': [True, False],
                'validation_fraction': [0.1],
                'max_iter': [500]
            },
            # SGD solver
            {
                'hidden_layer_sizes': [(64, 32), (128, 64), (100,)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': [0.001, 0.01],
                'power_t': [0.5],
                'momentum': [0.9],
                'nesterovs_momentum': [True, False],
                'early_stopping': [True, False],
                'validation_fraction': [0.1],
                'max_iter': [500]
            },
            # LBFGS solver
            {
                'hidden_layer_sizes': [(64, 32), (128, 64), (100,)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant'],
                'learning_rate_init': [0.001, 0.01],
                'early_stopping': [True, False],
                'validation_fraction': [0.1],
                'max_iter': [500]
            }
        ]
        
        return param_grids
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        param_grids = self.get_param_grid()
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for param_grid in param_grids:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    base_model = self.model.__class__(
                        random_state=42,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.1
                    )
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
                logger.warning(f"Grid search failed for {param_grid['solver'][0]} solver: {str(e)}")
                continue
        
        if best_model is not None:
            self.model = best_model
            self.best_params = best_params
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {best_score:.4f}")