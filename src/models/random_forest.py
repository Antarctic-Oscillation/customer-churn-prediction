from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from .base_model import BaseChurnModel
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestChurnModel(BaseChurnModel):
    def __init__(self):
        super().__init__("Random Forest")

    def build_model(self, params: dict = None):
        """Build Random Forest model with filtered parameters"""
        default_params = {"random_state": 42, "n_jobs": -1}
        if params:
            filtered_params = self._filter_params(params)
            default_params.update(filtered_params)

        self.model = RandomForestClassifier(**default_params)
        return self.model

    def _filter_params(self, params: dict) -> dict:
        """Filter Random Forest parameters to avoid compatibility issues"""
        filtered = params.copy()

        if "max_features" in filtered:
            max_features = filtered["max_features"]
            if isinstance(max_features, str):
                valid_strings = ["auto", "sqrt", "log2", "None"]
                if max_features not in valid_strings and max_features not in [
                    "sqrt",
                    "log2",
                    "auto",
                ]:
                    # Convert invalid string to valid one
                    if max_features in ["auto"]:
                        filtered["max_features"] = (
                            "sqrt"  # 'auto' is deprecated in newer versions
                        )
                    else:
                        del filtered["max_features"]
                        warnings.warn(
                            f"Removed invalid max_features='{max_features}'. Valid options: {valid_strings}"
                        )

        if "min_samples_split" in filtered and isinstance(
            filtered["min_samples_split"], float
        ):
            if filtered["min_samples_split"] < 1.0:
                # Ensure it's a valid fraction
                if filtered["min_samples_split"] <= 0.0:
                    del filtered["min_samples_split"]
                    warnings.warn("Removed invalid min_samples_split <= 0.0")

        if "min_samples_leaf" in filtered and isinstance(
            filtered["min_samples_leaf"], float
        ):
            if filtered["min_samples_leaf"] < 1.0:
                # Ensure it's a valid fraction
                if filtered["min_samples_leaf"] <= 0.0:
                    del filtered["min_samples_leaf"]
                    warnings.warn("Removed invalid min_samples_leaf <= 0.0")

        if "oob_score" in filtered and filtered["oob_score"]:
            if "bootstrap" in filtered and not filtered["bootstrap"]:
                del filtered["oob_score"]
                warnings.warn("Removed oob_score=True because bootstrap=False")

        return filtered

    def get_param_grid(self) -> list:
        """Return parameter grids with compatible combinations"""
        param_grids = [
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10, 0.1, 0.2],
                "min_samples_leaf": [1, 2, 4, 0.1, 0.2],
                "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
                "bootstrap": [True],
                "oob_score": [False, True],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "bootstrap": [False],
                "oob_score": [False],
                "class_weight": ["balanced", None],
            },
        ]

        return param_grids[0]  # Return the most comprehensive one

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train with parameter filtering"""

        param_grids = self.get_param_grid()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                grid_search = RandomizedSearchCV(
                    self.model.__class__(random_state=42, n_jobs=-1),
                    param_distributions=param_grids,
                    n_iter=40,
                    cv=3,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=1,
                    random_state=42,
                )
                grid_search.fit(X_train, y_train)

                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
                logger.info(f"Best parameters: {self.best_params}")

        except Exception as e:
            logger.error(f"Grid search failed: {str(e)}")
            # Fallback to default parameters
            self.model.fit(X_train, y_train)
