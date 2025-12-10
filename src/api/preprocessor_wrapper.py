import joblib
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class PreprocessorWrapper:
    def __init__(self, path: str):
        self.path = path
        self.preprocessor = None
        self.feature_names = None
        self.expected_columns = None
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            logger.warning(f"Preprocessor file not found: {self.path}")
            return
        try:
            data = joblib.load(self.path)
            self.preprocessor = data["column_transformer"]
            self.feature_names = data["feature_names_out"]
            self.expected_columns = data.get("expected_columns", self.feature_names)
            logger.info(f"Preprocessor loaded with {len(self.feature_names)} features")
        except Exception as e:
            logger.exception(f"Failed to load preprocessor: {e}")
            self.preprocessor = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded")

        df = df.copy()
        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self.expected_columns]

        arr = self.preprocessor.transform(df)

        if not isinstance(arr, pd.DataFrame):
            arr = pd.DataFrame(arr, columns=self.feature_names)

        return arr
