import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

log = logging.getLogger(__name__)


class ChurnPreprocessor:
    """
    Preprocessor for Customer Churn:
    - Cleans data
    - Converts numeric/categorical columns
    - Target-encodes categorical features
    - Standard-scales numeric features
    - Ensures consistent feature order for API
    """

    def __init__(self):
        self.fitted = False
        self.column_transformer: ColumnTransformer = None
        self.feature_names_out = None

        self.categorical_cols = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]

        self.numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

        self.expected_columns = self.categorical_cols + self.numeric_cols

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)

        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self.expected_columns]

        return df

    def fit(self, df: pd.DataFrame, y: pd.Series):
        log.info("Fitting ChurnPreprocessor...")
        df = self._clean_dataframe(df)

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("cat", TargetEncoder(), self.categorical_cols),
                ("num", StandardScaler(), self.numeric_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self.column_transformer.fit(df, y)
        self.feature_names_out = self.categorical_cols + self.numeric_cols
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform()")

        df = self._clean_dataframe(df)
        arr = self.column_transformer.transform(df)
        return pd.DataFrame(arr, columns=self.feature_names_out)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series):
        return self.fit(df, y).transform(df)

    def save(self, path: str):
        if not self.fitted:
            raise RuntimeError("Cannot save before fitting")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "column_transformer": self.column_transformer,
                "feature_names_out": self.feature_names_out,
                "categorical_cols": self.categorical_cols,
                "numeric_cols": self.numeric_cols,
                "expected_columns": self.expected_columns,
            },
            path,
        )

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls()
        obj.column_transformer = data["column_transformer"]
        obj.feature_names_out = data["feature_names_out"]
        obj.categorical_cols = data.get("categorical_cols", obj.categorical_cols)
        obj.numeric_cols = data.get("numeric_cols", obj.numeric_cols)
        obj.expected_columns = data.get("expected_columns", obj.expected_columns)
        obj.fitted = True
        return obj
