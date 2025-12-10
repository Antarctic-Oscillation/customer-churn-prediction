import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_telecom_data(self):
        print(self.data_path)
        if self.data_path.exists():
            return pd.read_csv(self.data_path)
        print(self.data_path)

        print("Downloading dataset from Kaggle...")
        return self._download_kaggle_dataset()

    def _download_kaggle_dataset(self):
        import kaggle  # requires Kaggle API setup

        kaggle.api.dataset_download_files(
            "blastchar/telco-customer-churn", path=self.data_path.parent, unzip=True
        )
        return pd.read_csv(
            self.data_path.parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
