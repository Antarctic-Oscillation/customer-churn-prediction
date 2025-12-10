import pandas as pd
import logging
import argparse
from rich import print as rprint
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.data_processing.preprocessing import ChurnPreprocessor
from src.models.xgboost_model import XGBoostChurnModel
from src.models.logistic_regression import LogisticRegressionChurnModel
from src.models.random_forest import RandomForestChurnModel
from src.models.svm_model import SVMChurnModel
from src.models.neural_network import NeuralNetworkChurnModel
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "xgboost": XGBoostChurnModel,
    "logistic": LogisticRegressionChurnModel,
    "random_forest": RandomForestChurnModel,
    "svm": SVMChurnModel,
    "neural": NeuralNetworkChurnModel,
}

MODEL_STORE_DIR = Path("model_store")
MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)

def run_training(model_key: str, X_train, y_train, X_val, y_val, preprocessor: ChurnPreprocessor):
    logger.info(f"Training model: {model_key}")
    model_cls = MODEL_REGISTRY[model_key]
    model_obj = model_cls()
    model_obj.build_model()
    model_obj.train(X_train, y_train)

    results = model_obj.evaluate(X_val, y_val)

    model_path = MODEL_STORE_DIR / f"{model_key}_model.pkl"
    prep_path = MODEL_STORE_DIR / "preprocessor.pkl"
    model_obj.save_model(model_path)
    preprocessor.save(prep_path)
    logger.info(f"{model_key} model saved to {model_path}")

    return results


def generate_markdown_report(all_results: dict, output_path: str = "reports/model_report.md"):
    md_lines = ["# Customer Churn Model Performance Report\n"]
    md_lines.append(f"Generated: {pd.Timestamp.now()}\n")
    
    for model_name, res in all_results.items():
        md_lines.append(f"## {model_name}\n")
        md_lines.append(f"- ROC-AUC: {res['roc_auc']:.4f}")
        md_lines.append(f"- Best Params: `{res.get('best_params', {})}`")
        md_lines.append(f"- CV Mean Score: {res.get('cv_scores_mean', None)}\n")
        md_lines.append("### Classification Report\n")
        md_lines.append(f"```\n{res['classification_report']}\n```\n")
        md_lines.append("### Confusion Matrix\n")
        md_lines.append(f"```\n{res['confusion_matrix']}\n```\n")
    
    Path(output_path).write_text("\n".join(md_lines))
    logger.info(f"Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true", help="Train all models instead of a single one"
    )
    parser.add_argument(
        "--model", type=str, choices=list(MODEL_REGISTRY.keys()), help="Train a specific model"
    )
    args = parser.parse_args()

    logger.info("Loading data...")
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    logger.info("Splitting into train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Preprocessing...")
    preprocessor = ChurnPreprocessor()
    X_train_prep = preprocessor.fit_transform(X_train, y_train)
    X_val_prep = preprocessor.transform(X_val)

    all_results = {}

    if args.all:
        for model_key in MODEL_REGISTRY.keys():
            res = run_training(model_key, X_train_prep, y_train, X_val_prep, y_val, preprocessor)
            all_results[model_key] = res
    elif args.model:
        res = run_training(args.model, X_train_prep, y_train, X_val_prep, y_val, preprocessor)
        all_results[args.model] = res
    else:
        # default to training XGBoost
        res = run_training("xgboost", X_train_prep, y_train, X_val_prep, y_val, preprocessor)
        all_results["xgboost"] = res

    rprint(all_results)
    generate_markdown_report(all_results)


if __name__ == "__main__":
    main()
