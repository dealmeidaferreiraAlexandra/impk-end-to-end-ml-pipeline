from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "train.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "Logistic Regression": ARTIFACTS_DIR / "logistic_regression.joblib",
    "Random Forest": ARTIFACTS_DIR / "random_forest.joblib",
    "SVM": ARTIFACTS_DIR / "svm.joblib",
}

MODEL_VERSIONS = {
    "Logistic Regression": "lr_v1",
    "Random Forest": "rf_v1",
    "SVM": "svm_v1",
}

METRICS_PATH = ARTIFACTS_DIR / "metrics.json"