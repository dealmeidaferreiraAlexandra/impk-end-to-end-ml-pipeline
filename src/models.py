from __future__ import annotations

import json
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import ARTIFACTS_DIR, METRICS_PATH, MODEL_PATHS, MODEL_VERSIONS
from .features import engineer_features

NUMERIC_COLS = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "HasCabin",
    "FarePerPerson",
]

CATEGORICAL_COLS = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "TicketPrefix",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )

def get_estimators():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
    }

def train_and_save_models(df: pd.DataFrame, target_col: str = "Survived") -> Tuple[Dict[str, Pipeline], pd.DataFrame, str]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

    working = engineer_features(df)
    working = working.dropna(subset=[target_col]).copy()

    X = working[FEATURE_COLS].copy()
    y = working[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    trained_models: Dict[str, Pipeline] = {}
    rows = []

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, estimator in get_estimators().items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "Version": MODEL_VERSIONS[name],
            "Accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "Precision": round(float(precision_score(y_test, preds)), 4),
            "Recall": round(float(recall_score(y_test, preds)), 4),
            "F1": round(float(f1_score(y_test, preds)), 4),
            "ROC_AUC": round(float(roc_auc_score(y_test, proba)), 4),
        }
        rows.append(row)

        trained_models[name] = pipeline
        joblib.dump(pipeline, MODEL_PATHS[name])

    metrics_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    metrics_df.to_json(METRICS_PATH, orient="records", indent=2)

    best_model_name = metrics_df.iloc[0]["Model"]
    return trained_models, metrics_df, best_model_name

def load_saved_artifacts():
    models: Dict[str, Pipeline] = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            models[name] = joblib.load(path)

    metrics_df = None
    if METRICS_PATH.exists():
        metrics_df = pd.read_json(METRICS_PATH)

    best_model_name = None
    if metrics_df is not None and not metrics_df.empty:
        best_model_name = metrics_df.iloc[0]["Model"]

    return models, metrics_df, best_model_name

def predict_single_row(models: Dict[str, Pipeline], row_df: pd.DataFrame):
    if row_df.empty:
        raise ValueError("No row provided for prediction.")

    working = engineer_features(row_df)
    X = working[FEATURE_COLS].copy()

    output = {}
    for name, pipeline in models.items():
        survival_prob = float(pipeline.predict_proba(X)[0, 1])
        pred = int(survival_prob >= 0.5)
        output[name] = {
            "prediction": pred,
            "label": "Survived" if pred == 1 else "Did not survive",
            "survival_prob": survival_prob,
            "uncertainty": round(1.0 - survival_prob, 4),
        }

    return output