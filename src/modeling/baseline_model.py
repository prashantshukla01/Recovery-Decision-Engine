"""Baseline Calibrated Logistic Regression Model (scikit-learn + Isotonic Calibration)."""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.simulation.generator import calculate_payday_proximity

BASELINE_FILE_PATH = "data/baseline_model.pkl"


def create_baseline_pipeline() -> Pipeline:
    """Creates a feature preprocessor + calibrated logistic regression pipeline."""
    categorical_features = ["decline_code", "action"]
    numeric_features = ["payday_prox", "retry_count", "customer_tenure_months", "subscription_value", "hours_since_failure"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    base_estimator = LogisticRegression(C=1.0, max_iter=1000)
    calibrated_model = CalibratedClassifierCV(estimator=base_estimator, method="isotonic", cv=3)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", calibrated_model),
        ]
    )
    return pipeline


def prepare_baseline_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds calculated feature columns like payday_prox to DataFrame for pipeline consumption."""
    df_feat = df.copy()
    df_feat["payday_prox"] = [calculate_payday_proximity(d) for d in df["day_of_month"]]
    return df_feat


def fit_baseline_model(train_df: pd.DataFrame) -> Pipeline:
    """Fits the baseline calibrated model on training DataFrame."""
    pipeline = create_baseline_pipeline()
    df_feat = prepare_baseline_df(train_df)
    X = df_feat[["decline_code", "action", "payday_prox", "retry_count", "customer_tenure_months", "subscription_value", "hours_since_failure"]]
    y = df_feat["outcome"].values
    pipeline.fit(X, y)
    return pipeline


def predict_baseline_p_success(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predicts point-estimate recovery probabilities P(success=1)."""
    df_feat = prepare_baseline_df(df)
    X = df_feat[["decline_code", "action", "payday_prox", "retry_count", "customer_tenure_months", "subscription_value", "hours_since_failure"]]
    probs = pipeline.predict_proba(X)[:, 1]
    return probs


def save_baseline_model(pipeline: Pipeline, path: str = BASELINE_FILE_PATH):
    """Saves baseline pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved baseline model to {path}")


def load_baseline_model(path: str = BASELINE_FILE_PATH) -> Pipeline:
    """Loads baseline pipeline from disk. Fits automatically if file is missing."""
    if not os.path.exists(path):
        print(f"[MODEL] Baseline model artifact '{path}' missing. Auto-fitting...")
        train_path = "data/train.csv"
        if not os.path.exists(train_path):
            from src.simulation.generator import generate_dataset
            train_df, _ = generate_dataset(num_events=500, seed=42)
        else:
            train_df = pd.read_csv(train_path)

        pipeline = fit_baseline_model(train_df)
        save_baseline_model(pipeline, path=path)
        return pipeline

    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
