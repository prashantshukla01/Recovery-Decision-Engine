"""PyMC Hierarchical Bayesian Logistic Regression Model for Failure Recovery.

Model Spec:
    logit(p_i) = alpha + u[decline_code_i] + v[action_i] + beta^T x_i
    u ~ Normal(0, sigma_decline), sigma_decline ~ HalfNormal(1)
    v ~ Normal(0, sigma_action),  sigma_action  ~ HalfNormal(1)
    alpha ~ Normal(0, 2)
    beta ~ Normal(0, 1)
"""

import math
import os
import pickle
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Tuple, Optional
from src.simulation.generator import calculate_payday_proximity, DECLINE_CODES, ACTIONS

MODEL_FILE_PATH = "data/pymc_model_idata.pkl"


class FeaturePreprocessor:
    """Preprocesses FailureContext & action records into numerical arrays for modeling."""

    def __init__(self):
        self.decline_codes = list(DECLINE_CODES)
        self.actions = list(ACTIONS)
        self.decline_map = {code: i for i, code in enumerate(self.decline_codes)}
        self.action_map = {act: i for i, act in enumerate(self.actions)}
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame):
        """Fits standardization scalers on numeric columns."""
        for col in ["customer_tenure_months", "subscription_value", "hours_since_failure"]:
            self.means[col] = float(df[col].mean())
            self.stds[col] = float(df[col].std()) if df[col].std() > 1e-6 else 1.0

    def transform_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transforms a DataFrame into decline_indices, action_indices, feature_matrix."""
        decline_idx = np.array([self.decline_map.get(c, 0) for c in df["decline_code"]], dtype=int)
        action_idx = np.array([self.action_map.get(a, 0) for a in df["action"]], dtype=int)

        payday_prox = np.array([calculate_payday_proximity(d) for d in df["day_of_month"]], dtype=float)
        retries = df["retry_count"].values.astype(float)
        tenure_std = (df["customer_tenure_months"].values - self.means.get("customer_tenure_months", 0.0)) / self.stds.get("customer_tenure_months", 1.0)
        sub_val_std = (df["subscription_value"].values - self.means.get("subscription_value", 0.0)) / self.stds.get("subscription_value", 1.0)
        hours_std = (df["hours_since_failure"].values - self.means.get("hours_since_failure", 0.0)) / self.stds.get("hours_since_failure", 1.0)

        # Matrix X: [payday_prox, retries, tenure_std, sub_val_std, hours_std]
        X = np.column_stack([payday_prox, retries, tenure_std, sub_val_std, hours_std])
        return decline_idx, action_idx, X

    def transform_single(self, decline_code: str, action: str, day_of_month: int, retry_count: int,
                         customer_tenure_months: int, subscription_value: float, hours_since_failure: float) -> Tuple[int, int, np.ndarray]:
        """Transforms a single failure context into decline_idx, action_idx, x_vector (shape 1, K)."""
        d_idx = self.decline_map.get(decline_code, 0)
        a_idx = self.action_map.get(action, 0)

        payday_prox = calculate_payday_proximity(day_of_month)
        tenure_std = (customer_tenure_months - self.means.get("customer_tenure_months", 0.0)) / self.stds.get("customer_tenure_months", 1.0)
        sub_val_std = (subscription_value - self.means.get("subscription_value", 0.0)) / self.stds.get("subscription_value", 1.0)
        hours_std = (hours_since_failure - self.means.get("hours_since_failure", 0.0)) / self.stds.get("hours_since_failure", 1.0)

        x_vec = np.array([[payday_prox, float(retry_count), tenure_std, sub_val_std, hours_std]])
        return d_idx, a_idx, x_vec


def build_pymc_model(
    decline_idx: np.ndarray,
    action_idx: np.ndarray,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    num_decline_codes: int = 5,
    num_actions: int = 7,
) -> pm.Model:
    """Constructs the PyMC Bayesian Hierarchical Logistic Regression model."""
    num_features = X.shape[1]

    with pm.Model() as model:
        # Data containers
        d_idx_obs = pm.Data("d_idx_obs", decline_idx)
        a_idx_obs = pm.Data("a_idx_obs", action_idx)
        X_obs = pm.Data("X_obs", X)

        # Global Intercept
        alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)

        # Partial Pooling over Decline Codes
        sigma_decline = pm.HalfNormal("sigma_decline", sigma=1.0)
        u_decline = pm.Normal("u_decline", mu=0.0, sigma=sigma_decline, shape=num_decline_codes)

        # Partial Pooling over Actions
        sigma_action = pm.HalfNormal("sigma_action", sigma=1.0)
        v_action = pm.Normal("v_action", mu=0.0, sigma=sigma_action, shape=num_actions)

        # Predictor Coefficients
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=num_features)

        # Logit calculation
        logit_p = alpha + u_decline[d_idx_obs] + v_action[a_idx_obs] + pm.math.dot(X_obs, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

        # Likelihood
        if y is not None:
            pm.Bernoulli("obs", p=p, observed=y)

    return model


def fit_bayesian_model(
    train_df: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 42,
) -> Tuple[az.InferenceData, FeaturePreprocessor]:
    """Fits the PyMC model on training data and returns InferenceData and fitted preprocessor."""
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(train_df)
    d_idx, a_idx, X = preprocessor.transform_df(train_df)
    y = train_df["outcome"].values

    pymc_model = build_pymc_model(
        decline_idx=d_idx,
        action_idx=a_idx,
        X=X,
        y=y,
        num_decline_codes=len(preprocessor.decline_codes),
        num_actions=len(preprocessor.actions),
    )

    with pymc_model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=False,
        )

    return idata, preprocessor


def check_convergence(idata: az.InferenceData) -> Tuple[bool, Dict[str, float]]:
    """Checks convergence metrics (R-hat and divergences)."""
    summary = az.summary(idata, var_names=["alpha", "sigma_decline", "sigma_action", "beta"])
    r_hat_max = float(summary["r_hat"].max())

    num_divergences = 0
    if hasattr(idata, "sample_stats") and "divergencing" in idata.sample_stats:
        num_divergences = int(idata.sample_stats.diverging.sum().values)

    is_converged = (r_hat_max <= 1.05) and (num_divergences == 0)
    metrics = {
        "r_hat_max": r_hat_max,
        "num_divergences": float(num_divergences),
    }
    return is_converged, metrics


def save_model_artifacts(idata: az.InferenceData, preprocessor: FeaturePreprocessor, path: str = MODEL_FILE_PATH):
    """Saves fitted InferenceData and Preprocessor to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"idata": idata, "preprocessor": preprocessor}, f)
    print(f"Saved PyMC model artifacts to {path}")


def load_model_artifacts(path: str = MODEL_FILE_PATH) -> Tuple[az.InferenceData, FeaturePreprocessor]:
    """Loads fitted InferenceData and Preprocessor from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifacts not found at {path}. Run model fitting first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["idata"], data["preprocessor"]
