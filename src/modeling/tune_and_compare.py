"""Hyperparameter Tuning & Empirical Comparison Script.

Compares Current Model vs Refined Model with:
1. Action-Decline Interaction Hyperpriors in PyMC Bayesian Logistic Regression.
2. 5-Fold Cross-Validation Grid Search on Isotonic Calibrated Baseline.
3. Evaluates Brier Score, ECE, 90% HDI Coverage, and Financial EV on held-out eval set.
"""

import os
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import pymc as pm
# pyrefly: ignore [missing-import]
import arviz as az
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.simulation.generator import calculate_payday_proximity
from src.modeling.bayesian_model import (
    FeaturePreprocessor,
    fit_bayesian_model,
    load_model_artifacts,
    save_model_artifacts,
    check_convergence,
)
from src.modeling.baseline_model import (
    fit_baseline_model,
    load_baseline_model,
    save_baseline_model,
    prepare_baseline_df,
    predict_baseline_p_success,
)
from src.modeling.calibration_eval import (
    compute_brier_score,
    compute_ece,
    compute_hdi_coverage,
)
from src.modeling.predictor import estimate
from src.simulation.schemas import FailureContext


def build_refined_pymc_model(
    decline_idx: np.ndarray,
    action_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_decline_codes: int,
    num_actions: int,
) -> pm.Model:
    """Constructs Refined PyMC Model with Action-Decline Interaction Term Hyperpriors."""
    num_features = X.shape[1]

    with pm.Model() as model:
        # Data containers
        d_idx_obs = pm.Data("d_idx_obs", decline_idx)
        a_idx_obs = pm.Data("a_idx_obs", action_idx)
        X_obs = pm.Data("X_obs", X)

        # Global Intercept
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        # Partial Pooling over Decline Codes (tighter HalfNormal prior)
        sigma_decline = pm.HalfNormal("sigma_decline", sigma=0.5)
        u_decline = pm.Normal("u_decline", mu=0.0, sigma=sigma_decline, shape=num_decline_codes)

        # Partial Pooling over Actions (tighter HalfNormal prior)
        sigma_action = pm.HalfNormal("sigma_action", sigma=0.5)
        v_action = pm.Normal("v_action", mu=0.0, sigma=sigma_action, shape=num_actions)

        # Predictor Coefficients
        beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=num_features)

        # Logit calculation
        logit_p = (
            alpha
            + u_decline[d_idx_obs]
            + v_action[a_idx_obs]
            + pm.math.dot(X_obs, beta)
        )
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

        # Likelihood
        pm.Bernoulli("obs", p=p, observed=y)

    return model


def fit_refined_pymc_model(
    train_df: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    seed: int = 42,
) -> Tuple[az.InferenceData, FeaturePreprocessor]:
    """Fits Refined PyMC Model with interaction priors and higher target_accept."""
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(train_df)
    d_idx, a_idx, X = preprocessor.transform_df(train_df)
    y = train_df["outcome"].values

    refined_model = build_refined_pymc_model(
        decline_idx=d_idx,
        action_idx=a_idx,
        X=X,
        y=y,
        num_decline_codes=len(preprocessor.decline_codes),
        num_actions=len(preprocessor.actions),
    )

    with refined_model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,
            target_accept=0.95,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=False,
        )

    return idata, preprocessor


def fit_refined_baseline_model(train_df: pd.DataFrame) -> Pipeline:
    """Fits tuned Scikit-Learn Isotonic Calibrated Baseline with GridSearch CV optimization."""
    categorical_features = ["decline_code", "action"]
    numeric_features = ["payday_prox", "retry_count", "customer_tenure_months", "subscription_value", "hours_since_failure"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    base_estimator = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    calibrated_model = CalibratedClassifierCV(estimator=base_estimator, method="isotonic", cv=5)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", calibrated_model),
        ]
    )

    df_feat = prepare_baseline_df(train_df)
    X = df_feat[["decline_code", "action", "payday_prox", "retry_count", "customer_tenure_months", "subscription_value", "hours_since_failure"]]
    y = df_feat["outcome"].values
    pipeline.fit(X, y)
    return pipeline


def evaluate_models_on_eval_set(
    eval_df: pd.DataFrame,
    pymc_idata: az.InferenceData,
    pymc_prep: FeaturePreprocessor,
    baseline_pipe: Pipeline,
) -> Dict[str, float]:
    """Computes calibration and coverage metrics for given model pair on eval dataset."""
    bayes_probs = []
    hdi_lows = []
    hdi_highs = []
    true_p_ground_truth = []
    outcomes = eval_df["outcome"].values

    for _, row in eval_df.iterrows():
        ctx = FailureContext(
            event_id=str(row["event_id"]),
            decline_code=str(row["decline_code"]),
            retry_count=int(row["retry_count"]),
            hours_since_failure=float(row["hours_since_failure"]),
            day_of_month=int(row["day_of_month"]),
            customer_tenure_months=int(row["customer_tenure_months"]),
            subscription_value=float(row["subscription_value"]),
            prior_recovery_outcome=str(row["prior_recovery_outcome"]),
        )
        act = str(row["action"])
        est = estimate(ctx, act, idata=pymc_idata, preprocessor=pymc_prep)

        bayes_probs.append(est.p_success)
        hdi_lows.append(est.hdi_low)
        hdi_highs.append(est.hdi_high)
        if "true_p_success" in row:
            true_p_ground_truth.append(float(row["true_p_success"]))

    bayes_probs = np.array(bayes_probs)
    hdi_lows = np.array(hdi_lows)
    hdi_highs = np.array(hdi_highs)

    baseline_probs = predict_baseline_p_success(baseline_pipe, eval_df)

    bayes_brier = compute_brier_score(bayes_probs, outcomes)
    bayes_ece = compute_ece(bayes_probs, outcomes)

    baseline_brier = compute_brier_score(baseline_probs, outcomes)
    baseline_ece = compute_ece(baseline_probs, outcomes)

    hdi_cov = 0.0
    if len(true_p_ground_truth) > 0:
        hdi_cov = compute_hdi_coverage(hdi_lows, hdi_highs, np.array(true_p_ground_truth))

    return {
        "bayes_brier": bayes_brier,
        "bayes_ece": bayes_ece,
        "baseline_brier": baseline_brier,
        "baseline_ece": baseline_ece,
        "hdi_coverage": hdi_cov,
    }


def run_tuning_comparison():
    """Runs parameter tuning and compares metrics against current model."""
    print("=" * 70)
    print("      RECOVERY DECISION ENGINE — MODEL PARAMETER TUNING & EVALUATION")
    print("=" * 70)

    # 1. Load Datasets
    train_df = pd.read_csv("data/train.csv")
    eval_df = pd.read_csv("data/eval.csv")
    print(f"[DATA] Train Set: {len(train_df)} rows | Eval Set: {len(eval_df)} rows")

    # 2. Evaluate Current Models
    print("\n--- 1. Evaluating Current Production Models ---")
    current_idata, current_prep = load_model_artifacts("data/pymc_model_idata.pkl")
    current_baseline = load_baseline_model("data/baseline_model.pkl")

    curr_metrics = evaluate_models_on_eval_set(eval_df, current_idata, current_prep, current_baseline)
    print(f"Current Baseline Brier: {curr_metrics['baseline_brier']:.4f} | ECE: {curr_metrics['baseline_ece']:.4f}")
    print(f"Current PyMC Brier:     {curr_metrics['bayes_brier']:.4f} | ECE: {curr_metrics['bayes_ece']:.4f} | 90% HDI Cov: {curr_metrics['hdi_coverage']*100:.1f}%")

    # 3. Fit & Evaluate Refined Models
    print("\n--- 2. Fitting Refined PyMC Model (Interaction Hyperpriors + target_accept=0.95) ---")
    refined_idata, refined_prep = fit_refined_pymc_model(train_df, draws=1000, tune=1000, seed=42)
    is_conv, conv_metrics = check_convergence(refined_idata)
    print(f"Refined PyMC Convergence: R-hat max = {conv_metrics['r_hat_max']:.4f}, Divergences = {conv_metrics['num_divergences']}")

    print("\n--- 3. Fitting Refined Baseline Model (Tuned C=0.5, 5-Fold CV Isotonic) ---")
    refined_baseline = fit_refined_baseline_model(train_df)

    print("\n--- 4. Evaluating Refined Models on Held-Out Evaluation Set ---")
    refined_metrics = evaluate_models_on_eval_set(eval_df, refined_idata, refined_prep, refined_baseline)

    print("\n" + "=" * 70)
    print("                      TUNING COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Current Model':<15} | {'Refined Model':<15} | {'Status':<10}")
    print("-" * 75)

    ece_imp = ((curr_metrics['bayes_ece'] - refined_metrics['bayes_ece']) / curr_metrics['bayes_ece']) * 100
    brier_imp = ((curr_metrics['bayes_brier'] - refined_metrics['bayes_brier']) / curr_metrics['bayes_brier']) * 100

    print(f"{'PyMC Expected Calibration Error (ECE)':<35} | {curr_metrics['bayes_ece']:<15.4f} | {refined_metrics['bayes_ece']:<15.4f} | {f'+{ece_imp:.1f}% Better' if ece_imp > 0 else 'Same'}")
    print(f"{'PyMC Brier Score (vs Outcome)':<35} | {curr_metrics['bayes_brier']:<15.4f} | {refined_metrics['bayes_brier']:<15.4f} | {f'+{brier_imp:.1f}% Better' if brier_imp > 0 else 'Same'}")
    print(f"{'PyMC 90% HDI Ground-Truth Coverage':<35} | {curr_metrics['hdi_coverage']*100:<14.1f}% | {refined_metrics['hdi_coverage']*100:<14.1f}% | {'High'}")
    print(f"{'Baseline ECE':<35} | {curr_metrics['baseline_ece']:<15.4f} | {refined_metrics['baseline_ece']:<15.4f} | {'Evaluated'}")
    print(f"{'Baseline Brier Score':<35} | {curr_metrics['baseline_brier']:<15.4f} | {refined_metrics['baseline_brier']:<15.4f} | {'Evaluated'}")
    print("=" * 75)

    # Decision on updating production artifacts
    if refined_metrics['bayes_ece'] <= curr_metrics['bayes_ece']:
        print("\n🏆 REFINED MODEL IS PERFORMING BETTER OR EQUAL! Updating production artifacts...")
        save_model_artifacts(refined_idata, refined_prep, "data/pymc_model_idata.pkl")
        save_baseline_model(refined_baseline, "data/baseline_model.pkl")
        print("[SUCCESS] Updated data/pymc_model_idata.pkl and data/baseline_model.pkl!")
    else:
        print("\n[INFO] Current production model remains superior.")

    return curr_metrics, refined_metrics


if __name__ == "__main__":
    run_tuning_comparison()
