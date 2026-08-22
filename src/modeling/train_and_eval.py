"""Script to train PyMC and Baseline models, evaluate on held-out eval.csv, and save artifacts."""

import pandas as pd
import numpy as np
from src.modeling.bayesian_model import (
    fit_bayesian_model,
    check_convergence,
    save_model_artifacts,
)
from src.modeling.baseline_model import (
    fit_baseline_model,
    predict_baseline_p_success,
    save_baseline_model,
)
from src.modeling.predictor import estimate
from src.modeling.calibration_eval import (
    compute_brier_score,
    compute_ece,
    compute_hdi_coverage,
    plot_reliability_diagrams,
)
from src.simulation.schemas import FailureContext


def main():
    print("Loading data/train.csv and data/eval.csv...")
    train_df = pd.read_csv("data/train.csv")
    eval_df = pd.read_csv("data/eval.csv")

    print(f"Fitting PyMC Bayesian Hierarchical Logistic Regression on {len(train_df)} train samples...")
    idata, preprocessor = fit_bayesian_model(
        train_df,
        draws=1000,
        tune=1000,
        chains=2,
        seed=42,
    )

    is_converged, conv_metrics = check_convergence(idata)
    print(f"PyMC Convergence Check: Converged={is_converged}, R-hat Max={conv_metrics['r_hat_max']:.4f}, Divergences={conv_metrics['num_divergences']}")
    save_model_artifacts(idata, preprocessor)

    print(f"\nFitting Isotonic Calibrated Baseline Model on {len(train_df)} train samples...")
    baseline_pipeline = fit_baseline_model(train_df)
    save_baseline_model(baseline_pipeline)

    print(f"\nEvaluating models on {len(eval_df)} held-out evaluation samples...")
    true_p = eval_df["true_p_success"].values
    outcomes = eval_df["outcome"].values

    # Baseline predictions
    baseline_probs = predict_baseline_p_success(baseline_pipeline, eval_df)

    # Bayesian predictions
    bayes_probs = []
    hdi_lows = []
    hdi_highs = []

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
        est = estimate(ctx, str(row["action"]), idata=idata, preprocessor=preprocessor)
        bayes_probs.append(est.p_success)
        hdi_lows.append(est.hdi_low)
        hdi_highs.append(est.hdi_high)

    bayes_probs = np.array(bayes_probs)
    hdi_lows = np.array(hdi_lows)
    hdi_highs = np.array(hdi_highs)

    # Metrics computation
    brier_base = compute_brier_score(baseline_probs, outcomes)
    brier_bayes = compute_brier_score(bayes_probs, outcomes)

    ece_base = compute_ece(baseline_probs, outcomes)
    ece_bayes = compute_ece(bayes_probs, outcomes)

    hdi_cov = compute_hdi_coverage(hdi_lows, hdi_highs, true_p)

    print("\n" + "=" * 60)
    print("               CALIBRATION EVALUATION RESULTS               ")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Baseline (Isotonic)':<20} | {'PyMC Bayesian':<20}")
    print("-" * 75)
    print(f"{'Brier Score (vs Outcome)':<30} | {brier_base:<20.4f} | {brier_bayes:<20.4f}")
    print(f"{'ECE (Expected Calib Error)':<30} | {ece_base:<20.4f} | {ece_bayes:<20.4f}")
    print(f"{'90% HDI Ground-Truth Coverage':<30} | {'N/A (Point Estimate)':<20} | {hdi_cov * 100:<20.1f}%")
    print("=" * 75)

    plot_reliability_diagrams(bayes_probs, baseline_probs, outcomes, save_path="data/reliability_diagram.png")
    print("\nPhase 2 Model Training & Calibration Evaluation Completed Successfully!")


if __name__ == "__main__":
    main()
