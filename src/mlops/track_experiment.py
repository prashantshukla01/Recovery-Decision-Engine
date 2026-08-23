"""MLflow Experiment Tracking script logging hyperparameters, calibration metrics, business evaluation, and artifacts."""

import os
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv

load_dotenv()
from src.modeling.bayesian_model import load_model_artifacts, check_convergence
from src.modeling.baseline_model import load_baseline_model, predict_baseline_p_success
from src.modeling.predictor import estimate
from src.modeling.calibration_eval import compute_brier_score, compute_ece, compute_hdi_coverage
from src.evaluation.business_eval import run_ablation_study
from src.simulation.schemas import FailureContext


def log_experiment_to_mlflow():
    """Runs MLflow tracking run logging parameters, calibration, business metrics, and artifacts."""
    mlflow.set_experiment("Recovery_Decision_Engine")

    train_df = pd.read_csv("data/train.csv")
    eval_df = pd.read_csv("data/eval.csv")
    outcomes = eval_df["outcome"].values
    true_p = eval_df["true_p_success"].values

    idata, preprocessor = load_model_artifacts("data/pymc_model_idata.pkl")
    baseline_pipeline = load_baseline_model("data/baseline_model.pkl")

    with mlflow.start_run(run_name="PyMC_Hierarchical_vs_Baseline_Eval"):
        # Log Hyperparameters
        mlflow.log_params(
            {
                "model_type": "PyMC Hierarchical Logistic Regression",
                "baseline_type": "scikit-learn Isotonic Calibrated Classifier",
                "draws": 1000,
                "tune": 1000,
                "chains": 2,
                "target_accept": 0.90,
                "tau_threshold": 0.25,
                "train_sample_size": len(train_df),
                "eval_sample_size": len(eval_df),
            }
        )

        # Baseline evaluation
        baseline_probs = predict_baseline_p_success(baseline_pipeline, eval_df)
        brier_base = compute_brier_score(baseline_probs, outcomes)
        ece_base = compute_ece(baseline_probs, outcomes)

        # PyMC Bayesian evaluation
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

        brier_bayes = compute_brier_score(bayes_probs, outcomes)
        ece_bayes = compute_ece(bayes_probs, outcomes)
        hdi_cov = compute_hdi_coverage(hdi_lows, hdi_highs, true_p)

        # Business Evaluation & Ablation
        ablation_res = run_ablation_study(eval_df)
        full_biz = ablation_res["full_engine"]

        # Log Metrics to MLflow
        mlflow.log_metrics(
            {
                "baseline_brier_score": brier_base,
                "bayes_brier_score": brier_bayes,
                "baseline_ece": ece_base,
                "bayes_ece": ece_bayes,
                "bayes_90_hdi_coverage": hdi_cov,
                "rupees_recovered": full_biz["rupees_recovered"],
                "rupees_wasted": full_biz["rupees_wasted"],
                "rupees_avoided": full_biz["rupees_avoided"],
                "automation_rate": full_biz["automation_rate"],
                "escalation_rate": full_biz["escalation_rate"],
                "false_intervention_rate": full_biz["false_intervention_rate"],
                "ablation_false_intervention_delta": ablation_res["false_intervention_rate_delta"],
            }
        )

        # 6. Log Artifacts & Registered Models
        if os.path.exists("data/reliability_diagram.png"):
            mlflow.log_artifact("data/reliability_diagram.png", artifact_path="calibration_plots")
        if os.path.exists("data/decline_code_p_success.png"):
            mlflow.log_artifact("data/decline_code_p_success.png", artifact_path="calibration_plots")

        # Register Baseline Calibrated Model in MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=baseline_pipeline,
            name="baseline_model",
            registered_model_name="Recovery_Baseline_Isotonic_Model",
            serialization_format="cloudpickle",
        )
        print("Logged and registered 'Recovery_Baseline_Isotonic_Model' in MLflow Model Registry.")

        # Log PyMC artifact
        if os.path.exists("data/pymc_model_idata.pkl"):
            mlflow.log_artifact("data/pymc_model_idata.pkl", artifact_path="pymc_model_artifacts")
            print("Logged PyMC model artifact 'data/pymc_model_idata.pkl' to MLflow.")

        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            print(f"\nMLflow experiment run logged successfully! (Run ID: {run_id})")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    log_experiment_to_mlflow()
