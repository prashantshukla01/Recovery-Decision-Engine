"""pytest test suite for Phase 2 modeling and prediction layer."""

import pytest
import pandas as pd
import numpy as np
from src.simulation.schemas import FailureContext
from src.simulation.generator import generate_dataset, ACTIONS
from src.modeling.bayesian_model import fit_bayesian_model, check_convergence
from src.modeling.baseline_model import fit_baseline_model, predict_baseline_p_success
from src.modeling.predictor import estimate, ActionEstimate, ACTION_COST_MAP
from src.modeling.calibration_eval import compute_brier_score, compute_ece, compute_hdi_coverage


@pytest.fixture(scope="module")
def trained_models():
    """Module-level fixture fitting PyMC and Baseline models on a 200-row synthetic subset for fast testing."""
    _, train_df = generate_dataset(num_events=200, seed=123, sample_actions_per_event=True)
    idata, preprocessor = fit_bayesian_model(train_df, draws=200, tune=200, chains=2, seed=123)
    baseline_pipeline = fit_baseline_model(train_df)
    return idata, preprocessor, baseline_pipeline, train_df


def test_pymc_model_convergence(trained_models):
    """Verify PyMC model fits cleanly with low R-hat and zero divergences."""
    idata, _, _, _ = trained_models
    is_converged, metrics = check_convergence(idata)

    assert metrics["r_hat_max"] <= 1.05, f"R-hat max too high: {metrics['r_hat_max']}"
    assert metrics["num_divergences"] == 0, f"Divergences detected: {metrics['num_divergences']}"


def test_estimate_schema_and_hdi_ordering(trained_models):
    """Verify estimate() returns valid ActionEstimate objects with strict HDI ordering."""
    idata, preprocessor, _, _ = trained_models

    ctx = FailureContext(
        event_id="test_ctx_001",
        decline_code="insufficient_funds",
        retry_count=1,
        hours_since_failure=4.0,
        day_of_month=1,
        customer_tenure_months=12,
        subscription_value=999.0,
        prior_recovery_outcome="none",
    )

    for action in ACTIONS:
        est = estimate(ctx, action, idata=idata, preprocessor=preprocessor)

        assert isinstance(est, ActionEstimate)
        assert est.action == action
        assert 0.0 <= est.p_success <= 1.0
        assert 0.0 <= est.hdi_low <= 1.0
        assert 0.0 <= est.hdi_high <= 1.0
        assert est.hdi_low <= est.p_success <= est.hdi_high
        assert est.cost == ACTION_COST_MAP[action]


def test_hard_decline_low_probability_narrow_interval(trained_models):
    """Verify hard declines (stolen_card, do_not_honor) produce low p_success with narrow credible intervals."""
    idata, preprocessor, _, _ = trained_models

    for hard_code in ["stolen_card", "do_not_honor"]:
        ctx = FailureContext(
            event_id=f"test_hard_{hard_code}",
            decline_code=hard_code,
            retry_count=2,
            hours_since_failure=12.0,
            day_of_month=15,
            customer_tenure_months=6,
            subscription_value=499.0,
            prior_recovery_outcome="none",
        )

        for act in ["retry_now", "nudge_whatsapp", "voice_call"]:
            est = estimate(ctx, act, idata=idata, preprocessor=preprocessor)

            assert est.p_success <= 0.08, f"Hard decline {hard_code} produced p_success={est.p_success} > 0.08"
            hdi_width = est.hdi_high - est.hdi_low
            assert hdi_width <= 0.10, f"Hard decline {hard_code} produced wide interval width={hdi_width} > 0.10"


def test_calibration_eval_metrics():
    """Verify Brier Score, ECE, and HDI coverage calculation logic."""
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])

    brier = compute_brier_score(y_prob, y_true)
    assert 0.0 <= brier <= 0.2

    ece = compute_ece(y_prob, y_true, n_bins=5)
    assert 0.0 <= ece <= 0.5

    hdi_lows = np.array([0.05, 0.15, 0.70, 0.80])
    hdi_highs = np.array([0.15, 0.25, 0.90, 0.95])
    true_p = np.array([0.10, 0.20, 0.80, 0.85])

    cov = compute_hdi_coverage(hdi_lows, hdi_highs, true_p)
    assert cov == 1.0
