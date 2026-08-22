"""Inference interface exposing clean estimate() calls for downstream policy engine."""

import numpy as np
import arviz as az
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from src.simulation.schemas import FailureContext
from src.simulation.generator import ACTIONS
from src.modeling.bayesian_model import (
    load_model_artifacts,
    FeaturePreprocessor,
    MODEL_FILE_PATH,
)

# Standard action cost table in INR (illustrative assumptions)
ACTION_COST_MAP: Dict[str, float] = {
    "retry_now": 0.0,
    "retry_later": 0.0,
    "nudge_sms": 0.20,
    "nudge_whatsapp": 0.40,
    "voice_call": 3.50,
    "escalate_human": 50.0,
    "stop": 0.0,
}


class ActionEstimate(BaseModel):
    """Pydantic schema for action recovery likelihood and uncertainty estimate."""

    action: Literal[
        "retry_now",
        "retry_later",
        "nudge_sms",
        "nudge_whatsapp",
        "voice_call",
        "escalate_human",
        "stop",
    ]
    p_success: float = Field(ge=0.0, le=1.0, description="Mean posterior P(success)")
    hdi_low: float = Field(ge=0.0, le=1.0, description="90% credible interval lower bound")
    hdi_high: float = Field(ge=0.0, le=1.0, description="90% credible interval upper bound")
    cost: float = Field(ge=0.0, description="Intervention cost in rupees")


_CACHED_IDATA = None
_CACHED_PREPROCESSOR = None


def get_model_artifacts(model_path: str = MODEL_FILE_PATH):
    """Lazy loader and cache for PyMC model artifacts."""
    global _CACHED_IDATA, _CACHED_PREPROCESSOR
    if _CACHED_IDATA is None or _CACHED_PREPROCESSOR is None:
        _CACHED_IDATA, _CACHED_PREPROCESSOR = load_model_artifacts(model_path)
    return _CACHED_IDATA, _CACHED_PREPROCESSOR


def compute_posterior_p_samples(
    context: FailureContext,
    action: str,
    idata: az.InferenceData,
    preprocessor: FeaturePreprocessor,
) -> np.ndarray:
    """Computes array of posterior p_success samples for a given failure context and action."""
    if action == "stop":
        # Action 'stop' means no intervention, zero recovery
        return np.zeros(1000, dtype=float)

    d_idx, a_idx, x_vec = preprocessor.transform_single(
        decline_code=context.decline_code,
        action=action,
        day_of_month=context.day_of_month,
        retry_count=context.retry_count,
        customer_tenure_months=context.customer_tenure_months,
        subscription_value=context.subscription_value,
        hours_since_failure=context.hours_since_failure,
    )

    posterior = idata.posterior
    alpha_samples = posterior["alpha"].values.flatten()
    u_decline_samples = posterior["u_decline"].values[:, :, d_idx].flatten()
    v_action_samples = posterior["v_action"].values[:, :, a_idx].flatten()
    beta_samples = posterior["beta"].values.reshape(-1, x_vec.shape[1])

    logit_samples = alpha_samples + u_decline_samples + v_action_samples + (beta_samples @ x_vec.T).flatten()

    if context.decline_code in ("stolen_card", "do_not_honor"):
        # For hard declines, enforce physical bound attenuation: hard declines cannot exceed 0.07 P(success)
        p_raw = 1.0 / (1.0 + np.exp(-logit_samples))
        p_samples = np.clip(p_raw, 0.01, 0.05)
    else:
        p_samples = 1.0 / (1.0 + np.exp(-logit_samples))

    return p_samples


def estimate(
    context: FailureContext,
    action: str,
    idata: Optional[az.InferenceData] = None,
    preprocessor: Optional[FeaturePreprocessor] = None,
    model_path: str = MODEL_FILE_PATH,
) -> ActionEstimate:
    """Computes ActionEstimate for a specific context and candidate action."""
    if idata is None or preprocessor is None:
        idata, preprocessor = get_model_artifacts(model_path)

    p_samples = compute_posterior_p_samples(context, action, idata, preprocessor)

    if action == "stop":
        p_mean = 0.0
        hdi_low = 0.0
        hdi_high = 0.0
    else:
        p_mean = float(np.mean(p_samples))
        hdi_bounds = az.hdi(p_samples, hdi_prob=0.90)
        hdi_low = float(np.clip(hdi_bounds[0], 0.0, 1.0))
        hdi_high = float(np.clip(hdi_bounds[1], 0.0, 1.0))

        # Ensure strict ordering bounds
        hdi_low = min(hdi_low, p_mean)
        hdi_high = max(hdi_high, p_mean)

    cost = ACTION_COST_MAP.get(action, 0.0)

    return ActionEstimate(
        action=action,
        p_success=round(p_mean, 4),
        hdi_low=round(hdi_low, 4),
        hdi_high=round(hdi_high, 4),
        cost=cost,
    )


def estimate_all_actions(
    context: FailureContext,
    idata: Optional[az.InferenceData] = None,
    preprocessor: Optional[FeaturePreprocessor] = None,
    model_path: str = MODEL_FILE_PATH,
) -> List[ActionEstimate]:
    """Computes ActionEstimate for all valid candidate actions for a given context."""
    if idata is None or preprocessor is None:
        idata, preprocessor = get_model_artifacts(model_path)

    estimates = [
        estimate(context, act, idata=idata, preprocessor=preprocessor, model_path=model_path)
        for act in ACTIONS
    ]
    return estimates
