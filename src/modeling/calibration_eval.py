"""Calibration evaluation metrics: Brier Score, ECE, and 90% HDI Coverage."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


def compute_brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Brier score: mean squared difference between probability predictions and outcomes/ground-truth."""
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_prob)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_prob = np.mean(y_prob[in_bin])
            avg_true = np.mean(y_true[in_bin])
            ece += (bin_size / total_samples) * abs(avg_prob - avg_true)

    return float(ece)


def compute_hdi_coverage(hdi_lows: np.ndarray, hdi_highs: np.ndarray, true_p: np.ndarray) -> float:
    """Computes 90% HDI coverage: fraction of events where true_p_success falls in [hdi_low, hdi_high]."""
    in_interval = (true_p >= hdi_lows) & (true_p <= hdi_highs)
    return float(np.mean(in_interval))


def plot_reliability_diagrams(
    y_prob_bayes: np.ndarray,
    y_prob_baseline: np.ndarray,
    y_true: np.ndarray,
    save_path: str = "data/reliability_diagram.png",
    n_bins: int = 10,
):
    """Plots and saves reliability diagrams comparing Bayesian vs Baseline models."""
    sns.set_theme(style="whitegrid")
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0

    def calc_bins(probs):
        obs_freq = []
        counts = []
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (probs >= bin_lower) & (probs <= bin_upper) if i == n_bins - 1 else (probs >= bin_lower) & (probs < bin_upper)
            if np.sum(in_bin) > 0:
                obs_freq.append(np.mean(y_true[in_bin]))
                counts.append(np.sum(in_bin))
            else:
                obs_freq.append(np.nan)
                counts.append(0)
        return np.array(obs_freq), np.array(counts)

    obs_bayes, _ = calc_bins(y_prob_bayes)
    obs_base, _ = calc_bins(y_prob_baseline)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", linewidth=1.5)
    plt.plot(bin_centers, obs_bayes, "s-", color="#1f77b4", label="PyMC Hierarchical Bayesian", linewidth=2, markersize=8)
    plt.plot(bin_centers, obs_base, "o-", color="#ff7f0e", label="Isotonic Calibrated Baseline", linewidth=2, markersize=8)

    plt.xlabel("Predicted P(Success)", fontsize=12)
    plt.ylabel("Observed Frequency / Ground Truth", fontsize=12)
    plt.title("Reliability Diagram: Bayesian Model vs Baseline", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reliability diagram to {save_path}")
