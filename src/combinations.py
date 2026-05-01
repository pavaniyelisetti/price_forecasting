"""
combinations.py
---------------
Forecast combination methods for electricity price forecasting.

Three combination approaches are implemented, matching the project paper:
  1. Simple Average    — equal weights across all models
  2. Trimmed Mean      — drops the worst-performing model (HW)
  3. Weighted Average  — weights proportional to inverse validation MAE
"""

import numpy as np
import pandas as pd
from typing import dict as Dict


# ---------------------------------------------------------------------------
# Default weights from the paper
# Naïve: 0.35, NARX: 0.30, ARX: 0.25, HW: 0.10
# ---------------------------------------------------------------------------
PAPER_WEIGHTS = {
    "naive": 0.35,
    "narx":  0.30,
    "arx":   0.25,
    "hw":    0.10,
}


def simple_average(forecasts: dict) -> np.ndarray:
    """
    Equal-weight average of all model forecasts.

    Parameters
    ----------
    forecasts : dict mapping model name → 1-D np.ndarray of predictions

    Returns
    -------
    np.ndarray of combined forecasts
    """
    stacked = np.vstack(list(forecasts.values()))
    return stacked.mean(axis=0)


def trimmed_mean(forecasts: dict, drop: list[str] | None = None) -> np.ndarray:
    """
    Average after dropping the specified worst-performing model(s).

    Parameters
    ----------
    forecasts : dict mapping model name → predictions
    drop      : list of model names to exclude (default: ["hw"])

    Returns
    -------
    np.ndarray of combined forecasts
    """
    if drop is None:
        drop = ["hw"]

    kept = {k: v for k, v in forecasts.items() if k not in drop}
    if not kept:
        raise ValueError("All models were dropped — nothing to combine.")

    stacked = np.vstack(list(kept.values()))
    return stacked.mean(axis=0)


def weighted_average(
    forecasts: dict,
    weights: dict | None = None,
) -> np.ndarray:
    """
    Weighted average combination.

    Parameters
    ----------
    forecasts : dict mapping model name → predictions
    weights   : dict mapping model name → weight (must sum to ~1.0).
                Defaults to PAPER_WEIGHTS if None.

    Returns
    -------
    np.ndarray of combined forecasts
    """
    if weights is None:
        weights = PAPER_WEIGHTS

    # Validate weights sum to 1
    total = sum(weights[k] for k in forecasts if k in weights)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Weights should sum to 1.0, got {total:.4f}")

    combined = np.zeros(len(next(iter(forecasts.values()))))
    for name, preds in forecasts.items():
        w = weights.get(name, 0.0)
        combined += w * np.asarray(preds)

    return combined


def inverse_mae_weights(
    forecasts: dict,
    actuals: np.ndarray,
) -> dict:
    """
    Compute data-driven weights proportional to 1/MAE on a validation set.

    Parameters
    ----------
    forecasts : dict mapping model name → validation predictions
    actuals   : actual values on the validation set

    Returns
    -------
    dict mapping model name → normalized weight
    """
    mae_scores = {
        name: np.mean(np.abs(np.asarray(preds) - actuals))
        for name, preds in forecasts.items()
    }
    inv_mae = {name: 1.0 / score for name, score in mae_scores.items()}
    total   = sum(inv_mae.values())
    return {name: v / total for name, v in inv_mae.items()}


def combine_all(
    forecasts: dict,
    actuals: np.ndarray | None = None,
) -> dict:
    """
    Convenience function: compute all three combination methods at once.

    Parameters
    ----------
    forecasts : dict of model_name → np.ndarray predictions
    actuals   : optional actuals for inverse-MAE weight calculation

    Returns
    -------
    dict with keys: 'simple_avg', 'trimmed_mean', 'weighted_avg'
    """
    results = {
        "simple_avg":   simple_average(forecasts),
        "trimmed_mean": trimmed_mean(forecasts, drop=["hw"]),
        "weighted_avg": weighted_average(forecasts, weights=PAPER_WEIGHTS),
    }

    if actuals is not None:
        data_weights = inverse_mae_weights(forecasts, actuals)
        results["data_driven_avg"] = weighted_average(forecasts, weights=data_weights)
        print("Data-driven weights:", {k: f"{v:.3f}" for k, v in data_weights.items()})

    return results
