
"""
Evaluation metrics for METR-LA traffic forecasting.

Responsibilities:
1. Denormalize Z-score predictions using training statistics.
2. Calculate MAE, RMSE and MAPE.
3. Mask invalid values and zero-speed observations.

All reported metrics are calculated in mph.
"""

import pickle
from pathlib import Path

import numpy as np
import torch


# =========================================================
# SCALER
# =========================================================

def load_scaler(scaler_path):
    """
    Load scaler statistics fitted on training data.

    Supported formats:
        - dict containing 'mean' and 'std'
        - sklearn-like scaler with 'mean_' and 'scale_'

    Returns:
        mean, std
    """

    scaler_path = Path(scaler_path)

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_path}"
        )

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    if isinstance(scaler, dict):
        mean = scaler.get("mean")
        std = scaler.get("std")

    elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = scaler.mean_
        std = scaler.scale_

    else:
        raise ValueError(
            "Unsupported scaler format. "
            "Expected dict(mean, std) or an object with mean_ and scale_."
        )

    if mean is None or std is None:
        raise ValueError(
            "Scaler must contain both mean and std."
        )

    mean = float(np.asarray(mean).squeeze())
    std = float(np.asarray(std).squeeze())

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid scaler statistics.")

    return mean, std


# =========================================================
# DENORMALIZATION
# =========================================================

def denormalize(data, mean, std):
    """
    Convert Z-score normalized traffic speed back to mph.

    Formula:
        x_real = x_norm * std + mean

    Args:
        data: NumPy array or PyTorch Tensor.
        mean: Mean calculated from training data.
        std: Standard deviation calculated from training data.

    Returns:
        Denormalized data in mph.
    """

    if std <= 0:
        raise ValueError("std must be greater than 0.")

    if isinstance(data, torch.Tensor):
        return data * std + mean

    data = np.asarray(data)

    return data * std + mean


# =========================================================
# MASKING
# =========================================================

def _to_numpy(data):
    """Convert NumPy array or PyTorch Tensor to NumPy."""

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    return np.asarray(data)


def _get_valid_mask(y_true, y_pred):
    """
    Create mask for valid traffic observations.

    Exclude:
        - NaN and Inf values.
        - Zero or negative ground-truth speed.
        - Invalid predictions.
    """

    return (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true > 0)
    )


# =========================================================
# MAE
# =========================================================

def masked_mae(y_true, y_pred):
    """
    Mean Absolute Error (mph).
    """

    y_true = _to_numpy(y_true).astype(np.float64)
    y_pred = _to_numpy(y_pred).astype(np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    mask = _get_valid_mask(y_true, y_pred)

    if not np.any(mask):
        return float("nan")

    errors = np.abs(y_true[mask] - y_pred[mask])

    return float(np.mean(errors))


# =========================================================
# RMSE
# =========================================================

def masked_rmse(y_true, y_pred):
    """
    Root Mean Squared Error (mph).
    """

    y_true = _to_numpy(y_true).astype(np.float64)
    y_pred = _to_numpy(y_pred).astype(np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    mask = _get_valid_mask(y_true, y_pred)

    if not np.any(mask):
        return float("nan")

    errors = y_true[mask] - y_pred[mask]

    return float(np.sqrt(np.mean(np.square(errors))))


# =========================================================
# MAPE
# =========================================================

def masked_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (%).

    Zero and invalid ground-truth speeds are excluded.
    """

    y_true = _to_numpy(y_true).astype(np.float64)
    y_pred = _to_numpy(y_pred).astype(np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    mask = _get_valid_mask(y_true, y_pred)

    if not np.any(mask):
        return float("nan")

    percentage_errors = (
        np.abs(y_true[mask] - y_pred[mask])
        / np.abs(y_true[mask])
    )

    return float(np.mean(percentage_errors) * 100)


# =========================================================
# COMPLETE EVALUATION
# =========================================================

def evaluate_metrics(
    y_true,
    y_pred,
    mean,
    std,
):
    """
    Denormalize predictions and calculate all metrics.

    Args:
        y_true: Normalized ground-truth values.
        y_pred: Normalized predicted values.
        mean: Training speed mean.
        std: Training speed standard deviation.

    Returns:
        Dictionary containing MAE, RMSE and MAPE in mph / %.
    """

    # Convert normalized values back to mph
    y_true_real = denormalize(y_true, mean, std)
    y_pred_real = denormalize(y_pred, mean, std)

    return {
        "MAE": masked_mae(y_true_real, y_pred_real),
        "RMSE": masked_rmse(y_true_real, y_pred_real),
        "MAPE": masked_mape(y_true_real, y_pred_real),
    }


if __name__ == "__main__":

    # Example scaler statistics from training data
    mean = 54.40592956542969
    std = 19.49374008178711

    # Example normalized values
    y_true = np.array([0.5, 0.8, 1.0, -2.79])
    y_pred = np.array([0.6, 0.7, 0.9, -2.79])

    results = evaluate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        mean=mean,
        std=std,
    )

    print("\n========== EVALUATION RESULTS ==========")

    for name, value in results.items():
        print(f"{name}: {value:.4f}")
