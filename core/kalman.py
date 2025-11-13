from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from .pairs import _prep_xy  # reuse our cleaning/alignment


def _kalman_filter_regression(
    y: np.ndarray,
    x: np.ndarray,
    delta: float = 1e-5,
    R: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """
    Time-varying linear regression via a 2D Kalman filter:
        y_t = alpha_t + beta_t * x_t + noise, noise ~ N(0, R)
    State (alpha, beta) follows a random walk with process noise Q.

    delta controls drift speed (larger => more adaptive). R is obs noise var.
    """

    T = len(y)
    if T != len(x):
        raise ValueError("x and y must have same length after alignment")

    # State: [alpha, beta]^T
    theta = np.zeros((2, 1))          # uninformative prior on levels
    P = np.eye(2) * 1e6               # large covariance (weak prior)

    # Process noise: Q = q * I
    q = float(delta / (1.0 - delta))
    Q = q * np.eye(2)

    alpha_hist = np.zeros(T)
    beta_hist  = np.zeros(T)
    y_hat_hist = np.zeros(T)
    resid_hist = np.zeros(T)

    for t in range(T):
        H = np.array([[1.0, x[t]]])   # design matrix for this observation (1x2)

        # Predict
        theta_pred = theta            # random walk
        P_pred = P + Q

        # Innovation
        y_pred = float(H @ theta_pred)
        S = float(H @ P_pred @ H.T + R)   # scalar
        K = (P_pred @ H.T) / S            # (2x1)

        resid = float(y[t] - y_pred)      # scalar

        # Update
        theta = theta_pred + K * resid    # (2x1)
        P = P_pred - K @ H @ P_pred

        # Save post-update stats
        alpha_hist[t] = theta[0, 0]
        beta_hist[t]  = theta[1, 0]
        y_hat = float((H @ theta).ravel())   # post-update prediction
        y_hat_hist[t] = y_hat
        resid_hist[t] = float(y[t] - y_hat)

    return {
        "alpha": alpha_hist,
        "beta": beta_hist,
        "y_hat": y_hat_hist,
        "resid": resid_hist,
    }


def kalman_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    roll_window: int = 60,
    delta: float = 1e-5,
    R: float = 1e-3,
) -> pd.DataFrame:
    """
    Dynamic spread between A and B using a Kalman-regressed hedge ratio.

    Returns DataFrame with:
      price_a, price_b, alpha, beta, fair_value_a, resid, z_resid
    """
    xx, yy = _prep_xy(series_a, series_b)
    if len(xx) < roll_window + 5:
        raise ValueError("Not enough overlap to compute rolling z-score reliably.")

    y_log = np.log(xx.values)  # A
    x_log = np.log(yy.values)  # B

    kf = _kalman_filter_regression(y_log, x_log, delta=delta, R=R)

    alpha = pd.Series(kf["alpha"], index=xx.index)
    beta  = pd.Series(kf["beta"],  index=xx.index)
    resid = pd.Series(kf["resid"], index=xx.index)

    # rolling z-score of residual
    roll_mean = resid.rolling(roll_window).mean()
    roll_std  = resid.rolling(roll_window).std(ddof=1)
    z_resid = (resid - roll_mean) / (roll_std.replace(0.0, np.nan))

    # Fair value of A implied by B at time t
    fair_log = alpha + beta * np.log(yy)
    fair_value = np.exp(fair_log)

    out = pd.DataFrame({
        "price_a": xx,
        "price_b": yy,
        "alpha": alpha,
        "beta": beta,
        "fair_value_a": fair_value,
        "resid": resid,
        "z_resid": z_resid,
    })

    return out
