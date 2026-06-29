import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def conformal_style_quantile(scores, interval_level):
    """Finite-sample conformal-style quantile for absolute residual scores."""
    scores = np.asarray(pd.Series(scores).dropna(), dtype=float)

    if len(scores) == 0:
        return np.nan

    q_level = np.ceil((len(scores) + 1) * interval_level) / len(scores)
    q_level = min(q_level, 1.0)

    return np.quantile(scores, q_level, method="higher")


def make_interval_rows(
    test_rows,
    interval_method,
    interval_level,
    lower_offset,
    upper_offset,
    rated_power_kw=250.0,
):
    """Apply residual offsets to point forecasts and return interval prediction rows."""
    out = test_rows.copy()

    out["interval_method"] = interval_method
    out["interval_level"] = interval_level
    out["lower_offset_norm"] = lower_offset
    out["upper_offset_norm"] = upper_offset

    out["lower_norm_raw"] = out["y_pred_norm"] + lower_offset
    out["upper_norm_raw"] = out["y_pred_norm"] + upper_offset

    out["lower_norm"] = out["lower_norm_raw"].clip(lower=0.0, upper=1.0)
    out["upper_norm"] = out["upper_norm_raw"].clip(lower=0.0, upper=1.0)

    out["covered"] = (
        (out["y_true_norm"] >= out["lower_norm"])
        & (out["y_true_norm"] <= out["upper_norm"])
    )

    out["interval_width_norm"] = out["upper_norm"] - out["lower_norm"]

    out["lower_kw_250"] = out["lower_norm"] * rated_power_kw
    out["upper_kw_250"] = out["upper_norm"] * rated_power_kw
    out["interval_width_kw_250"] = out["interval_width_norm"] * rated_power_kw

    return out


def kde_residual_quantiles(values, probs, grid_size=2048, bandwidth=0.01):
    """Estimate residual quantiles from a fixed-bandwidth Gaussian KDE."""
    values = np.asarray(pd.Series(values).dropna(), dtype=float)

    if len(values) < 20 or np.std(values) == 0:
        return np.quantile(values, probs)

    values_2d = values.reshape(-1, 1)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(values_2d)

    grid_margin = 4.0 * bandwidth
    grid_min = values.min() - grid_margin
    grid_max = values.max() + grid_margin
    grid = np.linspace(grid_min, grid_max, grid_size)

    density = np.exp(kde.score_samples(grid.reshape(-1, 1)))

    cdf = np.cumsum((density[:-1] + density[1:]) / 2.0 * np.diff(grid))
    cdf = np.insert(cdf, 0, 0.0)

    if cdf[-1] <= 0 or not np.isfinite(cdf[-1]):
        return np.quantile(values, probs)

    cdf = cdf / cdf[-1]

    return np.interp(probs, cdf, grid)


def interval_score(y_true, lower, upper, alpha):
    """Winkler-style interval score. Lower is better."""
    width = upper - lower

    below = y_true < lower
    above = y_true > upper

    penalty_below = (2.0 / alpha) * (lower - y_true)
    penalty_above = (2.0 / alpha) * (y_true - upper)

    return (
        width
        + np.where(below, penalty_below, 0.0)
        + np.where(above, penalty_above, 0.0)
    )


def select_high_variation_window(df, value_col="y_true_norm", time_col="target_time", window_size=96):
    """Select a time window with relatively high variation in the target column."""
    df = df.sort_values(time_col).reset_index(drop=True).copy()

    if len(df) <= window_size:
        return df

    rolling_std = df[value_col].rolling(
        window_size,
        min_periods=window_size // 2,
    ).std()

    end_idx = rolling_std.idxmax()

    if pd.isna(end_idx):
        return df.iloc[:window_size].copy()

    start_idx = max(0, int(end_idx) - window_size + 1)
    end_idx = min(len(df), start_idx + window_size)

    return df.iloc[start_idx:end_idx].copy()


def select_high_ramp_window(
    df,
    ramp_subset_col="ramp_subset",
    high_ramp_label="High-ramp",
    time_col="target_time",
    window_size=96,
):
    """Select a time window containing many high-ramp points."""
    df = df.sort_values(time_col).reset_index(drop=True).copy()

    if len(df) <= window_size:
        return df

    high_ramp_flag = df[ramp_subset_col].eq(high_ramp_label).astype(int)
    rolling_count = high_ramp_flag.rolling(
        window_size,
        min_periods=window_size // 2,
    ).sum()

    end_idx = rolling_count.idxmax()

    if pd.isna(end_idx):
        return df.iloc[:window_size].copy()

    start_idx = max(0, int(end_idx) - window_size + 1)
    end_idx = min(len(df), start_idx + window_size)

    return df.iloc[start_idx:end_idx].copy()
