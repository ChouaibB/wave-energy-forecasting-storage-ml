import numpy as np
import pandas as pd

def make_segment_safe_power_features(
    df,
    target_col,
    lag_steps,
    roll_mean_windows,
    roll_std_windows,
    segment_col="continuous_segment_id",
):
    """Create lag and rolling power features within continuous valid segments."""
    feature_df = df.copy()
    grouped_power = feature_df.groupby(segment_col, dropna=True)[target_col]

    for step in lag_steps:
        feature_df[f"power_lag_{step}"] = grouped_power.shift(step)

    past_power = grouped_power.shift(1)

    for window in roll_mean_windows:
        feature_df[f"power_roll_mean_{window}"] = (
            past_power
            .groupby(feature_df[segment_col], dropna=True)
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    for window in roll_std_windows:
        feature_df[f"power_roll_std_{window}"] = (
            past_power
            .groupby(feature_df[segment_col], dropna=True)
            .rolling(window=window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )

    return feature_df


def build_supervised_horizon_tables(
    feature_df,
    horizons,
    feature_cols,
    time_col,
    target_col,
    segment_col="continuous_segment_id",
):
    """Build one supervised forecasting table per horizon."""
    supervised_by_horizon = {}

    for horizon_steps, info in horizons.items():
        horizon_df = feature_df.copy()
        grouped_segment = horizon_df.groupby(segment_col, dropna=True)

        horizon_df["target_time"] = grouped_segment[time_col].shift(-horizon_steps)
        horizon_df["y_target_norm"] = grouped_segment[target_col].shift(-horizon_steps)

        horizon_df["horizon_steps"] = horizon_steps
        horizon_df["horizon_hours"] = info["horizon_hours"]
        horizon_df["horizon_label"] = info["horizon_label"]

        supervised_cols = (
            [time_col, "target_time", segment_col]
            + feature_cols
            + [
                "y_target_norm",
                target_col,
                "horizon_steps",
                "horizon_hours",
                "horizon_label",
            ]
        )

        horizon_df = horizon_df[supervised_cols].dropna(
            subset=feature_cols + ["y_target_norm"]
        )

        horizon_df = horizon_df.rename(columns={time_col: "origin_time"})
        supervised_by_horizon[horizon_steps] = horizon_df.reset_index(drop=True)

    return supervised_by_horizon

def build_expanding_rolling_origin_folds(
    supervised_by_horizon,
    n_folds,
    train_fraction,
    calibration_fraction,
    test_fraction,
):
    """Build expanding-window rolling-origin train/calibration/test fold definitions."""
    fold_records = {}
    updated_supervised_by_horizon = {}

    for horizon_steps, horizon_df in supervised_by_horizon.items():
        horizon_df = horizon_df.sort_values("origin_time").reset_index(drop=True).copy()
        horizon_df["sample_id"] = np.arange(len(horizon_df))
        updated_supervised_by_horizon[horizon_steps] = horizon_df

        n_rows = len(horizon_df)

        fold_end_positions = np.linspace(
            int(np.floor(0.80 * n_rows)),
            n_rows,
            n_folds,
            dtype=int,
        )

        if len(np.unique(fold_end_positions)) < n_folds:
            raise ValueError(
                f"Not enough supervised rows for {n_folds} folds at horizon {horizon_steps}."
            )

        horizon_fold_records = []

        for fold_id, fold_end in enumerate(fold_end_positions, start=1):
            train_end = int(np.floor(fold_end * train_fraction))
            calibration_end = train_end + int(np.floor(fold_end * calibration_fraction))
            test_end = fold_end

            split_blocks = {
                "train": (0, train_end),
                "calibration": (train_end, calibration_end),
                "test": (calibration_end, test_end),
            }

            for split_name, (start, end) in split_blocks.items():
                split_df = horizon_df.iloc[start:end][
                    [
                        "sample_id",
                        "origin_time",
                        "target_time",
                        "horizon_steps",
                        "horizon_hours",
                        "horizon_label",
                    ]
                ].copy()

                split_df["fold_id"] = fold_id
                split_df["split"] = split_name

                horizon_fold_records.append(split_df)

        fold_records[horizon_steps] = pd.concat(
            horizon_fold_records,
            ignore_index=True,
        )

    folds_df = pd.concat(fold_records.values(), ignore_index=True)

    folds_df = folds_df[
        [
            "horizon_steps",
            "horizon_hours",
            "horizon_label",
            "fold_id",
            "split",
            "sample_id",
            "origin_time",
            "target_time",
        ]
    ].sort_values(
        ["horizon_steps", "fold_id", "origin_time"]
    ).reset_index(drop=True)

    return folds_df, updated_supervised_by_horizon


def summarize_fold_splits(folds_df):
    """Summarize train/calibration/test counts and percentages by horizon and fold."""
    fold_split_counts = (
        folds_df
        .groupby(["horizon_steps", "horizon_label", "fold_id", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for split_name in ["train", "calibration", "test"]:
        if split_name not in fold_split_counts.columns:
            fold_split_counts[split_name] = 0

    fold_split_counts["total"] = (
        fold_split_counts["train"]
        + fold_split_counts["calibration"]
        + fold_split_counts["test"]
    )

    fold_split_summary = fold_split_counts[
        ["horizon_steps", "horizon_label", "fold_id"]
    ].copy()

    for split_name in ["train", "calibration", "test"]:
        pct = 100 * fold_split_counts[split_name] / fold_split_counts["total"]
        fold_split_summary[split_name] = (
            fold_split_counts[split_name].astype(int).astype(str)
            + " ("
            + pct.round(1).astype(str)
            + "%)"
        )

    fold_split_summary = fold_split_summary[
        ["horizon_steps", "horizon_label", "fold_id", "train", "calibration", "test"]
    ]

    fold_split_summary.columns.name = None

    return fold_split_summary

def format_params(params):
    """Format model parameters for compact table labels."""
    if not params:
        return ""

    return ", ".join(
        f"{key}={value}"
        for key, value in params.items()
    )


def make_model_setting_label(model_name, params_label):
    """Create a readable model-setting label."""
    if not params_label:
        return model_name

    return f"{model_name} ({params_label})"


def predict_baseline_model(model_name, params, data, target_col):
    """Predict with simple non-fitted baseline models."""
    if model_name == "Persistence":
        return data[target_col].to_numpy()

    if model_name == "RollingMean":
        return data[params["feature"]].to_numpy()

    raise ValueError(f"Unknown baseline model: {model_name}")


def fit_forecast_model(
    model_name,
    params,
    train_data,
    feature_cols,
    build_model_func,
    target_col="y_target_norm",
):
    """Fit a forecast model, returning None for non-fitted baselines."""
    if model_name in ["Persistence", "RollingMean"]:
        return None

    model = build_model_func(model_name, params)
    model.fit(train_data[feature_cols], train_data[target_col])

    return model


def predict_forecast_model(
    model_name,
    params,
    model,
    pred_data,
    feature_cols,
    target_col,
):
    """Predict with either a baseline model or a fitted sklearn model."""
    if model_name in ["Persistence", "RollingMean"]:
        return predict_baseline_model(
            model_name=model_name,
            params=params,
            data=pred_data,
            target_col=target_col,
        )

    return model.predict(pred_data[feature_cols])
