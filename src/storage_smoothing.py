import numpy as np
import pandas as pd


def compute_ramp_abs(series):
    """Compute absolute step-to-step ramp."""
    return series.diff().abs()


def compute_p95_ramp(series):
    """Compute p95 absolute step-to-step ramp."""
    ramp_abs = compute_ramp_abs(series).dropna()

    if len(ramp_abs) == 0:
        return np.nan

    return ramp_abs.quantile(0.95)


def smooth_with_trailing_rolling_mean(series, window_steps):
    """Smooth a time series using a trailing rolling mean."""
    return (
        series
        .rolling(window=window_steps, min_periods=1)
        .mean()
    )

def compute_storage_power_and_energy(raw_power, grid_power, time_step_hours=0.5):
    """
    Compute ideal storage exchange power and usable energy swing.

    The sign convention follows Notebook 05:

        storage_power = raw_power - grid_power

    Positive storage_power means storage absorbs excess WEC power.
    Negative storage_power means storage discharges to support the grid-side signal.
    """
    storage_power = raw_power - grid_power

    storage_energy = storage_power.cumsum() * time_step_hours
    storage_energy = storage_energy - storage_energy.min()

    return storage_power, storage_energy

def prepare_raw_power_variability_baseline(
    wec_df,
    dt_minutes=30,
    dt_hours=0.5,
):
    """Prepare raw WEC power, continuous segments, ramp metrics, and summaries."""
    power_df = wec_df[
        [
            "time",
            "wec_power_kw_250",
            "wec_power_norm_estimated",
        ]
    ].copy()

    power_df = power_df.rename(columns={"wec_power_kw_250": "p_wec_kw"})
    power_df["time"] = pd.to_datetime(power_df["time"])

    power_df = (
        power_df
        .dropna(subset=["p_wec_kw", "wec_power_norm_estimated"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    power_df["time_diff_minutes"] = (
        power_df["time"].diff().dt.total_seconds() / 60
    )

    power_df["is_new_segment"] = (
        power_df["time_diff_minutes"].isna()
        | (power_df["time_diff_minutes"] > dt_minutes)
    )

    power_df["segment_id"] = power_df["is_new_segment"].cumsum()

    power_df["ramp_kw_per_step"] = power_df["p_wec_kw"].diff()
    power_df.loc[power_df["is_new_segment"], "ramp_kw_per_step"] = np.nan

    power_df["abs_ramp_kw_per_step"] = power_df["ramp_kw_per_step"].abs()
    power_df["ramp_kw_per_hour"] = power_df["ramp_kw_per_step"] / dt_hours
    power_df["abs_ramp_kw_per_hour"] = power_df["ramp_kw_per_hour"].abs()

    high_ramp_threshold_kw = power_df["abs_ramp_kw_per_step"].quantile(0.95)

    power_df["is_high_ramp"] = (
        power_df["abs_ramp_kw_per_step"] >= high_ramp_threshold_kw
    ).fillna(False)

    power_summary = pd.DataFrame(
        [
            ["Samples", len(power_df), ""],
            ["Continuous segments", power_df["segment_id"].nunique(), ""],
            ["Start time", power_df["time"].min().strftime("%Y-%m-%d %H:%M"), ""],
            ["End time", power_df["time"].max().strftime("%Y-%m-%d %H:%M"), ""],
            ["Mean power", power_df["p_wec_kw"].mean(), "kW"],
            ["Median power", power_df["p_wec_kw"].median(), "kW"],
            ["95th percentile power", power_df["p_wec_kw"].quantile(0.95), "kW"],
            ["Maximum power", power_df["p_wec_kw"].max(), "kW"],
        ],
        columns=["Metric", "Value", "Unit"],
    )

    ramp_summary = pd.DataFrame(
        [
            ["Mean absolute ramp", power_df["abs_ramp_kw_per_step"].mean(), "kW / 30 min"],
            ["95th percentile absolute ramp", power_df["abs_ramp_kw_per_step"].quantile(0.95), "kW / 30 min"],
            ["Maximum absolute ramp", power_df["abs_ramp_kw_per_step"].max(), "kW / 30 min"],
            ["Mean absolute ramp rate", power_df["abs_ramp_kw_per_hour"].mean(), "kW / h"],
            ["95th percentile ramp rate", power_df["abs_ramp_kw_per_hour"].quantile(0.95), "kW / h"],
            ["High-ramp threshold", high_ramp_threshold_kw, "kW / 30 min"],
            ["High-ramp samples", int(power_df["is_high_ramp"].sum()), ""],
            ["High-ramp share", 100 * power_df["is_high_ramp"].mean(), "%"],
        ],
        columns=["Metric", "Value", "Unit"],
    )

    for summary_df in [power_summary, ramp_summary]:
        summary_df["Value"] = summary_df["Value"].apply(
            lambda x: round(x, 2) if isinstance(x, (int, float, np.floating)) else x
        )

    return power_df, power_summary, ramp_summary, high_ramp_threshold_kw

def prepare_selected_interval_timeseries(
    interval_df,
    selected_model,
    selected_interval_method,
    interval_level,
    high_ramp_threshold_kw,
    dt_minutes=30,
):
    """Select interval predictions and add continuous-segment/ramp columns."""
    selected_intervals = interval_df[
        (interval_df["split"] == "test")
        & (interval_df["model"] == selected_model)
        & (interval_df["interval_method"] == selected_interval_method)
        & (interval_df["interval_level"] == interval_level)
    ].copy()

    selected_intervals = selected_intervals.sort_values(
        ["horizon_steps", "fold_id", "target_time"]
    ).reset_index(drop=True)

    selected_intervals = selected_intervals.rename(
        columns={
            "y_true_kw_250": "p_wec_kw",
            "y_pred_kw_250": "p_forecast_kw",
            "lower_kw_250": "p_lower_kw",
            "upper_kw_250": "p_upper_kw",
        }
    )

    group_cols = ["horizon_steps", "fold_id"]

    selected_intervals["target_time_diff_minutes"] = (
        selected_intervals
        .groupby(group_cols)["target_time"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    selected_intervals["is_new_segment"] = (
        selected_intervals["target_time_diff_minutes"].isna()
        | (selected_intervals["target_time_diff_minutes"] > dt_minutes)
    )

    selected_intervals["segment_number"] = (
        selected_intervals
        .groupby(group_cols)["is_new_segment"]
        .cumsum()
    )

    selected_intervals["segment_id"] = (
        "h"
        + selected_intervals["horizon_steps"].astype(str)
        + "_fold"
        + selected_intervals["fold_id"].astype(str)
        + "_seg"
        + selected_intervals["segment_number"].astype(str)
    )

    segment_group_cols = ["horizon_steps", "fold_id", "segment_id"]

    selected_intervals["raw_ramp_kw_per_step"] = (
        selected_intervals
        .groupby(segment_group_cols)["p_wec_kw"]
        .diff()
    )

    selected_intervals["abs_raw_ramp_kw_per_step"] = (
        selected_intervals["raw_ramp_kw_per_step"].abs()
    )

    selected_intervals["is_high_ramp"] = (
        selected_intervals["abs_raw_ramp_kw_per_step"] >= high_ramp_threshold_kw
    ).fillna(False)

    return selected_intervals


def build_storage_smoothing_scenarios(
    selected_intervals,
    smoothing_windows_min,
    smoothing_window_labels,
    dt_minutes=30,
):
    """Build no-smoothing, observed, forecast, and lower-bound smoothing scenarios."""
    scenario_parts = []

    for (horizon_steps, fold_id), fold_df in selected_intervals.groupby(
        ["horizon_steps", "fold_id"],
        observed=True,
    ):
        fold_df = fold_df.copy()

        scenario_a = fold_df.copy()
        scenario_a["scenario"] = "A"
        scenario_a["scenario_name"] = "No smoothing"
        scenario_a["smoothing_window_min"] = 0
        scenario_a["smoothing_window_label"] = "No smoothing"
        scenario_a["p_grid_kw"] = scenario_a["p_wec_kw"]
        scenario_parts.append(scenario_a)

        for window_min in smoothing_windows_min:
            window_steps = int(window_min / dt_minutes)
            window_label = smoothing_window_labels[window_min]

            scenario_b = fold_df.copy()
            scenario_b["scenario"] = "B"
            scenario_b["scenario_name"] = "Observed-power smoothing"
            scenario_b["smoothing_window_min"] = window_min
            scenario_b["smoothing_window_label"] = window_label
            scenario_b["p_grid_kw"] = (
                scenario_b
                .groupby("segment_id")["p_wec_kw"]
                .transform(
                    lambda s: smooth_with_trailing_rolling_mean(
                        series=s,
                        window_steps=window_steps,
                    )
                )
            )
            scenario_parts.append(scenario_b)

            scenario_c = fold_df.copy()
            scenario_c["scenario"] = "C"
            scenario_c["scenario_name"] = "Forecast-informed smoothing"
            scenario_c["smoothing_window_min"] = window_min
            scenario_c["smoothing_window_label"] = window_label
            scenario_c["p_grid_kw"] = (
                scenario_c
                .groupby("segment_id")["p_forecast_kw"]
                .transform(
                    lambda s: smooth_with_trailing_rolling_mean(
                        series=s,
                        window_steps=window_steps,
                    )
                )
            )
            scenario_parts.append(scenario_c)

            scenario_d = fold_df.copy()
            scenario_d["scenario"] = "D"
            scenario_d["scenario_name"] = "Uncertainty-aware smoothing"
            scenario_d["smoothing_window_min"] = window_min
            scenario_d["smoothing_window_label"] = window_label
            scenario_d["p_grid_kw"] = (
                scenario_d
                .groupby("segment_id")["p_lower_kw"]
                .transform(
                    lambda s: smooth_with_trailing_rolling_mean(
                        series=s,
                        window_steps=window_steps,
                    )
                )
            )
            scenario_parts.append(scenario_d)

    return pd.concat(scenario_parts, ignore_index=True)

def add_storage_balance_columns(storage_timeseries, dt_hours=0.5):
    """Add ramp, storage-power, storage-energy, and SoC columns."""
    storage_timeseries = storage_timeseries.copy()

    storage_timeseries["p_st_kw"] = (
        storage_timeseries["p_wec_kw"] - storage_timeseries["p_grid_kw"]
    )

    storage_timeseries["p_st_lower_kw"] = (
        storage_timeseries["p_lower_kw"] - storage_timeseries["p_grid_kw"]
    )

    storage_timeseries["p_st_upper_kw"] = (
        storage_timeseries["p_upper_kw"] - storage_timeseries["p_grid_kw"]
    )

    storage_timeseries["storage_power_envelope_width_kw"] = (
        storage_timeseries["p_st_upper_kw"] - storage_timeseries["p_st_lower_kw"]
    )

    group_cols = [
        "scenario",
        "smoothing_window_min",
        "horizon_steps",
        "fold_id",
        "segment_id",
    ]

    fold_group_cols = [
        "scenario",
        "scenario_name",
        "smoothing_window_min",
        "smoothing_window_label",
        "horizon_steps",
        "horizon_hours",
        "horizon_label",
        "fold_id",
    ]

    storage_timeseries["raw_ramp_kw_per_step"] = (
        storage_timeseries
        .groupby(group_cols, observed=True)["p_wec_kw"]
        .diff()
    )

    storage_timeseries["grid_ramp_kw_per_step"] = (
        storage_timeseries
        .groupby(group_cols, observed=True)["p_grid_kw"]
        .diff()
    )

    storage_timeseries["abs_raw_ramp_kw_per_step"] = (
        storage_timeseries["raw_ramp_kw_per_step"].abs()
    )

    storage_timeseries["abs_grid_ramp_kw_per_step"] = (
        storage_timeseries["grid_ramp_kw_per_step"].abs()
    )

    storage_timeseries["storage_energy_increment_kwh"] = (
        storage_timeseries["p_st_kw"] * dt_hours
    )

    storage_timeseries["storage_energy_raw_kwh"] = (
        storage_timeseries
        .groupby(group_cols, observed=True)["storage_energy_increment_kwh"]
        .cumsum()
    )

    storage_timeseries["segment_energy_min_kwh"] = (
        storage_timeseries
        .groupby(group_cols, observed=True)["storage_energy_raw_kwh"]
        .transform("min")
    )

    storage_timeseries["segment_energy_max_kwh"] = (
        storage_timeseries
        .groupby(group_cols, observed=True)["storage_energy_raw_kwh"]
        .transform("max")
    )

    storage_timeseries["storage_energy_shifted_kwh"] = (
        storage_timeseries["storage_energy_raw_kwh"]
        - storage_timeseries["segment_energy_min_kwh"]
    )

    storage_timeseries["segment_required_energy_kwh"] = (
        storage_timeseries["segment_energy_max_kwh"]
        - storage_timeseries["segment_energy_min_kwh"]
    )

    fold_required_energy = (
        storage_timeseries
        .groupby(fold_group_cols, observed=True)["segment_required_energy_kwh"]
        .max()
        .reset_index()
        .rename(columns={"segment_required_energy_kwh": "required_energy_kwh"})
    )

    storage_timeseries = storage_timeseries.merge(
        fold_required_energy,
        on=fold_group_cols,
        how="left",
    )

    storage_timeseries["soc"] = np.where(
        storage_timeseries["required_energy_kwh"] > 0,
        storage_timeseries["storage_energy_shifted_kwh"]
        / storage_timeseries["required_energy_kwh"],
        0.0,
    )

    storage_timeseries["abs_storage_power_kw"] = storage_timeseries["p_st_kw"].abs()
    storage_timeseries["charge_power_kw"] = storage_timeseries["p_st_kw"].clip(lower=0)
    storage_timeseries["discharge_power_kw"] = (
        -storage_timeseries["p_st_kw"]
    ).clip(lower=0)

    storage_timeseries["charge_energy_kwh"] = (
        storage_timeseries["charge_power_kw"] * dt_hours
    )

    storage_timeseries["discharge_energy_kwh"] = (
        storage_timeseries["discharge_power_kw"] * dt_hours
    )

    storage_timeseries["net_balancing_energy_kwh"] = (
        storage_timeseries["p_st_kw"] * dt_hours
    )

    return storage_timeseries


def summarize_storage_smoothing_metrics(
    storage_timeseries,
    rated_power_kw,
    dt_hours=0.5,
):
    """Summarize storage-smoothing metrics by fold and across folds."""
    fold_group_cols = [
        "scenario",
        "scenario_name",
        "smoothing_window_min",
        "smoothing_window_label",
        "horizon_steps",
        "horizon_hours",
        "horizon_label",
        "fold_id",
    ]

    summary_group_cols = [
        "scenario",
        "scenario_name",
        "smoothing_window_min",
        "smoothing_window_label",
        "horizon_steps",
        "horizon_hours",
        "horizon_label",
    ]

    storage_metric_rows = []

    for group_key, group_df in storage_timeseries.groupby(
        fold_group_cols,
        observed=True,
    ):
        (
            scenario,
            scenario_name,
            smoothing_window_min,
            smoothing_window_label,
            horizon_steps,
            horizon_hours,
            horizon_label,
            fold_id,
        ) = group_key

        raw_ramp_p95 = group_df["abs_raw_ramp_kw_per_step"].quantile(0.95)
        grid_ramp_p95 = group_df["abs_grid_ramp_kw_per_step"].quantile(0.95)

        raw_ramp_max = group_df["abs_raw_ramp_kw_per_step"].max()
        grid_ramp_max = group_df["abs_grid_ramp_kw_per_step"].max()

        required_energy_kwh = group_df["required_energy_kwh"].max()
        throughput_kwh = group_df["abs_storage_power_kw"].sum() * dt_hours

        mean_p_wec_kw = group_df["p_wec_kw"].mean()
        raw_energy_kwh = group_df["p_wec_kw"].sum() * dt_hours
        grid_energy_kwh = group_df["p_grid_kw"].sum() * dt_hours

        charge_energy_kwh = group_df["charge_energy_kwh"].sum()
        discharge_energy_kwh = group_df["discharge_energy_kwh"].sum()
        net_balancing_energy_kwh = group_df["net_balancing_energy_kwh"].sum()

        ramp_p95_reduction_percent = np.nan
        if raw_ramp_p95 > 0:
            ramp_p95_reduction_percent = 100 * (1 - grid_ramp_p95 / raw_ramp_p95)

        ramp_max_reduction_percent = np.nan
        if raw_ramp_max > 0:
            ramp_max_reduction_percent = 100 * (1 - grid_ramp_max / raw_ramp_max)

        equivalent_full_cycles = 0.0
        required_energy_hours_at_mean_power = 0.0
        required_energy_hours_at_rated_power = 0.0

        if required_energy_kwh > 0:
            equivalent_full_cycles = throughput_kwh / (2 * required_energy_kwh)

            if mean_p_wec_kw > 0:
                required_energy_hours_at_mean_power = (
                    required_energy_kwh / mean_p_wec_kw
                )

            required_energy_hours_at_rated_power = (
                required_energy_kwh / rated_power_kw
            )

        grid_to_raw_energy_ratio = np.nan
        if raw_energy_kwh != 0:
            grid_to_raw_energy_ratio = grid_energy_kwh / raw_energy_kwh

        storage_metric_rows.append(
            {
                "scenario": scenario,
                "scenario_name": scenario_name,
                "smoothing_window_min": smoothing_window_min,
                "smoothing_window_label": smoothing_window_label,
                "horizon_steps": horizon_steps,
                "horizon_hours": horizon_hours,
                "horizon_label": horizon_label,
                "fold_id": fold_id,
                "n_samples": len(group_df),
                "mean_p_wec_kw": mean_p_wec_kw,
                "mean_p_grid_kw": group_df["p_grid_kw"].mean(),
                "raw_energy_kwh": raw_energy_kwh,
                "grid_energy_kwh": grid_energy_kwh,
                "grid_to_raw_energy_ratio": grid_to_raw_energy_ratio,
                "charge_energy_kwh": charge_energy_kwh,
                "discharge_energy_kwh": discharge_energy_kwh,
                "net_balancing_energy_kwh": net_balancing_energy_kwh,
                "raw_ramp_p95_kw_per_step": raw_ramp_p95,
                "grid_ramp_p95_kw_per_step": grid_ramp_p95,
                "ramp_p95_reduction_percent": ramp_p95_reduction_percent,
                "raw_ramp_max_kw_per_step": raw_ramp_max,
                "grid_ramp_max_kw_per_step": grid_ramp_max,
                "ramp_max_reduction_percent": ramp_max_reduction_percent,
                "max_charge_power_kw": group_df["p_st_kw"].max(),
                "max_discharge_power_kw": abs(group_df["p_st_kw"].min()),
                "storage_power_rating_kw": group_df["abs_storage_power_kw"].max(),
                "p95_abs_storage_power_kw": (
                    group_df["abs_storage_power_kw"].quantile(0.95)
                ),
                "required_energy_kwh": required_energy_kwh,
                "required_energy_hours_at_mean_power": (
                    required_energy_hours_at_mean_power
                ),
                "required_energy_hours_at_rated_power": (
                    required_energy_hours_at_rated_power
                ),
                "throughput_kwh": throughput_kwh,
                "equivalent_full_cycles": equivalent_full_cycles,
                "soc_min": group_df["soc"].min(),
                "soc_max": group_df["soc"].max(),
                "mean_storage_envelope_width_kw": (
                    group_df["storage_power_envelope_width_kw"].mean()
                ),
                "possible_discharge_from_lower_bound_kw": (
                    max(0, -group_df["p_st_lower_kw"].min())
                ),
                "possible_charge_from_upper_bound_kw": (
                    max(0, group_df["p_st_upper_kw"].max())
                ),
            }
        )

    storage_metrics_by_fold = pd.DataFrame(storage_metric_rows)

    storage_metrics_summary = (
        storage_metrics_by_fold
        .groupby(summary_group_cols, observed=True)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["scenario", "smoothing_window_min", "horizon_steps"])
    )

    return storage_metrics_by_fold, storage_metrics_summary

def select_representative_high_ramp_case(
    selected_intervals,
    storage_timeseries,
    horizon_steps,
    smoothing_window_min,
    context_hours,
    dt_hours=0.5,
):
    """Select a representative high-ramp case and return its storage time series."""
    context_steps = int(context_hours / dt_hours)

    case_candidates = selected_intervals[
        selected_intervals["horizon_steps"] == horizon_steps
    ].copy()

    case_candidates = case_candidates.sort_values(
        ["fold_id", "segment_id", "target_time"]
    ).reset_index(drop=True)

    case_candidates["position_in_segment"] = (
        case_candidates
        .groupby(["fold_id", "segment_id"])
        .cumcount()
    )

    case_candidates["segment_length"] = (
        case_candidates
        .groupby(["fold_id", "segment_id"])["target_time"]
        .transform("size")
    )

    case_candidates["steps_after"] = (
        case_candidates["segment_length"]
        - case_candidates["position_in_segment"]
        - 1
    )

    case_candidates = case_candidates[
        (case_candidates["position_in_segment"] >= context_steps)
        & (case_candidates["steps_after"] >= context_steps)
        & case_candidates["abs_raw_ramp_kw_per_step"].notna()
    ].copy()

    if case_candidates.empty:
        raise ValueError(
            "No high-ramp case has enough context before and after the selected point."
        )

    case_row = case_candidates.loc[
        case_candidates["abs_raw_ramp_kw_per_step"].idxmax()
    ]

    case_time = case_row["target_time"]
    case_fold_id = case_row["fold_id"]
    case_segment_id = case_row["segment_id"]

    case_start = case_time - pd.Timedelta(hours=context_hours)
    case_end = case_time + pd.Timedelta(hours=context_hours)

    case_filter = (
        (storage_timeseries["horizon_steps"] == horizon_steps)
        & (storage_timeseries["fold_id"] == case_fold_id)
        & (storage_timeseries["segment_id"] == case_segment_id)
        & (storage_timeseries["target_time"] >= case_start)
        & (storage_timeseries["target_time"] <= case_end)
        & (
            (
                (storage_timeseries["scenario"] == "A")
                & (storage_timeseries["smoothing_window_min"] == 0)
            )
            | (
                (storage_timeseries["scenario"].isin(["B", "C", "D"]))
                & (
                    storage_timeseries["smoothing_window_min"]
                    == smoothing_window_min
                )
            )
        )
    )

    case_ts = storage_timeseries[case_filter].copy()

    case_ts = case_ts.sort_values(
        ["scenario", "target_time"]
    ).reset_index(drop=True)

    case_metadata = {
        "case_time": case_time,
        "case_fold_id": case_fold_id,
        "case_segment_id": case_segment_id,
        "case_row": case_row,
    }

    return case_ts, case_metadata

def plot_storage_smoothing_balance(output_path=None, show=True):
    """Plot the conceptual storage-smoothing balance used in Notebook 05."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    wec_box = (0.05, 0.58, 0.22, 0.18)
    target_box = (0.39, 0.58, 0.24, 0.18)
    grid_box = (0.75, 0.58, 0.18, 0.18)
    storage_box = (0.34, 0.18, 0.34, 0.20)

    boxes = [wec_box, target_box, grid_box, storage_box]

    for x, y, width, height in boxes:
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                linewidth=1.2,
                edgecolor="black",
                facecolor="white",
            )
        )

    ax.text(
        wec_box[0] + wec_box[2] / 2,
        wec_box[1] + wec_box[3] / 2,
        "Estimated\nWEC power\n$p_{wec}$",
        ha="center",
        va="center",
        fontsize=12,
    )

    ax.text(
        target_box[0] + target_box[2] / 2,
        target_box[1] + target_box[3] / 2,
        "Smoothing target\nselection rule",
        ha="center",
        va="center",
        fontsize=12,
    )

    ax.text(
        grid_box[0] + grid_box[2] / 2,
        grid_box[1] + grid_box[3] / 2,
        "Grid export\n$p_{grid}$",
        ha="center",
        va="center",
        fontsize=12,
    )

    ax.text(
        storage_box[0] + storage_box[2] / 2,
        storage_box[1] + storage_box[3] / 2,
        "Storage buffer\n$p_{st} = p_{wec} - p_{grid}$",
        ha="center",
        va="center",
        fontsize=12,
    )

    ax.annotate(
        "",
        xy=(target_box[0], wec_box[1] + wec_box[3] / 2),
        xytext=(wec_box[0] + wec_box[2], wec_box[1] + wec_box[3] / 2),
        arrowprops=dict(arrowstyle="->", lw=1.4),
    )

    ax.annotate(
        "",
        xy=(grid_box[0], grid_box[1] + grid_box[3] / 2),
        xytext=(target_box[0] + target_box[2], target_box[1] + target_box[3] / 2),
        arrowprops=dict(arrowstyle="->", lw=1.4),
    )

    ax.annotate(
        "",
        xy=(storage_box[0] + storage_box[2] / 2, storage_box[1] + storage_box[3]),
        xytext=(target_box[0] + target_box[2] / 2, target_box[1]),
        arrowprops=dict(arrowstyle="->", lw=1.4),
    )

    ax.text(
        0.5,
        0.05,
        "$p_{st} > 0$: storage charges    |    $p_{st} < 0$: storage discharges",
        ha="center",
        va="center",
        fontsize=11,
    )

    ax.set_title("Conceptual storage-smoothing balance", fontsize=14, pad=10)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
