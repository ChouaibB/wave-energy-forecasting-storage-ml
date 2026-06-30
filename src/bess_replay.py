import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plot_distribution_panel(
    ax,
    df,
    metric_col,
    title,
    xlabel,
    scenario_color_map,
    n_bins=24,
):
    """Plot one distribution panel with histogram share and smoothed line."""
    positive_values = df.loc[df[metric_col] > 0, metric_col].dropna()

    if positive_values.empty:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Share of active samples (%)")
        return

    x_max = positive_values.quantile(0.995)

    if not np.isfinite(x_max) or x_max <= 0:
        x_max = positive_values.max()

    bin_edges = np.linspace(0, x_max, n_bins + 1)

    for scenario, scenario_df in df.groupby("scenario", observed=True):
        scenario_name = scenario_df["scenario_name"].iloc[0]
        values = scenario_df.loc[scenario_df[metric_col] > 0, metric_col].dropna()

        if values.empty:
            continue

        counts, _ = np.histogram(values, bins=bin_edges)

        if counts.sum() == 0:
            continue

        share_percent = 100 * counts / counts.sum()
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        smoothed_share = (
            pd.Series(share_percent)
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        ax.bar(
            bin_centres,
            share_percent,
            width=np.diff(bin_edges),
            alpha=0.25,
            color=scenario_color_map[scenario],
            align="center",
            label="_nolegend_",
        )

        ax.plot(
            bin_centres,
            smoothed_share,
            linewidth=2.0,
            color=scenario_color_map[scenario],
            label=scenario_name,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of active samples (%)")
    ax.legend(loc="best")


def plot_replay_distribution_grid(
    df,
    output_path,
    scenario_color_map,
    n_bins=24,
    show=True,
):
    """
    Plot a 2x2 grid of distribution panels:
    charge request, discharge request, charge C-rate proxy, discharge C-rate proxy.
    """
    plot_df = df.copy()

    plot_df["charge_request_kw"] = plot_df["p_bess_request_kw"].clip(lower=0)
    plot_df["discharge_request_kw"] = -plot_df["p_bess_request_kw"].clip(upper=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    _plot_distribution_panel(
        ax=axes[0, 0],
        df=plot_df,
        metric_col="charge_request_kw",
        title="Charge request distribution",
        xlabel="Charge request (kW)",
        scenario_color_map=scenario_color_map,
        n_bins=n_bins,
    )

    _plot_distribution_panel(
        ax=axes[0, 1],
        df=plot_df,
        metric_col="discharge_request_kw",
        title="Discharge request distribution",
        xlabel="Discharge request (kW)",
        scenario_color_map=scenario_color_map,
        n_bins=n_bins,
    )

    _plot_distribution_panel(
        ax=axes[1, 0],
        df=plot_df,
        metric_col="charge_c_rate_proxy",
        title="Charge C-rate proxy distribution",
        xlabel="Charge C-rate proxy",
        scenario_color_map=scenario_color_map,
        n_bins=n_bins,
    )

    _plot_distribution_panel(
        ax=axes[1, 1],
        df=plot_df,
        metric_col="discharge_c_rate_proxy",
        title="Discharge C-rate proxy distribution",
        xlabel="Discharge C-rate proxy",
        scenario_color_map=scenario_color_map,
        n_bins=n_bins,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes

def compute_energy_balanced_start_soc(
    power_kw,
    dt_hours,
    nominal_energy_capacity_kwh,
    soc_min,
    soc_max,
):
    """Choose an initial SOC that centers the replay energy swing in the SOC window."""
    cumulative_energy_kwh = pd.Series(power_kw).cumsum() * dt_hours

    energy_midpoint_kwh = 0.5 * (
        cumulative_energy_kwh.min()
        + cumulative_energy_kwh.max()
    )

    soc_midpoint = 0.5 * (soc_min + soc_max)

    start_soc = (
        soc_midpoint
        - energy_midpoint_kwh / nominal_energy_capacity_kwh
    )

    return float(np.clip(start_soc, soc_min, soc_max))


def replay_simses_profile(
    profile_df,
    battery,
    nominal_energy_capacity_kwh,
):
    """Replay one prepared BESS power profile through a SimSES battery object."""
    replay_rows = []

    profile_df = profile_df.sort_values("target_time").reset_index(drop=True)

    for _, row in profile_df.iterrows():
        battery.step(
            power_setpoint=float(row["p_bess_request_w"]),
            dt=float(row["dt_seconds"]),
        )

        state = battery.state

        replay_rows.append(
            {
                "scenario": row["scenario"],
                "scenario_name": row["scenario_name"],
                "fold_id": row["fold_id"],
                "segment_id": row["segment_id"],
                "target_time": row["target_time"],
                "p_bess_request_kw": row["p_bess_request_kw"],
                "p_bess_actual_kw": state.power / 1000,
                "p_tracking_error_kw": (
                    state.power / 1000
                    - row["p_bess_request_kw"]
                ),
                "dt_seconds": row["dt_seconds"],
                "soc": state.soc,
                "terminal_voltage_v": state.v,
                "current_a": state.i,
                "temperature_c": state.T,
                "loss_kw": state.loss / 1000,
                "heat_kw": state.heat / 1000,
                "soh_Q": state.soh_Q,
                "soh_R": state.soh_R,
                "i_max_charge_a": state.i_max_charge,
                "i_max_discharge_a": state.i_max_discharge,
            }
        )

    replay_df = pd.DataFrame(replay_rows)

    replay_df["throughput_increment_kwh"] = (
        replay_df["p_bess_actual_kw"].abs()
        * replay_df["dt_seconds"]
        / 3600
    )

    replay_df["cumulative_throughput_kwh"] = (
        replay_df["throughput_increment_kwh"].cumsum()
    )

    replay_df["equivalent_full_cycles"] = (
        replay_df["cumulative_throughput_kwh"]
        / (2 * nominal_energy_capacity_kwh)
    )

    return replay_df

def simulate_lumped_temperature_response(
    heat_kw,
    dt_seconds,
    ambient_temperature_c=25.0,
    initial_temperature_c=25.0,
    thermal_resistance_k_per_kw=2.0,
    thermal_capacity_kwh_per_k=3.0,
):
    """
    Simulate a simple lumped thermal response driven by heat generation.

    The discrete update uses:
        T_{t+1} = T_t + ((Q_gen - Q_cool) * dt_hours) / C_th

    where:
        Q_gen  = heat generation (kW)
        Q_cool = (T_t - T_amb) / R_th  (kW)
        C_th   = thermal capacity (kWh / K)
    """
    heat_kw = pd.Series(heat_kw).fillna(0.0).astype(float)
    dt_hours = float(dt_seconds) / 3600.0

    temperatures = []
    temperature_c = float(initial_temperature_c)

    for q_gen_kw in heat_kw:
        q_cool_kw = (temperature_c - ambient_temperature_c) / thermal_resistance_k_per_kw
        delta_temperature = ((q_gen_kw - q_cool_kw) * dt_hours) / thermal_capacity_kwh_per_k
        temperature_c = temperature_c + delta_temperature
        temperatures.append(temperature_c)

    return pd.Series(temperatures, index=heat_kw.index)
