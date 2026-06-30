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
