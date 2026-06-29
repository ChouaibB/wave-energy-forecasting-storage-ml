from matplotlib.lines import Line2D


QC_FLAG_LABELS = {
    -1: "NaN / missing QC flag",
     0: "0 No QC performed",
     1: "1 Good data",
     2: "2 Probably good data",
     3: "3 Bad data, potentially correctable",
     4: "4 Bad data",
     5: "5 Value changed",
     6: "6 Below detection / quantification",
     7: "7 Nominal value",
     8: "8 Interpolated value",
     9: "9 Missing value",
}


QC_FLAG_COLORS = {
    -1: "lightgray",
     0: "gray",
     1: "tab:blue",
     2: "gold",
     3: "orange",
     4: "red",
     5: "purple",
     6: "brown",
     7: "pink",
     8: "cyan",
     9: "black",
}


def plot_qc_flag_axis(ax, time_values, qc_values, point_size=10):
    """Plot QC flags as vertical strips on an existing matplotlib axis."""
    qc_values = qc_values.copy().fillna(-1)

    for qc_code, color in QC_FLAG_COLORS.items():
        mask = qc_values == qc_code

        ax.vlines(
            time_values.loc[mask],
            0,
            qc_values.loc[mask],
            color=color,
            linewidth=0.8,
            alpha=0.8,
        )

        ax.scatter(
            time_values.loc[mask],
            qc_values.loc[mask],
            color=color,
            s=point_size,
        )

    ax.set_ylabel("QC")
    ax.set_ylim(-1.5, 9.5)
    ax.set_yticks(range(-1, 10))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


def make_qc_legend_items(marker_size=6):
    """Create legend handles for the Copernicus in-situ QC flag codes."""
    return [
        Line2D(
            [0],
            [0],
            color=QC_FLAG_COLORS[code],
            marker="o",
            linestyle="None",
            markersize=marker_size,
            label=QC_FLAG_LABELS[code],
        )
        for code in range(-1, 10)
    ]
