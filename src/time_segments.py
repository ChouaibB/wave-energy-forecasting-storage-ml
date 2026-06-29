import pandas as pd


def add_continuous_target_segments(
    df,
    time_col,
    target_col,
    expected_step="30min",
    availability_col="target_available",
    segment_col="continuous_segment_id",
):
    """Add availability and continuous-segment columns for a regularly sampled target."""
    output_df = df.copy()

    output_df[availability_col] = output_df[target_col].notna()

    expected_step = pd.Timedelta(expected_step)
    time_gap = output_df[time_col].diff().ne(expected_step)
    availability_change = output_df[availability_col].ne(
        output_df[availability_col].shift()
    )

    output_df[segment_col] = (time_gap | availability_change).cumsum()
    output_df.loc[~output_df[availability_col], segment_col] = pd.NA

    return output_df


def summarize_continuous_segments(
    df,
    time_col,
    target_col,
    segment_col="continuous_segment_id",
    sample_hours=0.5,
):
    """Summarize continuous non-missing target segments."""
    segment_summary = (
        df.loc[df[segment_col].notna()]
        .groupby(segment_col, dropna=True)
        .agg(
            start_time=(time_col, "min"),
            end_time=(time_col, "max"),
            n_records=(target_col, "size"),
        )
        .reset_index()
    )

    segment_summary["duration_hours"] = (
        segment_summary["n_records"] * sample_hours
    )

    return segment_summary
