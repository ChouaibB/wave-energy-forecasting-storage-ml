import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def build_generic_wec_power_matrix(
    wec_assumptions,
    hm0_grid=None,
    te_grid=None,
):
    """Build the generic illustrative WEC power matrix used in Notebook 02."""
    if hm0_grid is None:
        hm0_grid = np.arange(0.0, 7.01, 0.25)

    if te_grid is None:
        te_grid = np.arange(4.0, 16.01, 0.25)

    hm0_mesh, te_mesh = np.meshgrid(hm0_grid, te_grid, indexing="ij")

    rated_power_kw = wec_assumptions["rated_power_kw"]
    cut_in_hm0_m = wec_assumptions["cut_in_hm0_m"]
    rated_hm0_m = wec_assumptions["rated_hm0_m"]
    high_wave_limit_hm0_m = wec_assumptions["storm_cutout_hm0_m"]
    preferred_te_s = wec_assumptions["preferred_te_s"]
    period_response_width_s = wec_assumptions["period_response_width_s"]

    height_response = (hm0_mesh - cut_in_hm0_m) / (
        rated_hm0_m - cut_in_hm0_m
    )
    height_response = np.clip(height_response, 0, 1)
    height_response = height_response**2

    period_response = np.exp(
        -0.5 * ((te_mesh - preferred_te_s) / period_response_width_s) ** 2
    )

    generic_power_norm = height_response * period_response
    generic_power_norm = np.clip(generic_power_norm, 0, 1)

    generic_power_matrix_kw = rated_power_kw * generic_power_norm
    generic_power_matrix_kw[hm0_mesh >= high_wave_limit_hm0_m] = 0.0

    generic_power_matrix_df = pd.DataFrame(
        generic_power_matrix_kw,
        index=np.round(hm0_grid, 2),
        columns=np.round(te_grid, 2),
    )

    return {
        "hm0_grid": hm0_grid,
        "te_grid": te_grid,
        "hm0_mesh": hm0_mesh,
        "te_mesh": te_mesh,
        "generic_power_matrix_kw": generic_power_matrix_kw,
        "generic_power_matrix_df": generic_power_matrix_df,
    }

def estimate_wec_power_from_matrix(
    wec_df,
    hm0_grid,
    te_grid,
    generic_power_matrix_kw,
    wec_assumptions,
    hm0_col="hm0_m",
    te_col="te_approx_s",
):
    """Map observed sea states to estimated WEC power using a generic power matrix."""
    output_df = wec_df.copy()

    power_interpolator = RegularGridInterpolator(
        points=(hm0_grid, te_grid),
        values=generic_power_matrix_kw,
        bounds_error=False,
        fill_value=np.nan,
    )

    output_df["matrix_input_missing_flag"] = (
        output_df[[hm0_col, te_col]].isna().any(axis=1)
    )

    output_df["te_outside_matrix_flag"] = (
        (output_df[te_col] < te_grid.min())
        | (output_df[te_col] > te_grid.max())
    )

    output_df["hm0_outside_matrix_flag"] = (
        (output_df[hm0_col] < hm0_grid.min())
        | (output_df[hm0_col] > hm0_grid.max())
    )

    hm0_eval = output_df[hm0_col].clip(hm0_grid.min(), hm0_grid.max())
    te_eval = output_df[te_col].clip(te_grid.min(), te_grid.max())
    interp_points = np.column_stack([hm0_eval, te_eval])

    output_df["wec_power_kw_estimated"] = power_interpolator(interp_points)

    output_df.loc[
        output_df["matrix_input_missing_flag"],
        "wec_power_kw_estimated",
    ] = np.nan

    output_df.loc[
        output_df[hm0_col] < wec_assumptions["cut_in_hm0_m"],
        "wec_power_kw_estimated",
    ] = 0.0

    output_df.loc[
        output_df[hm0_col] >= wec_assumptions["storm_cutout_hm0_m"],
        "wec_power_kw_estimated",
    ] = 0.0

    output_df["wec_power_norm_estimated"] = (
        output_df["wec_power_kw_estimated"] / wec_assumptions["rated_power_kw"]
    )

    return output_df

