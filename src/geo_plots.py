from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs


ETOPO_2022_60S_URL = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/"
    "60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
)


def plot_near_coast_station_map(
    station_name,
    station_latitude,
    station_longitude,
    city_name,
    city_latitude,
    city_longitude,
    subset_min_longitude,
    subset_max_longitude,
    subset_min_latitude,
    subset_max_latitude,
    map_min_longitude,
    map_max_longitude,
    map_min_latitude,
    map_max_latitude,
    output_path,
    figure_title,
    city_label=None,
    subset_label="Copernicus subset box",
    station_color="red",
    city_color="tab:blue",
    subset_box_color="red",
    station_marker="o",
    city_marker="^",
    figure_width=12,
    figure_height=8,
    max_depth_for_colorbar_m=2500,
    max_land_elevation_for_colorbar_m=1200,
    station_label_dx=0.05,
    station_label_dy=0.03,
    city_label_dx=0.05,
    city_label_dy=0.00,
    station_label_fontsize=13,
    city_label_fontsize=13,
    title_fontsize=14,
    axis_label_fontsize=12,
    legend_fontsize=11,
    colorbar_label_fontsize=12,
    colorbar_tick_fontsize=10,
    save_dpi=300,
    show=True,
):
    # Use city_name as the label if a custom city_label is not provided
    if city_label is None:
        city_label = city_name

    # Make sure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the bathymetry/topography dataset and select the local map region
    etopo = xr.open_dataset(ETOPO_2022_60S_URL, engine="netcdf4")

    elevation_m = etopo["z"].sel(
        lon=slice(map_min_longitude, map_max_longitude),
        lat=slice(map_min_latitude, map_max_latitude),
    ).load()

    # Separate sea depth and land elevation
    water_depth_m = -elevation_m.where(elevation_m < 0)
    land_elevation_m = elevation_m.where(elevation_m > 0)

    # Create a blue colormap for water depth
    base_water_cmap = plt.get_cmap("Blues")
    water_cmap = LinearSegmentedColormap.from_list(
        "truncated_blues",
        base_water_cmap(np.linspace(0.30, 1.00, 256)),
    )

    # Create a brown colormap for land elevation
    base_land_cmap = plt.get_cmap("YlOrBr")
    land_cmap = LinearSegmentedColormap.from_list(
        "truncated_browns",
        base_land_cmap(np.linspace(0.20, 0.95, 256)),
    )

    # Create the figure and axis
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot land elevation
    land_plot = ax.pcolormesh(
        land_elevation_m["lon"],
        land_elevation_m["lat"],
        land_elevation_m,
        shading="auto",
        cmap=land_cmap,
        vmin=0,
        vmax=max_land_elevation_for_colorbar_m,
        transform=ccrs.PlateCarree(),
    )

    # Plot water depth
    water_plot = ax.pcolormesh(
        water_depth_m["lon"],
        water_depth_m["lat"],
        water_depth_m,
        shading="auto",
        cmap=water_cmap,
        vmin=0,
        vmax=max_depth_for_colorbar_m,
        transform=ccrs.PlateCarree(),
    )

    # Draw coastline as the 0 m elevation contour
    ax.contour(
        elevation_m["lon"],
        elevation_m["lat"],
        elevation_m,
        levels=[0],
        colors="black",
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
    )

    # Plot the subset box
    subset_lons = [
        subset_min_longitude,
        subset_max_longitude,
        subset_max_longitude,
        subset_min_longitude,
        subset_min_longitude,
    ]

    subset_lats = [
        subset_min_latitude,
        subset_min_latitude,
        subset_max_latitude,
        subset_max_latitude,
        subset_min_latitude,
    ]

    ax.plot(
        subset_lons,
        subset_lats,
        linestyle="--",
        linewidth=1.8,
        color=subset_box_color,
        transform=ccrs.PlateCarree(),
        label=subset_label,
    )

    # Plot the station
    ax.scatter(
        station_longitude,
        station_latitude,
        s=90,
        marker=station_marker,
        color=station_color,
        transform=ccrs.PlateCarree(),
        label=station_name,
        zorder=5,
    )

    # Plot the city
    ax.scatter(
        city_longitude,
        city_latitude,
        s=90,
        marker=city_marker,
        color=city_color,
        transform=ccrs.PlateCarree(),
        label=city_label,
        zorder=5,
    )

    # Add station label
    ax.text(
        station_longitude + station_label_dx,
        station_latitude + station_label_dy,
        station_name,
        color=station_color,
        fontsize=station_label_fontsize,
        fontweight="bold",
        transform=ccrs.PlateCarree(),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
    )

    # Add city label
    ax.text(
        city_longitude + city_label_dx,
        city_latitude + city_label_dy,
        city_label,
        color=city_color,
        fontsize=city_label_fontsize,
        fontweight="bold",
        transform=ccrs.PlateCarree(),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
    )

    # Format the map
    ax.set_title(figure_title, fontsize=title_fontsize)
    ax.set_xlabel("Longitude", fontsize=axis_label_fontsize)
    ax.set_ylabel("Latitude", fontsize=axis_label_fontsize)
    ax.set_xlim(map_min_longitude, map_max_longitude)
    ax.set_ylim(map_min_latitude, map_max_latitude)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower left", fontsize=legend_fontsize)

    # Add water depth colorbar
    water_colorbar = fig.colorbar(
        water_plot,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    water_colorbar.set_label("Water depth (m)", fontsize=colorbar_label_fontsize)
    water_colorbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    # Add land elevation colorbar
    land_colorbar = fig.colorbar(
        land_plot,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.10,
    )
    land_colorbar.set_label(
        "Land elevation (m)",
        fontsize=colorbar_label_fontsize,
    )
    land_colorbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    # Save the figure
    plt.tight_layout()
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")

    # Show the figure in the notebook if requested
    if show:
        plt.show()

    return fig, ax
