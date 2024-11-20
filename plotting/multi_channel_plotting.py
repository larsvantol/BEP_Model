import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, LogNorm, Normalize

from .boundary_condition_plotting import (
    eta_and_tide_plot,
    phi_at_boundary,
    salt_concentration_split,
    split_flow,
    split_water_level,
)
from .eta_plotting import eta_plot
from .salt_plotting import salt_plot
from .v_plotting import v_from_phi_plot


def multi_channel_eta_plot(t_series, data: dict, cmap="seismic", filename=None):

    x_series_sea = data["sea_channel"]["x_series"]
    X_sea, T_sea = np.meshgrid(x_series_sea, t_series)
    eta_solution_sea = data["sea_channel"]["tidal_model"].eta(X_sea, T_sea)
    phi_solution_sea = data["sea_channel"]["tidal_model"].phi(X_sea, T_sea)

    x_series_side = data["side_channel"]["x_series"]
    X_side, T_side = np.meshgrid(x_series_side, t_series)
    eta_solution_side = data["side_channel"]["tidal_model"].eta(X_side, T_side)
    phi_solution_side = data["side_channel"]["tidal_model"].phi(X_side, T_side)

    x_series_main = data["main_channel"]["x_series"]
    X_main, T_main = np.meshgrid(x_series_main, t_series)
    eta_solution_main = data["main_channel"]["tidal_model"].eta(X_main, T_main)
    phi_solution_main = data["main_channel"]["tidal_model"].phi(X_main, T_main)

    eta_min = np.min(
        [np.min(eta_solution_main), np.min(eta_solution_sea), np.min(eta_solution_side)]
    )
    eta_max = np.max(
        [np.max(eta_solution_main), np.max(eta_solution_sea), np.max(eta_solution_side)]
    )

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15, 10),
        width_ratios=[np.max(x_series_main), np.max(x_series_sea), np.max(x_series_side)],
        squeeze=False,
        constrained_layout=True,
    )

    eta_plot(
        ax=ax[0, 0],
        x_series=x_series_main,
        t_series=t_series,
        eta_solution=eta_solution_main,
        phi_solution=phi_solution_main,
        title="Main channel",
        cbar=False,
        bounds=(eta_min, eta_max),
    )
    ax[0, 0].invert_xaxis()

    eta_plot(
        ax=ax[0, 1],
        x_series=x_series_sea,
        t_series=t_series,
        eta_solution=eta_solution_sea,
        phi_solution=phi_solution_sea,
        title="Sea channel",
        cbar=False,
        bounds=(eta_min, eta_max),
    )

    eta_plot(
        ax=ax[0, 2],
        x_series=x_series_side,
        t_series=t_series,
        eta_solution=eta_solution_side,
        phi_solution=phi_solution_side,
        title="Side channel",
        cbar=False,
        bounds=(eta_min, eta_max),
    )

    cmapable = plt.cm.ScalarMappable(cmap=cmap, norm=CenteredNorm(0, np.max([eta_max, -eta_min])))

    cbar = fig.colorbar(cmapable, ax=ax)
    cbar.set_ticks(np.arange(eta_min, eta_max, 0.1))  # type: ignore
    cbar.set_label("Amplitude [km]")

    fig.suptitle("Water height ($\\eta$)", fontsize=16)

    if filename:
        # Create a directory if it does not exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ext in ["png", "pdf", "svg"]:
            print(f"{filename}.{ext}")
            plt.savefig(f"{filename}.{ext}", format=ext)
    else:
        plt.show()


def multi_channel_v_from_phi_plot(t_series, data: dict, cmap="seismic", filename=None):

    x_series_sea = data["sea_channel"]["x_series"]
    X_sea, T_sea = np.meshgrid(x_series_sea, t_series)
    phi_solution_sea = data["sea_channel"]["tidal_model"].phi(X_sea, T_sea)

    x_series_side = data["side_channel"]["x_series"]
    X_side, T_side = np.meshgrid(x_series_side, t_series)
    phi_solution_side = data["side_channel"]["tidal_model"].phi(X_side, T_side)

    x_series_main = data["main_channel"]["x_series"]
    X_main, T_main = np.meshgrid(x_series_main, t_series)
    phi_solution_main = data["main_channel"]["tidal_model"].phi(X_main, T_main)

    phi_min = np.min(
        [np.min(phi_solution_main), np.min(phi_solution_sea), np.min(phi_solution_side)]
    )
    phi_max = np.max(
        [np.max(phi_solution_main), np.max(phi_solution_sea), np.max(phi_solution_side)]
    )
    v_min = phi_min / (
        data["main_channel"]["tidal_model"].properties.width
        * data["main_channel"]["tidal_model"].properties.height
    )
    v_max = phi_max / (
        data["main_channel"]["tidal_model"].properties.width
        * data["main_channel"]["tidal_model"].properties.height
    )

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15, 10),
        width_ratios=[np.max(x_series_main), np.max(x_series_sea), np.max(x_series_side)],
        squeeze=False,
        constrained_layout=True,
    )

    v_from_phi_plot(
        ax=ax[0, 0],
        x_series=x_series_main,
        t_series=t_series,
        channel_properties=data["main_channel"]["tidal_model"].properties,
        phi_solution=phi_solution_main,
        title="Main channel",
        cbar=False,
        bounds=(v_min, v_max),
    )
    ax[0, 0].invert_xaxis()

    v_from_phi_plot(
        ax=ax[0, 1],
        x_series=x_series_sea,
        t_series=t_series,
        channel_properties=data["sea_channel"]["tidal_model"].properties,
        phi_solution=phi_solution_sea,
        title="Sea channel",
        cbar=False,
        bounds=(v_min, v_max),
    )

    v_from_phi_plot(
        ax=ax[0, 2],
        x_series=x_series_side,
        t_series=t_series,
        channel_properties=data["side_channel"]["tidal_model"].properties,
        phi_solution=phi_solution_side,
        title="Side channel",
        cbar=False,
        bounds=(v_min, v_max),
    )

    cmapable = plt.cm.ScalarMappable(cmap=cmap, norm=CenteredNorm(0, np.max([v_max, -v_min])))

    cbar = fig.colorbar(cmapable, ax=ax)
    cbar.set_ticks(np.linspace(v_min, v_max, 20))  # type: ignore
    cbar.set_label("Velocity [m/s]")

    fig.suptitle("Flow ($\\frac{\\phi}{w h}$)", fontsize=16)

    if filename:
        # Create a directory if it does not exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ext in ["png", "pdf", "svg"]:
            plt.savefig(f"{filename}.{ext}", format=ext)
    else:
        plt.show()


def multi_channel_salt_plot(
    t_series,
    data: dict,
    intrusion_length=None,
    percentage=None,
    salt_log_scale=False,
    cmap="viridis",
    filename=None,
):

    sea_concentration: float = data["sea_channel"]["sea_concentration"]

    x_series_sea = data["sea_channel"]["salt_model"].x_domain()
    X_sea, T_sea = np.meshgrid(x_series_sea, t_series)
    phi_solution_sea = data["sea_channel"]["tidal_model"].phi(X_sea, T_sea)
    salt_solution_sea = (
        data["sea_channel"]["salt_model"]
        .salt_concentration(data["sea_channel"]["salt_model"].i_domain(), t_series)
        .transpose()
    )

    x_series_side = data["side_channel"]["salt_model"].x_domain()
    X_side, T_side = np.meshgrid(x_series_side, t_series)
    phi_solution_side = data["side_channel"]["tidal_model"].phi(X_side, T_side)
    salt_solution_side = (
        data["side_channel"]["salt_model"]
        .salt_concentration(data["side_channel"]["salt_model"].i_domain(), t_series)
        .transpose()
    )

    x_series_main = data["main_channel"]["salt_model"].x_domain()
    X_main, T_main = np.meshgrid(x_series_main, t_series)
    phi_solution_main = data["main_channel"]["tidal_model"].phi(X_main, T_main)
    salt_solution_main = (
        data["main_channel"]["salt_model"]
        .salt_concentration(data["main_channel"]["salt_model"].i_domain(), t_series)
        .transpose()
    )

    if intrusion_length:
        rest = data["sea_channel"]["tidal_model"].properties.length - intrusion_length

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15, 10),
        width_ratios=[np.max(x_series_main), np.max(x_series_sea), np.max(x_series_side)],
        squeeze=False,
        constrained_layout=True,
    )

    if salt_log_scale:
        min_salt_bound = np.min(
            [
                np.min(salt_solution_main[salt_solution_main > 0]),
                np.min(salt_solution_sea[salt_solution_sea > 0]),
                np.min(salt_solution_side[salt_solution_side > 0]),
            ]
        )
        salt_bounds = (min_salt_bound, sea_concentration)
    else:
        salt_bounds = (0, sea_concentration)

    salt_plot(
        ax=ax[0, 0],
        x_series=x_series_main,
        t_series=t_series,
        salt_solution=salt_solution_main,
        bounds=salt_bounds,
        phi_solution=phi_solution_main,
        title="Main channel",
        cbar=False,
        cmap=cmap,
        legend=False,
        log_scale=salt_log_scale,
    )
    ax[0, 0].invert_xaxis()

    if percentage:
        level = percentage * sea_concentration
        ax[0, 0].contour(
            X_main,
            T_main,
            salt_solution_main,
            levels=[level],
            colors="red",
            linestyles="--",
            linewidths=1,
        )
    if intrusion_length:
        if rest < 0:
            ax[0, 0].axvline(-1 * rest, color="white", linestyle="--", linewidth=1)

    salt_plot(
        ax=ax[0, 1],
        x_series=x_series_sea,
        t_series=t_series,
        salt_solution=salt_solution_sea,
        bounds=salt_bounds,
        phi_solution=phi_solution_sea,
        title="Sea channel",
        cbar=False,
        cmap=cmap,
        legend=False,
        log_scale=salt_log_scale,
    )
    if percentage:
        level = percentage * sea_concentration
        ax[0, 1].contour(
            X_sea,
            T_sea,
            salt_solution_sea,
            levels=[level],
            colors="red",
            linestyles="--",
            linewidths=1,
        )
    if intrusion_length:
        if rest >= 0:
            ax[0, 1].axvline(rest, color="white", linestyle="--", linewidth=1)

    salt_plot(
        ax=ax[0, 2],
        x_series=x_series_side,
        t_series=t_series,
        salt_solution=salt_solution_side,
        bounds=salt_bounds,
        phi_solution=phi_solution_side,
        title="Side channel",
        cbar=False,
        cmap=cmap,
        legend=False,
        log_scale=salt_log_scale,
    )
    if percentage:
        level = percentage * sea_concentration
        ax[0, 2].contour(
            X_side,
            T_side,
            salt_solution_side,
            levels=[level],
            colors="red",
            linestyles="--",
            linewidths=1,
        )

    if salt_log_scale:

        log_min_bound = np.floor(np.log10(salt_bounds[0]))
        log_max_bound = np.ceil(np.log10(salt_bounds[1]))

        norm = LogNorm(vmin=np.power(10, log_min_bound), vmax=np.power(10, log_max_bound))
        lev_exp_ticks = np.arange(
            log_min_bound,
            log_max_bound + 1,
            1,
        )
        levs_ticks = np.power(10, lev_exp_ticks)
    else:
        norm = Normalize(vmin=salt_bounds[0], vmax=salt_bounds[1])
        levs_ticks = np.linspace(salt_bounds[0], salt_bounds[1], 20)

    cmapable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    cbar = fig.colorbar(cmapable, ax=ax)
    cbar.set_ticks(levs_ticks)  # type: ignore
    cbar.set_label("Concentration ($s$/$s_0$) [-]")

    fig.suptitle("Salt Concentration ($s$)", fontsize=16)

    if filename:
        # Create a directory if it does not exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ext in ["png", "pdf", "svg"]:
            plt.savefig(f"{filename}.{ext}", format=ext)
    else:
        plt.show()


def multi_channel_boundary_conditions_plot(t_series, data: dict, filename=None):

    x_series_sea = data["sea_channel"]["x_series"]
    X_sea, T_sea = np.meshgrid(x_series_sea, t_series)
    eta_solution_sea = data["sea_channel"]["tidal_model"].eta(X_sea, T_sea)
    phi_solution_sea = data["sea_channel"]["tidal_model"].phi(X_sea, T_sea)
    salt_solution_sea = (
        data["sea_channel"]["salt_model"]
        .salt_concentration(data["sea_channel"]["salt_model"].i_domain(), t_series)
        .transpose()
    )

    x_series_side = data["side_channel"]["x_series"]
    X_side, T_side = np.meshgrid(x_series_side, t_series)
    eta_solution_side = data["side_channel"]["tidal_model"].eta(X_side, T_side)
    phi_solution_side = data["side_channel"]["tidal_model"].phi(X_side, T_side)
    salt_solution_side = (
        data["side_channel"]["salt_model"]
        .salt_concentration(data["side_channel"]["salt_model"].i_domain(), t_series)
        .transpose()
    )

    x_series_main = data["main_channel"]["x_series"]
    X_main, T_main = np.meshgrid(x_series_main, t_series)
    eta_solution_main = data["main_channel"]["tidal_model"].eta(X_main, T_main)
    phi_solution_main = data["main_channel"]["tidal_model"].phi(X_main, T_main)
    salt_solution_main = (
        data["main_channel"]["salt_model"]
        .salt_concentration(data["main_channel"]["salt_model"].i_domain(), t_series)
        .transpose()
    )

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        squeeze=False,
        constrained_layout=True,
    )

    split_water_level(
        ax=ax[0, 0],
        t_domain=t_series,
        eta_solution_sea=eta_solution_sea,
        eta_solution_side=eta_solution_side,
        eta_solution_main=eta_solution_main,
        tide=data["sea_channel"]["tidal_model"].tide,
    )

    split_flow(
        ax=ax[0, 1],
        t_domain=t_series,
        phi_solution_sea=phi_solution_sea,
        phi_solution_side=phi_solution_side,
        phi_solution_main=phi_solution_main,
    )

    salt_concentration_split(
        ax=ax[0, 2],
        t_domain=t_series,
        salt_solution_sea=salt_solution_sea,
        salt_solution_side=salt_solution_side,
        salt_solution_main=salt_solution_main,
    )

    eta_and_tide_plot(
        ax=ax[1, 0],
        t_series=t_series,
        eta_solution=eta_solution_sea,
        tidal_model=data["sea_channel"]["tidal_model"],
        boundary="x = L",
        title="Sea channel at x = L",
    )

    phi_at_boundary(
        ax=ax[1, 1],
        t_series=t_series,
        phi_solution=phi_solution_side,
        boundary="x = L",
        title="Side channel at x = L",
    )

    phi_at_boundary(
        ax=ax[1, 2],
        t_series=t_series,
        phi_solution=phi_solution_main,
        boundary="x = L",
        title="Main channel at x = L",
    )

    fig.suptitle("Boundary conditions", fontsize=16)

    if filename:
        # Create a directory if it does not exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ext in ["png", "pdf", "svg"]:
            plt.savefig(f"{filename}.{ext}", format=ext)
    else:
        plt.show()
