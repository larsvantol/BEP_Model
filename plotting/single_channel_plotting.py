import os

import matplotlib.pyplot as plt
import numpy as np

from model.salt_model import SaltModel
from model.tidal_model import TidalModel

from .boundary_condition_plotting import eta_and_tide_plot, phi_at_boundary
from .eta_plotting import eta_plot
from .salt_plotting import salt_plot
from .v_plotting import v_from_phi_plot


def plot_single_channel_flow_and_eta_with_bc(
    x_series, t_series, tidal_model: TidalModel, cmap="seismic"
):

    X, T = np.meshgrid(x_series, t_series)
    eta_solution = tidal_model.eta(X, T)
    phi_solution = tidal_model.phi(X, T)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)

    # Eta plot

    eta_plot(ax[0, 0], x_series, t_series, eta_solution, phi_solution=phi_solution, cmap=cmap)

    # Phi plot

    v_from_phi_plot(ax[0, 1], x_series, t_series, phi_solution, tidal_model.properties, cmap=cmap)

    # Boundary conditions

    eta_and_tide_plot(ax[1, 0], t_series, eta_solution, tidal_model, boundary="x = 0")
    phi_at_boundary(ax[1, 1], t_series, phi_solution, boundary="x = L")

    fig.suptitle("Single channel flow and eta with boundary conditions")

    plt.tight_layout()
    plt.show()


def plot_salt_flow_and_eta_in_single_channel(
    x_series,
    t_series,
    salt_model: SaltModel,
    sea_concentration: float,
    percentage=None,
    intrusion_length=None,
    cmap="seismic",
    x_ax_reverse=False,
    salt_log_scale=False,
    filename=None,
):

    fig, ax = plt.subplots(1, 3, figsize=(20, 10), squeeze=False)
    NUM_OF_SPATIAL_POINTS = len(x_series)

    X, T = np.meshgrid(x_series, t_series)
    eta_solution = salt_model.channel_tidal_model.eta(X, T)
    phi_solution = salt_model.channel_tidal_model.phi(X, T)
    salt_solution = salt_model.salt_concentration(
        np.arange(NUM_OF_SPATIAL_POINTS), t_series
    ).transpose()

    # Salt solution
    if salt_log_scale:
        min_salt_bound = np.min(salt_solution[salt_solution > 0])
        salt_bounds = (min_salt_bound, sea_concentration)
    else:
        salt_bounds = (0, sea_concentration)

    salt_plot(
        ax=ax[0, 0],
        x_series=x_series,
        t_series=t_series,
        salt_solution=salt_solution,
        bounds=salt_bounds,
        phi_solution=phi_solution,
        cmap="viridis",
        log_scale=salt_log_scale,
    )

    if percentage:
        level = percentage * sea_concentration
        ax[0, 0].contour(
            x_series,
            t_series,
            salt_solution,
            levels=[level],
            colors="red",
            linestyles="--",
            linewidths=1,
        )
    if intrusion_length:
        ax[0, 0].axvline(intrusion_length, color="white", linestyle="--", linewidth=1)

    # Flow solution

    v_from_phi_plot(
        ax[0, 1],
        x_series,
        t_series,
        phi_solution,
        salt_model.channel_tidal_model.properties,
        cmap=cmap,
    )

    # Eta solution

    eta_plot(ax[0, 2], x_series, t_series, eta_solution, phi_solution=phi_solution, cmap=cmap)

    if x_ax_reverse:
        for axis in ax[0]:
            axis.invert_xaxis()

    fig.tight_layout()

    if filename:
        # Create a directory if it does not exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename)
    else:
        plt.show()
