import numpy as np
from matplotlib.colors import CenteredNorm
from matplotlib.lines import Line2D

from .formatters import hours_formatter, kilometers_formatter, tide_formatter


def eta_plot(
    ax,
    x_series,
    t_series,
    eta_solution,
    phi_solution=None,
    cmap="seismic",
    title="Tidal wave",
    cbar=True,
    bounds=None,
):
    if bounds is None:
        eta_min = np.min(eta_solution)
        eta_max = np.max(eta_solution)
    else:
        eta_min, eta_max = bounds

    eta_contour = ax.contourf(
        x_series,
        t_series,
        eta_solution,
        levels=100,
        cmap=cmap,
        norm=CenteredNorm(),
        vmin=eta_min,
        vmax=eta_max,
    )

    if cbar:
        cbar = ax.get_figure().colorbar(eta_contour, ax=ax)
        cbar.set_label("$\\eta$ [m]")

    if phi_solution is not None:
        CS = ax.contour(
            x_series,
            t_series,
            phi_solution,
            levels=[0],
            colors="black",
            linestyles="dashed",
        )
        ax.clabel(CS, inline=True, fontsize=10, fmt="Slack tide ($\\phi = 0$)")
        zero_flow_line = Line2D([0], [0], color="black", lw=2, linestyle="--")

        if cbar:
            cbar.add_lines(levels=[0], colors=["black"], linewidths=2)

        ax.legend([zero_flow_line], ["Slack tide ($\\phi = 0$)"])

    ax.set_xlabel("$x$ [km]")
    ax.xaxis.set_major_formatter(kilometers_formatter)
    ax.tick_params(axis="x", rotation=45)

    ax.set_ylabel("$t$ [h]")
    # ax.yaxis.set_major_formatter(tide_formatter)
    ax.yaxis.set_major_formatter(hours_formatter)

    ax.set_title(title)
