import numpy as np
from matplotlib.colors import CenteredNorm
from matplotlib.lines import Line2D

from .formatters import hours_formatter, kilometers_formatter


def v_from_phi_plot(
    ax,
    x_series,
    t_series,
    phi_solution,
    channel_properties,
    cmap="seismic",
    title="Flow velocity",
    cbar=True,
    bounds=None,
):

    v_solution = phi_solution / (channel_properties.width * channel_properties.height)

    if bounds is None:
        v_min = np.min(v_solution)
        v_max = np.max(v_solution)
    else:
        v_min, v_max = bounds

    velocity_contour = ax.contourf(
        x_series,
        t_series,
        v_solution,
        levels=100,
        cmap=cmap,
        norm=CenteredNorm(vcenter=0, halfrange=max(abs(v_min), abs(v_max))),
    )

    if cbar:
        cbar = ax.get_figure().colorbar(velocity_contour, ax=ax)
        cbar.set_label("$v$ [m/s]")
        cbar.formatter.set_useOffset(False)  # type: ignore

    slack_tide_contour = ax.contour(
        x_series,
        t_series,
        phi_solution,
        levels=[0],
        colors="black",
        linestyles="dashed",
    )
    ax.clabel(slack_tide_contour, inline=True, fontsize=10, fmt="Slack tide ($\\phi = 0$)")
    zero_flow_line = Line2D([0], [0], color="black", lw=2, linestyle="--")

    if cbar:
        cbar.add_lines(levels=[0], colors=["black"], linewidths=2)
    ax.legend([zero_flow_line], ["Slack tide ($\\phi = 0$)"])

    ax.set_xlabel("$x$ [km]")
    ax.xaxis.set_major_formatter(kilometers_formatter)
    ax.tick_params(axis="x", rotation=45)

    ax.set_ylabel("$t$ [h]")
    ax.yaxis.set_major_formatter(hours_formatter)

    ax.set_title(title)
