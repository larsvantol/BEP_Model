import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

from model.salt_model import characteristic_intrusion_length

from .formatters import hours_formatter, kilometers_formatter


def salt_plot(
    ax,
    x_series,
    t_series,
    salt_solution,
    bounds: tuple,
    phi_solution=None,
    cmap="viridis",
    title="Salt Concentration",
    cbar=True,
    legend=True,
    log_scale=False,
):

    if not (bounds[0] <= np.min(salt_solution) < np.max(salt_solution) < bounds[1] * 1.1):
        print(
            f"WARNING: Salt concentration range is {np.min(salt_solution)} to {np.max(salt_solution)}, while bounds are {bounds}"
        )

    # Title
    ax.set_title(title)

    # X axis
    ax.set_xlabel("$x$ [km]")
    ax.xaxis.set_major_formatter(kilometers_formatter)
    ax.tick_params(axis="x", rotation=45)

    # Y axis
    ax.set_ylabel("$t$ [h]")
    ax.yaxis.set_major_formatter(hours_formatter)

    # Check dimensions
    if salt_solution.shape != (len(t_series), len(x_series)):
        raise ValueError(
            f"Salt solution shape is {salt_solution.shape} while it should be (t, x) {(len(t_series), len(x_series))}"
        )

    # Salt plot
    num_of_levels = 1000
    if log_scale:
        log_min_bound = np.floor(np.log10(bounds[0]))
        log_max_bound = np.ceil(np.log10(bounds[1]))

        norm = LogNorm(vmin=np.power(10, log_min_bound), vmax=np.power(10, log_max_bound))
        salt_solution = np.ma.masked_where(
            salt_solution == 0, salt_solution
        )  # Necessary for log scale, since log(0) is undefined
        levels_exp = np.linspace(
            log_min_bound,
            log_max_bound,
            num_of_levels,
        )
        levels = np.power(10, levels_exp)
        lev_exp_ticks = np.arange(
            log_min_bound,
            log_max_bound + 1,
            1,
        )
        levs_ticks = np.power(10, lev_exp_ticks)

        ax.set_facecolor("black")  # So that the masked values are black
    else:
        norm = Normalize(vmin=bounds[0], vmax=bounds[1])
        levels = num_of_levels
        levs_ticks = np.linspace(bounds[0], bounds[1], 20)

    salt_contour = ax.contourf(
        x_series,
        t_series,
        salt_solution,
        cmap=cmap,
        levels=levels,
        norm=norm,
    )

    if cbar:
        cbar = ax.get_figure().colorbar(salt_contour, ax=ax)
        if not log_scale:
            cbar.formatter.set_useOffset(False)  # type: ignore
        cbar.set_label("Concentration ($s$/$s_0$) [-]")
        cbar.set_ticks(levs_ticks)

    # Sea concentration line
    sea_concentration_line = Line2D([0], [0], color="red", lw=2)

    if cbar:
        cbar.add_lines(levels=[bounds[1]], colors=["red"], linewidths=2)

    if phi_solution is not None:
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

        if legend:
            ax.legend(
                [zero_flow_line, sea_concentration_line],
                ["Slack tide ($\\phi = 0$)", "Sea water\nconcentration"],
            )
    else:
        if legend:
            ax.legend([sea_concentration_line], ["Sea water\nconcentration"])
