from model.tide import height_from_tide

from .formatters import hours_formatter


def eta_and_tide_plot(ax, t_series, eta_solution, tidal_model, boundary, title=None):

    if boundary == "x = 0":
        index = 0
    elif boundary == "x = L":
        index = -1
    else:
        raise ValueError(f"Invalid boundary condition: {boundary}")

    T_MIN = t_series[0]
    T_MAX = t_series[-1]

    ax.plot(t_series, eta_solution[:, index], linestyle="-", label="Solution")
    ax.plot(
        t_series,
        height_from_tide(tidal_model.tide, t_series),
        linestyle="--",
        label="Boundary condition",
    )
    ax.set_xlabel("Time [h]")
    ax.xaxis.set_major_formatter(hours_formatter)
    ax.set_xlim([T_MIN, T_MAX])
    ax.set_ylabel("Amplitude")
    if title is None:
        ax.set_title(f"Tidal wave at {boundary}")
    else:
        ax.set_title(title)
    ax.grid()
    ax.legend()


def phi_at_boundary(ax, t_series, phi_solution, boundary, title=None):

    if boundary == "x = 0":
        index = 0
    elif boundary == "x = L":
        index = -1
    else:
        raise ValueError(f"Invalid boundary condition: {boundary}")

    T_MIN = t_series[0]
    T_MAX = t_series[-1]

    ax.plot(t_series, phi_solution[:, index], linestyle="-")
    ax.set_xlabel("Time [h]")
    ax.xaxis.set_major_formatter(hours_formatter)
    ax.set_xlim([T_MIN, T_MAX])
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.set_ylabel("Flow rate (m^3/s)")
    if title is None:
        ax.set_title(f"Flow rate at {boundary}")
    else:
        ax.set_title(title)
    ax.grid()
    ax.legend()


def split_water_level(ax, t_domain, eta_solution_sea, eta_solution_side, eta_solution_main, tide):
    index = 0
    ax.plot(t_domain, eta_solution_sea[:, index], label="Sea Channel")
    ax.plot(t_domain, eta_solution_main[:, index], label="Main Channel")
    ax.plot(t_domain, eta_solution_side[:, index], label="Side Channel")

    ax.plot(t_domain, height_from_tide(tide, t_domain), label="Tide (seaside)", linestyle="--")
    ax.set_title("Water height at split")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Amplitude [km]")
    ax.xaxis.set_major_formatter(hours_formatter)
    ax.legend()
    ax.grid()


def split_flow(ax, t_domain, phi_solution_sea, phi_solution_side, phi_solution_main):
    index = 0
    ax.plot(t_domain, phi_solution_sea[:, index], label="Sea Channel")
    ax.plot(t_domain, phi_solution_main[:, index], label="Main Channel")
    ax.plot(t_domain, phi_solution_side[:, index], label="Side Channel")

    sum_of_channels = (
        phi_solution_sea[:, index] + phi_solution_main[:, index] + phi_solution_side[:, index]
    )

    ax.plot(
        t_domain,
        sum_of_channels,
        label="Sum of all channels",
        linestyle="--",
    )
    ax.set_title(r"$\Phi_{i}$ at boundary $x = 0$")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Flow (m^3/s)")
    ax.xaxis.set_major_formatter(hours_formatter)
    ax.legend()
    ax.grid()


def salt_concentration_split(
    ax, t_domain, salt_solution_sea, salt_solution_side, salt_solution_main
):
    index = 0
    ax.plot(t_domain, salt_solution_sea[:, index], label="Sea Channel")
    ax.plot(t_domain, salt_solution_main[:, index], label="Main Channel")
    ax.plot(t_domain, salt_solution_side[:, index], label="Side Channel")

    ax.set_title("Salt concentration at split")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Concentration [kg/m^3]")
    ax.xaxis.set_major_formatter(hours_formatter)
    ax.legend()
    ax.grid()
