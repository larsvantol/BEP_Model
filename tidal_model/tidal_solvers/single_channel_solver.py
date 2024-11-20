import numpy as np

from model.tidal_model import BoundaryCondition, TidalModel


def single_channel_tidal_solver(
    tidal_model: TidalModel,
):

    sea_boundary_condition = BoundaryCondition.TIDE
    if tidal_model.properties.length == np.inf:
        land_boundary_condition = BoundaryCondition.INF_BOUNDED
    else:
        land_boundary_condition = BoundaryCondition.ZERO_TIDAL_FLOW

    a = []
    b = []

    # Sea boundary condition
    if sea_boundary_condition == BoundaryCondition.TIDE:
        a.append([1, 1])
        b.append(tidal_model.tide.complex_amplitude)
    else:
        raise ValueError("Only TIDE boundary condition is supported for sea boundary")

    # Land boundary condition
    if land_boundary_condition == BoundaryCondition.ZERO_TIDAL_FLOW:
        a.append(
            [
                np.exp(-tidal_model.lambda_ * tidal_model.properties.length),
                -np.exp(tidal_model.lambda_ * tidal_model.properties.length),
            ]
        )
        b.append(0)
    elif land_boundary_condition == BoundaryCondition.INF_BOUNDED:
        a.append([0, 1])
        b.append(0)
    else:
        raise ValueError("Only ZERO_TIDAL_FLOW and INF_BOUNDED boundary conditions are supported")

    alpha, beta = np.linalg.solve(a, b)
    tidal_model.set_alpha_beta(alpha, beta)
    # return tidal_model
