import numpy as np

from model.tidal_model import BoundaryCondition, TidalModel


def multi_channel_tidal_solver(
    sea_channel_tidal_model: TidalModel,
    side_channel_tidal_model: TidalModel,
    main_channel_tidal_model: TidalModel,
):

    # Check sum of flow
    sum_of_flow = (
        sea_channel_tidal_model.properties.tidal_averaged_flow
        + side_channel_tidal_model.properties.tidal_averaged_flow
        + main_channel_tidal_model.properties.tidal_averaged_flow
    )
    if not np.isclose(sum_of_flow, 0):
        raise ValueError(f"The sum of the tidal averaged flows should be zero, got {sum_of_flow}")

    # (1): sea_channel
    # (2): side_channel
    # (3): main_channel

    # Matrix A and vector b of the system of equations
    a = []
    b = []

    #######
    # Add the split boundary conditions (all x=0)
    #######

    # Equal water level (1) and (2) at the split
    a.append([1, 1, -1, -1, 0, 0])
    b.append(0)

    # Equal water level (2) and (3) at the split
    a.append([0, 0, 1, 1, -1, -1])
    b.append(0)

    # Flow is preserved at the split
    a.append(
        [
            sea_channel_tidal_model.flow_prefactor,
            -sea_channel_tidal_model.flow_prefactor,
            side_channel_tidal_model.flow_prefactor,
            -side_channel_tidal_model.flow_prefactor,
            main_channel_tidal_model.flow_prefactor,
            -main_channel_tidal_model.flow_prefactor,
        ]
    )
    b.append(0)

    #######
    # Now the boundary conditions of the channels
    #######

    # First the b.c.
    sea_boundary_condition = BoundaryCondition.TIDE

    if main_channel_tidal_model.properties.length == np.inf:
        main_boundary_condition = BoundaryCondition.INF_BOUNDED
    else:
        main_boundary_condition = BoundaryCondition.ZERO_TIDAL_FLOW

    if side_channel_tidal_model.properties.length == np.inf:
        side_boundary_condition = BoundaryCondition.INF_BOUNDED
    else:
        side_boundary_condition = BoundaryCondition.ZERO_TIDAL_FLOW

    # Sea channel
    if sea_boundary_condition == BoundaryCondition.TIDE:
        a.append(
            [
                np.exp(
                    -sea_channel_tidal_model.lambda_ * sea_channel_tidal_model.properties.length
                ),
                np.exp(sea_channel_tidal_model.lambda_ * sea_channel_tidal_model.properties.length),
                0,
                0,
                0,
                0,
            ]
        )
        b.append(sea_channel_tidal_model.tide.complex_amplitude)
    else:
        raise ValueError("Only TIDE boundary condition is supported for sea boundary")

    # Side channel
    if side_boundary_condition == BoundaryCondition.INF_BOUNDED:
        a.append([0, 0, 0, 1, 0, 0])
        b.append(0)
    elif side_boundary_condition == BoundaryCondition.ZERO_TIDAL_FLOW:
        a.append(
            [
                0,
                0,
                np.exp(
                    -side_channel_tidal_model.lambda_ * side_channel_tidal_model.properties.length
                ),
                -np.exp(
                    side_channel_tidal_model.lambda_ * side_channel_tidal_model.properties.length
                ),
                0,
                0,
            ]
        )
        b.append(0)

    # Main channel

    if main_boundary_condition == BoundaryCondition.INF_BOUNDED:
        a.append([0, 0, 0, 0, 0, 1])
        b.append(0)
    elif main_boundary_condition == BoundaryCondition.ZERO_TIDAL_FLOW:
        a.append(
            [
                0,
                0,
                0,
                0,
                np.exp(
                    -main_channel_tidal_model.lambda_ * main_channel_tidal_model.properties.length
                ),
                -np.exp(
                    main_channel_tidal_model.lambda_ * main_channel_tidal_model.properties.length
                ),
            ]
        )
        b.append(0)

    # Checks
    if len(a) != 6:
        raise ValueError("The number of equations should be 6")
    if len(a) != len(b):
        raise ValueError("The number of equations and the number of values should match")

    # Solve the system of equations
    (
        sea_channel_alpha,
        sea_channel_beta,
        side_channel_alpha,
        side_channel_beta,
        main_channel_alpha,
        main_channel_beta,
    ) = np.linalg.solve(a, b)

    # Set the alpha and beta coefficients
    sea_channel_tidal_model.set_alpha_beta(sea_channel_alpha, sea_channel_beta)
    side_channel_tidal_model.set_alpha_beta(side_channel_alpha, side_channel_beta)
    main_channel_tidal_model.set_alpha_beta(main_channel_alpha, main_channel_beta)
