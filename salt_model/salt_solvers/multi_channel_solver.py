import time

import numpy as np

from model.salt_model import SaltModel


def _append_no_transport_main_side_bc(
    blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model
):
    blocks.append(
        [
            # sea
            np.zeros((n, sea_salt_model.M_matrix.shape[1])),
            # side
            np.zeros((n, side_salt_model.M_matrix.shape[1])),
            # main
            np.zeros((n, main_salt_model.M_matrix.shape[1] - 2 * n)),
            -(main_salt_model.C_matrix / main_salt_model.delta_x),  # m-2
            main_salt_model.B_matrix(main_salt_model.channel_tidal_model.properties.length)
            + (main_salt_model.C_matrix / main_salt_model.delta_x),  # m-1
        ]
    )
    b_vector.append([np.zeros((n, 1))])


def _append_inf_main_bc(blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model):
    identity_matrix = np.identity(n)

    main_bc = [
        # sea
        np.zeros((n, sea_salt_model.M_matrix.shape[1])),
        # side
        np.zeros((n, side_salt_model.M_matrix.shape[1])),
        # main
        np.zeros((n, main_salt_model.M_matrix.shape[1] - 1 * n)),
        identity_matrix,  # m-1
    ]

    blocks.append(main_bc)
    b_vector.append([np.zeros((n, 1))])


def _append_inf_side_bc(blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model):
    identity_matrix = np.identity(n)

    main_bc = [
        # sea
        np.zeros((n, sea_salt_model.M_matrix.shape[1])),
        # side
        np.zeros((n, side_salt_model.M_matrix.shape[1] - 1 * n)),
        identity_matrix,  # m-1
        # main
        np.zeros((n, main_salt_model.M_matrix.shape[1])),
    ]

    blocks.append(main_bc)
    b_vector.append([np.zeros((n, 1))])


def _append_no_transport_side_bc(
    blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model
):
    blocks.append(
        [
            # sea
            np.zeros((n, sea_salt_model.M_matrix.shape[1])),
            # side
            np.zeros((n, side_salt_model.M_matrix.shape[1] - 2 * n)),
            -(side_salt_model.C_matrix / side_salt_model.delta_x),  # m-2
            side_salt_model.B_matrix(side_salt_model.channel_tidal_model.properties.length)
            + (side_salt_model.C_matrix / side_salt_model.delta_x),  # m-1
            # main
            np.zeros((n, main_salt_model.M_matrix.shape[1])),
        ]
    )
    b_vector.append([np.zeros((n, 1))])


def _append_constant_concentration_sea_side(
    blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model, sea_concentration
):
    zero_matrix = np.zeros((n, n))
    identity_matrix = np.identity(n)

    s_0: np.ndarray = np.array(
        [[sea_concentration] if i == (n - 1) / 2 else [0] for i in range(n)],
        dtype=complex,
    )
    sea_side_bc = [
        zero_matrix
        for _ in range(
            sea_salt_model.num_grid_points
            + side_salt_model.num_grid_points
            + main_salt_model.num_grid_points
        )
    ]
    # x=L is (m-1)+offset and since the sea channel is first, the index is m-1
    sea_side_bc[sea_salt_model.num_grid_points - 1] = identity_matrix

    blocks.append(sea_side_bc)
    b_vector.append([s_0])


def _append_conservation_of_mass(
    blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model
):
    blocks.append(
        [
            # sea
            sea_salt_model.B_matrix(0) - (sea_salt_model.C_matrix / sea_salt_model.delta_x),  # 0
            (sea_salt_model.C_matrix / sea_salt_model.delta_x),  # 1
            np.zeros((n, sea_salt_model.M_matrix.shape[1] - 2 * n)),
            # side
            side_salt_model.B_matrix(0) - (side_salt_model.C_matrix / side_salt_model.delta_x),  # 0
            (side_salt_model.C_matrix / side_salt_model.delta_x),  # 1
            np.zeros((n, side_salt_model.M_matrix.shape[1] - 2 * n)),
            # main
            main_salt_model.B_matrix(0) - (main_salt_model.C_matrix / main_salt_model.delta_x),  # 0
            (main_salt_model.C_matrix / main_salt_model.delta_x),  # 1
            np.zeros((n, main_salt_model.M_matrix.shape[1] - 2 * n)),
        ]
    )
    b_vector.append([np.zeros((n, 1))])


def _append_equal_concentration(
    blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model
):
    identity_matrix = np.identity(n)
    # Equal concentration (sea and side)
    blocks.append(
        [
            # Sea
            identity_matrix,
            np.zeros((n, sea_salt_model.M_matrix.shape[1] - n)),
            # Side
            -identity_matrix,
            np.zeros((n, side_salt_model.M_matrix.shape[1] - n)),
            # Main
            np.zeros((n, main_salt_model.M_matrix.shape[1])),
        ]
    )
    b_vector.append([np.zeros((n, 1))])

    # Equal concentration (side and main)

    blocks.append(
        [
            # Sea
            np.zeros((n, sea_salt_model.M_matrix.shape[1])),
            # Side
            identity_matrix,
            np.zeros((n, side_salt_model.M_matrix.shape[1] - n)),
            # Main
            -identity_matrix,
            np.zeros((n, main_salt_model.M_matrix.shape[1] - n)),
        ]
    )
    b_vector.append([np.zeros((n, 1))])


def _append_interior(blocks, b_vector, sea_salt_model, side_salt_model, main_salt_model):
    # Sea channel
    blocks.append(
        [
            sea_salt_model.M_matrix,
            np.zeros((sea_salt_model.M_matrix.shape[0], side_salt_model.M_matrix.shape[1])),
            np.zeros((sea_salt_model.M_matrix.shape[0], main_salt_model.M_matrix.shape[1])),
        ]
    )
    b_vector.append([np.zeros((sea_salt_model.M_matrix.shape[0], 1))])

    # Side channel
    blocks.append(
        [
            np.zeros((side_salt_model.M_matrix.shape[0], sea_salt_model.M_matrix.shape[1])),
            side_salt_model.M_matrix,
            np.zeros((side_salt_model.M_matrix.shape[0], main_salt_model.M_matrix.shape[1])),
        ]
    )
    b_vector.append([np.zeros((side_salt_model.M_matrix.shape[0], 1))])

    # Main channel
    blocks.append(
        [
            np.zeros((main_salt_model.M_matrix.shape[0], sea_salt_model.M_matrix.shape[1])),
            np.zeros((main_salt_model.M_matrix.shape[0], side_salt_model.M_matrix.shape[1])),
            main_salt_model.M_matrix,
        ]
    )
    b_vector.append([np.zeros((main_salt_model.M_matrix.shape[0], 1))])


def _check_dimensions(matrix, b, n, sea_salt_model, side_salt_model, main_salt_model):
    if matrix.ndim != 2:
        raise ValueError(f"Matrix is not 2D, but has shape: {matrix.shape}")
    if b.ndim != 1:
        raise ValueError(f"b vector is not 1D, but has shape: {b.shape}")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square, but has shape: {matrix.shape}")
    if matrix.shape[0] != b.shape[0]:
        raise ValueError(
            f"Matrix and b vector are not compatible, but have shapes: {matrix.shape} and {b.shape}"
        )

    total_grid_points = (
        sea_salt_model.num_grid_points
        + side_salt_model.num_grid_points
        + main_salt_model.num_grid_points
    )
    if matrix.shape[0] != n * total_grid_points:
        raise ValueError(
            f"Matrix and b vector have shapes: {matrix.shape} and {b.shape}, but should have shape {(n * total_grid_points, n * total_grid_points)} and {(n * total_grid_points,)}"
        )


def multi_channel_salt_solver(
    sea_salt_model: SaltModel,
    side_salt_model: SaltModel,
    main_salt_model: SaltModel,
    sea_concentration: float,
) -> None:

    n = 5

    b_vector = []
    blocks = []

    ###############################################
    # Create block matrix
    ###############################################

    # Interior of the block matrix

    _append_interior(blocks, b_vector, sea_salt_model, side_salt_model, main_salt_model)

    # Equal concentration (sea, side) and (side, main)

    _append_equal_concentration(
        blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model
    )

    # Conservation of mass (sea, side and main)

    _append_conservation_of_mass(
        blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model
    )

    # Fill in the sea channel boundary condition (concentration is sea_concentration)

    _append_constant_concentration_sea_side(
        blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model, sea_concentration
    )

    # Fill in the side channel boundary condition (no transport)

    if side_salt_model.channel_tidal_model.properties.length == np.inf:
        _append_inf_side_bc(blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model)
    else:
        _append_no_transport_side_bc(
            blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model
        )

    # Fill in the main channel boundary condition (no transport)

    if main_salt_model.channel_tidal_model.properties.length == np.inf:
        _append_inf_main_bc(blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model)
    else:
        _append_no_transport_main_side_bc(
            blocks, b_vector, n, sea_salt_model, side_salt_model, main_salt_model
        )

    ###############################################
    # Combine block matrix
    ###############################################

    matrix = np.block(blocks)
    b = np.block(b_vector).squeeze()

    ###############################################
    # Checks
    ###############################################

    # Checks
    _check_dimensions(matrix, b, n, sea_salt_model, side_salt_model, main_salt_model)

    print(f"Matrix shape: {matrix.shape}")
    print(f"b vector shape: {b.shape}")

    ###############################################
    # Solve
    ###############################################

    print("Solving...")
    start = time.time()
    x = np.linalg.solve(matrix, b)
    end = time.time()
    print(f"Solved in {end - start:.2f} seconds")

    ###############################################
    # Split solution
    ###############################################

    sea_channel_salt_vector = x[: sea_salt_model.M_matrix.shape[1]]

    side_channel_salt_vector = x[
        sea_salt_model.M_matrix.shape[1] : sea_salt_model.M_matrix.shape[1]
        + side_salt_model.M_matrix.shape[1]
    ]

    main_channel_salt_vector = x[
        sea_salt_model.M_matrix.shape[1] + side_salt_model.M_matrix.shape[1] :
    ]

    # Check dimensions
    if sea_channel_salt_vector.shape[0] != sea_salt_model.num_grid_points * n:
        raise ValueError(
            f"Sea channel salt vector has incorrect dimensions, shape is {sea_channel_salt_vector.shape} while it should be {(sea_salt_model.num_grid_points*n, 1)}"
        )
    if side_channel_salt_vector.shape[0] != side_salt_model.num_grid_points * n:
        raise ValueError(
            f"Side channel salt vector has incorrect dimensions, shape is {side_channel_salt_vector.shape} while it should be {(side_salt_model.num_grid_points*n, 1)}"
        )
    if main_channel_salt_vector.shape[0] != main_salt_model.num_grid_points * n:
        raise ValueError(
            f"Main channel salt vector has incorrect dimensions, shape is {main_channel_salt_vector.shape} while it should be {(main_salt_model.num_grid_points*n, 1)}"
        )

    # Set the solution
    sea_salt_model.set_solution(sea_channel_salt_vector)
    side_salt_model.set_solution(side_channel_salt_vector)
    main_salt_model.set_solution(main_channel_salt_vector)
