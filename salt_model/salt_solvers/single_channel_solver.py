import time

import numpy as np

from model.salt_model import SaltModel


def _append_tide_sea_side_bc(blocks, b_vector, n, salt_model, sea_concentration):
    zero_matrix = np.zeros((n, n))
    identity_matrix = np.identity(n)

    s_0: np.ndarray = np.array(
        [[sea_concentration] if i == (n - 1) / 2 else [0] for i in range(n)],
        dtype=complex,
    )

    sea_side_bc = [zero_matrix for _ in range(salt_model.num_grid_points)]
    sea_side_bc[0] = identity_matrix

    blocks.append(sea_side_bc)
    b_vector.append([s_0])


def _append_inf_land_side_bc(blocks, b_vector, n, salt_model):
    zero_matrix = np.zeros((n, n))
    identity_matrix = np.identity(n)

    land_side_bc = [zero_matrix for _ in range(salt_model.num_grid_points)]

    land_side_bc[salt_model.num_grid_points - 1] = identity_matrix

    blocks.append(land_side_bc)
    b_vector.append([np.zeros((n, 1))])


def _append_no_transport_land_side_bc(blocks, b_vector, n, salt_model):
    zero_matrix = np.zeros((n, n))
    identity_matrix = np.identity(n)

    land_side_bc = [zero_matrix for _ in range(salt_model.num_grid_points)]

    land_side_bc[salt_model.num_grid_points - 2] = (
        salt_model.channel_tidal_model.properties.effective_diffusion_coefficient
        / salt_model.delta_x
    ) * identity_matrix

    land_side_bc[salt_model.num_grid_points - 1] = (
        salt_model.channel_tidal_model.properties.tidal_averaged_flow
        - (
            salt_model.channel_tidal_model.properties.effective_diffusion_coefficient
            / salt_model.delta_x
        )
    ) * identity_matrix

    blocks.append(land_side_bc)
    b_vector.append([np.zeros((n, 1))])


def _append_interior(blocks, b_vector, n, salt_model):
    blocks.append([salt_model.M_matrix])
    b_vector.append([np.zeros((n * (salt_model.num_grid_points - 2), 1))])


def _check_dimensions(matrix, b, n, salt_model):
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
    if matrix.shape[0] != n * salt_model.num_grid_points:
        raise ValueError(
            f"Matrix and b vector have shapes: {matrix.shape} and {b.shape}, but should have shape {(n * salt_model.num_grid_points, n * salt_model.num_grid_points)} and {(n * salt_model.num_grid_points,)}"
        )


def single_channel_salt_solver(salt_model: SaltModel, sea_concentration: float) -> None:

    n = 5

    b_vector = []
    blocks = []

    # Fill in the sea side boundary condition
    _append_tide_sea_side_bc(blocks, b_vector, n, salt_model, sea_concentration)

    # Fill in the interior of the block matrix
    _append_interior(blocks, b_vector, n, salt_model)

    # Fill in the land side boundary condition
    if salt_model.channel_tidal_model.properties.length == np.inf:
        _append_inf_land_side_bc(blocks, b_vector, n, salt_model)
    else:
        _append_no_transport_land_side_bc(blocks, b_vector, n, salt_model)

    # Combine the block matrix into a single matrix
    matrix = np.block(blocks)
    b = np.block(b_vector).squeeze()

    # Checks
    _check_dimensions(matrix, b, n, salt_model)

    print(f"Matrix shape: {matrix.shape}")
    print(f"b vector shape: {b.shape}")

    # Solve
    print("Solving...", end="")
    start = time.time()
    x = np.linalg.solve(matrix, b)
    end = time.time()

    print(f"\t[Done] (in {end - start:.2f} s)")

    # Set the solution in the model
    salt_model.set_solution(x)
