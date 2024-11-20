import warnings

import numpy as np
from tqdm.auto import tqdm

from model.tidal_model import TidalModel
from model.tide import TideFrequency


class SaltModel:
    """
    Single channel salt model, which solves for the salt concentration.
    It uses a tidal model to calculate the flow.
    """

    def __init__(
        self,
        channel_tidal_model: TidalModel,
        num_grid_points: int,
        length: float | None = None,
    ):

        self.channel_tidal_model: TidalModel = channel_tidal_model
        self.tide: TideFrequency = channel_tidal_model.tide

        self.n: int = 5

        self.num_grid_points: int = num_grid_points
        self.length: float = self._set_length(length)
        self.delta_x: float = self.length / (num_grid_points - 1)

        self.A_matrix: np.ndarray = self._calc_A_matrix()
        self.C_matrix: np.ndarray = self._calc_C_matrix()
        self.M_matrix: np.ndarray = self._calc_M_matrix()

        self.salt_space_vector: np.ndarray = np.array([])

    ############################################################
    # Public methods
    ############################################################

    def x_domain(self):
        return np.linspace(0, self.length, self.num_grid_points)

    def i_domain(self):
        return np.arange(self.num_grid_points)

    def salt_complex(self, i, t):

        time_vector = np.array(
            # For n = 5 this would be [exp(-2wti), exp(-wti), 1, exp(wti), exp(2wti)]
            [
                np.exp(1j * (j - (self.n - 1) / 2) * self.tide.angular_frequency * t)
                for j in range(self.n)
            ]
        )

        # i is the index of the grid point
        return np.dot(np.squeeze(self.salt_space_vector[i]), time_vector)

    def salt_concentration(self, i, t):
        return np.real(self.salt_complex(i, t))

    def set_solution(self, salt_space_vector: np.ndarray):
        self.salt_space_vector = np.array(
            [salt_space_vector[i * self.n : (i + 1) * self.n] for i in range(self.num_grid_points)]
        )
        # check if the solution shape is correct
        if self.salt_space_vector.shape != (self.num_grid_points, self.n):
            raise ValueError(
                f"Solution shape is incorrect, salt_space_vector is shape {self.salt_space_vector.shape} while it should be {(self.num_grid_points, self.n)}"
            )

    ############################################################
    # Private methods
    ############################################################

    # A, B and C are the matrices for the equation:
    #           A s + B(x) s' + C s'' = 0
    # (s is the salt vector)

    def _calc_A_matrix(self) -> np.ndarray:
        if self.n % 2 == 0:
            raise ValueError("n must be odd")

        return np.diag(
            [(i - (self.n - 1) / 2) * 1j * self.tide.angular_frequency for i in range(self.n)]
        )

    def B_matrix(self, x) -> np.ndarray:
        lower_diagonal = np.eye(
            self.n, k=-1
        ) * self.channel_tidal_model.complex_phi_tidal_space_part(x)
        main_diagonal = (
            np.eye(self.n) * 2 * np.real(self.channel_tidal_model.properties.tidal_averaged_flow)
        )
        upper_diagonal = np.eye(self.n, k=1) * np.conjugate(
            self.channel_tidal_model.complex_phi_tidal_space_part(x)
        )

        b_matrix = lower_diagonal + main_diagonal + upper_diagonal

        prefactor = 1 / (
            2
            * self.channel_tidal_model.properties.width
            * self.channel_tidal_model.properties.height
        )

        return prefactor * (b_matrix)

    def _calc_C_matrix(self) -> np.ndarray:
        return -self.channel_tidal_model.properties.effective_diffusion_coefficient * np.eye(self.n)

    def _calc_M_matrix(self) -> np.ndarray:
        """Generates the block matrix for the salt model.
        It first generates a list of matrices (blocks) and then combines them into a single matrix,
        with dimensions (n * num_grid_points) x (n * num_grid_points)"""

        print("Generating blocks...", end="")
        zero_matrix = np.zeros((self.n, self.n))
        # identity_matrix = np.eye(self.n)

        # Initialize the list of blocks,
        # all blocks are zero matrices and (some) will be filled in later
        blocks = [
            [zero_matrix for _ in range(self.num_grid_points)]
            for _ in range(self.num_grid_points - 2)
        ]

        # Fill in the interior points
        for i in tqdm(range(1, self.num_grid_points - 1), leave=False):

            # at i = 1, x = delta_x etc.
            x = i * self.delta_x
            x = round(x, 10)  # To avoid floating point errors

            blocks[i - 1][i - 1] = self.C_matrix / (self.delta_x**2)
            blocks[i - 1][i] = (
                self.A_matrix
                - (self.B_matrix(x) / self.delta_x)
                - 2 * (self.C_matrix / (self.delta_x**2))
            )
            blocks[i - 1][i + 1] = (self.B_matrix(x) / self.delta_x) + (
                self.C_matrix / (self.delta_x**2)
            )

        print("[DONE]")

        print("Combining blocks...", end="")
        result = np.block(blocks)

        print("[DONE]")

        return result

    def _set_length(self, salt_model_length: float | None) -> float:
        """Sets the length of the channel, has a few checks to see if the length is valid"""

        if salt_model_length == np.inf:
            raise ValueError("Salt model length is infinite")

        # Infinite channel
        if self.channel_tidal_model.properties.length == np.inf:
            if salt_model_length is None:
                raise ValueError(
                    "Channel length is infinite, but no length is given for the salt model"
                )

            return salt_model_length

        # Finite channel
        if salt_model_length is None:
            return self.channel_tidal_model.properties.length

        if salt_model_length > self.channel_tidal_model.properties.length:
            raise ValueError("Channel length is smaller than the given length for the salt model")

        if salt_model_length < self.channel_tidal_model.properties.length:
            warnings.warn("Channel length is larger than the given length for the salt model")

        return salt_model_length
