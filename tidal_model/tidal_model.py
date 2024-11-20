import numpy as np

from model.channel_properties import ChannelProperties
from model.tide import TideFrequency


class TidalModel:
    def __init__(
        self,
        properties: ChannelProperties,
        tide: TideFrequency,
    ):
        self.properties = properties
        self.tide = tide

        self.lambda_ = self._calc_lambda()

        self.flow_prefactor = (
            self.properties.width * 1j * self.tide.angular_frequency
        ) / self.lambda_

        self.alpha = None
        self.beta = None

    ############################################################
    # Public methods
    ############################################################

    def set_alpha_beta(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    # Eta

    def complex_eta_space_part(self, x):
        self._check_x_in_domain(x)
        return self.alpha * np.exp(-self.lambda_ * x) + self.beta * np.exp(self.lambda_ * x)

    def complex_eta(self, x, t):
        self._check_x_in_domain(x)
        return self.complex_eta_space_part(x) * np.exp(1j * self.tide.angular_frequency * t)

    def eta(self, x, t):
        return np.real(self.complex_eta(x, t))

    # Phi

    def complex_phi_tidal_space_part(self, x):
        if np.any(x == np.inf) or np.any(x == -np.inf):
            raise ValueError("x cannot be infinity")
        self._check_x_in_domain(x)

        return self.flow_prefactor * (
            self.alpha * np.exp(-self.lambda_ * x) - self.beta * np.exp(self.lambda_ * x)
        )

    def complex_phi(self, x, t):
        self._check_x_in_domain(x)
        return self.properties.tidal_averaged_flow + (
            self.complex_phi_tidal_space_part(x) * np.exp(1j * self.tide.angular_frequency * t)
        )

    def phi(self, x, t):
        return np.real(self.complex_phi(x, t))

    ############################################################
    # Private methods
    ############################################################

    def _check_x_in_domain(self, x):
        if type(x) is float or type(x) is int:
            if x < 0 or x > self.properties.length:
                raise ValueError(
                    f"x should be in the range [0, L] ([0, {self.properties.length}]), got {x}"
                )
        if type(x) is np.ndarray:
            if np.any(x < 0) or np.any(x > self.properties.length):
                raise ValueError(
                    f"All x should be in the range [0, L] ([0, {self.properties.length}])."
                )

    def _calc_lambda(self):
        omega = self.tide.angular_frequency
        return (1 / self.properties.wave_velocity) * np.sqrt(
            -(omega**2) + self.properties.friction_factor * omega * 1j
        )
