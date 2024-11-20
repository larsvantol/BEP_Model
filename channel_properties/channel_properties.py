from functools import cached_property

import numpy as np
from scipy.constants import g


class ChannelProperties:

    def __init__(
        self,
        width: float,
        height: float,
        length: float,
        tidal_averaged_flow: float,
        drag_coefficient: float,
        tidal_velocity_amplitude: float,
        diffusion_prefactor: float = 0.0125,
    ):
        self.width: float = width
        self.height: float = height
        self.length: float = length
        self.tidal_averaged_flow: float = tidal_averaged_flow
        self.drag_coefficient: float = drag_coefficient
        self.tidal_velocity_amplitude: float = tidal_velocity_amplitude

        if diffusion_prefactor < 0.005 or diffusion_prefactor > 0.020:
            raise ValueError("Diffusion prefactor out of range")
        self.diffusion_prefactor: float = diffusion_prefactor

    @cached_property
    def wave_velocity(self) -> float:
        """Calculate the wave velocity of the eta/phi wave."""
        return np.sqrt(g * self.height)

    @cached_property
    def friction_factor(self) -> float:
        """Calculate the friction factor, assumes this tidal velocity amplitude is a constant"""
        # ToDo: Een check invoeren voor de tidal velocity amplitude?
        return (self.drag_coefficient / self.height) * (
            8 * self.tidal_velocity_amplitude / (3 * np.pi)
        )

    @cached_property
    def effective_diffusion_coefficient(self) -> float:
        """Calculate the effective diffusion coefficient, needs a diffusion prefactor
        between 0.005 and 0.020 that depends on the channels empirical data"""
        return (
            self.diffusion_prefactor
            * ((self.width**2) * self.tidal_velocity_amplitude)
            / (np.sqrt(self.drag_coefficient) * self.height)
        )
