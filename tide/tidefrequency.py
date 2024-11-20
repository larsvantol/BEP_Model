from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np


@dataclass
class TideFrequency:
    period: float  # hours
    amplitude: float  # meters
    phase: float  # radians

    @cached_property
    def complex_amplitude(self) -> complex:
        return self.amplitude * np.cos(self.phase) + 1j * self.amplitude * np.sin(self.phase)

    @cached_property
    def angular_frequency(self) -> float:
        if self.period == 0:
            return 0
        return 2 * np.pi / (self.period * 60 * 60)

    @cached_property
    def frequency(self) -> float:
        if self.period == 0:
            return 0
        return 1 / (self.period * 60 * 60)

    @classmethod
    def from_complex_amplitude(cls, complex_amplitude, period) -> Self:
        amplitude = np.sign(np.real(complex_amplitude)) * np.abs(complex_amplitude)
        phase = np.arctan(np.imag(complex_amplitude), np.real(complex_amplitude))
        return cls(period, amplitude, phase)


def height_from_tide(tide: TideFrequency, t: float | np.ndarray) -> float | np.ndarray:
    return np.real(tide.complex_amplitude * np.exp(1j * tide.angular_frequency * t))
