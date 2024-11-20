from enum import Enum


class BoundaryCondition(Enum):
    """Enum for the boundary conditions of the single channel tidal model"""

    TIDE = 1
    INF_BOUNDED = 2
    ZERO_TIDAL_FLOW = 3
