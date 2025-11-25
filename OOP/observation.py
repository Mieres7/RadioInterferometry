from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class Observation:

    experiment: int
    frecuency: float | np.ndarray = field(default_factory=lambda:np.ndarray([]))
    bandwidth: float  
    active_antennas: np.ndarray


