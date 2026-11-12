"""Deterministic collection/update scheduling, independent of the simulator."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CollectionSchedule:
    fixed_steps: int = 16
    update_to_data_ratio: float | None = None

    def __post_init__(self):
        if self.fixed_steps < 1:
            raise ValueError("fixed collection steps must be positive")
        if self.update_to_data_ratio is not None and (
                not math.isfinite(self.update_to_data_ratio) or not 0 < self.update_to_data_ratio <= 1):
            raise ValueError("update-to-data ratio must be in (0,1]")

    def steps(self, iteration):
        """Zero-based global iteration; 0.4 gives 2,3,2,3 steps, including on resume."""
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.update_to_data_ratio is None:
            return self.fixed_steps
        ratio = self.update_to_data_ratio
        return math.floor((iteration+1)/ratio+1e-9)-math.floor(iteration/ratio+1e-9)
